"""
Bill processing pipeline — entry point.
Config-driven, dependency-injected pipeline. No backward compatibility with old orchestrator.

Usage:
  python main.py [--config config.yaml] [--input ROOT] [--file PATH] [--output-dir DIR]
  --file: process single file; otherwise discover bills from --input and run batch.
  Policy JSON must exist at config.policy_path (e.g. policy_allowances.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Load .env before config so OPENAI_API_KEY / LLM_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.models import BillResult
from utils.config import load_config
from utils.logger import setup_logging, get_logger
from providers.factory import create_provider
from services.ocr_service import OCRService
from services.vision_service import VisionService
from services.decision_service import DecisionService
from services.post_processing_service import PostProcessingService
from pipeline.bill_pipeline import BillProcessingPipeline
from pipeline.batch_processor import BatchProcessor

logger = get_logger(__name__)


def _load_policy(path: Path) -> tuple[dict, str]:
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}. Provide policy JSON at config.policy_path.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Policy file must be a JSON object")
    return data, "v1"


def _build_pipeline(config, policy: dict, policy_hash: str) -> BillProcessingPipeline:
    llm_config = config.llm
    vision_provider = (llm_config.vision_provider or llm_config.provider).strip() or "ollama"
    decision_provider = (llm_config.decision_provider or llm_config.provider).strip() or "ollama"
    vision_url = (llm_config.vision_base_url or llm_config.base_url).strip()
    decision_url = (llm_config.decision_base_url or llm_config.base_url).strip()
    api_key = (llm_config.api_key or "").strip()
    if decision_provider == "openai" and not api_key:
        logger.error(
            "OpenAI is configured for the decision LLM but no API key is set. "
            "Set OPENAI_API_KEY or LLM_API_KEY in .env (or export in your shell)."
        )
        raise ValueError("OpenAI API key required for decision_provider=openai. Set OPENAI_API_KEY or LLM_API_KEY.")
    if vision_provider == "openai" and not api_key:
        logger.error(
            "OpenAI is configured for the vision LLM but no API key is set. "
            "Set OPENAI_API_KEY or LLM_API_KEY in .env (or export in your shell)."
        )
        raise ValueError("OpenAI API key required for vision_provider=openai. Set OPENAI_API_KEY or LLM_API_KEY.")
    vision_llm = create_provider(
        vision_provider,
        base_url=vision_url,
        api_key=api_key,
        model=llm_config.vision_model,
        timeout_sec=llm_config.timeout_sec,
    )
    decision_llm = create_provider(
        decision_provider,
        base_url=decision_url,
        api_key=api_key,
        model=llm_config.decision_model,
        timeout_sec=llm_config.timeout_sec,
    )
    ext = config.extraction
    vision = VisionService(
        vision_llm,
        model=llm_config.vision_model,
        max_retries=llm_config.max_retries,
        retry_delay_sec=llm_config.retry_delay_sec,
        json_mode=ext.extraction_json_mode,
        reasoning_fallback=ext.extraction_reasoning_fallback,
    )
    decision = DecisionService(
        decision_llm,
        model=llm_config.decision_model,
        max_retries=llm_config.max_retries,
        retry_delay_sec=llm_config.retry_delay_sec,
    )
    ocr = OCRService(engine=config.ocr.engine, dpi=config.ocr.dpi)
    post = PostProcessingService()
    return BillProcessingPipeline(
        ocr,
        vision,
        decision,
        post,
        policy=policy,
        policy_version_hash=policy_hash,
        fallback_threshold=ext.fallback_threshold,
        fallback_enabled=ext.fallback_enabled,
        extraction_strategy=ext.strategy,
        vision_if_ocr_below=ext.confidence_threshold,
        ocr_extraction=ext.ocr_extraction,
    )


def _discover_bills(root: Path) -> list[tuple[str, Path, str, str]]:
    """
    Discover bill files under root. Returns list of (source_folder, path, expense_type, employee_id).
    Deduplicates by (source_folder, filename), keeping the first occurrence.
    source_folder is the path relative to root (e.g. 'commute/kartik').
    """
    from extraction.discovery import iter_bills
    root_resolved = root.resolve()
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, Path, str, str]] = []
    for expense_type, employee_id, path in iter_bills(root):
        path = path.resolve()
        try:
            folder = str(path.parent.relative_to(root_resolved))
        except ValueError:
            folder = path.parent.name or "."
        key = (folder, path.name)
        if key in seen:
            logger.debug("Skipping duplicate in folder %s: %s", folder, path.name)
            continue
        seen.add(key)
        out.append((folder, path, (expense_type or "meal").strip(), (employee_id or "").strip()))
    return out


DECISIONS_CSV_COLUMNS = [
    "source_folder", "trace_id", "file", "policy_version_hash", "employee_id", "expense_type",
    "amount", "approved_amount", "month", "decision", "confidence_score",
    "reasoning", "violated_rules", "extraction_source", "ocr_confidence",
    "processing_time_sec",
]


def _bill_result_to_csv_row(r: BillResult, source_folder: str = "") -> dict[str, str]:
    """Map BillResult to a flat dict for decisions CSV."""
    sb = r.structured_bill
    dec = r.decision
    meta = r.metadata or {}
    vr = dec.get("violated_rules") or []
    violated = ",".join(str(x) for x in vr) if isinstance(vr, list) else str(vr)
    return {
        "source_folder": source_folder,
        "trace_id": r.trace_id,
        "file": r.file_name,
        "policy_version_hash": r.policy_version_hash or "",
        "employee_id": str(sb.get("employee_id", "")),
        "expense_type": str(sb.get("expense_type", "")),
        "amount": str(sb.get("amount", "")),
        "approved_amount": str(dec.get("approved_amount") if dec.get("approved_amount") is not None else ""),
        "month": str(sb.get("month", "")),
        "decision": str(dec.get("decision", "")),
        "confidence_score": str(dec.get("confidence_score", "")),
        "reasoning": str(dec.get("reasoning", "")),
        "violated_rules": violated,
        "extraction_source": r.extraction_source or "",
        "ocr_confidence": str(meta.get("ocr_confidence", "")),
        "processing_time_sec": str(meta.get("processing_time_sec", "")),
    }


def _write_decisions_csv(
    results: list[BillResult] | list[tuple[str, BillResult]],
    path: Path,
) -> None:
    """Overwrite decisions CSV with one row per bill result. Results may be (source_folder, BillResult) tuples."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DECISIONS_CSV_COLUMNS)
        writer.writeheader()
        for item in results:
            if isinstance(item, tuple):
                folder, r = item
                writer.writerow(_bill_result_to_csv_row(r, source_folder=folder))
            else:
                writer.writerow(_bill_result_to_csv_row(item))


def main() -> int:
    parser = argparse.ArgumentParser(description="Bill processing pipeline (config + DI)")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--input", "-i", default=None, help="Input root for batch")
    parser.add_argument("--file", "-f", default=None, help="Single file to process")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.log_level:
        config = config.with_overrides(log_level=args.log_level)
    if args.output_dir:
        config = config.with_overrides(output_dir=args.output_dir)
    setup_logging(config.log_level)

    policy_path = Path(config.policy_path)
    if not policy_path.is_absolute():
        policy_path = Path(config.output_dir) / policy_path.name
    try:
        policy_json, policy_hash = _load_policy(policy_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    pipeline = _build_pipeline(config, policy_json, policy_hash)

    if args.file:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "single_output.json"
        file_path = Path(args.file)
        # Multi-page PDF: extract one bill per page via process_multi
        if file_path.suffix.lower() == ".pdf":
            results = pipeline.process_multi(args.file)
            with open(out_path, "w") as f:
                json.dump({
                    "file": file_path.name,
                    "trace_id": results[0].trace_id if results else "",
                    "bill_count": len(results),
                    "bills": [
                        {
                            "extraction_source": r.extraction_source,
                            "structured_bill": r.structured_bill,
                            "decision": r.decision,
                            "metadata": r.metadata,
                            "ocr_extracted": r.ocr_extracted,
                        }
                        for r in results
                    ],
                }, f, indent=2)
            logger.info("Multi-page PDF: %s -> %s bills -> %s", args.file, len(results), out_path)
        else:
            results = pipeline.process(args.file)
            with open(out_path, "w") as f:
                json.dump({
                    "file": file_path.name,
                    "trace_id": results[0].trace_id if results else "",
                    "bill_count": len(results),
                    "bills": [
                        {
                            "extraction_source": r.extraction_source,
                            "structured_bill": r.structured_bill,
                            "decision": r.decision,
                            "metadata": r.metadata,
                            "ocr_extracted": r.ocr_extracted,
                        }
                        for r in results
                    ],
                }, f, indent=2)
            logger.info("Single file result: %s -> %s bills -> %s", args.file, len(results), out_path)
        return 0

    input_root = Path(args.input or config.input_root)
    if not input_root.is_dir():
        logger.error("Input root is not a directory: %s", input_root)
        return 1
    files_with_folders = _discover_bills(input_root)
    if not files_with_folders:
        logger.warning("No bills found under %s", input_root)
        return 0
    folders_in_batch = sorted({item[0] for item in files_with_folders})
    logger.info(
        "Batch: %s files from %s folder(s): %s",
        len(files_with_folders),
        len(folders_in_batch),
        ", ".join(folders_in_batch) if folders_in_batch else ".",
    )
    batch = BatchProcessor(pipeline, max_workers=config.max_workers)
    results_with_folders, metrics = batch.process_batch(files_with_folders, stop_on_first_error=False)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_output.json"
    with open(out_path, "w") as f:
        json.dump(
            [
                {
                    "source_folder": folder,
                    "trace_id": r.trace_id,
                    "file": r.file_name,
                    "extraction_source": r.extraction_source,
                    "structured_bill": r.structured_bill,
                    "decision": r.decision,
                    "metadata": r.metadata,
                    "ocr_extracted": r.ocr_extracted,
                }
                for folder, r in results_with_folders
            ],
            f,
            indent=2,
        )
    decisions_path = out_dir / "decisions.csv"
    _write_decisions_csv(results_with_folders, decisions_path)
    logger.info("Batch complete: %s", metrics.to_dict())
    logger.info("Output: %s", out_path)
    logger.info("Decisions CSV: %s", decisions_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
