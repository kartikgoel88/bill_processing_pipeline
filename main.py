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
import json
import sys
from pathlib import Path

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
    vision_llm = create_provider(
        llm_config.provider,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        model=llm_config.vision_model,
        timeout_sec=llm_config.timeout_sec,
    )
    decision_llm = create_provider(
        llm_config.provider,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        model=llm_config.decision_model,
        timeout_sec=llm_config.timeout_sec,
    )
    vision = VisionService(
        vision_llm,
        model=llm_config.vision_model,
        max_retries=llm_config.max_retries,
        retry_delay_sec=llm_config.retry_delay_sec,
    )
    decision = DecisionService(
        decision_llm,
        model=llm_config.decision_model,
        max_retries=llm_config.max_retries,
        retry_delay_sec=llm_config.retry_delay_sec,
    )
    ocr = OCRService(engine=config.ocr.engine)
    post = PostProcessingService()
    ext = config.extraction
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
    )


def _discover_bills(root: Path) -> list[Path]:
    from extraction.discovery import iter_bills
    return [p for _, _, p in iter_bills(root)]


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
    files = _discover_bills(input_root)
    if not files:
        logger.warning("No bills found under %s", input_root)
        return 0
    batch = BatchProcessor(pipeline)
    results, metrics = batch.process_batch(files, stop_on_first_error=False)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_output.json"
    with open(out_path, "w") as f:
        json.dump(
            [
                {
                    "trace_id": r.trace_id,
                    "file": r.file_name,
                    "extraction_source": r.extraction_source,
                    "structured_bill": r.structured_bill,
                    "decision": r.decision,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    logger.info("Batch complete: %s", metrics.to_dict())
    logger.info("Output: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
