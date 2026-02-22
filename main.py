"""
Production-grade Employee Reimbursement Processing Pipeline — entry point.

Two modes:
  1. Policy allowances (PDF → OCR → LLM → JSON): --policy-allowances
  2. Bill processing (batch): default (requires policy_allowances.json)

Usage:
  python main.py [--policy-allowances] [--input ROOT] [--dry-run] [--workers N] [--output-dir DIR]

- Policy allowances (--policy-allowances): PDF → OCR → LLM → JSON (client_location_allowance, fuel 2W/4W, meal_allowance) → policy_allowances.json.
- Bill processing: requires policy_allowances.json in output-dir (run --policy-allowances first). Traverses ROOT (expense_type/employee_id/bills)
  → OCR → validate → LLM fallback if needed → decision LLM (policy JSON + bill JSON) → FinalOutput per bill.
- Output: batch_output.json, decisions.csv, audit_trail.log.
- --dry-run: only count bills, no OCR/LLM.
- --workers N: use N processes for folder-level parallelism (default 1).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from commons.config import PipelineConfig, DEFAULT_BASE_URL
from llm.llm_client import OpenAICompatibleClient
from policy.policy_pipeline_allowances import run_policy_allowances_pipeline
from commons.orchestrator import ReimbursementOrchestrator
from commons.exceptions import PolicyExtractionError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


def create_llm_client(config: PipelineConfig) -> OpenAICompatibleClient:
    """Build default LLM client (single endpoint). Use create_vision_client / create_decision_client for split endpoints."""
    return OpenAICompatibleClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        default_model=config.llm_decision_model,
    )


def create_vision_client(config: PipelineConfig) -> OpenAICompatibleClient:
    """Build LLM client for vision (bill image extraction). Uses LLM_VISION_BASE_URL / LLM_VISION_API_KEY if set."""
    base_url = config.llm_vision_base_url if config.llm_vision_base_url else config.llm_base_url
    api_key = config.llm_vision_api_key if config.llm_vision_api_key is not None else config.llm_api_key
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key or "",
        default_model=config.llm_vision_model,
    )


def create_decision_client(config: PipelineConfig) -> OpenAICompatibleClient:
    """Build LLM client for decision (and policy extraction). Uses LLM_DECISION_BASE_URL / LLM_DECISION_API_KEY if set."""
    base_url = config.llm_decision_base_url if config.llm_decision_base_url else config.llm_base_url
    api_key = config.llm_decision_api_key if config.llm_decision_api_key is not None else config.llm_api_key
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key or "",
        default_model=config.llm_decision_model,
    )


# ---------------------------------------------------------------------------
# Policy (allowances only — load from policy_allowances.json)
# ---------------------------------------------------------------------------


def load_policy_allowances(output_dir: Path, config: PipelineConfig) -> tuple[dict[str, Any], str]:
    """Load policy_allowances.json. Raises SystemExit with message if missing."""
    path = output_dir / config.policy_allowances_output_path
    if not path.exists():
        print(
            f"Policy file not found: {path}. Run with --policy-allowances first to create it.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"Invalid policy file: {path} (not a JSON object).", file=sys.stderr)
            sys.exit(1)
        return (data, "allowances")
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to load policy from {path}: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Bill processing (batch)
# ---------------------------------------------------------------------------


def run_batch_single_process(
    input_root: Path,
    config: PipelineConfig,
    policy_json: dict[str, Any],
    policy_version_hash: str,
    vision_client: Any,
    decision_client: Any,
    audit_log_path: Path,
) -> list[dict[str, Any]]:
    """Run bill processing in current process; return list of FinalOutputSchema as dicts."""
    orch = ReimbursementOrchestrator(
        policy_json=policy_json,
        policy_version_hash=policy_version_hash,
        vision_client=vision_client,
        decision_client=decision_client,
        ocr_confidence_threshold=config.ocr_confidence_threshold,
        vision_model=config.llm_vision_model,
        decision_model=config.llm_decision_model,
        max_retries=config.llm_max_retries,
        retry_delay_sec=config.llm_retry_delay_sec,
        audit_log_path=audit_log_path,
        vision_extractor=config.vision_extractor,
        donut_model_id=config.donut_model_id,
        layoutlm_model_id=config.layoutlm_model_id,
        extraction_strategy=config.extraction_strategy,
        output_all_extractors=config.output_all_extractors,
    )
    results = orch.process_batch_from_folder(input_root, dry_run=config.dry_run)
    return [r.model_dump(mode="json") for r in results]


def run_batch_multiprocessing(
    input_root: Path,
    config: PipelineConfig,
    policy_json: dict[str, Any],
    policy_version_hash: str,
    vision_client: Any,
    decision_client: Any,
    audit_log_path: Path,
    max_workers: int,
) -> list[dict[str, Any]]:
    """
    Split by top-level (expense_type) folders, process each in a worker, merge results.
    Policy dict and hash are passed to workers (allowances or schema shape).
    """
    from bills.bill_folder_reader import iter_bills
    from multiprocessing import Pool

    root = Path(input_root)
    expense_folders = sorted({p.parent.parent for _, _, p in iter_bills(root)})
    if not expense_folders:
        expense_folders = [root]
    if max_workers < 2:
        return run_batch_single_process(
            root, config, policy_json, policy_version_hash,
            vision_client, decision_client, audit_log_path,
        )

    config_dict = _config_to_dict(config)
    paths_to_process = [str(f) for f in expense_folders[: max_workers * 2]] or [str(root)]

    def worker(
        folder_path: str,
        pol_json: dict[str, Any],
        pol_hash: str,
        cfg_dict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        cfg = PipelineConfig.from_env(**cfg_dict)
        v_client = create_vision_client(cfg)
        d_client = create_decision_client(cfg)
        return run_batch_single_process(
            Path(folder_path),
            cfg,
            pol_json,
            pol_hash,
            v_client,
            d_client,
            audit_log_path,
        )

    with Pool(processes=min(max_workers, len(paths_to_process))) as pool:
        result_lists = pool.starmap(
            worker,
            [(p, policy_json, policy_version_hash, config_dict) for p in paths_to_process],
        )
    merged: list[dict[str, Any]] = []
    for L in result_lists:
        merged.extend(L)
    return merged


def _config_to_dict(c: PipelineConfig) -> dict[str, Any]:
    """Serialize config for worker processes."""
    return {
        "input_root": c.input_root,
        "policy_pdf_path": c.policy_pdf_path,
        "dry_run": c.dry_run,
        "llm_base_url": c.llm_base_url,
        "llm_api_key": c.llm_api_key,
        "llm_vision_base_url": c.llm_vision_base_url,
        "llm_vision_api_key": c.llm_vision_api_key,
        "llm_decision_base_url": c.llm_decision_base_url,
        "llm_decision_api_key": c.llm_decision_api_key,
        "extraction_strategy": c.extraction_strategy,
        "output_all_extractors": c.output_all_extractors,
        "vision_extractor": c.vision_extractor,
        "donut_model_id": c.donut_model_id,
        "layoutlm_model_id": c.layoutlm_model_id,
        "llm_vision_model": c.llm_vision_model,
        "llm_decision_model": c.llm_decision_model,
        "ocr_confidence_threshold": c.ocr_confidence_threshold,
        "llm_max_retries": c.llm_max_retries,
        "llm_retry_delay_sec": c.llm_retry_delay_sec,
        "policy_cache_path": c.policy_cache_path,
        "policy_version_hash_path": c.policy_version_hash_path,
        "audit_log_path": c.audit_log_path,
    }


def save_batch_output(results: list[dict[str, Any]], path: Path) -> None:
    """Save list of FinalOutputSchema (as dicts) to JSON. Policy JSON (policy_used) is omitted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [{k: v for k, v in r.items() if k != "policy_used"} for r in results]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logging.getLogger(__name__).info("Saved batch output to %s", path)


def save_decisions_csv(results: list[dict[str, Any]], path: Path) -> None:
    """Export each bill decision to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    rows = []
    for r in results:
        decision = r.get("decision") or {}
        bill = r.get("structured_bill") or {}
        meta = r.get("metadata") or {}
        rows.append({
            "trace_id": r.get("trace_id", ""),
            "file": r.get("file", ""),
            "policy_version_hash": (r.get("policy_version_hash") or "")[:16],
            "employee_id": bill.get("employee_id", ""),
            "expense_type": bill.get("expense_type", ""),
            "amount": bill.get("amount", ""),
            "approved_amount": decision.get("approved_amount", ""),
            "month": bill.get("month", ""),
            "decision": decision.get("decision", ""),
            "confidence_score": decision.get("confidence_score", ""),
            "reasoning": decision.get("reasoning", ""),
            "violated_rules": "|".join(decision.get("violated_rules") or []),
            "extraction_source": r.get("extraction_source", ""),
            "ocr_confidence": meta.get("ocr_confidence"),
            "processing_time_sec": meta.get("processing_time_sec"),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    logging.getLogger(__name__).info("Saved decisions CSV to %s", path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Employee Reimbursement Processing: policy ingestion + bill batch (rules from Policy PDF)",
    )
    parser.add_argument(
        "--policy-allowances",
        action="store_true",
        help="Run PDF → OCR → LLM → JSON with allowances shape (client_location_allowance, fuel 2W/4W, meal_allowance); write policy_allowances.json",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Root folder for bills (expense_type/employee_id/files). Default: INPUT_ROOT or test_input",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count bills; no OCR or LLM",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for outputs (default: OUTPUT_DIR from env or test_output)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.input is not None:
        overrides["input_root"] = args.input
    if args.dry_run:
        overrides["dry_run"] = True
    if args.workers is not None:
        overrides["max_workers"] = args.workers
    if args.log_level:
        overrides["log_level"] = args.log_level

    config = PipelineConfig.from_env(**overrides)
    setup_logging(config.log_level)
    log = logging.getLogger(__name__)

    out_dir = Path(args.output_dir or config.output_dir)
    audit_path = out_dir / config.audit_log_path
    out_json = out_dir / config.output_json_path
    out_csv = out_dir / config.output_csv_path

    # Step 1: Build vision and decision clients (can use same or different endpoints)
    vision_client = create_vision_client(config)
    decision_client = create_decision_client(config)
    if args.policy_allowances:
        try:
            prompt_path = Path(config.policy_allowances_prompt_path)
            if not prompt_path.is_absolute():
                prompt_path = Path.cwd() / prompt_path
            out_path = out_dir / config.policy_allowances_output_path
            run_policy_allowances_pipeline(
                config.policy_pdf_path,
                decision_client,
                output_json_path=out_path,
                system_prompt_path=prompt_path if prompt_path.exists() else None,
                model=config.llm_decision_model,
                max_retries=config.llm_max_retries,
            )
            log.info("Policy allowances pipeline complete. Output: %s", out_path)
        except (PolicyExtractionError, FileNotFoundError) as e:
            log.error("Policy allowances pipeline failed: %s", e)
            return 1
        return 0

    # Step 2: Bill processing — requires policy_allowances.json (allowances pipeline only)
    policy_json, policy_version_hash = load_policy_allowances(out_dir, config)
    log.info("Using policy from %s for decision LLM", out_dir / config.policy_allowances_output_path)

    workers = args.workers if args.workers is not None else config.max_workers
    input_root = Path(args.input or config.input_root)

    if workers and workers > 1:
        results = run_batch_multiprocessing(
            input_root, config, policy_json, policy_version_hash,
            vision_client, decision_client, audit_path, workers,
        )
    else:
        results = run_batch_single_process(
            input_root, config, policy_json, policy_version_hash,
            vision_client, decision_client, audit_path,
        )

    save_batch_output(results, out_json)
    save_decisions_csv(results, out_csv)

    approved = sum(1 for r in results if (r.get("decision") or {}).get("decision") == "APPROVED")
    rejected = sum(1 for r in results if (r.get("decision") or {}).get("decision") == "REJECTED")
    review = sum(1 for r in results if (r.get("decision") or {}).get("decision") == "NEEDS_REVIEW")

    print("Batch complete.")
    print(f"  total_processed: {len(results)}")
    print(f"  approved: {approved}")
    print(f"  rejected: {rejected}")
    print(f"  needs_review: {review}")
    print(f"  output: {out_json}, {out_csv}")
    print(f"  audit_log: {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
