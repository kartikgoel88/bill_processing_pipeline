#!/usr/bin/env python3
"""
Self-test pipeline: run the bill processing pipeline with test_input and compare
outputs against test_expected. Use --update-expected to refresh expected from current run.

Usage:
  python scripts/run_self_test.py                    # run pipeline, then compare to test_expected
  python scripts/run_self_test.py --update-expected # run pipeline, write outputs to test_expected
  python scripts/run_self_test.py --run-only        # run pipeline only (no comparison)
  python scripts/run_self_test.py --compare-only    # compare existing test_output to test_expected (no run)

Input layout:
  - Full: input_root/expense_type/employee_id/*.pdf  (e.g. test_input/meal/emp_id/file.pdf)
  - Single expense: input_root/expense_type/*.pdf    (e.g. test_input/commute/*.pdf → employee_id "unknown")
After changing test input or fixing extraction/validation, run with --update-expected to refresh
test_expected/batch_output.json so the self-test passes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Project root = parent of scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TEST_INPUT = PROJECT_ROOT / "test_input"
TEST_OUTPUT = PROJECT_ROOT / "test_output"
TEST_EXPECTED = PROJECT_ROOT / "test_expected"
BATCH_JSON = "batch_output.json"


def _normalize_record(r: dict) -> dict:
    """Produce a comparable record (no trace_id, timing, raw OCR). Includes file name for tracking."""
    bill = r.get("structured_bill") or {}
    decision = r.get("decision") or {}
    return {
        "file": r.get("file", ""),
        "employee_id": bill.get("employee_id", ""),
        "expense_type": bill.get("expense_type", ""),
        "amount": bill.get("amount"),
        "month": bill.get("month", ""),
        "bill_date": str(bill.get("bill_date") or ""),
        "vendor_name": (bill.get("vendor_name") or "").strip(),
        "decision": (decision.get("decision") or "").strip().upper(),
        "approved_amount": decision.get("approved_amount"),
    }


def _sort_key(rec: dict) -> tuple:
    """Sort by bill identity; use file then amount so order is stable and trackable."""
    amt = rec.get("amount")
    amt_val = float(amt) if amt is not None else 0.0
    return (
        rec.get("employee_id", ""),
        rec.get("month", ""),
        rec.get("expense_type", ""),
        rec.get("file", ""),
        str(rec.get("bill_date") or ""),
        (rec.get("vendor_name") or "").strip(),
        amt_val,
    )


def load_and_normalize_batch(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    # Raw pipeline output has structured_bill + decision; expected may already be normalized (flat)
    normalized = []
    for r in data:
        if "structured_bill" in r and "decision" in r and isinstance(r.get("decision"), dict):
            normalized.append(_normalize_record(r))
        else:
            # Already normalized (e.g. test_expected)
            normalized.append({
                "file": r.get("file", ""),
                "employee_id": r.get("employee_id", ""),
                "expense_type": r.get("expense_type", ""),
                "amount": r.get("amount"),
                "month": r.get("month", ""),
                "bill_date": str(r.get("bill_date") or ""),
                "vendor_name": (r.get("vendor_name") or "").strip(),
                "decision": (str(r.get("decision") or "").strip().upper()),
                "approved_amount": r.get("approved_amount"),
            })
    normalized.sort(key=_sort_key)
    return normalized


def compare_records(actual: list[dict], expected: list[dict]) -> list[str]:
    """Compare normalized lists. Return list of diff messages (empty if match)."""
    diffs: list[str] = []
    if len(actual) != len(expected):
        diffs.append(f"Count mismatch: actual={len(actual)}, expected={len(expected)}")

    # Compare field-by-field for each index (after sort, order should align by employee/month/type/amount)
    for i, (a, e) in enumerate(zip(actual, expected)):
        prefix = f"[{i}] {a.get('file') or a.get('employee_id')}/{a.get('month')}/{a.get('expense_type')}"
        for key in ("file", "employee_id", "expense_type", "month", "decision"):
            av, ev = a.get(key), e.get(key)
            if str(av).strip() != str(ev).strip():
                diffs.append(f"{prefix} {key}: actual={av!r}, expected={ev!r}")
        for key in ("amount", "approved_amount"):
            av, ev = a.get(key), e.get(key)
            if av is None and ev is None:
                continue
            if av is None or ev is None:
                if av != ev:
                    diffs.append(f"{prefix} {key}: actual={av!r}, expected={ev!r}")
                continue
            try:
                af, ef = float(av), float(ev)
                if abs(af - ef) > 0.01:
                    diffs.append(f"{prefix} {key}: actual={af}, expected={ef}")
            except (TypeError, ValueError):
                if av != ev:
                    diffs.append(f"{prefix} {key}: actual={av!r}, expected={ev!r}")

    # If expected has more, report missing
    for i in range(len(actual), len(expected)):
        e = expected[i]
        diffs.append(f"Missing in actual: expected record {i} {e.get('employee_id')}/{e.get('month')}")

    return diffs


def run_pipeline(input_dir: Path, output_dir: Path) -> int:
    """Run main.py with test input/output. Policy JSON must exist at output_dir/policy_allowances.json."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--config", str(PROJECT_ROOT / "config.yaml"),
        "--input", str(input_dir),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-test: run pipeline with test_input, compare to test_expected")
    parser.add_argument("--input-dir", type=Path, default=TEST_INPUT, help="Input root (default: test_input)")
    parser.add_argument("--output-dir", type=Path, default=TEST_OUTPUT, help="Output dir for run (default: test_output)")
    parser.add_argument("--expected-dir", type=Path, default=TEST_EXPECTED, help="Expected outputs (default: test_expected)")
    parser.add_argument("--run-only", action="store_true", help="Only run pipeline, do not compare")
    parser.add_argument("--compare-only", action="store_true", help="Only compare output-dir to expected-dir (no run)")
    parser.add_argument("--update-expected", action="store_true", help="After run, write normalized output to expected-dir")
    args = parser.parse_args()

    if not args.compare_only:
        print("Running pipeline...")
        print(f"  input:  {args.input_dir}")
        print(f"  output: {args.output_dir}")
        if not args.input_dir.is_dir():
            print(f"Error: input dir not found: {args.input_dir}", file=sys.stderr)
            return 1
        policy_path = args.output_dir / "policy_allowances.json"
        if not policy_path.exists():
            print(f"Warning: {policy_path} not found. Create it (e.g. minimal policy JSON) for batch run.", file=sys.stderr)
        code = run_pipeline(args.input_dir, args.output_dir)
        if code != 0:
            print(f"Pipeline exited with code {code}", file=sys.stderr)
            return code
        print("Pipeline run finished.")

    actual_path = args.output_dir / BATCH_JSON
    actual = load_and_normalize_batch(actual_path)

    if args.update_expected:
        args.expected_dir.mkdir(parents=True, exist_ok=True)
        out_file = args.expected_dir / BATCH_JSON
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(actual, f, indent=2)
        print(f"Updated expected: {out_file} ({len(actual)} records)")
        return 0

    if args.run_only:
        print(f"Run complete. Output: {actual_path} ({len(actual)} records)")
        return 0

    expected_path = args.expected_dir / BATCH_JSON
    if not expected_path.exists():
        print(f"No expected file at {expected_path}. Run with --update-expected to create from current output.")
        return 1

    expected = load_and_normalize_batch(expected_path)
    diffs = compare_records(actual, expected)

    if not diffs:
        print("Self-test passed: output matches expected.")
        return 0

    print("Self-test failed: output differs from expected.", file=sys.stderr)
    for d in diffs:
        print(f"  {d}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
