"""
Self-test pipeline: compare test_output to test_expected without running the pipeline.
Run the full self-test (run + compare) via: python scripts/run_self_test.py
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_self_test.py"
TEST_OUTPUT = ROOT / "test_output"
TEST_EXPECTED = ROOT / "test_expected"


def test_self_test_script_exists() -> None:
    assert SCRIPT.exists(), f"Self-test script not found: {SCRIPT}"


def test_expected_dir_has_batch_output() -> None:
    """Expected baseline must exist for compare-only to succeed."""
    path = TEST_EXPECTED / "batch_output.json"
    if not path.exists():
        pytest.skip("test_expected/batch_output.json not found; run: python scripts/run_self_test.py --update-expected")


def test_compare_only_matches_when_output_equals_expected() -> None:
    """Run compare-only self-test: test_output vs test_expected. Passes when they match."""
    if not (TEST_EXPECTED / "batch_output.json").exists():
        pytest.skip("test_expected/batch_output.json not found")
    if not (TEST_OUTPUT / "batch_output.json").exists():
        pytest.skip("test_output/batch_output.json not found (run pipeline first)")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--compare-only"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Self-test compare failed. stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
