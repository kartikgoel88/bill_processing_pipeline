# Expected outputs for self-test

This directory holds the **expected** normalized outputs used when comparing a pipeline run to a baseline.

## Files

- **batch_output.json** – Normalized batch results (one record per bill: employee_id, expense_type, amount, month, decision, approved_amount, etc.). No trace_id or timing so comparison is stable.

## Usage

1. **First time / after changing test_input or desired baseline**  
   Run the pipeline, then save current output as expected:
   ```bash
   python scripts/run_self_test.py --update-expected
   ```
   This runs the pipeline with `test_input` → `test_output`, then writes the normalized `batch_output.json` into `test_expected/`.

2. **Run self-test (run + compare)**  
   ```bash
   python scripts/run_self_test.py
   ```
   Runs the pipeline with `test_input` → `test_output`, then compares `test_output/batch_output.json` to `test_expected/batch_output.json`. Exits 0 if they match, 1 and prints diffs if not.

3. **Compare only (no run)**  
   ```bash
   python scripts/run_self_test.py --compare-only
   ```
   Compares existing `test_output/batch_output.json` to `test_expected/batch_output.json` without running the pipeline.

4. **Run only (no compare)**  
   ```bash
   python scripts/run_self_test.py --run-only
   ```
   Runs the pipeline only; no comparison.

## Comparison rules

- Records are sorted by (employee_id, month, expense_type, amount) before comparison.
- Compared fields: employee_id, expense_type, month, decision, amount, approved_amount (numeric tolerance 0.01).
- trace_id, processing_time_sec, ocr_confidence, and raw OCR text are ignored.
