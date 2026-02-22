# Employee Reimbursement Processing Pipeline

Production-grade pipeline for employee reimbursement processing. **Reimbursement rules are not hardcoded**; they are **dynamically extracted from a Policy PDF** and passed to the decision LLM.

## High-level flow

### Step 1: Policy allowances (run once per policy update)

1. Read policy PDF (pypdf or OCR fallback).
2. Use LLM to extract **allowances** JSON: `client_location_allowance`, `fuel_reimbursement_two_wheeler`, `fuel_reimbursement_four_wheeler`, `meal_allowance`.
3. Write **policy_allowances.json** (required for bill processing).

### Step 2: Bill processing (batch)

1. Read bills from folder: `root/{fuel,meal,commute}/{EMP001,EMP002,...}/` (expense_type / employee_id).
2. Extract bill via OCR.
3. Validate critical fields: **amount**, **month** (YYYY-MM). If missing/invalid → LLM extraction fallback.
4. If still invalid → **auto REJECTED** (no decision LLM).
5. Pass **structured policy JSON + bill JSON + expense type + employee monthly total** to decision LLM.
6. LLM returns decision: **APPROVED** | **REJECTED** | **NEEDS_REVIEW**.

## Flows you can achieve

| Flow | How to run | What it does |
|------|------------|--------------|
| **Policy allowances** | `python main.py --policy-allowances --output-dir ./out` | Read policy PDF → OCR (pypdf or fallback) → LLM extracts allowances JSON → write **policy_allowances.json**. Run once per policy update; required before bill processing. |
| **Bill processing (batch)** | `python main.py --input ROOT --output-dir ./out` | Traverse `ROOT/{fuel,meal,commute}/{employee_id}/` → OCR + vision extraction → fusion or vision_first → validate (amount, month) → LLM fallback if needed → auto-reject if invalid → decision LLM (policy + bill) → meal cap → **batch_output.json**, **decisions.csv**, **audit_trail.log**. |
| **Dry-run (count only)** | `python main.py --input ROOT --dry-run` | Same folder traversal; only counts bills. No OCR or LLM. |
| **Parallel by expense type** | `python main.py --input ROOT --workers N` | Same as batch; splits work by top-level folders (fuel/meal/commute) and runs up to N processes. |
| **Self-test (run + compare)** | `python scripts/run_self_test.py` | Run pipeline with **test_input** → compare **test_output** to **test_expected**; fails if diff. |
| **Self-test (run only)** | `python scripts/run_self_test.py --run-only` | Run pipeline with test_input; no comparison. |
| **Self-test (compare only)** | `python scripts/run_self_test.py --compare-only` | Compare existing test_output to test_expected; no run. |
| **Self-test (update baseline)** | `python scripts/run_self_test.py --update-expected` | Run pipeline, then save normalized output as **test_expected** for future comparisons. |
| **All-extractors comparison** | Set `OUTPUT_ALL_EXTRACTORS=1` + run batch | For each bill, additionally run Tesseract, EasyOCR, Donut, Qwen-VL; write results to **all_extractor_outputs** in batch_output.json. Decision still uses main flow (fusion or vision_first). |

**Single-file processing** is available programmatically via `ReimbursementOrchestrator.process_file(path, expense_type=..., employee_id=...)`; the CLI runs batch only.

## Folder structure

```
root/
    fuel/
        EMP001/
            bill1.pdf
        EMP002/
            bill2.jpg
    meal/
        EMP001/
            bill3.jpg
    commute/
        EMP003/
            bill4.pdf
```

- **Top-level** = expense_type (fuel | meal | commute)
- **Second-level** = employee_id
- **Files** = bills (PDF, JPG, PNG, etc.)

## Setup

```bash
uv sync
# or: pip install -e .
# Optional: .env with LLM_BASE_URL (default http://localhost:11434/v1), POLICY_PDF_PATH, INPUT_ROOT, etc.
```

## Usage

```bash
# 1) Create policy_allowances.json (PDF → OCR → LLM → allowances JSON). Run once per policy update.
python main.py --policy-allowances --output-dir ./out

# 2) Process bills (requires policy_allowances.json in output-dir; run step 1 first)
python main.py --input test_input --output-dir ./out

# Specify input and output
python main.py --input /path/to/root_folder --output-dir ./out

# Dry-run: only count bills
python main.py --input test_input --dry-run

# Multiprocessing (parallel by expense-type folder)
python main.py --input test_input --workers 4

# Log level
python main.py --input test_input --log-level DEBUG
```

## Outputs

- **batch_output.json** — List of final results per bill: `trace_id`, `policy_version_hash`, `extraction_source`, `structured_bill`, `policy_used`, `decision`, `metadata`.
- **decisions.csv** — One row per bill: trace_id, employee_id, expense_type, amount, month, decision, confidence_score, reasoning, violated_rules, etc.
- **audit_trail.log** — Append-only audit log (trace_id, stage, payload).
- **policy_allowances.json** — Policy allowances (from `--policy-allowances`); required for bill processing.

## Self-test pipeline

Run the pipeline with **test_input** and compare outputs to a baseline in **test_expected**:

```bash
# Run pipeline, then compare test_output to test_expected (fails if diff)
python scripts/run_self_test.py

# Only run the pipeline (no comparison)
python scripts/run_self_test.py --run-only

# Only compare existing test_output to test_expected (no run)
python scripts/run_self_test.py --compare-only

# Update baseline: run pipeline and save normalized output as expected
python scripts/run_self_test.py --update-expected
```

- **Input**: `test_input/` (expense_type/employee_id/bills).
- **Output**: `test_output/batch_output.json` (and decisions.csv).
- **Expected**: `test_expected/batch_output.json` (normalized: no trace_id/timing).
- Comparison is by (employee_id, month, expense_type, amount); fields compared: decision, amount, approved_amount, etc. See `test_expected/README.md`.

## Module structure

| Module | Role |
|--------|------|
| `policy/policy_reader.py` | Read policy PDF; pypdf + OCR fallback |
| `policy/policy_extractor_allowances.py` | LLM extraction → allowances JSON (client_location_allowance, fuel 2W/4W, meal_allowance) |
| `policy/policy_pipeline_allowances.py` | PDF → OCR → LLM → policy_allowances.json |
| `prompts/system_prompt_policy_allowances.txt` | System prompt for allowances shape |
| `bills/bill_folder_reader.py` | Traverse root → (expense_type, employee_id, path) |
| `bills/ocr_engine.py` | Tesseract or EasyOCR on PDF/images (set `OCR_ENGINE=easyocr`; install with `pip install .[ocr]`) |
| `bills/bill_extractor.py` | Parse OCR/LLM into ReimbursementSchema |
| `decision/validator.py` | Critical (amount, month) + validate_structured_bill |
| `llm_client.py` | OpenAI-compatible API; retry; vision |
| `decision_engine.py` | Decision LLM: policy JSON + bill JSON only (no hardcoded rules) |
| `orchestrator.py` | Load policy → OCR → validate → LLM fallback → decision → FinalOutput + audit |
| `main.py` | CLI: --policy-allowances, batch, multiprocessing, audit |

## Policy (allowances shape)

Policy is extracted via `--policy-allowances` and saved as **policy_allowances.json** with keys: `client_location_allowance`, `fuel_reimbursement_two_wheeler`, `fuel_reimbursement_four_wheeler`, `meal_allowance`. The decision LLM receives this JSON.

## Decision engine (no hardcoded rules)

The decision LLM receives:

1. **Policy allowances JSON** (from policy_allowances.json).
2. **Extracted bill JSON**.
3. **Expense type**.
4. **Employee's current monthly total**.

It returns strict JSON: `decision`, `confidence_score`, `reasoning`, `violated_rules`. Rules are **not** hardcoded in code; the LLM uses only the provided policy JSON.

## Configuration

- **Extraction strategy:** `EXTRACTION_STRATEGY` = `fusion` (default) or `vision_first`. With `vision_first`, Donut/LayoutLMv3/Qwen-VL/vision_llm is tried first; Tesseract (keyword proximity + bounding box + numeric validation) is used only as fallback.
- **Vision/document:** `VISION_EXTRACTOR` = `donut` | `layoutlm` | `vision_llm` | `qwen_vl`. For `qwen_vl`: set `VISION_BACKEND` = `ollama` (default; use API with `QWEN_VL_MODEL`), `huggingface` (run Qwen-VL locally; requires `pip install -e ".[document]"`), or `hf_api` (Hugging Face Inference API with `HF_TOKEN`; no local download).
- **LLM:** `LLM_BASE_URL` (default `http://localhost:11434/v1`), `LLM_API_KEY`, `LLM_VISION_MODEL`, `LLM_DECISION_MODEL`. Decision model supports e.g. `llama3.2`, `gpt-4o-mini`, Qwen3 (`qwen3:1.7b`, `qwen3:4b`).
- **Paths:** `POLICY_PDF_PATH`, `INPUT_ROOT`, `POLICY_ALLOWANCES_OUTPUT_PATH` (default `policy_allowances.json`), `AUDIT_LOG_PATH`, `OUTPUT_JSON_PATH`, `OUTPUT_CSV_PATH`.
- **Thresholds:** `OCR_CONFIDENCE_THRESHOLD`, `LLM_MAX_RETRIES`, `MAX_WORKERS`.
- **OCR:** `OCR_ENGINE` = `tesseract` (default) or `easyocr`. For EasyOCR install: `pip install .[ocr]`.
- **Compare all extractors:** `OUTPUT_ALL_EXTRACTORS=1` runs Tesseract, EasyOCR, Donut, and Qwen-VL on every bill and adds an `all_extractor_outputs` object to each result in `batch_output.json` (keys: `tesseract`, `easyocr`, `donut`, `qwen_vl`), so you can compare raw and structured outputs side by side.

## Extraction flow (complete)

**By default we do not use all extractors.** Only **one OCR** and **one vision/document** extractor run for the actual result. The decision LLM always receives a single fused or chosen extraction.

### What runs for the final result (decision)

| Config | What runs | Result used for decision |
|--------|-----------|---------------------------|
| **EXTRACTION_STRATEGY=fusion** (default) | 1. **One OCR** (Tesseract or EasyOCR per `OCR_ENGINE`) → raw text → `parse_structured_from_ocr` (keyword/bbox + numeric validation).<br>2. **One vision** (Donut, LayoutLMv3, vision_llm, or qwen_vl per `VISION_EXTRACTOR`).<br>3. **Fusion** of OCR + vision → validate. | Fused schema. |
| **EXTRACTION_STRATEGY=vision_first** | 1. **One vision** extractor (same as above) tried first.<br>2. If good (amount + month, confidence ≥ threshold) → use it.<br>3. Else **one OCR** (Tesseract/EasyOCR per `OCR_ENGINE`) as fallback → `parse_structured_from_ocr` → validate. | Vision schema or OCR fallback schema. |

So in normal runs you use **at most 2 extractors** for the result: one OCR engine and one vision/document engine. Which OCR engine is controlled by `OCR_ENGINE` (tesseract | easyocr). Which vision engine is controlled by `VISION_EXTRACTOR` (donut | layoutlm | vision_llm | qwen_vl).

### When all four run (comparison only)

Only when **OUTPUT_ALL_EXTRACTORS=1** do we **additionally** run all four:

1. **Tesseract** (OCR) → raw text + structured
2. **EasyOCR** (OCR) → raw text + structured (or error if not installed)
3. **Donut** (vision) → structured
4. **Qwen-VL** (vision LLM, model from `QWEN_VL_MODEL` or `LLM_VISION_MODEL`) → structured

Their outputs are written to **`all_extractor_outputs`** in each bill’s entry in `batch_output.json`. The **decision is still based on the main flow above** (fusion or vision_first), not on this comparison run. So:

- **Default:** 1 OCR + 1 vision (2 extractors) → fuse or choose → decision.
- **OUTPUT_ALL_EXTRACTORS=1:** same as above **plus** all four run for comparison; result in `all_extractor_outputs` only.

### Quick reference

| Extractors used for decision | When |
|------------------------------|------|
| Tesseract or EasyOCR (1) + Donut or LayoutLM or vision_llm or qwen_vl (1) | `EXTRACTION_STRATEGY=fusion` (default) |
| Vision (1) only, or OCR (1) fallback | `EXTRACTION_STRATEGY=vision_first` |
| Tesseract + EasyOCR + Donut + Qwen-VL (all 4) | Only when `OUTPUT_ALL_EXTRACTORS=1` (for comparison in JSON; decision still from main flow). |

## Programmatic usage

Use `main.py` and the package modules (`commons`, `bills`, `policy`, `decision`, `llm`) for policy ingestion, single-bill or batch processing, and the decision engine (policy JSON + bill JSON).

## Exception hierarchy

- `PolicyExtractionError` — Policy PDF extraction or schema validation failed.
- `BillExtractionError` — Bill extraction (OCR/LLM) failed.
- `ValidationError` — Critical or other validation failed.
- `DecisionError` — Decision LLM call or parse error.

All extend `BillProcessingError` (optional `trace_id`).
