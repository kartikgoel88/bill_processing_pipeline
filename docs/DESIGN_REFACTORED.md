# Refactored Bill Processing Pipeline — Design

This document describes the production-grade refactored architecture: clean layers, SOLID, dependency injection, and testability. **Old code and backward compatibility have been removed:** single entry point is `main.py` (config-driven pipeline); no legacy orchestrator, env-only config, or policy PDF pipeline.

## Folder Structure

```
/core
    interfaces.py   # Abstract interfaces (IOCRService, IVisionService, IDecisionService, IPostProcessingService, ILLMProvider)
    models.py      # DTOs: LLMResponse, ExtractionResult, BillResult, BatchMetrics
    schema.py      # Pydantic: LineItemSchema, ReimbursementSchema, OCRExtractionResult, DEFAULT_EXPENSE_TYPE
    exceptions.py  # BillProcessingError, OCRError, VisionExtractionError, DecisionError, PostProcessingError

/extraction
    ocr.py              # OCR: create_ocr_engine, run_engine_on_images (Tesseract/EasyOCR + preprocessors); image_io: PathImageReader/BytesImageReader for loading
    parser.py            # parse_structured_from_ocr, parse_llm_extraction, prompts
    numeric_validator.py # is_valid_amount (amount-in-context guardrails)
    discovery.py         # iter_bills(root) -> (expense_type, employee_id, file_path)

/services
    ocr_service.py           # IOCRService implementation (delegates to extraction.ocr)
    vision_service.py        # IVisionService; uses injected ILLMProvider
    decision_service.py      # IDecisionService; uses injected ILLMProvider
    post_processing_service.py  # IPostProcessingService (meal cap, etc.)

/providers
    base.py             # BaseLLMProvider(ILLMProvider): generate(), chat(), chat_vision()
    openai_provider.py  # OpenAIProvider
    ollama_provider.py  # OllamaProvider
    huggingface_provider.py  # HuggingFaceProvider
    factory.py          # create_provider(provider, base_url, model, ...)

/pipeline
    bill_pipeline.py    # BillProcessingPipeline.process(file_path) -> BillResult
    batch_processor.py  # BatchProcessor.process_batch(file_paths) -> (results, metrics)
    fallback.py         # ConfidenceFallbackStrategy (vision confidence < threshold -> OCR fallback)

/utils
    config.py   # AppConfig, load_config (YAML + env)
    logger.py   # get_logger, setup_logging
    retry.py    # with_retry (exponential backoff)
```

## Design Rules Applied

- **Interfaces**: Every external dependency (OCR, vision, decision LLM, post-processing) is behind an interface in `core/interfaces.py`. No service depends on a concrete LLM implementation.
- **Dependency injection**: `BillProcessingPipeline` receives `IOCRService`, `IVisionService`, `IDecisionService`, `IPostProcessingService` via constructor. It does not know which LLM is used.
- **No global state**: Config is passed or loaded once; no module-level clients.
- **No hardcoded model names**: All model names and URLs come from `AppConfig` (YAML + env).
- **Retry**: `utils/retry.py` provides `with_retry` with exponential backoff; services can use it or their own retry (e.g. decision service).
- **Custom exceptions**: `core/exceptions.py`; no generic `Exception` for business failures.
- **Structured data**: `core/models.py` uses dataclasses for `LLMResponse`, `BillResult`, `ExtractionResult`, `BatchMetrics`.

## LLM Abstraction

- **Interface**: `ILLMProvider` (in `core/interfaces.py`) defines `generate(prompt, **kwargs) -> LLMResponse` and `chat(messages, **kwargs) -> str`, `chat_vision(messages, **kwargs) -> str`.
- **Base**: `BaseLLMProvider` in `providers/base.py` implements the abstract base; concrete providers implement `generate()` and `chat()`.
- **Implementations**: `OpenAIProvider`, `OllamaProvider`, `HuggingFaceProvider`. New providers: add a class in `providers/` and register in `providers/factory.py` `create_provider()`.

## Pipeline Design

- **Single public API**: `BillProcessingPipeline.process(file_path: str | Path) -> BillResult`.
- **Flow**:  
  1. OCR (for fusion or fallback)  
  2. Vision extraction (injected `IVisionService`, which uses injected `ILLMProvider`)  
  3. If vision confidence < threshold and fallback enabled → replace with OCR-only extraction  
  4. If critical validation fails → auto REJECTED; else call decision service  
  5. Post-process (e.g. meal cap)  
  6. Return `BillResult`.

- **Batch**: `BatchProcessor(pipeline).process_batch(file_paths)` runs `pipeline.process()` per file, collects metrics (total_processed, approved_count, rejected_count, needs_review_count, failed_count, total_time_sec), does not duplicate pipeline logic.

## Fallback Strategy

- **Pattern**: `ConfidenceFallbackStrategy` in `pipeline/fallback.py`. When vision extraction confidence < configurable threshold, the pipeline uses OCR-only extraction and sets `extraction_source` to `ocr_fallback`.
- **Config**: `extraction.fallback_enabled` and `extraction.fallback_threshold` in `config.yaml` (or env).

## Testability

- **Mock LLMProvider**: In tests, use `unittest.mock.MagicMock(spec=ILLMProvider)` and set `mock.chat.return_value = '...'`, `mock.chat_vision.return_value = '...'`.
- **Fake services**: Implement `IOCRService`, `IVisionService`, `IDecisionService`, `IPostProcessingService` with fixed return values (see `tests/test_bill_pipeline_refactored.py`: `FakeOCRService`, `FakeVisionService`, `FakeDecisionService`, `FakePostProcessingService`).
- **Unit test pipeline**: Build `BillProcessingPipeline` with fakes; call `process(path)` and assert on `BillResult` fields. No real LLM or OCR.
- **Unit test batch**: Use same fakes; `BatchProcessor(pipeline).process_batch([path1, path2])` and assert on `results` and `metrics`.

## Config Options and Possible Flows

All behavior is driven by `config.yaml` (and env overrides). Below: every option, allowed values, and the flows they produce.

### Config reference

| Section | Option | Allowed / default | Effect |
|--------|--------|-------------------|--------|
| (root) | `input_root` | path, default `test_input` | Default root for batch discovery when `--input` not given. |
| | `output_dir` | path, default `test_output` | Where batch/single output JSON is written. |
| | `policy_path` | path | Path to policy JSON (e.g. meal caps). Required. |
| | `audit_log_path` | path | Audit log file. |
| | `log_level` | INFO, DEBUG, WARNING, ERROR | Logging level. |
| | `dry_run` | bool, default false | Reserved for future (e.g. skip writes). |
| | `max_workers` | int, default 1 | Reserved for future parallel batch; currently batch is sequential. |
| **llm** | `provider` | `ollama` \| `openai` \| `huggingface` | LLM backend for **both** vision and decision. |
| | `base_url` | URL | Provider endpoint (e.g. `http://localhost:11434/v1` for Ollama). |
| | `api_key` | string | API key when required (e.g. OpenAI). |
| | `model` | string | Default/text model name (used if vision/decision not set). |
| | `vision_model` | string | Model for vision extraction (image → structured bill). |
| | `decision_model` | string | Model for approve/reject decision. |
| | `max_retries`, `retry_delay_sec`, `timeout_sec` | numbers | Retry and timeout for LLM calls. |
| **ocr** | `engine` | `tesseract` \| `easyocr` | OCR engine used by `OCRService` (path/bytes → text). |
| **extraction** | `strategy` | `fusion` \| `vision_first` | When OCR runs: fusion = run OCR + vision; vision_first = vision only, OCR only on fallback. |
| | `fallback_enabled` | bool, default true | If true, when vision confidence < `fallback_threshold` use OCR-only extraction. |
| | `fallback_threshold` | float, default 0.6 | Confidence below this triggers OCR fallback. |
| | `confidence_threshold` | float, default 0.6 | In config only; currently unused (fallback uses `fallback_threshold`). |
| | `vision_extractor` | e.g. `donut` | In config only; vision path is currently always vision LLM (prompt + image). |
| | `vision_backend` | e.g. `ollama` | In config only; actual backend is `llm.provider`. |

Env overrides (see `utils/config.load_config`): `INPUT_ROOT`, `OUTPUT_DIR`, `LOG_LEVEL`, `DRY_RUN`, `MAX_WORKERS`, `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_VISION_MODEL`, `LLM_DECISION_MODEL`, `LLM_API_KEY`, `OCR_ENGINE`.

---

### Run modes (CLI)

- **Single file**  
  `python main.py --config config.yaml --file path/to/bill.pdf`  
  Runs the pipeline once for that file. Output: `output_dir/single_output.json`.

- **Batch**  
  `python main.py --config config.yaml [--input ROOT]`  
  Discovers bills under `--input` (or `config.input_root`) via `extraction.discovery.iter_bills` (structure: `root/expense_type/employee_id/*.pdf|...`). Runs `pipeline.process()` per file; output: `output_dir/batch_output.json` and metrics (approved/rejected/needs_review/failed counts, total time).

Optional: `--output-dir DIR`, `--log-level LEVEL`.

---

### Extraction flows (per file)

Each file is processed by the same pipeline; the path depends on **extraction.strategy**, **extraction.fallback_enabled**, and **extraction.fallback_threshold**.

#### 1. Fusion (`strategy: fusion`)

1. **OCR** runs first → raw text + confidence (engine from **ocr.engine**).
2. **Vision** runs on image bytes (first page for PDF) → structured bill + confidence (model from **llm.vision_model**, provider from **llm.provider**).
3. If **fallback_enabled** and vision confidence **< fallback_threshold** → replace extraction with **OCR-only** (re-parse OCR text to structured bill; `extraction_source = "ocr_fallback"`).
4. **Decision** on structured bill (model from **llm.decision_model**) → approve/reject/needs_review.
5. **Post-process** (e.g. meal cap) → final decision and amount.

So with fusion you always get OCR + vision; fallback only swaps which extraction result is used for decision.

#### 2. Vision-first (`strategy: vision_first`)

1. **OCR is not run** initially.
2. **Vision** runs on image bytes → structured bill + confidence.
3. If **fallback_enabled** and vision confidence **< fallback_threshold** → run **OCR** and use OCR-only extraction (`extraction_source = "ocr_fallback"`).
4. If vision fails (exception) → same OCR fallback.
5. **Decision** and **post-process** as above.

So with vision_first, OCR runs only when falling back (low confidence or vision error).

#### 3. Fallback disabled (`fallback_enabled: false`)

- **Fusion**: OCR + vision still run; if vision confidence is low, the low-confidence vision result is still used (no OCR fallback).
- **Vision-first**: Only vision runs; if confidence is low or vision fails, that (or error) is still used—no OCR.

#### 4. Critical validation

If the chosen extraction (vision or ocr_fallback) has invalid critical fields (e.g. amount ≤ 0 or missing month), the pipeline skips the decision LLM and returns **REJECTED** with reason “Critical validation failed”.

---

### OCR engine (`ocr.engine`)

- **tesseract** (default): Tesseract via `extraction.ocr` (with configurable preprocessing).
- **easyocr**: EasyOCR via `extraction.ocr`.

Used whenever OCR runs: in fusion (step 1) or in fallback (vision_first or fusion). **OCRService** is built with `config.ocr.engine` in `main.py`.

---

### LLM provider (`llm.provider`)

- **ollama**: Local Ollama; **base_url** typically `http://localhost:11434/v1`.
- **openai**: OpenAI API; **base_url** and **api_key** used.
- **huggingface**: Hugging Face inference; **base_url** and **api_key** as needed.

Same provider is used for:
- **Vision**: `vision_model` + `chat_vision()` (image + prompt → structured bill text).
- **Decision**: `decision_model` + `generate()` (structured bill + policy → approve/reject).

So changing `llm.provider` (and optionally `base_url`, `api_key`, `vision_model`, `decision_model`) switches the entire stack (vision + decision) to that backend.

---

### Flow summary matrix

| strategy     | fallback_enabled | When OCR runs              | When vision runs | Result used when vision low conf |
|-------------|------------------|----------------------------|------------------|-----------------------------------|
| fusion      | true             | Always (step 1)             | Always           | OCR-only (fallback)               |
| fusion      | false            | Always (step 1)             | Always           | Vision (no fallback)              |
| vision_first| true             | Only on fallback or error   | Always           | OCR-only (fallback)               |
| vision_first| false            | Never                       | Always           | Vision (no fallback)              |

---

## Example Usage

- **Config**: Copy `config.yaml` and set `llm.provider`, `llm.base_url`, `llm.model`, etc. Override with env: `LLM_PROVIDER`, `LLM_BASE_URL`, `INPUT_ROOT`, `OUTPUT_DIR`, etc.
- **Single file**: `python main.py --config config.yaml --file path/to/bill.pdf`
- **Batch**: `python main.py --config config.yaml --input test_input`
- **Policy**: Ensure policy JSON exists at `config.policy_path` (e.g. `test_output/policy_allowances.json`).

## Running Tests

```bash
pip install -e ".[dev]"   # or uv sync with dev
pytest tests/test_bill_pipeline_refactored.py -v
```

Tests use fake services and minimal image files; no live LLM or OCR required for the refactored pipeline tests.
