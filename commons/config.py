"""
Configuration for the reimbursement processing pipeline.
Loads from environment with sensible defaults; overridable for tests/Spark.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # optional: run with env vars only

DEFAULT_BASE_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration. Use dependency injection in components."""

    # Input
    input_root: str = field(
        default_factory=lambda: os.getenv("INPUT_ROOT", "test_input")
    )
    policy_pdf_path: str = field(
        default_factory=lambda: os.getenv("POLICY_PDF_PATH", "test_input/policy/company_policy.pdf")
    )
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes")
    )

    # LLM (default endpoint used for both vision and decision when not overridden)
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    # Optional: override per-capability so e.g. vision can be local (Ollama) and decision from OpenAI
    llm_vision_base_url: str | None = field(
        default_factory=lambda: os.getenv("LLM_VISION_BASE_URL") or None
    )
    llm_vision_api_key: str | None = field(
        default_factory=lambda: os.getenv("LLM_VISION_API_KEY") or None
    )
    llm_decision_base_url: str | None = field(
        default_factory=lambda: os.getenv("LLM_DECISION_BASE_URL") or None
    )
    llm_decision_api_key: str | None = field(
        default_factory=lambda: os.getenv("LLM_DECISION_API_KEY") or None
    )
    llm_vision_model: str = field(
        default_factory=lambda: os.getenv("LLM_VISION_MODEL", "llava")
    )
    # Vision/document extraction: "vision_llm" (default) | "donut" | "layoutlm"
    vision_extractor: str = field(
        default_factory=lambda: (os.getenv("VISION_EXTRACTOR", "donut").strip().lower())
    )
    donut_model_id: str = field(
        default_factory=lambda: os.getenv("DONUT_MODEL_ID", "naver-clova-ix/donut-base-finetuned-cord-v2"))
    layoutlm_model_id: str = field(
        default_factory=lambda: os.getenv("LAYOUTLM_MODEL_ID", "nielsr/layoutlmv3-finetuned-cord"))
    llm_decision_model: str = field(
        default_factory=lambda: os.getenv("LLM_DECISION_MODEL", "llama3.2")
    )

    # Thresholds
    ocr_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))
    )
    decision_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("DECISION_CONFIDENCE_THRESHOLD", "0.7"))
    )

    # Retry
    llm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3"))
    )
    llm_retry_delay_sec: float = field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_DELAY_SEC", "2.0"))
    )

    # Policy: general
    submission_window_days: int = field(
        default_factory=lambda: int(os.getenv("SUBMISSION_WINDOW_DAYS", "30"))
    )
    monthly_total_cap: float = field(
        default_factory=lambda: float(os.getenv("MONTHLY_TOTAL_CAP", "5000.0"))
    )

    # Policy: meal
    meal_max_daily_limit: float = field(
        default_factory=lambda: float(os.getenv("MEAL_MAX_DAILY_LIMIT", "50.0"))
    )
    meal_max_monthly_limit: float = field(
        default_factory=lambda: float(os.getenv("MEAL_MAX_MONTHLY_LIMIT", "500.0"))
    )

    # Policy: fuel
    fuel_max_per_km_rate: float = field(
        default_factory=lambda: float(os.getenv("FUEL_MAX_PER_KM_RATE", "0.5"))
    )
    fuel_monthly_cap: float = field(
        default_factory=lambda: float(os.getenv("FUEL_MONTHLY_CAP", "300.0"))
    )

    # Policy: commute
    commute_max_per_trip: float = field(
        default_factory=lambda: float(os.getenv("COMMUTE_MAX_PER_TRIP", "25.0"))
    )
    commute_monthly_cap: float = field(
        default_factory=lambda: float(os.getenv("COMMUTE_MONTHLY_CAP", "200.0"))
    )

    # Output
    output_dir: str = field(
        default_factory=lambda: os.getenv("OUTPUT_DIR", "test_output")
    )
    output_json_path: str = field(
        default_factory=lambda: os.getenv("OUTPUT_JSON_PATH", "batch_output.json")
    )
    output_csv_path: str = field(
        default_factory=lambda: os.getenv("OUTPUT_CSV_PATH", "decisions.csv")
    )
    output_dashboard_path: str = field(
        default_factory=lambda: os.getenv("OUTPUT_DASHBOARD_PATH", "dashboard_summary.json")
    )
    output_policy_path: str = field(
        default_factory=lambda: os.getenv("OUTPUT_POLICY_PATH", "policy_loaded.json")
    )
    policy_cache_path: str = field(
        default_factory=lambda: os.getenv("POLICY_CACHE_PATH", "policy_cache.json")
    )
    policy_version_hash_path: str = field(
        default_factory=lambda: os.getenv("POLICY_VERSION_HASH_PATH", "policy_version_hash.txt")
    )
    # Allowances pipeline (PDF → OCR → LLM → JSON, admin_billdesk shape)
    policy_allowances_output_path: str = field(
        default_factory=lambda: os.getenv("POLICY_ALLOWANCES_OUTPUT_PATH", "policy_allowances.json")
    )
    policy_allowances_prompt_path: str = field(
        default_factory=lambda: os.getenv("POLICY_ALLOWANCES_PROMPT_PATH", "prompts/system_prompt_policy_allowances.txt")
    )
    audit_log_path: str = field(
        default_factory=lambda: os.getenv("AUDIT_LOG_PATH", "audit_trail.log")
    )

    # Parallelism
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("MAX_WORKERS", "1"))  # 1 = sequential; >1 for multiprocessing
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    @classmethod
    def from_env(cls, **overrides: Optional[str | int | float | bool]) -> "PipelineConfig":
        """Build config from env with optional overrides (e.g. for tests)."""
        def _bool(s: str) -> bool:
            return (s or "").lower() in ("1", "true", "yes")

        defaults = {
            "input_root": os.getenv("INPUT_ROOT", "test_input"),
            "policy_pdf_path": os.getenv("POLICY_PDF_PATH", "test_input/policy/company_policy.pdf"),
            "dry_run": _bool(os.getenv("DRY_RUN", "false")),
            "llm_base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
            "llm_api_key": os.getenv("LLM_API_KEY", ""),
            "llm_vision_base_url": os.getenv("LLM_VISION_BASE_URL") or None,
            "llm_vision_api_key": os.getenv("LLM_VISION_API_KEY") or None,
            "llm_decision_base_url": os.getenv("LLM_DECISION_BASE_URL") or None,
            "llm_decision_api_key": os.getenv("LLM_DECISION_API_KEY") or None,
            "llm_vision_model": os.getenv("LLM_VISION_MODEL", "llava"),
            "vision_extractor": (os.getenv("VISION_EXTRACTOR", "donut") or "donut").strip().lower(),
            "donut_model_id": os.getenv("DONUT_MODEL_ID", "naver-clova-ix/donut-base-finetuned-cord-v2"),
            "layoutlm_model_id": os.getenv("LAYOUTLM_MODEL_ID", "nielsr/layoutlmv3-finetuned-cord"),
            "llm_decision_model": os.getenv("LLM_DECISION_MODEL", "llama3.2"),
            "ocr_confidence_threshold": float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6")),
            "decision_confidence_threshold": float(os.getenv("DECISION_CONFIDENCE_THRESHOLD", "0.7")),
            "llm_max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
            "llm_retry_delay_sec": float(os.getenv("LLM_RETRY_DELAY_SEC", "2.0")),
            "submission_window_days": int(os.getenv("SUBMISSION_WINDOW_DAYS", "30")),
            "monthly_total_cap": float(os.getenv("MONTHLY_TOTAL_CAP", "5000.0")),
            "meal_max_daily_limit": float(os.getenv("MEAL_MAX_DAILY_LIMIT", "50.0")),
            "meal_max_monthly_limit": float(os.getenv("MEAL_MAX_MONTHLY_LIMIT", "500.0")),
            "fuel_max_per_km_rate": float(os.getenv("FUEL_MAX_PER_KM_RATE", "0.5")),
            "fuel_monthly_cap": float(os.getenv("FUEL_MONTHLY_CAP", "300.0")),
            "commute_max_per_trip": float(os.getenv("COMMUTE_MAX_PER_TRIP", "25.0")),
            "commute_monthly_cap": float(os.getenv("COMMUTE_MONTHLY_CAP", "200.0")),
            "output_dir": os.getenv("OUTPUT_DIR", "test_output"),
            "output_json_path": os.getenv("OUTPUT_JSON_PATH", "batch_output.json"),
            "output_csv_path": os.getenv("OUTPUT_CSV_PATH", "decisions.csv"),
            "output_dashboard_path": os.getenv("OUTPUT_DASHBOARD_PATH", "dashboard_summary.json"),
            "output_policy_path": os.getenv("OUTPUT_POLICY_PATH", "policy_loaded.json"),
            "policy_cache_path": os.getenv("POLICY_CACHE_PATH", "policy_cache.json"),
            "policy_version_hash_path": os.getenv("POLICY_VERSION_HASH_PATH", "policy_version_hash.txt"),
            "policy_allowances_output_path": os.getenv("POLICY_ALLOWANCES_OUTPUT_PATH", "policy_allowances.json"),
            "policy_allowances_prompt_path": os.getenv("POLICY_ALLOWANCES_PROMPT_PATH", "prompts/system_prompt_policy_allowances.txt"),
            "audit_log_path": os.getenv("AUDIT_LOG_PATH", "audit_trail.log"),
            "max_workers": int(os.getenv("MAX_WORKERS", "1")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "dry_run": _bool(os.getenv("DRY_RUN", "false")),
        }
        defaults.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**defaults)
