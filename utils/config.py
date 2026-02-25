"""
Configuration loader: YAML + env overrides.
No hardcoded model names; all from config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.exceptions import ConfigError

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def _coerce_bool(s: Any) -> bool:
    if isinstance(s, bool):
        return s
    return (str(s).strip().lower() in ("1", "true", "yes")) if s else False


def _coerce_float(s: Any) -> float:
    if s is None or s == "":
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(s: Any) -> int:
    if s is None or s == "":
        return 0
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class LLMConfig:
    """LLM endpoint and model configuration."""

    provider: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama3.2"
    vision_model: str = "llava"
    decision_model: str = "llama3.2"
    max_retries: int = 3
    retry_delay_sec: float = 2.0
    timeout_sec: int = 120


@dataclass(frozen=True)
class OCRConfig:
    """OCR engine selection."""

    engine: str = "tesseract"  # tesseract | easyocr


@dataclass(frozen=True)
class ExtractionConfig:
    """Extraction strategy and thresholds."""

    strategy: str = "fusion"  # ocr_only | fusion | vision_first
    vision_extractor: str = "donut"  # donut | layoutlm | qwen_vl | vision_llm
    vision_backend: str = "ollama"
    confidence_threshold: float = 0.6
    fallback_enabled: bool = True
    fallback_threshold: float = 0.6


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration. Built from YAML + env."""

    input_root: str = "test_input"
    output_dir: str = "test_output"
    policy_path: str = "test_output/policy_allowances.json"
    audit_log_path: str = "test_output/audit_trail.log"
    log_level: str = "INFO"
    dry_run: bool = False
    max_workers: int = 1
    llm: LLMConfig = field(default_factory=LLMConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    def with_overrides(self, **overrides: Any) -> AppConfig:
        """Return new config with replaced keys (only top-level and nested replaced whole)."""
        d: dict[str, Any] = {
            "input_root": self.input_root,
            "output_dir": self.output_dir,
            "policy_path": self.policy_path,
            "audit_log_path": self.audit_log_path,
            "log_level": self.log_level,
            "dry_run": self.dry_run,
            "max_workers": self.max_workers,
            "llm": self.llm,
            "ocr": self.ocr,
            "extraction": self.extraction,
        }
        for k, v in overrides.items():
            if k in d and v is not None:
                d[k] = v
        return AppConfig(
            input_root=d["input_root"],
            output_dir=d["output_dir"],
            policy_path=d["policy_path"],
            audit_log_path=d["audit_log_path"],
            log_level=d["log_level"],
            dry_run=_coerce_bool(d["dry_run"]),
            max_workers=_coerce_int(d["max_workers"]),
            llm=d["llm"] if isinstance(d["llm"], LLMConfig) else LLMConfig(**d["llm"]),
            ocr=d["ocr"] if isinstance(d["ocr"], OCRConfig) else OCRConfig(**d["ocr"]),
            extraction=d["extraction"] if isinstance(d["extraction"], ExtractionConfig) else ExtractionConfig(**d["extraction"]),
        )


def _env_override(key: str, default: Any, coerce: type | Any = str) -> Any:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    if coerce is bool:
        return _coerce_bool(raw)
    if coerce is float:
        return _coerce_float(raw)
    if coerce is int:
        return _coerce_int(raw)
    return str(raw).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if not YAML_AVAILABLE:
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _config_from_dict(data: dict[str, Any]) -> AppConfig:
    """Build AppConfig from flat/nested dict. Env overrides applied in load_config."""
    llm_data = data.get("llm") or {}
    ext_data = data.get("extraction") or {}
    return AppConfig(
        input_root=str(data.get("input_root", "test_input")),
        output_dir=str(data.get("output_dir", "test_output")),
        policy_path=str(data.get("policy_path", "test_output/policy_allowances.json")),
        audit_log_path=str(data.get("audit_log_path", "test_output/audit_trail.log")),
        log_level=str(data.get("log_level", "INFO")),
        dry_run=_coerce_bool(data.get("dry_run", False)),
        max_workers=_coerce_int(data.get("max_workers", 1)),
        llm=LLMConfig(
            provider=str(llm_data.get("provider", "ollama")),
            base_url=str(llm_data.get("base_url", "http://localhost:11434/v1")),
            api_key=str(llm_data.get("api_key", "")),
            model=str(llm_data.get("model", "llama3.2")),
            vision_model=str(llm_data.get("vision_model", "llava")),
            decision_model=str(llm_data.get("decision_model", "llama3.2")),
            max_retries=_coerce_int(llm_data.get("max_retries", 3)),
            retry_delay_sec=_coerce_float(llm_data.get("retry_delay_sec", 2.0)),
            timeout_sec=_coerce_int(llm_data.get("timeout_sec", 120)),
        ),
        ocr=(
            OCRConfig(engine=str(ocr_data.get("engine", "tesseract")))
            if isinstance(ocr_data := data.get("ocr"), dict)
            else OCRConfig()
        ),
        extraction=ExtractionConfig(
            strategy=str(ext_data.get("strategy", "fusion")),
            vision_extractor=str(ext_data.get("vision_extractor", "donut")),
            vision_backend=str(ext_data.get("vision_backend", "ollama")),
            confidence_threshold=_coerce_float(ext_data.get("confidence_threshold", 0.6)),
            fallback_enabled=_coerce_bool(ext_data.get("fallback_enabled", True)),
            fallback_threshold=_coerce_float(ext_data.get("fallback_threshold", 0.6)),
        ),
    )


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load config from YAML file, then apply env overrides.
    Env vars: INPUT_ROOT, OUTPUT_DIR, LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL, etc.
    """
    path = Path(config_path) if config_path else Path("config.yaml")
    data = _load_yaml(path)
    cfg = _config_from_dict(data)
    # Env overrides (single source for deployment)
    overrides: dict[str, Any] = {}
    if os.getenv("INPUT_ROOT"):
        overrides["input_root"] = os.getenv("INPUT_ROOT")
    if os.getenv("OUTPUT_DIR"):
        overrides["output_dir"] = os.getenv("OUTPUT_DIR")
    if os.getenv("LOG_LEVEL"):
        overrides["log_level"] = os.getenv("LOG_LEVEL")
    if os.getenv("DRY_RUN") is not None:
        overrides["dry_run"] = _coerce_bool(os.getenv("DRY_RUN"))
    if os.getenv("MAX_WORKERS") is not None:
        overrides["max_workers"] = _coerce_int(os.getenv("MAX_WORKERS"))
    provider = os.getenv("LLM_PROVIDER")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    if provider or base_url or model:
        llm = cfg.llm
        overrides["llm"] = LLMConfig(
            provider=provider or llm.provider,
            base_url=base_url or llm.base_url,
            api_key=os.getenv("LLM_API_KEY", llm.api_key),
            model=model or llm.model,
            vision_model=os.getenv("LLM_VISION_MODEL", llm.vision_model),
            decision_model=os.getenv("LLM_DECISION_MODEL", llm.decision_model),
            max_retries=llm.max_retries,
            retry_delay_sec=llm.retry_delay_sec,
            timeout_sec=llm.timeout_sec,
        )
    if os.getenv("OCR_ENGINE"):
        ocr = cfg.ocr
        overrides["ocr"] = OCRConfig(engine=os.getenv("OCR_ENGINE", ocr.engine).strip().lower())
    if not overrides:
        return cfg
    return cfg.with_overrides(**overrides)
