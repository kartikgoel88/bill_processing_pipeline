"""Custom exceptions for the bill processing pipeline. No generic Exception usage."""

from __future__ import annotations


class BillProcessingError(Exception):
    """Base exception for pipeline failures."""

    def __init__(self, message: str, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or ""
        super().__init__(message)


class OCRError(BillProcessingError):
    """OCR extraction failed."""

    pass


class VisionExtractionError(BillProcessingError):
    """Vision/LLM extraction failed or returned invalid data."""

    pass


class DecisionError(BillProcessingError):
    """Decision LLM call or parse error."""

    pass


class PostProcessingError(BillProcessingError):
    """Post-processing (e.g. meal cap) failed."""

    pass


class ConfigError(BillProcessingError):
    """Invalid or missing configuration."""

    pass


class StructuredOutputError(DecisionError):
    """LLM output could not be parsed as valid JSON/schema."""

    pass


class BillExtractionError(BillProcessingError):
    """Bill extraction (OCR or vision) failed or returned invalid data."""

    pass
