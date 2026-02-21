"""Custom exceptions for the Employee Reimbursement Processing pipeline."""

from __future__ import annotations


class BillProcessingError(Exception):
    """Base exception for pipeline failures."""

    def __init__(self, message: str, trace_id: str | None = None) -> None:
        self.trace_id = trace_id
        super().__init__(message)


class PolicyExtractionError(BillProcessingError):
    """Policy PDF extraction or schema validation failed."""

    pass


class BillExtractionError(BillProcessingError):
    """Bill extraction (OCR or LLM) failed or returned invalid data."""

    pass


class ValidationError(BillProcessingError):
    """Structured validation failed (missing/invalid critical fields)."""

    pass


class DecisionError(BillProcessingError):
    """Decision engine LLM call or parse error."""

    pass


class StructuredOutputError(DecisionError):
    """LLM returned output that could not be parsed as valid JSON/schema."""

    pass
