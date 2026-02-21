"""
Pydantic schemas for Employee Reimbursement Processing.
Used across OCR, bill_extractor, validator, decision engine, and orchestrator.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Line item (bill line)
# ---------------------------------------------------------------------------


class LineItemSchema(BaseModel):
    """Single line item on the bill."""

    description: str = ""
    amount: float = 0.0
    quantity: int = 1
    code: str = ""

    @field_validator("amount")
    @classmethod
    def amount_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("amount must be >= 0")
        return v


# ---------------------------------------------------------------------------
# Reimbursement (structured extraction output)
# ---------------------------------------------------------------------------

EXPENSE_TYPES = ("fuel", "meal", "commute")


class ReimbursementSchema(BaseModel):
    """Structured reimbursement data from OCR or LLM extraction."""

    employee_id: str = ""
    expense_type: str = ""  # fuel | meal | commute
    amount: float = 0.0
    month: str = ""  # YYYY-MM
    bill_date: date | str = ""
    vendor_name: str = ""
    currency: str = "USD"
    category: str = ""
    line_items: list[LineItemSchema] = Field(default_factory=list)

    @field_validator("bill_date", mode="before")
    @classmethod
    def normalize_bill_date(cls, v: Any) -> date | str:
        if v is None or v == "":
            return ""
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            return v.strip()
        return str(v)

    @field_validator("expense_type")
    @classmethod
    def expense_type_lower(cls, v: str) -> str:
        return (v or "").strip().lower()

    @field_validator("month")
    @classmethod
    def month_stripped(cls, v: str) -> str:
        return (v or "").strip()


# ---------------------------------------------------------------------------
# OCR result
# ---------------------------------------------------------------------------


class OCRExtractionResult(BaseModel):
    """Result from OCR engine: raw text and confidence."""

    raw_text: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Decision (LLM final output)
# ---------------------------------------------------------------------------


class DecisionType(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class DecisionSchema(BaseModel):
    """Structured decision from the decision LLM."""

    decision: Literal["APPROVED", "REJECTED", "NEEDS_REVIEW"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    violated_rules: list[str] = Field(default_factory=list)

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, v: Any) -> str:
        if isinstance(v, str):
            u = v.upper().strip()
            if u in ("APPROVED", "REJECTED", "NEEDS_REVIEW"):
                return u
        raise ValueError("decision must be APPROVED, REJECTED, or NEEDS_REVIEW")


# ---------------------------------------------------------------------------
# Batch / aggregation
# ---------------------------------------------------------------------------


class BillDecisionRecord(BaseModel):
    """Single bill decision for aggregation."""

    file_path: str = ""
    amount: float = 0.0
    month: str = ""
    decision: str = ""
    confidence_score: float = 0.0
    reasoning: str = ""
    violated_rules: list[str] = Field(default_factory=list)
    extraction_source: str = "OCR"  # OCR | LLM
    ocr_confidence: float | None = None
    llm_confidence: float | None = None


class BatchResultSchema(BaseModel):
    """Aggregated result per employee per month."""

    employee_id: str = ""
    month: str = ""
    total_amount: float = 0.0
    decisions: list[BillDecisionRecord] = Field(default_factory=list)
    final_status: str = ""  # APPROVED | REJECTED | NEEDS_REVIEW


class BatchSummary(BaseModel):
    """Summary counts for the batch run."""

    total_processed: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    review_count: int = 0


class BatchOutput(BaseModel):
    """Final batch pipeline output."""

    trace_id: str = ""
    batch_summary: BatchSummary = Field(default_factory=BatchSummary)
    employee_results: list[BatchResultSchema] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final output per bill (production-grade pipeline)
# ---------------------------------------------------------------------------


class FinalOutputMetadata(BaseModel):
    """Metadata for a single bill processing result."""

    ocr_confidence: float | None = None
    llm_model: str = ""
    processing_time_sec: float = 0.0
    explainability_score: float | None = None


class FinalOutputSchema(BaseModel):
    """
    Final output format for each processed bill.
    trace_id, policy_version_hash, extraction_source, structured_bill, policy_used, decision, metadata, ocr_extracted.
    """

    trace_id: str = ""
    policy_version_hash: str = ""
    extraction_source: Literal["OCR", "LLM"] = "OCR"
    structured_bill: dict[str, Any] = Field(default_factory=dict)
    policy_used: dict[str, Any] = Field(default_factory=dict)
    decision: dict[str, Any] = Field(default_factory=dict)  # decision, confidence_score, reasoning, violated_rules
    metadata: FinalOutputMetadata = Field(default_factory=FinalOutputMetadata)
    ocr_extracted: dict[str, Any] = Field(
        default_factory=dict,
        description="OCR extraction result: raw_text, confidence (always present from OCR step)",
    )


