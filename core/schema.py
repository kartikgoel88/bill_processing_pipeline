"""
Pydantic schemas for bill extraction and OCR. Used by extraction/, pipeline, services.
"""
from __future__ import annotations

from datetime import date
from typing import Any

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
DEFAULT_EXPENSE_TYPE = "meal"
MEAL_EXPENSE_TYPE = "meal"


class ReimbursementSchema(BaseModel):
    """Structured reimbursement data from OCR or LLM extraction."""

    employee_id: str = ""
    expense_type: str = ""
    amount: float = 0.0
    month: str = ""
    bill_date: date | str = ""
    vendor_name: str = ""
    currency: str = "INR"
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
