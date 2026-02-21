"""
Structured validation for reimbursement data.
Critical: amount > 0, month YYYY-MM, month not future → critical_failure if any fail.
Other: employee_id match folder, expense_type match folder, vendor_name not empty.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from commons.schema import ReimbursementSchema

logger = logging.getLogger(__name__)

MONTH_YYYY_MM = re.compile(r"^(20\d{2})-(0[1-9]|1[0-2])$")


def _parse_month(s: str) -> tuple[int, int] | None:
    """Parse YYYY-MM to (year, month); return None if invalid."""
    if not s or not s.strip():
        return None
    m = MONTH_YYYY_MM.match(s.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _month_not_future(year: int, month: int) -> bool:
    """True if the month is not in the future."""
    today = date.today()
    if year < today.year:
        return True
    if year > today.year:
        return False
    return month <= today.month


class ValidationResult:
    """Result of reimbursement validation: valid, critical_failure, schema, fail_reasons."""

    __slots__ = ("valid", "critical_failure", "schema", "fail_reasons")

    def __init__(
        self,
        valid: bool,
        critical_failure: bool,
        schema: ReimbursementSchema | None = None,
        fail_reasons: list[str] | None = None,
    ) -> None:
        self.valid = valid
        self.critical_failure = critical_failure
        self.schema = schema
        self.fail_reasons = fail_reasons or []


def validate_reimbursement(
    schema: ReimbursementSchema,
    *,
    expected_employee_id: str | None = None,
    expected_expense_type: str | None = None,
) -> ValidationResult:
    """
    Validate structured reimbursement.

    Critical (any failure → critical_failure = True, trigger LLM fallback or auto-REJECT):
    - amount > 0
    - month valid format YYYY-MM
    - month not future

    Other:
    - employee_id must match expected_employee_id (if provided)
    - expense_type must match expected_expense_type (if provided)
    - vendor_name not empty
    """
    reasons: list[str] = []
    critical_failure = False

    # --- Critical ---
    if schema.amount is None:
        reasons.append("amount is null")
        critical_failure = True
    elif schema.amount == 0:
        reasons.append("amount is zero")
        critical_failure = True
    elif schema.amount < 0:
        reasons.append("amount must be > 0")
        critical_failure = True

    month_parsed = _parse_month(schema.month)
    if not schema.month or not schema.month.strip():
        reasons.append("month is null or empty")
        critical_failure = True
    elif not month_parsed:
        reasons.append("month has invalid format (required YYYY-MM)")
        critical_failure = True
    else:
        y, m = month_parsed
        if not _month_not_future(y, m):
            reasons.append("month must not be in the future")
            critical_failure = True

    # --- Other ---
    if expected_employee_id is not None and schema.employee_id != expected_employee_id:
        reasons.append(
            f"employee_id '{schema.employee_id}' does not match folder '{expected_employee_id}'"
        )
    if expected_expense_type is not None and schema.expense_type != expected_expense_type:
        reasons.append(
            f"expense_type '{schema.expense_type}' does not match folder '{expected_expense_type}'"
        )
    if not (schema.vendor_name and schema.vendor_name.strip()):
        reasons.append("vendor_name is empty")

    valid = len(reasons) == 0
    if not valid:
        logger.debug("Validation failed: critical_failure=%s reasons=%s", critical_failure, reasons)
    return ValidationResult(
        valid=valid,
        critical_failure=critical_failure,
        schema=schema,
        fail_reasons=reasons,
    )


def validate_structured_bill(
    schema: ReimbursementSchema,
    *,
    expected_employee_id: str | None = None,
    expected_expense_type: str | None = None,
) -> ValidationResult:
    """
    Validate an already-structured ReimbursementSchema (e.g. from LLM extraction).
    Same rules as validate_reimbursement.
    """
    return validate_reimbursement(
        schema,
        expected_employee_id=expected_employee_id,
        expected_expense_type=expected_expense_type,
    )


def validate_and_parse_ocr(
    raw_text: str,
    *,
    employee_id: str = "",
    expense_type: str = "",
) -> ValidationResult:
    """
    Parse OCR text into ReimbursementSchema (via bill_extractor) and validate.
    Single entry point for OCR path: parse then validate.
    """
    from bills.bill_extractor import parse_structured_from_ocr

    schema = parse_structured_from_ocr(
        raw_text, employee_id=employee_id, expense_type=expense_type
    )
    if schema is None:
        return ValidationResult(
            valid=False,
            critical_failure=True,
            schema=None,
            fail_reasons=["Failed to parse OCR into ReimbursementSchema"],
        )
    return validate_reimbursement(
        schema,
        expected_employee_id=employee_id or None,
        expected_expense_type=expense_type or None,
    )
