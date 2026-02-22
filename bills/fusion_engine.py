"""
Fusion Engine: field-level fusion of OCR and Vision LLM extraction results.
Runs both extractors; fuses per-field with conflict detection and metadata.
Uses numeric guardrails when OCR raw text is provided to reject amounts from invoice IDs etc.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from commons.schema import ReimbursementSchema
from bills.numeric_validator import is_valid_amount
from bills.bill_extractor import _correct_rupee_read_as_leading_2, _text_has_year_in_date_context

logger = logging.getLogger(__name__)

MONTH_YYYY_MM = re.compile(r"^(20\d{2})-(0[1-9]|1[0-2])$")
AMOUNT_MISMATCH_THRESHOLD = 0.20  # 20% relative difference → flag for review
TEXT_OCR_CONFIDENCE_THRESHOLD = 0.9  # Above this, prefer OCR for text fields


def _to_dict(schema: ReimbursementSchema | dict[str, Any]) -> dict[str, Any]:
    """Normalize to dict for fusion."""
    if isinstance(schema, ReimbursementSchema):
        return schema.model_dump(mode="json")
    return dict(schema)


def _valid_amount(value: Any) -> float | None:
    """Return float if valid positive amount, else None."""
    if value is None:
        return None
    try:
        v = float(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _valid_month(value: Any) -> str | None:
    """Return string if valid YYYY-MM, else None."""
    if value is None or not str(value).strip():
        return None
    s = str(value).strip()[:7]
    if MONTH_YYYY_MM.match(s):
        return s
    return None


@dataclass
class FusionMetadata:
    """Metadata produced by the fusion engine."""

    field_sources: dict[str, str] = field(default_factory=dict)  # field -> "OCR" | "LLM" | "fused"
    conflicts: list[str] = field(default_factory=list)
    final_confidence_score: float = 0.0
    has_major_numeric_conflict: bool = False  # amount/month mismatch > threshold → NEEDS_REVIEW


@dataclass
class FusionResult:
    """Result of fusing OCR and LLM extraction."""

    final_structured_data: dict[str, Any]
    fusion_metadata: FusionMetadata


def fuse_extractions(
    ocr_result: ReimbursementSchema | dict[str, Any],
    llm_result: ReimbursementSchema | dict[str, Any] | None,
    ocr_confidence: float = 0.0,
    ocr_raw_text: str = "",
) -> FusionResult:
    """
    Fuse OCR and LLM extraction results with field-level logic.

    - amount: When ocr_raw_text is provided, only accept amounts that pass numeric guardrails
      (e.g. near "Total"/"Net Payable"; reject near "Invoice No"/GST or embedded in alphanumeric).
      Prefer valid numeric; if both valid and equal → use OCR; if mismatch > 20% → flag review; prefer OCR.
    - month: Validate YYYY-MM; prefer valid; if both valid but different → flag review.
    - text fields: Prefer LLM unless OCR confidence > 0.9.

    Returns FusionResult with final_structured_data and fusion_metadata (field_sources, conflicts, final_confidence_score).
    """
    ocr = _to_dict(ocr_result)
    llm = _to_dict(llm_result) if llm_result is not None else {}
    field_sources: dict[str, str] = {}
    conflicts: list[str] = []
    has_major_numeric_conflict = False

    # Numeric guardrails: reject amounts that look like invoice numbers / IDs when we have OCR text
    ocr_amt = _valid_amount(ocr.get("amount"))
    llm_amt = _valid_amount(llm.get("amount")) if llm else None
    if ocr_raw_text and ocr_raw_text.strip():
        if ocr_amt is not None and not is_valid_amount(ocr_raw_text, ocr_amt):
            logger.debug("OCR amount %.2f rejected by numeric validator", ocr_amt)
            ocr_amt = None
        if llm_amt is not None and not is_valid_amount(ocr_raw_text, llm_amt):
            logger.debug("LLM amount %.2f rejected by numeric validator", llm_amt)
            llm_amt = None

    # Default: use OCR for everything, then override per field
    final: dict[str, Any] = dict(ocr)
    # Ensure keys from LLM exist in final
    for k in llm:
        if k not in final:
            final[k] = llm[k]

    # --- amount ---
    if ocr_amt is not None and llm_amt is not None:
        if ocr_amt == llm_amt:
            final["amount"] = ocr_amt
            field_sources["amount"] = "OCR"
        else:
            rel_diff = abs(ocr_amt - llm_amt) / max(ocr_amt, llm_amt, 1e-9)
            if rel_diff > AMOUNT_MISMATCH_THRESHOLD:
                conflicts.append(f"amount mismatch: OCR={ocr_amt} vs LLM={llm_amt} ({rel_diff:.1%})")
                has_major_numeric_conflict = True
            final["amount"] = ocr_amt  # prefer OCR for numeric reliability
            field_sources["amount"] = "OCR"
    elif ocr_amt is not None:
        final["amount"] = ocr_amt
        field_sources["amount"] = "OCR"
    elif llm_amt is not None:
        final["amount"] = llm_amt
        field_sources["amount"] = "LLM"
    else:
        final["amount"] = 0.0
        field_sources["amount"] = "none"

    # Rupee (₹) read as leading "2": e.g. 278.75 → 78.75 when 78.75 appears in OCR text
    if ocr_raw_text and final.get("amount"):
        amt = float(final["amount"])
        corrected = _correct_rupee_read_as_leading_2(ocr_raw_text, amt)
        if corrected != amt:
            logger.info("Fusion: amount corrected from %.2f to %.2f (rupee symbol read as 2)", amt, corrected)
            final["amount"] = corrected
    # Reject year (2000-2030) in date context (e.g. "Nov 14th 2024" mistaken as amount)
    if ocr_raw_text and final.get("amount"):
        amt = float(final["amount"])
        if amt == int(amt) and 2000 <= amt <= 2030 and _text_has_year_in_date_context(ocr_raw_text, amt):
            logger.info("Fusion: amount %.0f rejected (year in date context); using 0", amt)
            final["amount"] = 0.0

    # --- month ---
    ocr_month = _valid_month(ocr.get("month"))
    llm_month = _valid_month(llm.get("month")) if llm else None
    if ocr_month and llm_month:
        if ocr_month == llm_month:
            final["month"] = ocr_month
            field_sources["month"] = "OCR"
        else:
            conflicts.append(f"month mismatch: OCR={ocr_month} vs LLM={llm_month}")
            has_major_numeric_conflict = True
            final["month"] = ocr_month
            field_sources["month"] = "OCR"
    elif ocr_month:
        final["month"] = ocr_month
        field_sources["month"] = "OCR"
    elif llm_month:
        final["month"] = llm_month
        field_sources["month"] = "LLM"
    else:
        final["month"] = ""
        field_sources["month"] = "none"

    # --- text fields: prefer LLM unless OCR confidence > 0.9 ---
    text_fields = ["vendor_name", "currency", "category", "employee_id", "expense_type"]
    for f in text_fields:
        ocr_val = (ocr.get(f) or "")
        ocr_val = ocr_val if isinstance(ocr_val, str) else str(ocr_val)
        llm_val = (llm.get(f) or "") if llm else ""
        llm_val = llm_val if isinstance(llm_val, str) else str(llm_val)
        if ocr_confidence > TEXT_OCR_CONFIDENCE_THRESHOLD and ocr_val.strip():
            final[f] = ocr_val.strip() or llm_val.strip()
            field_sources[f] = "OCR"
        elif llm_val.strip():
            final[f] = llm_val.strip()
            field_sources[f] = "LLM"
        else:
            final[f] = ocr_val.strip()
            field_sources[f] = "OCR"

    # --- bill_date (treat like text; prefer valid) ---
    ocr_date = (ocr.get("bill_date") or "")
    llm_date = (llm.get("bill_date") or "") if llm else ""
    if ocr_confidence > TEXT_OCR_CONFIDENCE_THRESHOLD and ocr_date:
        final["bill_date"] = ocr_date
        field_sources["bill_date"] = "OCR"
    elif llm_date:
        final["bill_date"] = llm_date
        field_sources["bill_date"] = "LLM"
    else:
        final["bill_date"] = ocr_date or llm_date
        field_sources["bill_date"] = "OCR" if ocr_date else "LLM"

    # --- line_items: use from same source as amount for consistency ---
    amt_source = field_sources.get("amount", "OCR")
    if amt_source == "OCR" and ocr.get("line_items"):
        final["line_items"] = ocr["line_items"]
        field_sources["line_items"] = "OCR"
    elif llm.get("line_items"):
        final["line_items"] = llm["line_items"]
        field_sources["line_items"] = "LLM"
    else:
        final["line_items"] = ocr.get("line_items") or []
        field_sources["line_items"] = "OCR"

    # Final confidence: 1.0 minus penalty for conflicts; floor 0
    conflict_penalty = min(0.5, len(conflicts) * 0.15)
    base = (ocr_confidence + 0.85) / 2.0 if llm else ocr_confidence  # LLM placeholder 0.85
    final_confidence_score = max(0.0, min(1.0, base - conflict_penalty))

    metadata = FusionMetadata(
        field_sources=field_sources,
        conflicts=conflicts,
        final_confidence_score=final_confidence_score,
        has_major_numeric_conflict=has_major_numeric_conflict,
    )
    return FusionResult(final_structured_data=final, fusion_metadata=metadata)
