"""
Bill processing pipeline: single public method process(file_path) -> BillResult.
Does not know which LLM is used; all services injected via constructor.
Flow: OCR -> Vision extraction -> validate -> fallback if low confidence -> Decision -> Post-process.
"""

from __future__ import annotations

import time
import uuid
import logging
from pathlib import Path
from typing import Any

from core.interfaces import (
    IOCRService,
    IVisionService,
    IDecisionService,
    IPostProcessingService,
)
from core.models import BillResult, ExtractionResult
from utils.image_utils import image_bytes_from_path_for_vision

logger = logging.getLogger(__name__)


def _ocr_fallback(
    ocr_service: IOCRService,
    path: Path,
    context: dict[str, Any],
) -> ExtractionResult:
    """Use OCR only and parse to structured bill."""
    from extraction.parser import parse_structured_from_ocr
    from core.schema import ReimbursementSchema, DEFAULT_EXPENSE_TYPE
    raw_text, confidence = ocr_service.extract_from_path(path)
    schema = parse_structured_from_ocr(
        raw_text,
        employee_id=context.get("employee_id", ""),
        expense_type=context.get("expense_type", "meal"),
    )
    if schema is None:
        schema = ReimbursementSchema(
            employee_id=context.get("employee_id", ""),
            expense_type=context.get("expense_type", "") or DEFAULT_EXPENSE_TYPE,
            amount=0.0,
            month="",
            bill_date="",
            vendor_name="Unknown",
            currency="INR",
            category="",
            line_items=[],
        )
    bill_dict = schema.model_dump(mode="json")
    return ExtractionResult(
        structured_bill=bill_dict,
        confidence=confidence,
        source="ocr_fallback",
        ocr_raw_text=raw_text,
        ocr_confidence=confidence,
        critical_validation_failed=not _validate_critical(bill_dict),
    )


def _validate_critical(structured_bill: dict[str, Any]) -> bool:
    """True if amount > 0 and month present."""
    amt = structured_bill.get("amount")
    if amt is None:
        return False
    try:
        if float(amt) <= 0:
            return False
    except (TypeError, ValueError):
        return False
    month = (structured_bill.get("month") or "").strip()
    return bool(month)


class BillProcessingPipeline:
    """
    Production pipeline: process(file_path) -> BillResult.
    No global state; no knowledge of concrete LLM. All deps injected.
    """

    def __init__(
        self,
        ocr_service: IOCRService,
        vision_service: IVisionService,
        decision_service: IDecisionService,
        post_processing_service: IPostProcessingService,
        *,
        policy: dict[str, Any],
        policy_version_hash: str = "",
        fallback_threshold: float = 0.6,
        fallback_enabled: bool = True,
        extraction_strategy: str = "fusion",
        expense_type: str = "meal",
        employee_id: str = "",
    ) -> None:
        self._ocr = ocr_service
        self._vision = vision_service
        self._decision = decision_service
        self._post = post_processing_service
        self._policy = policy
        self._policy_hash = policy_version_hash or ""
        self._fallback_threshold = fallback_threshold
        self._fallback_enabled = fallback_enabled
        self._strategy = (extraction_strategy or "fusion").strip().lower()
        self._expense_type = expense_type
        self._employee_id = employee_id

    def process(self, file_path: str | Path) -> BillResult:
        """
        Run full pipeline for one file.
        vision_first: vision only, then OCR fallback if needed (no initial OCR).
        fusion: OCR first, then vision, then OCR fallback if low confidence.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        trace_id = str(uuid.uuid4())
        start = time.perf_counter()
        context = {
            "employee_id": self._employee_id,
            "expense_type": self._expense_type,
        }

        # 1) OCR only when strategy is fusion (run both); vision_first skips OCR until fallback
        ocr_text, ocr_conf = "", 0.0
        if self._strategy == "fusion":
            try:
                ocr_text, ocr_conf = self._ocr.extract_from_path(path)
            except Exception as e:
                logger.exception("OCR failed for %s: %s", path, e)

        # 2) Vision extraction
        image_bytes = image_bytes_from_path_for_vision(path)
        if not image_bytes:
            logger.warning("No image bytes for vision (e.g. pdf2image missing or failed); using OCR only")
        try:
            if image_bytes:
                logger.info("Running vision model for extraction: %s", path.name)
            extraction = self._vision.extract(image_bytes, context)
        except Exception as e:
            logger.warning("Vision extraction failed: %s; using OCR fallback", e)
            extraction = _ocr_fallback(self._ocr, path, context)

        # 3) Fallback: if confidence < threshold and fallback enabled, use OCR only
        if (
            self._fallback_enabled
            and extraction.confidence < self._fallback_threshold
            and extraction.source != "ocr_fallback"
        ):
            logger.info("Vision confidence %.2f < threshold %.2f; using OCR fallback", extraction.confidence, self._fallback_threshold)
            extraction = _ocr_fallback(self._ocr, path, context)

        logger.info("Extraction source: %s (confidence=%.2f)", extraction.source, extraction.confidence)

        # OCR fields for result: from initial OCR (fusion) or from fallback
        if extraction.source == "ocr_fallback":
            ocr_text = extraction.ocr_raw_text
            ocr_conf = extraction.ocr_confidence

        if extraction.critical_validation_failed or not _validate_critical(extraction.structured_bill):
            decision = {
                "decision": "REJECTED",
                "confidence_score": 1.0,
                "reasoning": "Critical validation failed",
                "violated_rules": ["Critical validation failed"],
                "approved_amount": None,
            }
        else:
            monthly_total = float(extraction.structured_bill.get("amount") or 0)
            decision = self._decision.get_decision(
                extraction.structured_bill,
                self._policy,
                self._expense_type,
                monthly_total,
                trace_id,
            )
            decision = self._post.apply(
                decision,
                extraction.structured_bill,
                self._policy,
                self._expense_type,
                remaining_day_cap=None,
            )

        elapsed = time.perf_counter() - start
        structured_bill_out = dict(extraction.structured_bill)
        if structured_bill_out.get("bill_date") and hasattr(structured_bill_out["bill_date"], "isoformat"):
            structured_bill_out["bill_date"] = str(structured_bill_out["bill_date"])

        return BillResult(
            trace_id=trace_id,
            file_name=path.name,
            extraction_source=extraction.source,
            structured_bill=structured_bill_out,
            decision=decision,
            policy_version_hash=self._policy_hash,
            metadata={
                "ocr_confidence": extraction.ocr_confidence,
                "processing_time_sec": round(elapsed, 4),
            },
            ocr_extracted={"raw_text": ocr_text, "confidence": ocr_conf},
            fusion_metadata=extraction.fusion_metadata,
        )
