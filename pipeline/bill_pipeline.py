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
from utils.image_utils import image_bytes_from_path_for_vision, image_bytes_list_from_path_for_vision

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


def _bills_from_extraction(extraction: ExtractionResult) -> list[dict[str, Any]]:
    """One or more structured bills from extraction (multiple when structured_bills is set)."""
    if extraction.structured_bills:
        return list(extraction.structured_bills)
    if extraction.structured_bill:
        return [extraction.structured_bill]
    return []


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
        vision_if_ocr_below: float = 0.6,
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
        self._vision_if_ocr_below = vision_if_ocr_below
        self._expense_type = expense_type
        self._employee_id = employee_id

    def process(self, file_path: str | Path) -> list[BillResult]:
        """
        Run full pipeline for one file. Returns one BillResult per extracted bill
        (one for single bill, multiple when vision returns multiple bills on one page).
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

        # 1) OCR: run for ocr_only and fusion (needed to decide whether to call vision)
        ocr_text, ocr_conf = "", 0.0
        if self._strategy in ("ocr_only", "fusion"):
            try:
                ocr_text, ocr_conf = self._ocr.extract_from_path(path)
            except Exception as e:
                logger.exception("OCR failed for %s: %s", path, e)

        # 2) OCR-only mode: use OCR result and skip vision entirely
        if self._strategy == "ocr_only":
            extraction = _ocr_fallback(self._ocr, path, context)
            if extraction.source == "ocr_fallback":
                ocr_text = extraction.ocr_raw_text
                ocr_conf = extraction.ocr_confidence
        else:
            # 3) Vision: for fusion, only call vision when OCR confidence is below threshold
            use_vision = True
            if self._strategy == "fusion" and ocr_conf >= self._vision_if_ocr_below:
                use_vision = False
                logger.info("OCR confidence %.2f >= threshold %.2f; skipping vision LLM", ocr_conf, self._vision_if_ocr_below)
                extraction = _ocr_fallback(self._ocr, path, context)
                ocr_text = extraction.ocr_raw_text
                ocr_conf = extraction.ocr_confidence

            if use_vision:
                image_bytes = image_bytes_from_path_for_vision(path)
                if not image_bytes:
                    logger.warning("No image bytes for vision (e.g. pdf2image missing or failed); using OCR only")
                    extraction = _ocr_fallback(self._ocr, path, context)
                    ocr_text = extraction.ocr_raw_text
                    ocr_conf = extraction.ocr_confidence
                else:
                    try:
                        logger.info("Running vision model for extraction: %s", path.name)
                        extraction = self._vision.extract(image_bytes, context)
                    except Exception as e:
                        logger.warning("Vision extraction failed: %s; using OCR fallback", e)
                        extraction = _ocr_fallback(self._ocr, path, context)
                        ocr_text = extraction.ocr_raw_text
                        ocr_conf = extraction.ocr_confidence

        # 3) Fallback: if confidence < threshold and fallback enabled, use OCR only
        if (
            self._fallback_enabled
            and extraction.confidence < self._fallback_threshold
            and extraction.source != "ocr_fallback"
        ):
            logger.info("Vision confidence %.2f < threshold %.2f; using OCR fallback", extraction.confidence, self._fallback_threshold)
            extraction = _ocr_fallback(self._ocr, path, context)

        logger.info("Extraction source: %s (confidence=%.2f)", extraction.source, extraction.confidence)

        if extraction.source == "ocr_fallback":
            ocr_text = extraction.ocr_raw_text
            ocr_conf = extraction.ocr_confidence

        bills_to_process = _bills_from_extraction(extraction)
        if not bills_to_process:
            elapsed = time.perf_counter() - start
            return [
                BillResult(
                    trace_id=trace_id,
                    file_name=path.name,
                    extraction_source=extraction.source,
                    structured_bill={},
                    decision={
                        "decision": "REJECTED",
                        "confidence_score": 1.0,
                        "reasoning": "No bill extracted",
                        "violated_rules": [],
                        "approved_amount": None,
                    },
                    policy_version_hash=self._policy_hash,
                    metadata={"processing_time_sec": round(elapsed, 4)},
                    ocr_extracted={"raw_text": ocr_text, "confidence": ocr_conf},
                    fusion_metadata=extraction.fusion_metadata,
                )
            ]

        results: list[BillResult] = []
        for structured_bill in bills_to_process:
            if extraction.critical_validation_failed or not _validate_critical(structured_bill):
                decision = {
                    "decision": "REJECTED",
                    "confidence_score": 1.0,
                    "reasoning": "Critical validation failed",
                    "violated_rules": ["Critical validation failed"],
                    "approved_amount": None,
                }
            else:
                monthly_total = float(structured_bill.get("amount") or 0)
                decision = self._decision.get_decision(
                    structured_bill,
                    self._policy,
                    self._expense_type,
                    monthly_total,
                    trace_id,
                )
                decision = self._post.apply(
                    decision,
                    structured_bill,
                    self._policy,
                    self._expense_type,
                    remaining_day_cap=None,
                )
            elapsed = time.perf_counter() - start
            structured_bill_out = dict(structured_bill)
            if structured_bill_out.get("bill_date") and hasattr(structured_bill_out["bill_date"], "isoformat"):
                structured_bill_out["bill_date"] = str(structured_bill_out["bill_date"])
            results.append(
                BillResult(
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
            )
        return results

    def process_multi(self, file_path: str | Path) -> list[BillResult]:
        """
        Process a file that may contain multiple bills (e.g. multi-page PDF).
        Runs vision extraction per page and returns one BillResult per page.
        Use for PDFs where each page is a separate bill.
        In ocr_only mode, runs OCR once on the whole file and returns one BillResult.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        trace_id = str(uuid.uuid4())
        context = {
            "employee_id": self._employee_id,
            "expense_type": self._expense_type,
        }

        if self._strategy == "ocr_only":
            extraction = _ocr_fallback(self._ocr, path, context)
            bills_to_process = _bills_from_extraction(extraction)
            start = time.perf_counter()
            if not bills_to_process:
                return [
                    BillResult(
                        trace_id=trace_id,
                        file_name=path.name,
                        extraction_source=extraction.source,
                        structured_bill={},
                        decision={
                            "decision": "REJECTED",
                            "confidence_score": 1.0,
                            "reasoning": "No bill extracted",
                            "violated_rules": [],
                            "approved_amount": None,
                        },
                        policy_version_hash=self._policy_hash,
                        metadata={"processing_time_sec": 0},
                        ocr_extracted={"raw_text": extraction.ocr_raw_text, "confidence": extraction.ocr_confidence},
                        fusion_metadata=extraction.fusion_metadata,
                    )
                ]
            results: list[BillResult] = []
            for structured_bill in bills_to_process:
                if extraction.critical_validation_failed or not _validate_critical(structured_bill):
                    decision = {
                        "decision": "REJECTED",
                        "confidence_score": 1.0,
                        "reasoning": "Critical validation failed",
                        "violated_rules": ["Critical validation failed"],
                        "approved_amount": None,
                    }
                else:
                    monthly_total = float(structured_bill.get("amount") or 0)
                    decision = self._decision.get_decision(
                        structured_bill, self._policy, self._expense_type, monthly_total, trace_id,
                    )
                    decision = self._post.apply(
                        decision, structured_bill, self._policy, self._expense_type, remaining_day_cap=None,
                    )
                elapsed = time.perf_counter() - start
                structured_bill_out = dict(structured_bill)
                if structured_bill_out.get("bill_date") and hasattr(structured_bill_out["bill_date"], "isoformat"):
                    structured_bill_out["bill_date"] = str(structured_bill_out["bill_date"])
                results.append(
                    BillResult(
                        trace_id=trace_id,
                        file_name=path.name,
                        extraction_source=extraction.source,
                        structured_bill=structured_bill_out,
                        decision=decision,
                        policy_version_hash=self._policy_hash,
                        metadata={"processing_time_sec": round(elapsed, 4)},
                        ocr_extracted={"raw_text": extraction.ocr_raw_text, "confidence": extraction.ocr_confidence},
                        fusion_metadata=extraction.fusion_metadata,
                    )
                )
            return results

        # fusion with high OCR confidence: use OCR once for whole file (one result)
        ocr_text, ocr_conf = "", 0.0
        if self._strategy == "fusion":
            try:
                ocr_text, ocr_conf = self._ocr.extract_from_path(path)
            except Exception as e:
                logger.exception("OCR failed for %s: %s", path, e)
            if ocr_conf >= self._vision_if_ocr_below:
                logger.info("OCR confidence %.2f >= threshold %.2f; skipping vision for multi-page", ocr_conf, self._vision_if_ocr_below)
                extraction = _ocr_fallback(self._ocr, path, context)
                bills_to_process = _bills_from_extraction(extraction)
                start = time.perf_counter()
                if not bills_to_process:
                    return [
                        BillResult(
                            trace_id=trace_id,
                            file_name=path.name,
                            extraction_source=extraction.source,
                            structured_bill={},
                            decision={
                                "decision": "REJECTED",
                                "confidence_score": 1.0,
                                "reasoning": "No bill extracted",
                                "violated_rules": [],
                                "approved_amount": None,
                            },
                            policy_version_hash=self._policy_hash,
                            metadata={"processing_time_sec": 0},
                            ocr_extracted={"raw_text": extraction.ocr_raw_text, "confidence": extraction.ocr_confidence},
                            fusion_metadata=extraction.fusion_metadata,
                        )
                    ]
                results = []
                for structured_bill in bills_to_process:
                    if extraction.critical_validation_failed or not _validate_critical(structured_bill):
                        decision = {
                            "decision": "REJECTED",
                            "confidence_score": 1.0,
                            "reasoning": "Critical validation failed",
                            "violated_rules": ["Critical validation failed"],
                            "approved_amount": None,
                        }
                    else:
                        monthly_total = float(structured_bill.get("amount") or 0)
                        decision = self._decision.get_decision(
                            structured_bill, self._policy, self._expense_type, monthly_total, trace_id,
                        )
                        decision = self._post.apply(
                            decision, structured_bill, self._policy, self._expense_type, remaining_day_cap=None,
                        )
                    elapsed = time.perf_counter() - start
                    structured_bill_out = dict(structured_bill)
                    if structured_bill_out.get("bill_date") and hasattr(structured_bill_out["bill_date"], "isoformat"):
                        structured_bill_out["bill_date"] = str(structured_bill_out["bill_date"])
                    results.append(
                        BillResult(
                            trace_id=trace_id,
                            file_name=path.name,
                            extraction_source=extraction.source,
                            structured_bill=structured_bill_out,
                            decision=decision,
                            policy_version_hash=self._policy_hash,
                            metadata={"processing_time_sec": round(elapsed, 4)},
                            ocr_extracted={"raw_text": extraction.ocr_raw_text, "confidence": extraction.ocr_confidence},
                            fusion_metadata=extraction.fusion_metadata,
                        )
                    )
                return results

        page_images = image_bytes_list_from_path_for_vision(path)
        if not page_images:
            logger.warning("No images for multi-page vision: %s", path)
            return []
        results = []
        for page_index, image_bytes in enumerate(page_images):
            start = time.perf_counter()
            try:
                extraction = self._vision.extract(image_bytes, context)
            except Exception as e:
                logger.warning("Vision extraction failed for page %s: %s", page_index + 1, e)
                extraction = ExtractionResult(
                    structured_bill={},
                    confidence=0.0,
                    source="vision_llm",
                    critical_validation_failed=True,
                )
            if (
                self._fallback_enabled
                and extraction.confidence < self._fallback_threshold
                and extraction.source != "ocr_fallback"
            ):
                extraction = _ocr_fallback(self._ocr, path, context)

            bills_to_process = _bills_from_extraction(extraction)
            if not bills_to_process:
                elapsed = time.perf_counter() - start
                results.append(
                    BillResult(
                        trace_id=trace_id,
                        file_name=path.name,
                        extraction_source=extraction.source,
                        structured_bill={},
                        decision={
                            "decision": "REJECTED",
                            "confidence_score": 1.0,
                            "reasoning": "No bill extracted",
                            "violated_rules": [],
                            "approved_amount": None,
                        },
                        policy_version_hash=self._policy_hash,
                        metadata={
                            "page_index": page_index,
                            "total_pages": len(page_images),
                            "processing_time_sec": round(elapsed, 4),
                        },
                        ocr_extracted={},
                        fusion_metadata=extraction.fusion_metadata,
                    )
                )
                continue

            for structured_bill in bills_to_process:
                if extraction.critical_validation_failed or not _validate_critical(structured_bill):
                    decision = {
                        "decision": "REJECTED",
                        "confidence_score": 1.0,
                        "reasoning": "Critical validation failed",
                        "violated_rules": ["Critical validation failed"],
                        "approved_amount": None,
                    }
                else:
                    monthly_total = float(structured_bill.get("amount") or 0)
                    decision = self._decision.get_decision(
                        structured_bill,
                        self._policy,
                        self._expense_type,
                        monthly_total,
                        trace_id,
                    )
                    decision = self._post.apply(
                        decision,
                        structured_bill,
                        self._policy,
                        self._expense_type,
                        remaining_day_cap=None,
                    )
                elapsed = time.perf_counter() - start
                structured_bill_out = dict(structured_bill)
                if structured_bill_out.get("bill_date") and hasattr(structured_bill_out["bill_date"], "isoformat"):
                    structured_bill_out["bill_date"] = str(structured_bill_out["bill_date"])
                results.append(
                    BillResult(
                        trace_id=trace_id,
                        file_name=path.name,
                        extraction_source=extraction.source,
                        structured_bill=structured_bill_out,
                        decision=decision,
                        policy_version_hash=self._policy_hash,
                        metadata={
                            "page_index": page_index,
                            "total_pages": len(page_images),
                            "processing_time_sec": round(elapsed, 4),
                        },
                        ocr_extracted={},
                        fusion_metadata=extraction.fusion_metadata,
                    )
                )
        return results
