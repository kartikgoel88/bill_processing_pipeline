"""
Orchestrator: runs the full bill processing pipeline with policy from store (no hardcoded rules).
Flow: OCR → Vision LLM extraction → Fusion → Validate → Decision → FinalOutput + audit.
"""
from __future__ import annotations

import io
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    convert_from_path = None  # type: ignore

from commons.schema import (
    DecisionSchema,
    FinalOutputMetadata,
    FinalOutputSchema,
    OCRExtractionResult,
    ReimbursementSchema,
)
from commons.exceptions import BillProcessingError
from bills.ocr_engine import extract_from_image, extract_from_pdf, extract_from_bytes
from bills.bill_extractor import parse_structured_from_ocr
from bills.bill_extractor import extract_bill_via_llm
from bills.fusion_engine import fuse_extractions
from bills.document_extractor import extract_bill_via_donut, extract_bill_via_layoutlm
from decision.validator import validate_reimbursement, validate_and_parse_ocr
from decision.decision_engine import (
    DecisionEngine,
    auto_rejected_decision,
    explainability_score_from_decision,
)
logger = logging.getLogger(__name__)


def _apply_meal_cap(
    decision: DecisionSchema,
    schema: ReimbursementSchema | None,
    policy_json: dict[str, Any],
    expense_type: str,
    *,
    remaining_day_cap: float | None = None,
) -> DecisionSchema:
    """
    Enforce policy meal_allowance.limit on approved_amount.
    Cap is per day (combined for all meal bills on same employee+date). When remaining_day_cap
    is provided (batch), use it so total approved for that day does not exceed limit; else use
    full limit (single-file or first bill of the day).
    approved_amount = min(approved_amount, effective_cap, bill amount).
    """
    if decision.decision != "APPROVED" or decision.approved_amount is None:
        return decision
    if (expense_type or "").lower() != "meal":
        return decision
    try:
        limit = policy_json.get("meal_allowance") or {}
        day_cap = limit.get("limit") if isinstance(limit, dict) else None
        if day_cap is None:
            return decision
        day_cap = float(day_cap)
    except (TypeError, ValueError):
        return decision
    effective_cap = min(day_cap, remaining_day_cap) if remaining_day_cap is not None else day_cap
    if effective_cap <= 0:
        approved = 0.0
    else:
        bill_amount = float(schema.amount) if schema is not None else (decision.approved_amount or 0)
        approved = min(decision.approved_amount, effective_cap, bill_amount)
    if approved == decision.approved_amount:
        return decision
    return DecisionSchema(
        decision=decision.decision,
        confidence_score=decision.confidence_score,
        reasoning=decision.reasoning,
        violated_rules=decision.violated_rules,
        approved_amount=approved,
    )


# Vision API: max dimension and JPEG quality to avoid Ollama 500 (huge payloads)
VISION_IMAGE_MAX_PX = 1024
VISION_IMAGE_JPEG_QUALITY = 85
VISION_PDF_DPI = 150


def _read_file_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _image_bytes_for_vision(path: Path) -> bytes:
    """
    Return image bytes suitable for the vision LLM (always raster, size-limited).
    For PDFs: render first page to JPEG. For images: load and optionally resize, then JPEG.
    Prevents 500 from Ollama when sending raw PDF bytes or huge images.
    """
    is_pdf = path.suffix.lower() == ".pdf"
    if is_pdf:
        if not PDF_AVAILABLE or convert_from_path is None:
            logger.warning("pdf2image not available; cannot send PDF to vision model")
            return b""
        pages = convert_from_path(str(path), dpi=VISION_PDF_DPI, first_page=1, last_page=1)
        if not pages:
            return b""
        img = pages[0]
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > VISION_IMAGE_MAX_PX:
            ratio = VISION_IMAGE_MAX_PX / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=VISION_IMAGE_JPEG_QUALITY)
        return buf.getvalue()
    # Image file (PNG, JPEG, etc.)
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > VISION_IMAGE_MAX_PX:
            ratio = VISION_IMAGE_MAX_PX / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=VISION_IMAGE_JPEG_QUALITY)
        return buf.getvalue()


def _ocr_from_path(path: Path) -> tuple[OCRExtractionResult, bytes]:
    """Run OCR and return (result, image_bytes for vision LLM). Image bytes are always raster (PDF→first page as JPEG)."""
    data = _read_file_bytes(path)
    is_pdf = path.suffix.lower() == ".pdf"
    if is_pdf:
        ocr_result = extract_from_pdf(path)
    else:
        ocr_result = extract_from_bytes(data, is_pdf=False)
    vision_bytes = _image_bytes_for_vision(path)
    return ocr_result, vision_bytes


def _ocr_result_from_path(path: Path) -> OCRExtractionResult:
    """Run OCR only (Tesseract/EasyOCR). Used when extraction_strategy=vision_first and vision fails."""
    if path.suffix.lower() == ".pdf":
        return extract_from_pdf(path)
    return extract_from_bytes(_read_file_bytes(path), is_pdf=False)


def _run_all_extractors(
    path: Path,
    image_bytes: bytes,
    employee_id: str,
    expense_type: str,
    donut_model_id: str,
    llm_client: Any,
    qwen_vl_model: str,
) -> dict[str, Any]:
    """
    Run OCR (Tesseract), EasyOCR, Donut, and Qwen-VL and return their outputs.
    Used when OUTPUT_ALL_EXTRACTORS=1 so you can compare all extractors in batch_output.json.
    Returns dict with keys: tesseract, easyocr, donut, qwen_vl. Each value is
    { raw_text?, confidence, structured } for OCR or { structured, confidence } for vision, or { error: "..." }.
    """
    import os as _os

    out: dict[str, Any] = {}

    def _schema_to_dict(s: ReimbursementSchema | None) -> dict[str, Any]:
        if s is None:
            return {}
        return s.model_dump(mode="json")

    # 1. Tesseract OCR
    try:
        if path.suffix.lower() == ".pdf":
            ocr_t = extract_from_pdf(path, engine_override="tesseract")
        else:
            ocr_t = extract_from_bytes(_read_file_bytes(path), is_pdf=False, engine_override="tesseract")
        schema_t = parse_structured_from_ocr(ocr_t.raw_text or "", employee_id=employee_id, expense_type=expense_type)
        out["tesseract"] = {
            "raw_text": ocr_t.raw_text or "",
            "confidence": ocr_t.confidence,
            "structured": _schema_to_dict(schema_t),
        }
    except Exception as e:
        out["tesseract"] = {"error": str(e)}

    # 2. EasyOCR
    try:
        from bills.ocr_engine import EASYOCR_AVAILABLE
        if not EASYOCR_AVAILABLE:
            out["easyocr"] = {"error": "easyocr not installed (pip install .[ocr])"}
        else:
            if path.suffix.lower() == ".pdf":
                ocr_e = extract_from_pdf(path, engine_override="easyocr")
            else:
                ocr_e = extract_from_bytes(_read_file_bytes(path), is_pdf=False, engine_override="easyocr")
            schema_e = parse_structured_from_ocr(ocr_e.raw_text or "", employee_id=employee_id, expense_type=expense_type)
            out["easyocr"] = {
                "raw_text": ocr_e.raw_text or "",
                "confidence": ocr_e.confidence,
                "structured": _schema_to_dict(schema_e),
            }
    except Exception as e:
        out["easyocr"] = {"error": str(e)}

    # 3. Donut
    try:
        if image_bytes:
            schema_d, conf_d = extract_bill_via_donut(
                image_bytes,
                employee_id=employee_id,
                expense_type=expense_type,
                model_id=donut_model_id,
            )
            out["donut"] = {"structured": _schema_to_dict(schema_d), "confidence": conf_d or 0.0}
        else:
            out["donut"] = {"error": "no image bytes (e.g. pdf2image not available)"}
    except Exception as e:
        out["donut"] = {"error": str(e)}

    # 4. Qwen-VL (vision LLM)
    try:
        if image_bytes and llm_client:
            schema_q, conf_q = extract_bill_via_llm(
                llm_client,
                image_bytes,
                employee_id=employee_id,
                expense_type=expense_type,
                vision_model=qwen_vl_model,
            )
            out["qwen_vl"] = {"structured": _schema_to_dict(schema_q), "confidence": conf_q or 0.0}
        else:
            out["qwen_vl"] = {"error": "no image bytes or vision client not configured"}
    except Exception as e:
        out["qwen_vl"] = {"error": str(e)}

    return out


def _get_structured_bill(
    image_bytes: bytes,
    employee_id: str,
    expense_type: str,
    ocr_confidence_threshold: float,
    llm_client: Any,
    vision_model: str,
    max_retries: int,
    retry_delay_sec: float,
    trace_id: str,
    vision_extractor: str = "vision_llm",
    donut_model_id: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
    layoutlm_model_id: str = "nielsr/layoutlmv3-finetuned-cord",
    extraction_strategy: str = "fusion",
    ocr_result: OCRExtractionResult | None = None,
    path: Path | None = None,
) -> tuple[
    ReimbursementSchema | None,
    str,
    float | None,
    float | None,
    bool,
    dict[str, Any],
    OCRExtractionResult | None,
]:
    """
    Run extraction per strategy; return (schema, extraction_source, ocr_conf, llm_conf, critical_failure, fusion_metadata, ocr_result_used).
    - vision_first: try vision/document (Donut, LayoutLMv3, Qwen-VL, vision_llm) first; use Tesseract only as fallback.
    - fusion: run OCR + vision, then fuse (current behaviour).
    """
    import os as _os

    strategy = (extraction_strategy or "fusion").strip().lower()
    extractor = (vision_extractor or "donut").strip().lower()
    if extractor == "layout":
        extractor = "layoutlm"
    extractor_name = extractor  # for extraction_source in response (donut, layoutlm, vision_llm, qwen_vl)
    # Qwen-VL: use vision_llm path; prefer QWEN_VL_MODEL, else use configured vision_model (LLM_VISION_MODEL) so no 404 when qwen2-vl not installed
    if extractor == "qwen_vl":
        vision_model_actual = _os.getenv("QWEN_VL_MODEL") or vision_model
        extractor = "vision_llm"
    else:
        vision_model_actual = vision_model

    empty_fusion = {
        "field_sources": {},
        "conflicts": [],
        "final_confidence_score": 0.0,
        "has_major_numeric_conflict": False,
    }

    def _run_vision_extractor() -> tuple[ReimbursementSchema | None, float | None]:
        """Run Donut, LayoutLM, or vision LLM; return (schema, confidence)."""
        if not image_bytes:
            return None, None
        try:
            if extractor == "donut":
                return extract_bill_via_donut(
                    image_bytes,
                    employee_id=employee_id,
                    expense_type=expense_type,
                    model_id=donut_model_id,
                )
            if extractor == "layoutlm":
                # LayoutLM needs OCR text for word alignment; run OCR if we have path
                ocr_text = ""
                if path:
                    _ocr = _ocr_result_from_path(path)
                    ocr_text = _ocr.raw_text or ""
                return extract_bill_via_layoutlm(
                    image_bytes,
                    ocr_text,
                    employee_id=employee_id,
                    expense_type=expense_type,
                    model_id=layoutlm_model_id,
                )
            if llm_client and extractor == "vision_llm":
                return extract_bill_via_llm(
                    llm_client,
                    image_bytes,
                    employee_id=employee_id,
                    expense_type=expense_type,
                    vision_model=vision_model_actual,
                    max_retries=max_retries,
                    retry_delay_sec=retry_delay_sec,
                )
        except Exception as e:
            logger.warning("Vision/document extraction failed trace_id=%s: %s", trace_id, e)
        return None, None

    def _vision_has_critical(s: ReimbursementSchema | None, conf: float | None) -> bool:
        if s is None:
            return False
        if (conf or 0) < ocr_confidence_threshold:
            return False
        if (s.amount or 0) <= 0 or not (s.month or "").strip():
            return False
        return True

    # --- vision_first: try vision first; Tesseract only as fallback ---
    if strategy == "vision_first" and path is not None:
        llm_schema, llm_conf = _run_vision_extractor()
        if _vision_has_critical(llm_schema, llm_conf):
            try:
                out_schema = ReimbursementSchema(**llm_schema.model_dump(mode="json"))
            except Exception:
                out_schema = llm_schema
            validation = validate_reimbursement(
                out_schema,
                expected_employee_id=employee_id or None,
                expected_expense_type=expense_type or None,
            )
            return (
                out_schema,
                extractor_name,
                None,
                llm_conf,
                validation.critical_failure,
                empty_fusion,
                None,
            )
        # Fallback: Tesseract (image_to_data) → parse_structured_from_ocr → numeric validation
        ocr_result_fb = _ocr_result_from_path(path)
        ocr_schema_fb = parse_structured_from_ocr(
            ocr_result_fb.raw_text,
            employee_id=employee_id,
            expense_type=expense_type,
        )
        if ocr_schema_fb is None:
            ocr_schema_fb = ReimbursementSchema(
                employee_id=employee_id,
                expense_type=expense_type or "meal",
                amount=0.0,
                month="",
                bill_date="",
                vendor_name="Unknown",
                currency="INR",
                category="",
                line_items=[],
            )
        try:
            fused_fb = ReimbursementSchema(**ocr_schema_fb.model_dump(mode="json"))
        except Exception as e:
            logger.warning("OCR fallback schema validation failed trace_id=%s: %s", trace_id, e)
            return None, "ocr_fallback", ocr_result_fb.confidence, None, True, empty_fusion, ocr_result_fb
        validation = validate_reimbursement(
            fused_fb,
            expected_employee_id=employee_id or None,
            expected_expense_type=expense_type or None,
        )
        return (
            fused_fb,
            "ocr_fallback",
            ocr_result_fb.confidence,
            None,
            validation.critical_failure,
            empty_fusion,
            ocr_result_fb,
        )

    # --- fusion: OCR + vision, then fuse (require ocr_result) ---
    if ocr_result is None:
        ocr_result = OCRExtractionResult()
    ocr_schema = parse_structured_from_ocr(
        ocr_result.raw_text,
        employee_id=employee_id,
        expense_type=expense_type,
    )
    if ocr_schema is None:
        ocr_schema = ReimbursementSchema(
            employee_id=employee_id,
            expense_type=expense_type or "meal",
            amount=0.0,
            month="",
            bill_date="",
            vendor_name="Unknown",
            currency="INR",
            category="",
            line_items=[],
        )

    llm_schema, llm_conf = _run_vision_extractor()
    if extractor not in ("donut", "layoutlm", "vision_llm") and llm_schema is None:
        logger.debug("Vision extractor=%s not run or failed; using OCR only", vision_extractor)

    fusion_result = fuse_extractions(
        ocr_schema,
        llm_schema,
        ocr_confidence=ocr_result.confidence,
        ocr_raw_text=ocr_result.raw_text or "",
    )
    fusion_metadata = {
        "field_sources": fusion_result.fusion_metadata.field_sources,
        "conflicts": fusion_result.fusion_metadata.conflicts,
        "final_confidence_score": fusion_result.fusion_metadata.final_confidence_score,
        "has_major_numeric_conflict": fusion_result.fusion_metadata.has_major_numeric_conflict,
    }

    try:
        fused_schema = ReimbursementSchema(**fusion_result.final_structured_data)
    except Exception as e:
        logger.warning("Fused data schema validation failed trace_id=%s: %s", trace_id, e)
        return (
            None,
            "fusion",
            ocr_result.confidence,
            llm_conf,
            True,
            fusion_metadata,
            ocr_result,
        )
    validation = validate_reimbursement(
        fused_schema,
        expected_employee_id=employee_id or None,
        expected_expense_type=expense_type or None,
    )
    if validation.critical_failure:
        return (
            fused_schema,
            "fusion",
            ocr_result.confidence,
            llm_conf,
            True,
            fusion_metadata,
            ocr_result,
        )
    return (
        fused_schema,
        "fusion",
        ocr_result.confidence,
        llm_conf,
        False,
        fusion_metadata,
        ocr_result,
    )


class ReimbursementOrchestrator:
    """
    Orchestrates: load policy (from cache or ingest) → for each bill: OCR → validate
    → LLM fallback if needed → decision engine (policy JSON + bill JSON) → FinalOutput + audit.
    """

    def __init__(
        self,
        *,
        policy_json: dict[str, Any],
        policy_version_hash: str,
        vision_client: Any,
        decision_client: Any,
        decision_engine: DecisionEngine | None = None,
        ocr_confidence_threshold: float = 0.6,
        vision_model: str = "llava",
        decision_model: str = "llama3.2",
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        audit_log_path: str | Path | None = None,
        vision_extractor: str = "donut",
        donut_model_id: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
        layoutlm_model_id: str = "nielsr/layoutlmv3-finetuned-cord",
        extraction_strategy: str = "fusion",
        output_all_extractors: bool = False,
    ) -> None:
        self.policy_json = policy_json
        self.policy_version_hash = policy_version_hash
        self.vision_client = vision_client
        self.decision_client = decision_client
        self.decision_engine = decision_engine or DecisionEngine(
            client=decision_client,
            decision_model=decision_model,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
        )
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.vision_model = vision_model
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        self.vision_extractor = (vision_extractor or "donut").strip().lower()
        self.donut_model_id = donut_model_id or "naver-clova-ix/donut-base-finetuned-cord-v2"
        self.layoutlm_model_id = layoutlm_model_id or "nielsr/layoutlmv3-finetuned-cord"
        self.extraction_strategy = (extraction_strategy or "fusion").strip().lower()
        self.output_all_extractors = bool(output_all_extractors)

    def _trace_id(self) -> str:
        return str(uuid.uuid4())

    def _audit_log(self, trace_id: str, stage: str, payload: dict[str, Any]) -> None:
        """Log to logger and optionally append to audit file."""
        logger.info(
            "AUDIT trace_id=%s stage=%s payload_keys=%s",
            trace_id,
            stage,
            list(payload.keys()),
            extra={"trace_id": trace_id, "stage": stage},
        )
        if self.audit_log_path:
            try:
                line = f"trace_id={trace_id} stage={stage} {payload}\n"
                with open(self.audit_log_path, "a") as f:
                    f.write(line)
            except OSError as e:
                logger.debug("Could not write audit log: %s", e)

    def process_file(
        self,
        path: str | Path,
        *,
        expense_type: str = "",
        employee_id: str = "",
        trace_id: str | None = None,
        is_pdf: bool | None = None,
    ) -> FinalOutputSchema:
        """
        Process a single bill file. expense_type and employee_id should be set from folder context.
        Returns FinalOutputSchema (trace_id, policy_version_hash, extraction_source, structured_bill, policy_used, decision, metadata).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        tid = trace_id or self._trace_id()
        start = time.perf_counter()
        strategy = (getattr(self, "extraction_strategy", None) or "fusion").strip().lower()

        if strategy == "vision_first":
            image_bytes = _image_bytes_for_vision(path)
            ocr_result = None
        else:
            ocr_result, image_bytes = _ocr_from_path(path)

        schema, extraction_source, ocr_conf, llm_conf, critical_failure, fusion_metadata, ocr_result_used = _get_structured_bill(
            image_bytes,
            employee_id=employee_id or "",
            expense_type=expense_type or "meal",
            ocr_confidence_threshold=self.ocr_confidence_threshold,
            llm_client=self.vision_client,
            vision_model=self.vision_model,
            max_retries=self.max_retries,
            retry_delay_sec=self.retry_delay_sec,
            trace_id=tid,
            vision_extractor=self.vision_extractor,
            donut_model_id=self.donut_model_id,
            layoutlm_model_id=self.layoutlm_model_id,
            extraction_strategy=strategy,
            ocr_result=ocr_result,
            path=path,
        )

        all_extractor_outputs: dict[str, Any] = {}
        if getattr(self, "output_all_extractors", False):
            import os as _os
            qwen_model = _os.getenv("QWEN_VL_MODEL") or self.vision_model
            all_extractor_outputs = _run_all_extractors(
                path,
                image_bytes,
                employee_id or "",
                expense_type or "meal",
                self.donut_model_id,
                self.vision_client,
                qwen_model,
            )

        # For single-file we don't have running monthly total; use bill amount as placeholder.
        monthly_total = schema.amount if schema else 0.0

        if critical_failure or schema is None:
            decision = auto_rejected_decision("Critical validation failed")
        else:
            decision = self.decision_engine.get_decision(
                schema,
                self.policy_json,
                expense_type or schema.expense_type or "meal",
                monthly_total,
                critical_failure=critical_failure,
                trace_id=tid,
            )
            # Conflict flag: major numeric mismatch → auto NEEDS_REVIEW
            if fusion_metadata.get("has_major_numeric_conflict"):
                decision = DecisionSchema(
                    decision="NEEDS_REVIEW",
                    confidence_score=decision.confidence_score,
                    reasoning=(decision.reasoning or "") + " [Fusion: numeric conflict between OCR and LLM]",
                    violated_rules=list(decision.violated_rules),
                    approved_amount=decision.approved_amount,
                )
            decision = _apply_meal_cap(decision, schema, self.policy_json, expense_type or (schema.expense_type if schema else "") or "meal")

        processing_time = time.perf_counter() - start
        expl_score = explainability_score_from_decision(decision) if schema else None

        structured_bill = schema.model_dump(mode="json") if schema else {}
        if structured_bill.get("bill_date") and hasattr(structured_bill["bill_date"], "isoformat"):
            structured_bill["bill_date"] = str(structured_bill["bill_date"])

        return FinalOutputSchema(
            trace_id=tid,
            file=path.name,
            policy_version_hash=self.policy_version_hash,
            extraction_source=extraction_source,  # type: ignore[arg-type]
            structured_bill=structured_bill,
            policy_used=self.policy_json,
            decision=decision.model_dump(mode="json"),
            metadata=FinalOutputMetadata(
                ocr_confidence=ocr_conf,
                llm_model=self.decision_engine.decision_model,
                processing_time_sec=round(processing_time, 4),
                explainability_score=expl_score,
            ),
            ocr_extracted=(ocr_result_used or OCRExtractionResult()).model_dump(mode="json"),
            fusion_metadata=fusion_metadata,
            all_extractor_outputs=all_extractor_outputs,
        )

    def process_batch_from_folder(
        self,
        root_folder: str | Path,
        *,
        dry_run: bool = False,
    ) -> list[FinalOutputSchema]:
        """
        Traverse root_folder (expense_type/employee_id/bills), process each bill,
        maintain monthly total per (employee_id, month), return list of FinalOutputSchema.
        """
        from collections import defaultdict

        from bills.bill_folder_reader import iter_bills

        root = Path(root_folder)
        if not root.is_dir():
            logger.warning("Root is not a directory: %s", root)
            return []

        if dry_run:
            count = sum(1 for _ in iter_bills(root))
            self._audit_log(self._trace_id(), "dry_run", {"bill_count": count})
            return []

        # Phase 1: collect all bills with (path, employee_id, expense_type, ..., ocr_for_output, fusion_metadata)
        strategy = (getattr(self, "extraction_strategy", None) or "fusion").strip().lower()
        phase1: list[tuple[Path, str, str, ReimbursementSchema | None, str, float | None, float | None, bool, OCRExtractionResult, dict[str, Any]]] = []
        for expense_type, employee_id, file_path in iter_bills(root):
            if strategy == "vision_first":
                image_bytes = _image_bytes_for_vision(file_path)
                ocr_result = None
            else:
                try:
                    ocr_result, image_bytes = _ocr_from_path(file_path)
                except Exception as e:
                    logger.exception("OCR failed for %s: %s", file_path, e)
                    phase1.append((file_path, employee_id, expense_type, None, "fusion", None, None, True, OCRExtractionResult(), {}))
                    continue

            tid = self._trace_id()
            schema, source, ocr_c, llm_c, critical_failure, fusion_metadata, ocr_result_used = _get_structured_bill(
                image_bytes,
                employee_id=employee_id,
                expense_type=expense_type,
                ocr_confidence_threshold=self.ocr_confidence_threshold,
                llm_client=self.vision_client,
                vision_model=self.vision_model,
                max_retries=self.max_retries,
                retry_delay_sec=self.retry_delay_sec,
                trace_id=tid,
                vision_extractor=self.vision_extractor,
                donut_model_id=self.donut_model_id,
                layoutlm_model_id=self.layoutlm_model_id,
                extraction_strategy=strategy,
                ocr_result=ocr_result,
                path=file_path,
            )
            phase1.append((file_path, employee_id, expense_type, schema, source, ocr_c, llm_c, critical_failure, ocr_result_used or OCRExtractionResult(), fusion_metadata))

        # Group by (employee_id, month) for running monthly total
        groups: dict[tuple[str, str], list[tuple[Path, str, str, ReimbursementSchema | None, str, float | None, float | None, bool, OCRExtractionResult, dict[str, Any]]]] = defaultdict(list)
        for path, emp_id, exp_type, schema, source, ocr_c, llm_c, crit, ocr_result, fusion_metadata in phase1:
            month = schema.month if schema and schema.month else "unknown"
            groups[(emp_id, month)].append((path, emp_id, exp_type, schema, source, ocr_c, llm_c, crit, ocr_result, fusion_metadata))

        results: list[FinalOutputSchema] = []
        meal_cap = None
        try:
            ma = self.policy_json.get("meal_allowance")
            if isinstance(ma, dict) and ma.get("limit") is not None:
                meal_cap = float(ma["limit"])
        except (TypeError, ValueError):
            pass
        day_meal_used: dict[tuple[str, str], float] = {}  # (employee_id, bill_date) -> total approved that day

        for (employee_id, month), items in sorted(groups.items()):
            # Process same-day meal bills in amount order (smaller first) so per-day cap allocation is deterministic
            def _item_sort_key(item: tuple) -> tuple:
                _path, _eid, _etype, s, _src, _o, _l, _c, _ocr, _meta = item
                if s is None:
                    return ("z", 0.0)
                bd = s.bill_date.isoformat() if hasattr(s.bill_date, "isoformat") else str(s.bill_date or "")
                amt = float(s.amount) if s.amount is not None else 0.0
                return (bd or "z", amt)

            items_sorted = sorted(items, key=_item_sort_key)
            # Running total for this employee in this month (employee-level; passed to decision LLM)
            monthly_total = 0.0
            for path, emp_id, expense_type, schema, source, ocr_c, llm_c, critical_failure, ocr_result, fusion_metadata in items_sorted:
                tid = self._trace_id()
                start = time.perf_counter()
                if critical_failure or schema is None:
                    decision = auto_rejected_decision("Critical validation failed")
                else:
                    monthly_total += schema.amount
                    decision = self.decision_engine.get_decision(
                        schema,
                        self.policy_json,
                        expense_type,
                        monthly_total,
                        critical_failure=critical_failure,
                        trace_id=tid,
                    )
                    if fusion_metadata.get("has_major_numeric_conflict"):
                        decision = DecisionSchema(
                            decision="NEEDS_REVIEW",
                            confidence_score=decision.confidence_score,
                            reasoning=(decision.reasoning or "") + " [Fusion: numeric conflict between OCR and LLM]",
                            violated_rules=list(decision.violated_rules),
                            approved_amount=decision.approved_amount,
                        )
                    remaining_day_cap = None
                    if (expense_type or "").lower() == "meal" and meal_cap is not None and schema is not None:
                        bill_date_str = (
                            schema.bill_date.isoformat()
                            if hasattr(schema.bill_date, "isoformat")
                            else str(schema.bill_date or "")
                        )
                        if bill_date_str:
                            used = day_meal_used.get((employee_id, bill_date_str), 0.0)
                            remaining_day_cap = max(0.0, meal_cap - used)
                    decision = _apply_meal_cap(
                        decision, schema, self.policy_json, expense_type,
                        remaining_day_cap=remaining_day_cap,
                    )
                    if (expense_type or "").lower() == "meal" and schema is not None and decision.approved_amount is not None:
                        bill_date_str = (
                            schema.bill_date.isoformat()
                            if hasattr(schema.bill_date, "isoformat")
                            else str(schema.bill_date or "")
                        )
                        if bill_date_str:
                            key = (employee_id, bill_date_str)
                            day_meal_used[key] = day_meal_used.get(key, 0.0) + decision.approved_amount

                processing_time = time.perf_counter() - start
                expl_score = explainability_score_from_decision(decision) if schema else None
                structured_bill = schema.model_dump(mode="json") if schema else {}
                if structured_bill.get("bill_date") and hasattr(structured_bill.get("bill_date"), "isoformat"):
                    structured_bill["bill_date"] = str(structured_bill["bill_date"])

                all_extractor_outputs_batch: dict[str, Any] = {}
                if getattr(self, "output_all_extractors", False):
                    import os as _os
                    qwen_model = _os.getenv("QWEN_VL_MODEL") or self.vision_model
                    all_extractor_outputs_batch = _run_all_extractors(
                        path,
                        _image_bytes_for_vision(path),
                        emp_id,
                        expense_type,
                        self.donut_model_id,
                        self.vision_client,
                        qwen_model,
                    )

                results.append(
                    FinalOutputSchema(
                        trace_id=tid,
                        file=path.name,
                        policy_version_hash=self.policy_version_hash,
                        extraction_source=source,  # type: ignore[arg-type]
                        structured_bill=structured_bill,
                        policy_used=self.policy_json,
                        decision=decision.model_dump(mode="json"),
                        metadata=FinalOutputMetadata(
                            ocr_confidence=ocr_c,
                            llm_model=self.decision_engine.decision_model,
                            processing_time_sec=round(processing_time, 4),
                            explainability_score=expl_score,
                        ),
                        ocr_extracted=ocr_result.model_dump(mode="json"),
                        fusion_metadata=fusion_metadata,
                        all_extractor_outputs=all_extractor_outputs_batch,
                    )
                )
                self._audit_log(tid, "decision", {"file": str(path), "decision": decision.decision})

        return results
