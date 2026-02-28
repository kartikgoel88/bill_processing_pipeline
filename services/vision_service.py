"""
Vision extraction service: image bytes + context -> structured bill + confidence.
Uses ILLMProvider (injected); supports donut/layoutlm/vision_llm via config.
"""

from __future__ import annotations

import logging
from typing import Any

from core.interfaces import IVisionService, ILLMProvider
from core.models import ExtractionResult
from core.exceptions import VisionExtractionError

from extraction.parser import (
    _bill_extraction_system_prompt,
    _bill_extraction_vision_prompt,
    _bill_extraction_from_text_prompt,
    parse_llm_extraction_multi,
)
logger = logging.getLogger(__name__)


def _vision_prompt() -> str:
    return f"{_bill_extraction_system_prompt()}\n\n---\n\n{_bill_extraction_vision_prompt()}"


def _parse_llm_extraction_to_bills(text: str, context: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Parse LLM vision output to one or more structured bill dicts.
    Returns (structured_bill, structured_bills). When multiple bills: structured_bills has all; structured_bill is first.
    """
    emp = context.get("employee_id") or ""
    exp = context.get("expense_type") or "meal"
    bills = parse_llm_extraction_multi(text, employee_id=emp, expense_type=exp)
    if not bills:
        return ({}, [])
    if len(bills) == 1:
        return (bills[0], [])
    return (bills[0], bills)


class VisionService(IVisionService):
    """Vision extraction using injected LLM provider. No concrete LLM imports."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        model: str = "llava",
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
    ) -> None:
        self._llm = llm_provider
        self._model = model
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec

    def extract(self, image_bytes: bytes, context: dict[str, Any]) -> ExtractionResult:
        if not image_bytes:
            return ExtractionResult(
                structured_bill={},
                confidence=0.0,
                source="vision_llm",
                critical_validation_failed=True,
            )
        logger.info("Calling vision LLM (model=%s)", self._model)
        prompt = _vision_prompt()
        try:
            from utils.image_utils import image_to_data_url
            data_url = image_to_data_url(image_bytes)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            messages = [{"role": "user", "content": content}]
            text = self._llm.chat_vision(messages, model=self._model, max_tokens=4096)
            logger.debug("Vision LLM response length: %s", len(text or ""))
        except Exception as e:
            logger.warning("Vision LLM failed: %s", e)
            raise VisionExtractionError(f"Vision extraction failed: {e}") from e
        try:
            structured_bill, structured_bills = _parse_llm_extraction_to_bills(text, context)
        except Exception as e:
            logger.warning("Parse LLM extraction failed: %s", e)
            return ExtractionResult(
                structured_bill={},
                confidence=0.0,
                source="vision_llm",
                critical_validation_failed=True,
            )
        if not structured_bill and not structured_bills:
            return ExtractionResult(
                structured_bill={},
                confidence=0.0,
                source="vision_llm",
                critical_validation_failed=True,
            )
        return ExtractionResult(
            structured_bill=structured_bill,
            confidence=0.85,
            source="vision_llm",
            structured_bills=structured_bills,
        )

    def extract_from_text(self, raw_text: str, context: dict[str, Any]) -> ExtractionResult:
        """
        Extract structured bill from raw OCR text using the same LLM (text-only, no image).
        Use when ocr_extraction=llm to get fields via LLM instead of regex parsing.
        """
        if not (raw_text and raw_text.strip()):
            return ExtractionResult(
                structured_bill={},
                confidence=0.0,
                source="ocr_llm",
                critical_validation_failed=True,
            )
        logger.info("Calling LLM for extraction from OCR text (model=%s)", self._model)
        system = _bill_extraction_system_prompt()
        user_content = _bill_extraction_from_text_prompt(raw_text)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        try:
            text = self._llm.chat(messages, model=self._model, max_tokens=4096)
        except Exception as e:
            logger.warning("LLM extraction from text failed: %s", e)
            raise VisionExtractionError(f"LLM extraction from OCR text failed: {e}") from e
        try:
            structured_bill, structured_bills = _parse_llm_extraction_to_bills(text, context)
        except Exception as e:
            logger.warning("Parse LLM extraction from text failed: %s", e)
            return ExtractionResult(
                structured_bill={},
                confidence=0.0,
                source="ocr_llm",
                critical_validation_failed=True,
            )
        if not structured_bill and not structured_bills:
            return ExtractionResult(
                structured_bill={},
                confidence=0.0,
                source="ocr_llm",
                critical_validation_failed=True,
            )
        return ExtractionResult(
            structured_bill=structured_bill,
            confidence=0.85,
            source="ocr_llm",
            structured_bills=structured_bills,
            ocr_raw_text=raw_text,
            ocr_confidence=0.9,
        )
