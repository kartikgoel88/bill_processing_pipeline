"""
Vision extraction service: image bytes + context -> structured bill + confidence.
Uses ILLMProvider (injected); supports donut/layoutlm/vision_llm via config.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from core.interfaces import IVisionService, ILLMProvider
from core.models import ExtractionResult
from core.exceptions import VisionExtractionError

from PIL import Image
from extraction.ocr import load_images_from_path

from extraction.parser import _bill_extraction_system_prompt, _bill_extraction_vision_prompt

from extraction.parser import parse_llm_extraction
from core.schema import ReimbursementSchema

logger = logging.getLogger(__name__)

VISION_IMAGE_MAX_PX = 1024
VISION_JPEG_QUALITY = 85
VISION_FIRST_PAGE_DPI = 150


def _image_bytes_from_path(path: Path) -> bytes:
    """First page as JPEG, size-limited. Uses extraction.load_images_from_path (first page only) then resize + encode."""

    path = Path(path)
    try:
        images = load_images_from_path(path, dpi=VISION_FIRST_PAGE_DPI, first_page_only=True)
    except (FileNotFoundError, RuntimeError):
        return b""
    if not images:
        return b""
    img = images[0]
    w, h = img.size
    if max(w, h) > VISION_IMAGE_MAX_PX:
        ratio = VISION_IMAGE_MAX_PX / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=VISION_JPEG_QUALITY)
    return buf.getvalue()


def _vision_prompt() -> str:
    
    return f"{_bill_extraction_system_prompt()}\n\n---\n\n{_bill_extraction_vision_prompt()}"


def _parse_llm_extraction(text: str, context: dict[str, Any]) -> dict[str, Any]:
    """Parse LLM vision output to structured bill dict."""
    
    emp = context.get("employee_id") or ""
    exp = context.get("expense_type") or "meal"
    schema = parse_llm_extraction(text, employee_id=emp, expense_type=exp)
    return schema.model_dump(mode="json") if isinstance(schema, ReimbursementSchema) else schema


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
            structured = _parse_llm_extraction(text, context)
        except Exception as e:
            logger.warning("Parse LLM extraction failed: %s", e)
            return ExtractionResult(
                structured_bill={},
                confidence=0.0,
                source="vision_llm",
                critical_validation_failed=True,
            )
        return ExtractionResult(
            structured_bill=structured,
            confidence=0.85,
            source="vision_llm",
        )


def image_bytes_for_vision(path: Path) -> bytes:
    """Helper: get image bytes for vision from file path (PDF or image)."""
    return _image_bytes_from_path(Path(path))
