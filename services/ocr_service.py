"""
OCR service: implements IOCRService using extraction.ocr engine API.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.interfaces import IOCRService
from core.exceptions import OCRError

logger = logging.getLogger(__name__)


def _extract_from_path(path: Path, engine_name: str, dpi: int = 300) -> tuple[str, float]:
    from extraction.image_io import PathImageReader
    from extraction.ocr import create_ocr_engine, run_engine_on_images

    path = Path(path)
    if not path.exists():
        raise OCRError(f"File not found: {path}", trace_id=None)
    eng = create_ocr_engine(engine_name)
    images = PathImageReader(path, dpi=dpi).read()
    text, confidence = run_engine_on_images(eng, images)
    if path.suffix.lower() == ".pdf" and images:
        logger.info(
            "OCR from PDF: %s pages, dpi=%s, engine=%s, combined length %s, avg confidence %.3f",
            len(images), dpi, eng.name, len(text), confidence,
        )
    return (text or "", float(confidence))


def _extract_from_bytes(data: bytes, is_pdf: bool, engine_name: str, dpi: int = 300) -> tuple[str, float]:
    from extraction.image_io import BytesImageReader
    from extraction.ocr import create_ocr_engine, run_engine_on_images

    eng = create_ocr_engine(engine_name)
    images = BytesImageReader(data, is_pdf=is_pdf, dpi=dpi).read()
    text, confidence = run_engine_on_images(eng, images)
    return (text or "", float(confidence))


class OCRService(IOCRService):
    """Production OCR service: path or bytes -> (text, confidence). Engine from config."""

    def __init__(self, engine: str = "tesseract", dpi: int = 300) -> None:
        self._engine = (engine or "tesseract").strip().lower()
        self._dpi = dpi

    def extract_from_path(self, path: Path) -> tuple[str, float]:
        try:
            return _extract_from_path(Path(path), self._engine, dpi=self._dpi)
        except Exception as e:
            raise OCRError(f"OCR failed: {e}", trace_id=None) from e

    def extract_from_bytes(self, data: bytes, is_pdf: bool) -> tuple[str, float]:
        try:
            return _extract_from_bytes(data, is_pdf, self._engine, dpi=self._dpi)
        except Exception as e:
            raise OCRError(f"OCR from bytes failed: {e}", trace_id=None) from e
