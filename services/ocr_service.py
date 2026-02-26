"""
OCR service: implements IOCRService using extraction.ocr engine API.
Uses image_io reader strategies (PathImageReader, BytesImageReader); one extraction path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.interfaces import IOCRService
from core.exceptions import OCRError

from extraction.ocr import create_ocr_engine, run_engine_on_images
from extraction.image_io import PathImageReader
from extraction.image_io import BytesImageReader
from extraction.pdf_text import extract_text_from_pdf, MIN_PDF_TEXT_LEN


if TYPE_CHECKING:
    from extraction.image_io import IImageReader

logger = logging.getLogger(__name__)


def _extract_from_reader(
    reader: IImageReader,
    engine_name: str,
    *,
    path: Path | None = None,
    dpi: int = 300,
) -> tuple[str, float]:
    """Run OCR on images from any reader (path, bytes, etc.). path optional for PDF logging."""
    

    eng = create_ocr_engine(engine_name)
    images = reader.read()
    text, confidence = run_engine_on_images(eng, images)
    if path is not None and path.suffix.lower() == ".pdf" and images:
        logger.info(
            "OCR from PDF: %s pages, dpi=%s, engine=%s, combined length %s, avg confidence %.3f",
            len(images), dpi, eng.name, len(text), confidence,
        )
    return (text or "", float(confidence))


def _extract_from_path(path: Path, engine_name: str, dpi: int = 300) -> tuple[str, float]:
    path = Path(path)
    if not path.exists():
        raise OCRError(f"File not found: {path}", trace_id=None)
    # For PDFs: try native text extraction first (digital PDFs have embedded text)
    if path.suffix.lower() == ".pdf":
        pdf_text = extract_text_from_pdf(path)
        if pdf_text and len(pdf_text.strip()) >= MIN_PDF_TEXT_LEN:
            logger.info(
                "Using native PDF text for %s (len=%s); skipping OCR",
                path.name,
                len(pdf_text),
            )
            return (pdf_text.strip(), 1.0)
    return _extract_from_reader(
        PathImageReader(path, dpi=dpi),
        engine_name,
        path=path,
        dpi=dpi,
    )


def _extract_from_bytes(data: bytes, is_pdf: bool, engine_name: str, dpi: int = 300) -> tuple[str, float]:

    return _extract_from_reader(
        BytesImageReader(data, is_pdf=is_pdf, dpi=dpi),
        engine_name,
        dpi=dpi,
    )


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
