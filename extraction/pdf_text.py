"""
Extract native text from PDFs (digital/vector PDFs with embedded text).
When a PDF has selectable text, this yields clean text without OCR.
For image-only (scanned) PDFs, returns empty or minimal text — caller should fall back to OCR.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum character count to consider PDF "text-based" (avoid treating scanned PDFs as text)
MIN_PDF_TEXT_LEN = 80


def extract_text_from_pdf(path: str | Path) -> str:
    """
    Extract text from all pages of a PDF using pypdf (native PDF text, no OCR).
    Returns empty string on error or if the PDF has no extractable text (e.g. scanned image).
    """
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text and isinstance(text, str):
                    parts.append(text.strip())
            except Exception as e:
                logger.debug("PDF page text extraction failed: %s", e)
        return "\n\n".join(parts).strip() if parts else ""
    except Exception as e:
        logger.debug("PDF text extraction failed for %s: %s", path, e)
        return ""
