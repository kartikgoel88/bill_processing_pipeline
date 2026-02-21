"""
Policy reader: read policy PDF and extract text via pypdf or OCR.
Used by the Policy Ingestion Pipeline (Step 1).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional pypdf for text-based PDFs
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    PdfReader = None  # type: ignore


def read_policy_pdf(
    path: str | Path,
    *,
    use_ocr_fallback: bool = True,
    skip_cover_pages: bool = True,
) -> str:
    """
    Read policy PDF and return extracted text.

    1. Tries pypdf first (fast for text-based PDFs).
    2. If insufficient text or pypdf fails, falls back to OCR via ocr_engine.
    3. When using OCR and skip_cover_pages=True, only pages 2 to n-1 are extracted
       (skips cover and trailing "Change Record" / end pages) for cleaner LLM input.

    :param path: Path to policy PDF file.
    :param use_ocr_fallback: If True, use OCR when pypdf yields little/no text.
    :param skip_cover_pages: If True and OCR is used, skip first and last page (cover/TOC).
    :return: Extracted text (may be empty).
    :raises FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy PDF not found: {path}")

    text = ""
    page_count: int | None = None

    # Try pypdf first
    if PYPDF_AVAILABLE and PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
            parts: list[str] = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            text = "\n".join(parts).strip()
            if len(text) >= 50:
                logger.info("Policy PDF text extracted with pypdf (%s chars)", len(text))
                return text
        except Exception as e:
            logger.debug("pypdf extraction failed for %s: %s", path, e)

    # Fall back to OCR for image-based/scanned PDFs
    if use_ocr_fallback:
        try:
            from bills.ocr_engine import extract_from_pdf
            first_page, last_page = None, None
            if skip_cover_pages:
                if page_count is None and PYPDF_AVAILABLE and PdfReader is not None:
                    try:
                        page_count = len(PdfReader(str(path)).pages)
                    except Exception:
                        pass
                if page_count is not None and page_count > 2:
                    first_page, last_page = 2, page_count - 1
                    logger.info("Policy OCR: skipping cover pages (using %s–%s of %s)", first_page, last_page, page_count)
            result = extract_from_pdf(path, dpi=150, first_page=first_page, last_page=last_page)
            text = (result.raw_text or "").strip()
            logger.info("Policy PDF text extracted via OCR (%s chars)", len(text))
            if text:
                logger.debug("Policy OCR text preview: %s...", (text[:200] + "..." if len(text) > 200 else text))
        except Exception as e:
            logger.warning("OCR fallback failed for policy PDF %s: %s", path, e)

    return text
