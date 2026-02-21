"""
OCR extraction using Tesseract with image preprocessing for low-quality scans/PDFs.
- PDF input → convert pages to images (configurable DPI), preprocess, then run Tesseract.
- Image input → preprocess then run Tesseract.
Preprocessing: grayscale, resize-up for small images, sharpen, contrast; optional OpenCV denoise/adaptive threshold/deskew.
Returns raw text and a confidence score (0-1).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from commons.schema import OCRExtractionResult

logger = logging.getLogger(__name__)

# One-time print of OCR backends (Tesseract + OpenCV/PIL)
_ocr_backends_printed = False


def _print_ocr_backends(use_opencv: bool = True) -> None:
    """Print which OCR method is being used: Tesseract (text), OpenCV or PIL (preprocessing)."""
    global _ocr_backends_printed
    if _ocr_backends_printed:
        return
    _ocr_backends_printed = True
    text_engine = "Tesseract" if TESSERACT_AVAILABLE else "none (Tesseract not available)"
    preprocess_engine = "OpenCV" if (use_opencv and CV2_AVAILABLE) else "PIL only"
    msg = f"OCR: text engine={text_engine}, preprocessing={preprocess_engine}"
    print(msg)
    logger.info(msg)


# Optional: pdf2image (requires poppler)
try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    convert_from_path = None  # type: ignore

# Optional: Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None  # type: ignore

# Optional: OpenCV for advanced preprocessing (denoise, adaptive threshold, deskew)
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Preprocessing for low-quality images
# ---------------------------------------------------------------------------

def _preprocess_pil_only(image: Image.Image) -> Image.Image:
    """
    Preprocess using only PIL: grayscale, resize up if small, sharpen, contrast.
    Safe to use when OpenCV is not installed.
    """
    # Grayscale often improves Tesseract on noisy color scans
    if image.mode != "L":
        image = image.convert("L")
    # Resize up small images (Tesseract works better with ~300+ px min dimension)
    min_side = min(image.size)
    if min_side > 0 and min_side < 300:
        scale = 300 / min_side
        new_w = max(300, int(image.width * scale))
        new_h = max(300, int(image.height * scale))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Slight sharpen to recover edges in blurry scans
    image = image.filter(ImageFilter.SHARPEN)
    # Boost contrast (helps faded/low-contrast scans)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    return image


def _preprocess_with_cv2(image: Image.Image) -> Image.Image:
    """
    When OpenCV is available: denoise, optional adaptive threshold, then return PIL.
    Keeps grayscale/resize/sharpen/contrast from PIL; adds denoise and optional binarization.
    """
    img_gray = image.convert("L")
    min_side = min(img_gray.size)
    if min_side > 0 and min_side < 300:
        scale = 300 / min_side
        new_w = max(300, int(img_gray.width * scale))
        new_h = max(300, int(img_gray.height * scale))
        img_gray = img_gray.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_gray = img_gray.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img_gray)
    img_gray = enhancer.enhance(1.3)

    if not CV2_AVAILABLE or cv2 is None or np is None:
        return img_gray

    arr = np.array(img_gray)
    # Denoise (helps speckled/low-quality scans)
    denoised = cv2.fastNlMeansDenoising(arr, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Optional: adaptive threshold for very faint or uneven lighting (can hurt good images)
    # Uncomment below to force binarization for very poor quality:
    # denoised = cv2.adaptiveThreshold(
    #     denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    return Image.fromarray(denoised)


def _deskew_cv2(image: Image.Image) -> Image.Image:
    """Detect skew angle and rotate image to straighten text. OpenCV only."""
    if not CV2_AVAILABLE or cv2 is None or np is None:
        return image
    gray = np.array(image.convert("L"))
    # Invert if background is white so minAreaRect sees text as foreground
    if np.mean(gray) > 127:
        gray = 255 - gray
    coords = np.column_stack(np.where(gray > 0))
    if coords.size < 100:
        return image
    try:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        if abs(angle) < 0.5:
            return image
        (h, w) = image.size[1], image.size[0]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        arr = np.array(image)
        rotated = cv2.warpAffine(
            arr,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(rotated)
    except Exception as e:
        logger.debug("Deskew failed: %s", e)
        return image


def preprocess_for_ocr(
    image: Image.Image,
    *,
    use_opencv: bool = True,
    deskew: bool = True,
) -> Image.Image:
    """
    Preprocess image for better OCR on low-quality scans/PDFs.
    - use_opencv: if True and OpenCV available, apply denoising.
    - deskew: if True and OpenCV available, try to straighten rotated text.
    Returns a single-band (grayscale) PIL Image suitable for Tesseract.
    """
    if use_opencv and CV2_AVAILABLE and deskew:
        image = _deskew_cv2(image)
    if use_opencv and CV2_AVAILABLE:
        image = _preprocess_with_cv2(image)
    else:
        image = _preprocess_pil_only(image)
    return image


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

# Tesseract config: PSM 6 = assume uniform block of text (good for receipts/bills)
TESSERACT_CONFIG = "--psm 6 --oem 3"


def _run_tesseract_on_image(
    image: Image.Image,
    *,
    config: str = TESSERACT_CONFIG,
) -> tuple[str, float]:
    """Run Tesseract on a PIL Image; return (text, mean_confidence)."""
    if not TESSERACT_AVAILABLE or pytesseract is None:
        raise RuntimeError("pytesseract is not installed")
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(image, config=config).strip()
    confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
    mean_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
    return text, min(1.0, max(0.0, mean_conf))


def extract_from_image(
    image: Union[Image.Image, Path, str],
    *,
    preprocess: bool = True,
    use_opencv: bool = True,
    deskew: bool = True,
) -> OCRExtractionResult:
    """
    Extract text from a single image (PIL Image, path to image file).
    When preprocess=True (default), applies preprocessing for low-quality images.
    """
    _print_ocr_backends(use_opencv=use_opencv)
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image, path string, or Path")
    if preprocess:
        image = preprocess_for_ocr(image, use_opencv=use_opencv, deskew=deskew)
    text, confidence = _run_tesseract_on_image(image)
    return OCRExtractionResult(raw_text=text, confidence=confidence)


def extract_from_pdf(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    first_page: int | None = None,
    last_page: int | None = None,
    *,
    preprocess: bool = True,
    use_opencv: bool = True,
    deskew: bool = True,
) -> OCRExtractionResult:
    """
    Convert PDF pages to images and run Tesseract on each (with optional preprocessing).
    Higher DPI (e.g. 300) improves quality for low-resolution PDFs.
    first_page/last_page are 1-based inclusive.
    """
    _print_ocr_backends(use_opencv=use_opencv)
    if not PDF_AVAILABLE or convert_from_path is None:
        raise RuntimeError("pdf2image is not installed; install it and poppler")
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    pages = convert_from_path(str(path), dpi=dpi, first_page=first_page, last_page=last_page)
    all_text: list[str] = []
    all_conf: list[float] = []
    for i, img in enumerate(pages):
        if preprocess:
            img = preprocess_for_ocr(img, use_opencv=use_opencv, deskew=deskew)
        text, conf = _run_tesseract_on_image(img)
        all_text.append(text)
        all_conf.append(conf)
    combined = "\n\n".join(all_text)
    avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0
    logger.info(
        "OCR from PDF: %s pages, dpi=%s, preprocess=%s, combined length %s, avg confidence %.3f",
        len(pages), dpi, preprocess, len(combined), avg_conf,
    )
    return OCRExtractionResult(raw_text=combined, confidence=avg_conf)


def extract_from_bytes(
    data: bytes,
    is_pdf: bool = False,
    dpi: int = 300,
    *,
    preprocess: bool = True,
    use_opencv: bool = True,
    deskew: bool = True,
) -> OCRExtractionResult:
    """
    Extract from in-memory bytes (image or PDF).
    Preprocessing is applied when preprocess=True (default) for low-quality content.
    """
    _print_ocr_backends(use_opencv=use_opencv)
    if is_pdf:
        if not PDF_AVAILABLE or convert_from_path is None:
            raise RuntimeError("pdf2image required for PDF bytes")
        pages = convert_from_path(io.BytesIO(data), dpi=dpi)
        all_text, all_conf = [], []
        for img in pages:
            if preprocess:
                img = preprocess_for_ocr(img, use_opencv=use_opencv, deskew=deskew)
            t, c = _run_tesseract_on_image(img)
            all_text.append(t)
            all_conf.append(c)
        combined = "\n\n".join(all_text)
        avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0
        return OCRExtractionResult(raw_text=combined, confidence=avg_conf)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if preprocess:
        image = preprocess_for_ocr(image, use_opencv=use_opencv, deskew=deskew)
    text, confidence = _run_tesseract_on_image(image)
    return OCRExtractionResult(raw_text=text, confidence=confidence)
