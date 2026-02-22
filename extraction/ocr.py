"""
OCR extraction with pluggable engines (Tesseract, EasyOCR) and preprocessing providers.
BaseOCREngine + TesseractEngine / EasyOCREngine; BasePreprocessor + PIL / OpenCV / Deskew.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None  # type: ignore

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore
    np = None  # type: ignore

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None  # type: ignore

_ocr_backends_printed = False
TESSERACT_CONFIG = "--psm 11 --oem 3"
MIN_SIDE_PX = 300


def _log_backend_once(engine_name: str, preprocessor_name: str) -> None:
    global _ocr_backends_printed
    if _ocr_backends_printed:
        return
    _ocr_backends_printed = True
    msg = f"OCR: engine={engine_name}, preprocessor={preprocessor_name}"
    logger.info(msg)


# ---------------------------------------------------------------------------
# Base preprocessor
# ---------------------------------------------------------------------------


class BasePreprocessor(ABC):
    """Abstract image preprocessor for OCR. Returns PIL Image (e.g. grayscale for Tesseract)."""

    @property
    def name(self) -> str:
        return "base"

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Return preprocessed image (same or new)."""
        ...


class NoOpPreprocessor(BasePreprocessor):
    """No preprocessing; return image as-is."""

    @property
    def name(self) -> str:
        return "none"

    def preprocess(self, image: Image.Image) -> Image.Image:
        return image


class PILPreprocessor(BasePreprocessor):
    """Grayscale, resize up if small, sharpen, contrast. No OpenCV."""

    @property
    def name(self) -> str:
        return "pil"

    def preprocess(self, image: Image.Image) -> Image.Image:
        if image.mode != "L":
            image = image.convert("L")
        min_side = min(image.size)
        if min_side > 0 and min_side < MIN_SIDE_PX:
            scale = MIN_SIDE_PX / min_side
            new_w = max(MIN_SIDE_PX, int(image.width * scale))
            new_h = max(MIN_SIDE_PX, int(image.height * scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        return image


class OpenCVDenoisePreprocessor(BasePreprocessor):
    """PIL-style steps plus OpenCV denoise. Requires cv2/numpy."""

    @property
    def name(self) -> str:
        return "opencv"

    def preprocess(self, image: Image.Image) -> Image.Image:
        img = image.convert("L")
        min_side = min(img.size)
        if min_side > 0 and min_side < MIN_SIDE_PX:
            scale = MIN_SIDE_PX / min_side
            new_w = max(MIN_SIDE_PX, int(img.width * scale))
            new_h = max(MIN_SIDE_PX, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img = img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        if CV2_AVAILABLE and cv2 is not None and np is not None:
            arr = np.array(img)
            denoised = cv2.fastNlMeansDenoising(arr, None, h=10, templateWindowSize=7, searchWindowSize=21)
            img = Image.fromarray(denoised)
        return img


class DeskewPreprocessor(BasePreprocessor):
    """Wraps another preprocessor; runs OpenCV deskew first (if available)."""

    def __init__(self, inner: BasePreprocessor | None = None) -> None:
        self._inner = inner or PILPreprocessor()

    @property
    def name(self) -> str:
        return f"deskew+{self._inner.name}"

    def preprocess(self, image: Image.Image) -> Image.Image:
        image = self._deskew(image)
        return self._inner.preprocess(image)

    def _deskew(self, image: Image.Image) -> Image.Image:
        if not CV2_AVAILABLE or cv2 is None or np is None:
            return image
        gray = np.array(image.convert("L"))
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
            h, w = image.size[1], image.size[0]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            arr = np.array(image)
            rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return Image.fromarray(rotated)
        except Exception as e:
            logger.debug("Deskew failed: %s", e)
            return image


# ---------------------------------------------------------------------------
# Base OCR engine
# ---------------------------------------------------------------------------


class BaseOCREngine(ABC):
    """Abstract OCR engine: PIL Image -> (text, confidence)."""

    @property
    def name(self) -> str:
        return "base"

    @abstractmethod
    def run(self, image: Image.Image) -> tuple[str, float]:
        """Extract text and confidence (0-1) from image."""
        ...


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR. Uses grayscale preprocessed image."""

    def __init__(self, config: str = TESSERACT_CONFIG, preprocessor: BasePreprocessor | None = None) -> None:
        if not TESSERACT_AVAILABLE or pytesseract is None:
            raise RuntimeError("pytesseract is not installed")
        self._config = config
        self._preprocessor = preprocessor or PILPreprocessor()

    @property
    def name(self) -> str:
        return "tesseract"

    def run(self, image: Image.Image) -> tuple[str, float]:
        image = self._preprocessor.preprocess(image)
        data = pytesseract.image_to_data(image, config=self._config, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(image, config=self._config).strip()
        confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
        mean_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        return text, min(1.0, max(0.0, mean_conf))


class EasyOCREngine(BaseOCREngine):
    """EasyOCR. Uses RGB; optional resize for small images. No grayscale/denoise."""

    _reader: Any = None

    def __init__(self, preprocessor: BasePreprocessor | None = None) -> None:
        if not EASYOCR_AVAILABLE or easyocr is None:
            raise RuntimeError("easyocr is not installed; pip install easyocr or .[ocr]")
        self._preprocessor = preprocessor or NoOpPreprocessor()

    @property
    def name(self) -> str:
        return "easyocr"

    def run(self, image: Image.Image) -> tuple[str, float]:
        image = image.convert("RGB")
        min_side = min(image.size)
        if min_side > 0 and min_side < MIN_SIDE_PX:
            scale = MIN_SIDE_PX / min_side
            new_w = max(MIN_SIDE_PX, int(image.width * scale))
            new_h = max(MIN_SIDE_PX, int(image.height * scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if EasyOCREngine._reader is None:
            EasyOCREngine._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        import numpy as np_arr
        arr = np_arr.array(image)
        result = EasyOCREngine._reader.readtext(arr)
        if not result:
            return "", 0.0
        texts = [item[1] for item in result]
        confidences = [item[2] for item in result]
        text = "\n".join(texts).strip()
        mean_conf = sum(confidences) / len(confidences)
        return text, min(1.0, max(0.0, float(mean_conf)))


# ---------------------------------------------------------------------------
# Factory: create engine and preprocessor by name
# ---------------------------------------------------------------------------


def create_preprocessor(kind: str = "auto", deskew: bool = True) -> BasePreprocessor:
    """Create preprocessor. kind: 'none' | 'pil' | 'opencv' | 'auto'."""
    k = (kind or "auto").strip().lower()
    if k == "none":
        return NoOpPreprocessor()
    if k == "pil":
        return DeskewPreprocessor(PILPreprocessor()) if deskew and CV2_AVAILABLE else PILPreprocessor()
    if k == "opencv":
        base = OpenCVDenoisePreprocessor() if CV2_AVAILABLE else PILPreprocessor()
        return DeskewPreprocessor(base) if deskew and CV2_AVAILABLE else base
    # auto: opencv+deskew if available, else pil
    if CV2_AVAILABLE:
        base = OpenCVDenoisePreprocessor()
        return DeskewPreprocessor(base) if deskew else base
    return DeskewPreprocessor(PILPreprocessor()) if deskew else PILPreprocessor()


def create_ocr_engine(
    engine: str = "tesseract",
    *,
    preprocessor: BasePreprocessor | None = None,
    preprocessor_kind: str = "auto",
    deskew: bool = True,
) -> BaseOCREngine:
    """Create OCR engine by name. preprocessor used for Tesseract; EasyOCR uses minimal resize only."""
    e = (engine or "tesseract").strip().lower()
    if e not in ("tesseract", "easyocr"):
        e = "tesseract"
    if e == "easyocr" and not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr not installed; pip install easyocr or .[ocr]")
    prep = preprocessor or create_preprocessor(kind=preprocessor_kind, deskew=deskew)
    if e == "tesseract":
        eng = TesseractEngine(preprocessor=prep)
    else:
        eng = EasyOCREngine(preprocessor=NoOpPreprocessor())
    _log_backend_once(eng.name, eng._preprocessor.name if hasattr(eng, "_preprocessor") else "default")
    return eng


def default_engine_name() -> str:
    """Engine name from OCR_ENGINE env; falls back to tesseract if easyocr requested but not installed."""
    name = (os.getenv("OCR_ENGINE") or "tesseract").strip().lower()
    if name == "easyocr" and not EASYOCR_AVAILABLE:
        return "tesseract"
    return name if name in ("tesseract", "easyocr") else "tesseract"


def run_engine_on_images(engine: BaseOCREngine, images: list[Image.Image]) -> tuple[str, float]:
    """Run OCR engine on each image; return combined text and average confidence."""
    if not images:
        return "", 0.0
    texts: list[str] = []
    confs: list[float] = []
    for img in images:
        text, conf = engine.run(img)
        texts.append(text)
        confs.append(conf)
    combined = "\n\n".join(texts)
    avg_conf = sum(confs) / len(confs)
    return combined, min(1.0, max(0.0, avg_conf))
