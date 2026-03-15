"""
OCR extraction with pluggable engines (Tesseract, EasyOCR) and preprocessing providers.
BaseOCREngine + TesseractEngine / EasyOCREngine; BasePreprocessor + PIL / OpenCV / Deskew.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
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


def _tesseract_data_to_text(data: dict[str, Any]) -> str:
    """
    Build text from Tesseract image_to_data preserving block/line layout.
    Words on the same line are space-separated; lines are newline-separated.
    Improves receipt/table extraction vs raw image_to_string.
    """
    n = len(data.get("text", []))
    if n == 0:
        return ""
    lines: dict[tuple[int, int, int], list[str]] = {}  # (block, par, line) -> words
    for i in range(n):
        text = (data.get("text") or [])[i]
        if not isinstance(text, str) or not text.strip():
            continue
        conf = data.get("conf", [])
        if conf and i < len(conf):
            try:
                if int(conf[i]) < 0:
                    continue  # Tesseract uses -1 for non-text
            except (TypeError, ValueError):
                pass
        block = int((data.get("block_num") or [0])[i]) if i < len(data.get("block_num") or []) else 0
        par = int((data.get("par_num") or [0])[i]) if i < len(data.get("par_num") or []) else 0
        line = int((data.get("line_num") or [0])[i]) if i < len(data.get("line_num") or []) else 0
        key = (block, par, line)
        lines.setdefault(key, []).append(text.strip())
    # Sort by block, par, line and join
    sorted_keys = sorted(lines.keys())
    return "\n".join(" ".join(lines[k]) for k in sorted_keys).strip()


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


def _make_rupee_templates() -> list[Any]:
    """Build small template images of ₹ for OpenCV matchTemplate. Returns list of grayscale template arrays."""
    if not CV2_AVAILABLE or cv2 is None or np is None:
        return []
    templates: list[Any] = []
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return []
    # Try font names that often include ₹ (DejaVu, Liberation, Arial, Noto)
    font_names = [
        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
        "LiberationSans-Regular.ttf", "arial.ttf", "Arial.ttf",
        "NotoSansDevanagari-Regular.ttf", "NotoSans-Regular.ttf",
    ]
    import sys
    if sys.platform == "darwin":
        font_names.extend(["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"])
    font_path: str | None = None
    for name in font_names:
        try:
            f = ImageFont.truetype(name, 24)
            img = Image.new("L", (40, 40), 255)
            d = ImageDraw.Draw(img)
            d.text((2, 2), "₹", font=f, fill=0)
            if np.array(img).min() < 255:
                font_path = name
                break
        except (OSError, IOError):
            continue
    if font_path is None:
        logger.debug("No font with ₹ found; rupee masking disabled")
        return []
    for size in (16, 24, 32):
        try:
            f = ImageFont.truetype(font_path, size)
            pad = max(4, size // 4)
            w, h = size + 2 * pad, size + 2 * pad
            img = Image.new("L", (w, h), 255)
            d = ImageDraw.Draw(img)
            d.text((pad, pad), "₹", font=f, fill=0)
            arr = np.array(img)
            if arr.min() < 255:
                templates.append(arr)
        except Exception:
            continue
    # Optional: load templates from folder (bold/light/rotated) for better detection
    _templates_dir = Path(__file__).resolve().parent / "templates" / "rupee"
    if _templates_dir.is_dir():
        for p in sorted(_templates_dir.glob("*.png")):
            try:
                timg = Image.open(p).convert("L")
                tarr = np.array(timg)
                if tarr.size > 0:
                    templates.append(tarr)
            except Exception as e:
                logger.debug("Could not load rupee template %s: %s", p, e)
    return templates


def _detect_rupee_regions(image_bgr: Any, templates: list[Any], threshold: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Run template matching; return list of (x, y, w, h) to mask. Merges overlapping boxes."""
    if not templates or image_bgr is None:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    boxes: list[tuple[int, int, int, int]] = []
    for tpl_gray in templates:
        th, tw = tpl_gray.shape[:2]
        if th > gray.shape[0] or tw > gray.shape[1]:
            continue
        try:
            res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                boxes.append((int(pt[0]), int(pt[1]), tw, th))
        except Exception:
            continue
    if not boxes:
        return []
    # Merge overlapping boxes (simple merge)
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes:
        overlap = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            if not (x + w < mx or x > mx + mw or y + h < my or y > my + mh):
                overlap = True
                break
        if not overlap:
            merged.append((x, y, w, h))
    return merged


def _shrink_rupee_boxes(
    boxes: list[tuple[int, int, int, int]],
    shrink_px: int,
    min_width: int = 4,
    min_height: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Shrink each box inward by shrink_px so masking does not erase digits next to ₹."""
    if shrink_px <= 0:
        return boxes
    out: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes:
        px = min(shrink_px, w // 2, h // 2)
        x2 = x + px
        y2 = y + px
        w2 = max(min_width, w - 2 * px)
        h2 = max(min_height, h - 2 * px)
        if w2 > 0 and h2 > 0:
            out.append((x2, y2, w2, h2))
    return out


class RupeeMaskPreprocessor(BasePreprocessor):
    """
    Mask ₹ symbol in the image (replace with white) before OCR using OpenCV template matching.
    Reduces OCR misreading ₹ as 2 or 7. Runs first, then delegates to inner preprocessor.
    """

    def __init__(
        self,
        inner: BasePreprocessor | None = None,
        threshold: float = 0.5,
        shrink_px: int = 2,
    ) -> None:
        self._inner = inner or PILPreprocessor()
        self._templates: list[Any] = []
        self._threshold = max(0.3, min(0.95, threshold))
        self._shrink_px = max(0, shrink_px)

    @property
    def name(self) -> str:
        return f"rupee_mask+{self._inner.name}"

    def preprocess(self, image: Image.Image) -> Image.Image:
        if not CV2_AVAILABLE or cv2 is None or np is None:
            return self._inner.preprocess(image)
        if not self._templates:
            self._templates = _make_rupee_templates()
        if not self._templates:
            return self._inner.preprocess(image)
        arr = np.array(image)
        if arr.ndim == 2:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        boxes = _detect_rupee_regions(img_bgr, self._templates, threshold=self._threshold)
        boxes = _shrink_rupee_boxes(boxes, self._shrink_px)
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 255, 255), -1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image_masked = Image.fromarray(img_rgb)
        return self._inner.preprocess(image_masked)


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
    """Tesseract OCR. Uses grayscale preprocessed image. Builds text from word-level data to preserve layout."""

    def __init__(self, config: str = TESSERACT_CONFIG, preprocessor: BasePreprocessor | None = None) -> None:
        if not TESSERACT_AVAILABLE or pytesseract is None:
            raise RuntimeError("pytesseract is not installed")
        # Prefer explicit binary (e.g. Homebrew 5.x) when multiple Tesseract installs exist
        tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
        if tesseract_cmd and os.path.isfile(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._config = config
        self._preprocessor = preprocessor or PILPreprocessor()

    @property
    def name(self) -> str:
        return "tesseract"

    def run(self, image: Image.Image) -> tuple[str, float]:
        image = self._preprocessor.preprocess(image)
        config = self._config
        try:
            return self._run_with_config(image, config)
        except Exception as e:
            err_msg = str(e)
            if "unknown command line argument" in err_msg and "--lang" in self._config:
                config = _tesseract_config_fallback_lang(self._config)
                return self._run_with_config(image, config)
            raise

    def _run_with_config(self, image: Image.Image, config: str) -> tuple[str, float]:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        text = _tesseract_data_to_text(data)
        if not text.strip():
            text = pytesseract.image_to_string(image, config=config).strip()
        confidences = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
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


def create_preprocessor(
    kind: str = "auto",
    deskew: bool = True,
    mask_rupee: bool = False,
    rupee_mask_threshold: float = 0.5,
    rupee_mask_shrink_px: int = 2,
) -> BasePreprocessor:
    """Create preprocessor. kind: 'none' | 'pil' | 'opencv' | 'auto'. mask_rupee: mask ₹ before OCR (template matching)."""
    k = (kind or "auto").strip().lower()
    if k == "none":
        base = NoOpPreprocessor()
    elif k == "pil":
        base = DeskewPreprocessor(PILPreprocessor()) if deskew and CV2_AVAILABLE else PILPreprocessor()
    elif k == "opencv":
        inner = OpenCVDenoisePreprocessor() if CV2_AVAILABLE else PILPreprocessor()
        base = DeskewPreprocessor(inner) if deskew and CV2_AVAILABLE else inner
    else:
        if CV2_AVAILABLE:
            inner = OpenCVDenoisePreprocessor()
            base = DeskewPreprocessor(inner) if deskew else inner
        else:
            base = DeskewPreprocessor(PILPreprocessor()) if deskew else PILPreprocessor()
    if mask_rupee and CV2_AVAILABLE:
        base = RupeeMaskPreprocessor(
            inner=base,
            threshold=rupee_mask_threshold,
            shrink_px=rupee_mask_shrink_px,
        )
    return base


def build_tesseract_config(
    psm: int = 11,
    oem: int = 3,
    lang: str = "eng",
) -> str:
    """Build Tesseract config string. psm 11 = sparse text; 6 = block; 3 = full page. oem 3 = LSTM only."""
    parts = [f"--psm {psm}", f"--oem {oem}"]
    if lang and lang.strip():
        parts.append(f"--lang {lang.strip()}")  # Tesseract 4.1+; fallback to -l in run() if unsupported
    return " ".join(parts)


def _tesseract_config_fallback_lang(config: str) -> str:
    """Replace --lang with -l for older Tesseract that don't support --lang."""
    if "--lang " in config:
        return config.replace("--lang ", "-l ")
    return config


def create_ocr_engine(
    engine: str = "tesseract",
    *,
    preprocessor: BasePreprocessor | None = None,
    preprocessor_kind: str = "auto",
    deskew: bool = True,
    mask_rupee: bool = False,
    rupee_mask_threshold: float = 0.5,
    rupee_mask_shrink_px: int = 2,
    tesseract_psm: int = 11,
    tesseract_oem: int = 3,
    tesseract_lang: str = "eng",
) -> BaseOCREngine:
    """Create OCR engine by name. preprocessor used for Tesseract; EasyOCR uses minimal resize only."""
    e = (engine or "tesseract").strip().lower()
    if e not in ("tesseract", "easyocr"):
        e = "tesseract"
    if e == "easyocr" and not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr not installed; pip install easyocr or .[ocr]")
    prep = preprocessor or create_preprocessor(
        kind=preprocessor_kind,
        deskew=deskew,
        mask_rupee=mask_rupee,
        rupee_mask_threshold=rupee_mask_threshold,
        rupee_mask_shrink_px=rupee_mask_shrink_px,
    )
    if e == "tesseract":
        config = build_tesseract_config(psm=tesseract_psm, oem=tesseract_oem, lang=tesseract_lang or "eng")
        eng = TesseractEngine(config=config, preprocessor=prep)
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
