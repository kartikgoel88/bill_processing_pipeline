"""Image encoding helpers (e.g. for vision API)."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from extraction.image_io import PathImageReader, BytesImageWriter

# First-page vision defaults (PDF/image -> single JPEG for vision LLM)
VISION_IMAGE_MAX_PX = 1024
VISION_JPEG_QUALITY = 85
# Higher DPI helps capture small or corner receipts (missing-bills reduction)
VISION_FIRST_PAGE_DPI = 250


def _resize_to_max_px(img: Image.Image, max_px: int = VISION_IMAGE_MAX_PX) -> Image.Image:
    """Resize image so longest side is at most max_px."""
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    ratio = max_px / max(w, h)
    return img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)


def image_bytes_from_path_for_vision(path: Path) -> bytes:
    """First page as JPEG, size-limited. For PDF/image -> bytes suitable for vision API."""
    try:
        images = PathImageReader(path, dpi=VISION_FIRST_PAGE_DPI, first_page_only=True).read()
    except (FileNotFoundError, RuntimeError):
        return b""
    if not images:
        return b""
    img = _resize_to_max_px(images[0])
    return BytesImageWriter(format="JPEG", quality=VISION_JPEG_QUALITY).write([img])


def image_bytes_list_from_path_for_vision(path: Path) -> list[bytes]:
    """
    All pages as JPEG bytes (one per page), size-limited.
    Use for multi-bill PDFs: run vision extraction per page and get one bill per page.
    For single-page PDF or image file, returns a list of one element.
    """
    try:
        images = PathImageReader(path, dpi=VISION_FIRST_PAGE_DPI, first_page_only=False).read()
    except (FileNotFoundError, RuntimeError):
        return []
    if not images:
        return []
    writer = BytesImageWriter(format="JPEG", quality=VISION_JPEG_QUALITY)
    out: list[bytes] = []
    for img in images:
        resized = _resize_to_max_px(img)
        out.append(writer.write([resized]))
    return out


def image_to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """Encode image bytes as data URL for vision API."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{b64}"
