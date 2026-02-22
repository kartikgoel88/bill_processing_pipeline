"""Image encoding helpers (e.g. for vision API)."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from extraction.image_io import PathImageReader, BytesImageWriter

# First-page vision defaults (PDF/image -> single JPEG for vision LLM)
VISION_IMAGE_MAX_PX = 1024
VISION_JPEG_QUALITY = 85
VISION_FIRST_PAGE_DPI = 150


def image_bytes_from_path_for_vision(path: Path) -> bytes:
    """First page as JPEG, size-limited. For PDF/image -> bytes suitable for vision API."""
    try:
        images = PathImageReader(path, dpi=VISION_FIRST_PAGE_DPI, first_page_only=True).read()
    except (FileNotFoundError, RuntimeError):
        return b""
    if not images:
        return b""
    img = images[0]
    w, h = img.size
    if max(w, h) > VISION_IMAGE_MAX_PX:
        ratio = VISION_IMAGE_MAX_PX / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    return BytesImageWriter(format="JPEG", quality=VISION_JPEG_QUALITY).write([img])


def image_to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """Encode image bytes as data URL for vision API."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{b64}"
