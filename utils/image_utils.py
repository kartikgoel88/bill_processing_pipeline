"""Image encoding helpers (e.g. for vision API)."""

from __future__ import annotations

import base64


def image_to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """Encode image bytes as data URL for vision API."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{b64}"
