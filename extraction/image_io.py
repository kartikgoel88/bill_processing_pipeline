"""
Image reader and writer strategies: load/save images from paths, bytes, and extendible to DB/cache.
Used by OCR and vision pipelines; add new strategies (e.g. DatabaseImageReader) as needed.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    convert_from_path = None  # type: ignore


# ---------------------------------------------------------------------------
# Reader strategies
# ---------------------------------------------------------------------------


class IImageReader(ABC):
    """Strategy to read one or more images from a source (path, bytes, DB, cache, etc.)."""

    @abstractmethod
    def read(self) -> list[Image.Image]:
        """Load images; caller must close or discard them. PDF -> multiple; image file -> single."""
        ...


class PathImageReader(IImageReader):
    """Read images from a file path (PDF -> pages; image file -> single)."""

    def __init__(
        self,
        path: Path | str,
        *,
        dpi: int = 300,
        first_page_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.dpi = dpi
        self.first_page_only = first_page_only

    def read(self) -> list[Image.Image]:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        if self.path.suffix.lower() == ".pdf":
            if not PDF_AVAILABLE or convert_from_path is None:
                raise RuntimeError("pdf2image is not installed; install it and poppler")
            kwargs: dict[str, Any] = {"dpi": self.dpi}
            if self.first_page_only:
                kwargs["first_page"] = 1
                kwargs["last_page"] = 1
            pages = convert_from_path(str(self.path), **kwargs)
            return [p.convert("RGB") if p.mode != "RGB" else p for p in pages]
        img = Image.open(self.path).convert("RGB")
        return [img]


class BytesImageReader(IImageReader):
    """Read images from in-memory bytes (PDF or image)."""

    def __init__(
        self,
        data: bytes,
        *,
        is_pdf: bool,
        dpi: int = 300,
    ) -> None:
        self.data = data
        self.is_pdf = is_pdf
        self.dpi = dpi

    def read(self) -> list[Image.Image]:
        if self.is_pdf:
            if not PDF_AVAILABLE or convert_from_path is None:
                raise RuntimeError("pdf2image required for PDF bytes")
            pages = convert_from_path(io.BytesIO(self.data), dpi=self.dpi)
            return [p.convert("RGB") if p.mode != "RGB" else p for p in pages]
        img = Image.open(io.BytesIO(self.data)).convert("RGB")
        return [img]


# ---------------------------------------------------------------------------
# Writer strategies
# ---------------------------------------------------------------------------


class IImageWriter(ABC):
    """Strategy to write one or more images to a destination (path, bytes, DB, cache, etc.)."""

    @abstractmethod
    def write(self, images: list[Image.Image], **kwargs: Any) -> Any:
        """Write images; return value is strategy-specific (e.g. path, bytes, key)."""
        ...


class PathImageWriter(IImageWriter):
    """Write images to the filesystem (directory or single file)."""

    def __init__(
        self,
        path: Path | str,
        *,
        prefix: str = "page",
        format: str = "PNG",
        single_file: bool | None = None,
    ) -> None:
        self.path = Path(path)
        self.prefix = prefix
        self.format = format
        # If path has a suffix (file), write one image to it; else (directory) write prefix_1, prefix_2, ...
        self.single_file = single_file if single_file is not None else bool(self.path.suffix)

    def write(self, images: list[Image.Image], **kwargs: Any) -> Path | list[Path]:
        if not images:
            return self.path if self.path.suffix else self.path / f"{self.prefix}_0.{self.format.lower()}"
        fmt = kwargs.get("format", self.format)
        if len(images) == 1 and (self.path.suffix or kwargs.get("single_file", self.single_file)):
            out = self.path if self.path.suffix else self.path / f"{self.prefix}.{fmt.lower()}"
            out.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(out, format=fmt, **{k: v for k, v in kwargs.items() if k not in ("format", "single_file")})
            return out
        self.path.mkdir(parents=True, exist_ok=True)
        out_paths: list[Path] = []
        for i, img in enumerate(images):
            out = self.path / f"{self.prefix}_{i + 1}.{fmt.lower()}"
            img.save(out, format=fmt, **{k: v for k, v in kwargs.items() if k != "format"})
            out_paths.append(out)
        return out_paths


class BytesImageWriter(IImageWriter):
    """Write images to bytes (single image -> bytes; multiple -> list of bytes)."""

    def __init__(self, *, format: str = "JPEG", quality: int = 85) -> None:
        self.format = format
        self.quality = quality

    def write(self, images: list[Image.Image], **kwargs: Any) -> bytes | list[bytes]:
        if not images:
            return b""
        fmt = kwargs.get("format", self.format)
        quality = kwargs.get("quality", self.quality)
        if len(images) == 1:
            buf = io.BytesIO()
            images[0].save(buf, format=fmt, quality=quality, **{k: v for k, v in kwargs.items() if k not in ("format", "quality")})
            return buf.getvalue()
        result: list[bytes] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format=fmt, quality=quality, **{k: v for k, v in kwargs.items() if k not in ("format", "quality")})
            result.append(buf.getvalue())
        return result
