"""Extraction: OCR engines, preprocessing, bill parsing, and file discovery."""

from extraction.image_io import (
    IImageReader,
    IImageWriter,
    PathImageReader,
    BytesImageReader,
    PathImageWriter,
    BytesImageWriter,
)
from extraction.ocr import (
    create_ocr_engine,
    create_preprocessor,
    run_engine_on_images,
    default_engine_name,
    BaseOCREngine,
    BasePreprocessor,
    TesseractEngine,
    EasyOCREngine,
)
from extraction.parser import (
    parse_structured_from_ocr,
    parse_llm_extraction,
    _bill_extraction_system_prompt,
    _bill_extraction_vision_prompt,
)
from extraction.discovery import iter_bills

__all__ = [
    "IImageReader",
    "IImageWriter",
    "PathImageReader",
    "BytesImageReader",
    "PathImageWriter",
    "BytesImageWriter",
    "create_ocr_engine",
    "create_preprocessor",
    "run_engine_on_images",
    "default_engine_name",
    "BaseOCREngine",
    "BasePreprocessor",
    "TesseractEngine",
    "EasyOCREngine",
    "parse_structured_from_ocr",
    "parse_llm_extraction",
    "_bill_extraction_system_prompt",
    "_bill_extraction_vision_prompt",
    "iter_bills",
]
