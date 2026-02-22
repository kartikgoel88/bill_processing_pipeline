"""Extraction: OCR engines, preprocessing, bill parsing, and file discovery."""

from extraction.ocr import (
    create_ocr_engine,
    create_preprocessor,
    load_images_from_path,
    load_images_from_bytes,
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
    "create_ocr_engine",
    "create_preprocessor",
    "load_images_from_path",
    "load_images_from_bytes",
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
