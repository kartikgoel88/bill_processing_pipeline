"""Fallback strategy: when vision confidence < threshold, use alternative (e.g. OCR)."""

from __future__ import annotations


class ConfidenceFallbackStrategy:
    """Configurable: should_fallback(confidence, threshold) and fallback source label."""

    def __init__(self, source_label: str = "ocr_fallback") -> None:
        self._source_label = source_label

    def should_fallback(self, confidence: float, threshold: float) -> bool:
        return confidence < threshold

    def get_fallback_source(self) -> str:
        return self._source_label
