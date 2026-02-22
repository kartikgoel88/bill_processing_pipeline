"""
Abstract interfaces for the bill processing pipeline.
Every external dependency is behind an interface; no service depends on concrete LLM/OCR impl.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from core.models import LLMResponse, ExtractionResult, BillResult


class ILLMProvider(ABC):
    """Abstract LLM provider: text/chat and optional vision. Used by vision and decision services."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate text from prompt. kwargs may include messages, model, max_tokens, temperature."""
        ...

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Chat completion; returns content string."""
        ...

    def chat_vision(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Vision-capable chat. Default: delegate to chat."""
        return self.chat(messages, **kwargs)


class IOCRService(ABC):
    """Abstract OCR: file path or bytes -> raw text + confidence."""

    @abstractmethod
    def extract_from_path(self, path: Path) -> tuple[str, float]:
        """Extract text from file. Returns (raw_text, confidence in [0,1])."""
        ...

    @abstractmethod
    def extract_from_bytes(self, data: bytes, is_pdf: bool) -> tuple[str, float]:
        """Extract from in-memory bytes. Returns (raw_text, confidence)."""
        ...


class IVisionService(ABC):
    """Abstract vision extraction: image bytes + context -> structured bill + confidence."""

    @abstractmethod
    def extract(self, image_bytes: bytes, context: dict[str, Any]) -> ExtractionResult:
        """
        Extract structured bill from image.
        context: employee_id, expense_type, etc.
        Returns ExtractionResult with structured_bill, confidence, source.
        """
        ...


class IDecisionService(ABC):
    """Abstract decision: bill + policy -> APPROVED | REJECTED | NEEDS_REVIEW."""

    @abstractmethod
    def get_decision(
        self,
        structured_bill: dict[str, Any],
        policy: dict[str, Any],
        expense_type: str,
        monthly_total: float,
        trace_id: str,
    ) -> dict[str, Any]:
        """Return decision dict: decision, confidence_score, reasoning, violated_rules, approved_amount."""
        ...


class IPostProcessingService(ABC):
    """Abstract post-processing: apply meal cap, normalize, enrich metadata."""

    @abstractmethod
    def apply(
        self,
        decision: dict[str, Any],
        structured_bill: dict[str, Any],
        policy: dict[str, Any],
        expense_type: str,
        remaining_day_cap: float | None = None,
    ) -> dict[str, Any]:
        """Apply policy rules (e.g. meal cap) and return updated decision dict."""
        ...


class IFallbackStrategy(Protocol):
    """Strategy: when primary vision confidence < threshold, use fallback extraction."""

    def should_fallback(self, confidence: float, threshold: float) -> bool:
        """True if fallback should be used."""
        ...

    def get_fallback_source(self) -> str:
        """Label for extraction_source when fallback was used (e.g. 'ocr_fallback')."""
        ...
