"""
Data models for the refactored pipeline.
Uses dataclasses for DTOs; Pydantic schemas (ReimbursementSchema, etc.) in core.schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """Structured response from an LLM provider."""

    text: str
    model: str = ""
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of vision/OCR extraction for one bill."""

    structured_bill: dict[str, Any]
    confidence: float
    source: str  # e.g. "donut", "vision_llm", "ocr_fallback", "fusion"
    ocr_raw_text: str = ""
    ocr_confidence: float = 0.0
    fusion_metadata: dict[str, Any] = field(default_factory=dict)
    critical_validation_failed: bool = False


@dataclass
class BillResult:
    """Final result of processing one bill (single public output of pipeline)."""

    trace_id: str
    file_name: str
    extraction_source: str
    structured_bill: dict[str, Any]
    decision: dict[str, Any]
    policy_version_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ocr_extracted: dict[str, Any] = field(default_factory=dict)
    fusion_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchMetrics:
    """Metrics collected during batch processing."""

    total_processed: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    needs_review_count: int = 0
    failed_count: int = 0
    total_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Export for logging/serialization."""
        return {
            "total_processed": self.total_processed,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "needs_review_count": self.needs_review_count,
            "failed_count": self.failed_count,
            "total_time_sec": round(self.total_time_sec, 4),
        }
