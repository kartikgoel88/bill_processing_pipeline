"""
Unit tests for the refactored bill processing pipeline.
Demonstrates: mocking ILLMProvider, testing pipeline with injected fakes, no tight coupling.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.interfaces import (
    IOCRService,
    IVisionService,
    IDecisionService,
    IPostProcessingService,
    ILLMProvider,
)
from core.models import ExtractionResult, LLMResponse, BillResult
from pipeline.bill_pipeline import BillProcessingPipeline
from pipeline.batch_processor import BatchProcessor


# ---------------------------------------------------------------------------
# Fake implementations (test doubles)
# ---------------------------------------------------------------------------


class FakeOCRService(IOCRService):
    """Returns fixed text and confidence for testing."""

    def __init__(self, raw_text: str = "Total 100.00", confidence: float = 0.9) -> None:
        self.raw_text = raw_text
        self.confidence = confidence

    def extract_from_path(self, path: Path) -> tuple[str, float]:
        return (self.raw_text, self.confidence)

    def extract_from_bytes(self, data: bytes, is_pdf: bool) -> tuple[str, float]:
        return (self.raw_text, self.confidence)


class FakeVisionService(IVisionService):
    """Returns configurable ExtractionResult without calling any LLM."""

    def __init__(
        self,
        structured_bill: dict | None = None,
        confidence: float = 0.85,
        source: str = "vision_llm",
    ) -> None:
        self.structured_bill = structured_bill or {
            "employee_id": "E1",
            "expense_type": "meal",
            "amount": 100.0,
            "month": "2024-01",
            "bill_date": "2024-01-15",
            "vendor_name": "Test Vendor",
            "currency": "INR",
            "category": "",
            "line_items": [],
        }
        self.confidence = confidence
        self.source = source

    def extract(self, image_bytes: bytes, context: dict) -> ExtractionResult:
        return ExtractionResult(
            structured_bill=self.structured_bill,
            confidence=self.confidence,
            source=self.source,
        )


class FakeDecisionService(IDecisionService):
    """Returns fixed decision for testing."""

    def __init__(
        self,
        decision: str = "APPROVED",
        approved_amount: float | None = 100.0,
    ) -> None:
        self.decision = decision
        self.approved_amount = approved_amount

    def get_decision(
        self,
        structured_bill: dict,
        policy: dict,
        expense_type: str,
        monthly_total: float,
        trace_id: str,
    ) -> dict:
        return {
            "decision": self.decision,
            "confidence_score": 0.9,
            "reasoning": "Test",
            "violated_rules": [],
            "approved_amount": self.approved_amount,
        }


class FakePostProcessingService(IPostProcessingService):
    """Pass-through; no cap applied in test."""

    def apply(
        self,
        decision: dict,
        structured_bill: dict,
        policy: dict,
        expense_type: str,
        remaining_day_cap: float | None = None,
    ) -> dict:
        return decision


# ---------------------------------------------------------------------------
# Mock LLM provider (for services that need ILLMProvider)
# ---------------------------------------------------------------------------


def test_mock_llm_provider_interface() -> None:
    """Show how to mock ILLMProvider: implement chat and generate."""
    mock = MagicMock(spec=ILLMProvider)
    mock.chat.return_value = '{"decision": "APPROVED", "confidence_score": 0.9}'
    mock.chat_vision.return_value = '{"amount": 50.0, "month": "2024-01"}'
    assert mock.chat([{"role": "user", "content": "test"}]) == '{"decision": "APPROVED", "confidence_score": 0.9}'
    assert mock.chat_vision([]) == '{"amount": 50.0, "month": "2024-01"}'


# ---------------------------------------------------------------------------
# Pipeline unit tests (no real files; all fakes)
# ---------------------------------------------------------------------------


@pytest.fixture
def policy_json() -> dict:
    return {
        "meal_allowance": {"limit": 50.0},
        "submission_window_days": 30,
    }


@pytest.fixture
def pipeline(policy_json: dict) -> BillProcessingPipeline:
    ocr = FakeOCRService(raw_text="Total 75.00\nDate 2024-02-01", confidence=0.85)
    vision = FakeVisionService(
        structured_bill={
            "employee_id": "E1",
            "expense_type": "meal",
            "amount": 75.0,
            "month": "2024-02",
            "bill_date": "2024-02-01",
            "vendor_name": "Cafe",
            "currency": "INR",
            "category": "",
            "line_items": [],
        },
        confidence=0.9,
    )
    decision = FakeDecisionService(decision="APPROVED", approved_amount=75.0)
    post = FakePostProcessingService()
    return BillProcessingPipeline(
        ocr,
        vision,
        decision,
        post,
        policy=policy_json,
        policy_version_hash="test_hash",
        fallback_threshold=0.6,
        fallback_enabled=True,
        extraction_strategy="fusion",
        vision_if_ocr_below=0.9,  # OCR 0.85 < 0.9 so vision still runs in this fixture
        expense_type="meal",
        employee_id="E1",
    )


def _minimal_image_path(tmp_path: Path) -> Path:
    """Write a minimal valid PNG so pipeline image_bytes step does not fail."""
    try:
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), color="white").save(buf, format="PNG")
        p = tmp_path / "bill.png"
        p.write_bytes(buf.getvalue())
        return p
    except ImportError:
        p = tmp_path / "bill.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
        return p


def test_pipeline_process_returns_bill_result(tmp_path: Path, pipeline: BillProcessingPipeline) -> None:
    """Pipeline.process() returns list of BillResult with expected fields."""
    bill_file = _minimal_image_path(tmp_path)
    results = pipeline.process(bill_file)
    assert isinstance(results, list)
    assert len(results) >= 1
    result = results[0]
    assert isinstance(result, BillResult)
    assert result.trace_id
    assert result.file_name == bill_file.name
    assert result.decision.get("decision") == "APPROVED"
    assert result.structured_bill.get("amount") == 75.0
    assert result.extraction_source == "vision_llm"


def test_pipeline_fallback_when_vision_low_confidence(tmp_path: Path, policy_json: dict) -> None:
    """When vision confidence < threshold, fallback to OCR path (source becomes ocr_fallback)."""
    ocr = FakeOCRService(raw_text="Total 30.00\n2024-03", confidence=0.7)
    vision = FakeVisionService(confidence=0.4, source="vision_llm")  # below threshold
    decision = FakeDecisionService(decision="APPROVED", approved_amount=30.0)
    post = FakePostProcessingService()
    pipeline = BillProcessingPipeline(
        ocr,
        vision,
        decision,
        post,
        policy=policy_json,
        fallback_threshold=0.6,
        fallback_enabled=True,
        vision_if_ocr_below=0.6,  # OCR 0.7 >= 0.6 so we skip vision and use OCR (ocr_fallback)
    )
    bill_file = _minimal_image_path(tmp_path)
    results = pipeline.process(bill_file)
    assert len(results) >= 1
    assert results[0].extraction_source == "ocr_fallback"


def test_pipeline_ocr_only_mode(tmp_path: Path, policy_json: dict) -> None:
    """OCR-only mode uses OCR only; vision is never called (extraction_source is ocr_fallback)."""
    ocr = FakeOCRService(raw_text="Total 50.00\nDate 2024-01-10", confidence=0.8)
    vision = FakeVisionService(confidence=0.9)
    decision = FakeDecisionService(decision="APPROVED", approved_amount=50.0)
    post = FakePostProcessingService()
    pipeline = BillProcessingPipeline(
        ocr,
        vision,
        decision,
        post,
        policy=policy_json,
        fallback_threshold=0.6,
        fallback_enabled=True,
        extraction_strategy="ocr_only",
        vision_if_ocr_below=0.6,
    )
    bill_file = _minimal_image_path(tmp_path)
    results = pipeline.process(bill_file)
    assert len(results) >= 1
    assert results[0].extraction_source == "ocr_fallback"
    # Vision was skipped; result comes from OCR parsing
    assert "50" in str(results[0].structured_bill.get("amount", ""))


def test_pipeline_fusion_skips_vision_when_ocr_confident(tmp_path: Path, policy_json: dict) -> None:
    """Fusion: when OCR confidence >= vision_if_ocr_below, vision LLM is skipped."""
    ocr = FakeOCRService(raw_text="Total 60.00\n2024-04", confidence=0.95)
    vision = FakeVisionService(confidence=0.9)
    decision = FakeDecisionService(decision="APPROVED", approved_amount=60.0)
    post = FakePostProcessingService()
    pipeline = BillProcessingPipeline(
        ocr,
        vision,
        decision,
        post,
        policy=policy_json,
        fallback_threshold=0.6,
        fallback_enabled=True,
        extraction_strategy="fusion",
        vision_if_ocr_below=0.6,
    )
    bill_file = _minimal_image_path(tmp_path)
    results = pipeline.process(bill_file)
    assert len(results) >= 1
    assert results[0].extraction_source == "ocr_fallback"
    # Vision was skipped because OCR confidence 0.95 >= 0.6


def test_batch_processor_collects_metrics(tmp_path: Path, pipeline: BillProcessingPipeline) -> None:
    """BatchProcessor returns results and metrics without duplicating pipeline logic."""
    a = _minimal_image_path(tmp_path)
    b = tmp_path / "b.png"
    if a.name == "bill.png":
        import shutil
        shutil.copy(a, b)
    else:
        b.write_bytes(a.read_bytes())
    batch = BatchProcessor(pipeline)
    results, metrics = batch.process_batch([a, b])
    assert len(results) == 2
    assert metrics.total_processed == 2
    assert metrics.approved_count == 2
    assert metrics.total_time_sec >= 0


def test_batch_processor_continues_on_error(tmp_path: Path, pipeline: BillProcessingPipeline) -> None:
    """When one file fails, batch continues and metrics record failed_count."""
    ok_file = _minimal_image_path(tmp_path)
    batch = BatchProcessor(pipeline)
    results, metrics = batch.process_batch(
        [ok_file, tmp_path / "nonexistent.txt"],
        stop_on_first_error=False,
    )
    assert len(results) == 1
    assert metrics.failed_count == 1
