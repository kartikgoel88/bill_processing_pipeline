"""
Unit tests for decision engine JSON guardrails.
Tests: invalid JSON repair, retry logic, fallback trigger.
"""
from __future__ import annotations

import pytest

from decision.decision_engine import (
    safe_parse_decision_json,
    fallback_decision,
    parse_decision_with_retry,
    SafeJsonParser,
)
from commons.exceptions import StructuredOutputError


def test_invalid_json_repair() -> None:
    """Repair logic: trailing commas, markdown fence, and invalid escapes are fixed."""
    # Trailing comma
    raw1 = '{"decision": "APPROVED", "confidence_score": 0.9, "reasoning": "ok", "violated_rules": [],}'
    out1 = safe_parse_decision_json(raw1)
    assert out1["decision"] == "APPROVED"
    assert out1["confidence_score"] == 0.9

    # Wrapped in markdown
    raw2 = '```json\n{"decision": "REJECTED", "confidence_score": 1.0, "reasoning": "over limit", "violated_rules": ["cap"]}\n```'
    out2 = safe_parse_decision_json(raw2)
    assert out2["decision"] == "REJECTED"
    assert out2["violated_rules"] == ["cap"]

    # Invalid JSON raises
    with pytest.raises(StructuredOutputError):
        safe_parse_decision_json("not json at all")

    # Trailing comma in array
    raw3 = '{"decision": "NEEDS_REVIEW", "confidence_score": 0.5, "reasoning": "x", "violated_rules": ["a",]}'
    out3 = safe_parse_decision_json(raw3)
    assert out3["decision"] == "NEEDS_REVIEW"
    assert out3["violated_rules"] == ["a"]


def test_retry_logic() -> None:
    """parse_decision_with_retry uses parser once; raises on invalid JSON."""
    parser = SafeJsonParser(trace_id="tid-1")
    valid = '{"decision": "APPROVED", "confidence_score": 0.8, "reasoning": "ok", "violated_rules": []}'
    data = parse_decision_with_retry(valid, parser, "tid-1")
    assert data["decision"] == "APPROVED"

    parser2 = SafeJsonParser(trace_id="tid-2")
    with pytest.raises(StructuredOutputError):
        parse_decision_with_retry("invalid { json", parser2, "tid-2")


def test_fallback_trigger() -> None:
    """fallback_decision returns NEEDS_REVIEW with fixed fields; pipeline does not crash."""
    d = fallback_decision()
    assert d.decision == "NEEDS_REVIEW"
    assert d.confidence_score == 0.0
    assert d.reasoning == "LLM_OUTPUT_INVALID_JSON"
    assert d.violated_rules == []
