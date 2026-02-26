"""
Decision service: structured bill + policy -> APPROVED | REJECTED | NEEDS_REVIEW.
Uses injected ILLMProvider; no concrete LLM dependency.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.interfaces import IDecisionService, ILLMProvider
from core.exceptions import DecisionError, StructuredOutputError

logger = logging.getLogger(__name__)


def _decision_system_prompt() -> str:
    from prompts import load_prompt
    return load_prompt("system_prompt_decision.txt")


def _build_user_prompt(
    structured_bill: dict[str, Any],
    policy: dict[str, Any],
    expense_type: str,
    monthly_total: float,
) -> str:
    parts = [
        "## Reimbursement policy JSON",
        json.dumps(policy, indent=2),
        "",
        "## Extracted bill JSON",
        json.dumps(structured_bill, indent=2),
        "",
        "## Expense type",
        expense_type,
        "",
        "## Employee's current monthly total (including this bill)",
        str(monthly_total),
    ]
    return "\n".join(parts)


def _critical_validation_reason(structured_bill: dict[str, Any]) -> str:
    """Exact reason for critical validation failure (amount/month)."""
    reasons: list[str] = []
    amt = structured_bill.get("amount")
    if amt is None:
        reasons.append("amount missing")
    else:
        try:
            if float(amt) <= 0:
                reasons.append("amount is zero or negative")
        except (TypeError, ValueError):
            reasons.append("amount invalid or not a number")
    month = (structured_bill.get("month") or "").strip()
    if not month:
        reasons.append("month missing")
    if not reasons:
        return "Critical validation failed"
    return "Critical validation failed: " + "; ".join(reasons)


def _normalize_decision(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["decision"] = (data.get("decision") or "NEEDS_REVIEW").upper().strip()
    if out["decision"] not in ("APPROVED", "REJECTED", "NEEDS_REVIEW"):
        out["decision"] = "NEEDS_REVIEW"
    try:
        out["confidence_score"] = max(0.0, min(1.0, float(data.get("confidence_score", 0.5))))
    except (TypeError, ValueError):
        out["confidence_score"] = 0.5
    out["reasoning"] = str(data.get("reasoning", "")).strip()
    vr = data.get("violated_rules")
    out["violated_rules"] = list(vr) if isinstance(vr, list) else []
    aa = data.get("approved_amount")
    try:
        out["approved_amount"] = float(aa) if aa is not None else None
    except (TypeError, ValueError):
        out["approved_amount"] = None
    return out


def _parse_decision_json(raw: str) -> dict[str, Any]:
    import re
    s = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    start = s.find("{")
    if start < 0:
        raise StructuredOutputError("No JSON object in response", trace_id="")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                s = s[start : i + 1]
                break
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return json.loads(s)


class DecisionService(IDecisionService):
    """Decision LLM via injected provider. Deterministic params (temperature=0, top_p=0.1)."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        model: str = "llama3.2",
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
    ) -> None:
        self._llm = llm_provider
        self._model = model
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec

    def get_decision(
        self,
        structured_bill: dict[str, Any],
        policy: dict[str, Any],
        expense_type: str,
        monthly_total: float,
        trace_id: str,
    ) -> dict[str, Any]:
        if not structured_bill or (structured_bill.get("amount") or 0) <= 0:
            reason = _critical_validation_reason(structured_bill or {})
            return _normalize_decision({
                "decision": "REJECTED",
                "confidence_score": 1.0,
                "reasoning": reason,
                "violated_rules": [reason],
                "approved_amount": None,
            })
        user_content = _build_user_prompt(
            structured_bill, policy, expense_type, monthly_total
        )
        messages = [
            {"role": "system", "content": _decision_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        try:
            raw = self._llm.chat(
                messages,
                model=self._model,
                max_tokens=1024,
                temperature=0.0,
                top_p=0.1,
            )
        except Exception as e:
            raise DecisionError(f"Decision LLM failed: {e}", trace_id=trace_id) from e
        try:
            data = _parse_decision_json(raw)
        except (json.JSONDecodeError, StructuredOutputError) as e:
            logger.warning("Decision JSON parse failed; fallback NEEDS_REVIEW: %s", e)
            return _normalize_decision({
                "decision": "NEEDS_REVIEW",
                "confidence_score": 0.0,
                "reasoning": "LLM_OUTPUT_INVALID_JSON",
                "violated_rules": [],
                "approved_amount": None,
            })
        return _normalize_decision(data)
