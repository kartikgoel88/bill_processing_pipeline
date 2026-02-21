"""
Decision engine: LLM-driven approval using structured policy JSON + bill JSON only.
Rules are NOT hardcoded; the LLM receives policy JSON and decides based on it.
Returns DecisionSchema (APPROVED | REJECTED | NEEDS_REVIEW).

Enterprise-grade structured output guardrails:
- Self-healing retry on invalid JSON (limit 2 attempts)
- Deterministic LLM params (temperature=0, top_p=0.1, no streaming)
- Safe fallback (NEEDS_REVIEW) on parse failure without crashing pipeline
- SafeJsonParser with repair and StructuredOutputError
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from commons.schema import DecisionSchema, ReimbursementSchema
from commons.exceptions import DecisionError, StructuredOutputError
from prompts import load_prompt

logger = logging.getLogger(__name__)

# Max LLM attempts: 1 initial + 1 self-healing retry = 2 total
DECISION_JSON_MAX_ATTEMPTS = 2
DECISION_LLM_MAX_TOKENS = 1024
DECISION_LLM_TEMPERATURE = 0.0
DECISION_LLM_TOP_P = 0.1


def _decision_system_prompt() -> str:
    return load_prompt("system_prompt_decision.txt")


def _self_healing_user_prompt() -> str:
    return load_prompt("self_healing_decision.txt")


def _build_decision_prompt(
    bill: ReimbursementSchema,
    policy_json: dict[str, Any],
    expense_type: str,
    monthly_total: float,
) -> str:
    """Build user message: policy JSON + bill JSON + expense type + monthly total."""
    bill_dict = bill.model_dump(mode="json")
    if hasattr(bill_dict.get("bill_date"), "isoformat"):
        bill_dict["bill_date"] = bill.bill_date.isoformat() if bill.bill_date else ""
    elif isinstance(bill_dict.get("bill_date"), str):
        pass
    parts = [
        "## Reimbursement policy JSON",
        json.dumps(policy_json, indent=2),
        "",
        "## Extracted bill JSON",
        json.dumps(bill_dict, indent=2),
        "",
        "## Expense type",
        expense_type,
        "",
        "## Employee's current monthly total (including this bill)",
        str(monthly_total),
    ]
    return "\n".join(parts)


def _fix_common_json_issues(s: str) -> str:
    """Remove trailing commas and other common invalid JSON."""
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _fix_invalid_json_escapes(s: str) -> str:
    """
    Fix invalid backslash escapes in JSON string.
    LLMs often produce e.g. \\p or \\$ in strings; JSON only allows \\ \" \\/ \\b \\f \\n \\r \\t \\uXXXX.
    """
    def repl(m: re.Match) -> str:
        return "\\\\" + m.group(1)

    return re.sub(
        r'\\(?!["\\\\/bfnrt]|u[0-9a-fA-F]{4})(.)',
        repl,
        s,
    )


def _extract_json_object(text: str) -> str:
    """Extract the first {...} object from text (brace-balanced)."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


# ---------------------------------------------------------------------------
# Unit-testable: invalid JSON repair
# ---------------------------------------------------------------------------


def safe_parse_decision_json(text: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response; strip markdown, fix common issues and invalid escapes.
    Raises StructuredOutputError (wrapping json.JSONDecodeError) on failure.
    Used by SafeJsonParser and by tests (test_invalid_json_repair).
    """
    stripped = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if m:
        stripped = m.group(1).strip()
    stripped = _extract_json_object(stripped)
    stripped = _fix_common_json_issues(stripped)
    stripped = _fix_invalid_json_escapes(stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(
            f"Invalid JSON: {e}",
            trace_id="",
        ) from e


# ---------------------------------------------------------------------------
# Safe fallback (unit-testable: test_fallback_trigger)
# ---------------------------------------------------------------------------


def fallback_decision() -> DecisionSchema:
    """
    Return safe NEEDS_REVIEW decision when LLM output cannot be parsed.
    Do NOT crash pipeline. Used after all retries exhausted.
    """
    return DecisionSchema(
        decision="NEEDS_REVIEW",
        confidence_score=0.0,
        reasoning="LLM_OUTPUT_INVALID_JSON",
        violated_rules=[],
        approved_amount=None,
    )


# ---------------------------------------------------------------------------
# SafeJsonParser: wrap parsing with optional repair logging
# ---------------------------------------------------------------------------


class SafeJsonParser:
    """
    Wraps JSON parsing with repair steps and consistent error handling.
    Integrates with logging (raw output, repaired output, trace_id).
    """

    def __init__(self, trace_id: str = "") -> None:
        self.trace_id = trace_id
        self._last_raw: str = ""
        self._last_repaired: str = ""

    def parse(self, raw: str) -> dict[str, Any]:
        """
        Parse raw LLM output to decision dict. Applies strip, extract object, fix commas/escapes.
        Sets _last_raw and _last_repaired for logging. Raises StructuredOutputError on failure.
        """
        self._last_raw = raw or ""
        stripped = (raw or "").strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
        if m:
            stripped = m.group(1).strip()
        stripped = _extract_json_object(stripped)
        stripped = _fix_common_json_issues(stripped)
        stripped = _fix_invalid_json_escapes(stripped)
        self._last_repaired = stripped
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            raise StructuredOutputError(
                f"Invalid JSON: {e}",
                trace_id=self.trace_id,
            ) from e

    @property
    def last_raw(self) -> str:
        return self._last_raw

    @property
    def last_repaired(self) -> str:
        return self._last_repaired


def _normalize_decision_data(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure types for DecisionSchema."""
    out: dict[str, Any] = {}
    out["decision"] = (data.get("decision") or "NEEDS_REVIEW").upper().strip()
    if out["decision"] not in ("APPROVED", "REJECTED", "NEEDS_REVIEW"):
        out["decision"] = "NEEDS_REVIEW"
    try:
        out["confidence_score"] = float(data.get("confidence_score", 0.5))
    except (TypeError, ValueError):
        out["confidence_score"] = 0.5
    out["confidence_score"] = max(0.0, min(1.0, out["confidence_score"]))
    out["reasoning"] = str(data.get("reasoning", "")).strip()
    vr = data.get("violated_rules")
    out["violated_rules"] = list(vr) if isinstance(vr, list) else []
    aa = data.get("approved_amount")
    try:
        out["approved_amount"] = float(aa) if aa is not None else None
    except (TypeError, ValueError):
        out["approved_amount"] = None
    return out


def auto_rejected_decision(reason: str) -> DecisionSchema:
    """Return a REJECTED decision without calling LLM (e.g. critical validation failed)."""
    return DecisionSchema(
        decision="REJECTED",
        confidence_score=1.0,
        reasoning=reason,
        violated_rules=[reason],
        approved_amount=None,
    )


def explainability_score_from_decision(decision: DecisionSchema) -> float:
    """
    Bonus: simple explainability score 0–1.
    Higher when reasoning is non-empty and violated_rules are present when relevant.
    """
    score = 0.0
    if decision.reasoning and len(decision.reasoning) > 10:
        score += 0.5
    if decision.violated_rules:
        score += 0.3
    if 0 <= decision.confidence_score <= 1:
        score += 0.2 * decision.confidence_score
    return min(1.0, score)


# ---------------------------------------------------------------------------
# Unit-testable: retry logic (attempt count and self-healing path)
# ---------------------------------------------------------------------------


def parse_decision_with_retry(
    raw: str,
    parser: SafeJsonParser,
    trace_id: str,
) -> dict[str, Any]:
    """
    Try to parse raw once. Raises StructuredOutputError on failure.
    Caller is responsible for retrying LLM with self-healing prompt when this raises.
    Used so tests can assert on parse_decision_with_retry without calling LLM (test_retry_logic).
    """
    return parser.parse(raw)


class DecisionEngine:
    """
    Calls decision LLM with structured policy JSON + bill JSON + expense type + monthly total.
    No hardcoded rules; LLM uses only the provided policy JSON.
    Deterministic output (temperature=0, top_p=0.1), self-healing retry, safe fallback.
    """

    def __init__(
        self,
        client: Any,
        decision_model: str,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        decision_max_tokens: int = DECISION_LLM_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.decision_model = decision_model
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.decision_max_tokens = decision_max_tokens

    def _call_llm(self, messages: list[dict[str, Any]]) -> str:
        """Call LLM with deterministic params: temperature=0, top_p=0.1, no streaming, max_tokens."""
        kwargs: dict[str, Any] = {
            "model": self.decision_model,
            "max_tokens": self.decision_max_tokens,
            "temperature": DECISION_LLM_TEMPERATURE,
            "top_p": DECISION_LLM_TOP_P,
            "stream": False,
        }
        if hasattr(self.client, "chat_with_retry"):
            return self.client.chat_with_retry(
                messages,
                max_retries=self.max_retries,
                retry_delay_sec=self.retry_delay_sec,
                **kwargs,
            )
        return self.client.chat(messages, **kwargs)

    def get_decision(
        self,
        bill: ReimbursementSchema,
        policy_json: dict[str, Any],
        expense_type: str,
        monthly_total: float,
        *,
        critical_failure: bool = False,
        trace_id: str = "",
    ) -> DecisionSchema:
        """
        If critical_failure is True, return auto REJECTED without calling LLM.
        Otherwise call LLM with deterministic params; on invalid JSON retry once with
        self-healing prompt (max 2 attempts). If still invalid, return fallback (NEEDS_REVIEW).
        """
        if critical_failure:
            logger.info(
                "Critical validation failed; auto REJECTED (no LLM call) trace_id=%s",
                trace_id,
            )
            return auto_rejected_decision(
                "Critical validation failed: amount or month invalid"
            )

        user_content = _build_decision_prompt(
            bill, policy_json, expense_type, monthly_total
        )
        messages = [
            {"role": "system", "content": _decision_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        parser = SafeJsonParser(trace_id=trace_id)
        last_error: Exception | None = None
        raw = ""

        for attempt in range(DECISION_JSON_MAX_ATTEMPTS):
            try:
                raw = self._call_llm(messages)
                logger.info(
                    "Decision LLM raw output attempt=%s trace_id=%s len=%s",
                    attempt + 1,
                    trace_id,
                    len(raw or ""),
                )
                if attempt > 0:
                    logger.info(
                        "Repaired/output used after self-healing attempt=%s trace_id=%s repaired_preview=%s",
                        attempt + 1,
                        trace_id,
                        (parser.last_repaired or raw or "")[:200],
                    )
                data = parser.parse(raw or "")
                if attempt > 0:
                    logger.info(
                        "Self-healing parse succeeded attempt=%s trace_id=%s",
                        attempt + 1,
                        trace_id,
                    )
                data = _normalize_decision_data(data)
                return DecisionSchema(**data)
            except (StructuredOutputError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(
                    "Decision JSON parse failed attempt=%s trace_id=%s error=%s raw_preview=%s repaired_preview=%s",
                    attempt + 1,
                    trace_id,
                    e,
                    (parser.last_raw or "")[:300],
                    (parser.last_repaired or "")[:300],
                )
                if attempt < DECISION_JSON_MAX_ATTEMPTS - 1:
                    messages = [
                        {"role": "system", "content": _decision_system_prompt()},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": parser.last_raw or raw},
                        {"role": "user", "content": _self_healing_user_prompt()},
                    ]
                    continue
                break
            except Exception as e:
                logger.exception(
                    "Decision LLM request failed trace_id=%s: %s", trace_id, e
                )
                raise DecisionError(
                    f"Decision LLM request failed: {e}",
                    trace_id=trace_id,
                ) from e

        logger.error(
            "Decision JSON parse failed after %s attempts; using fallback trace_id=%s last_error=%s",
            DECISION_JSON_MAX_ATTEMPTS,
            trace_id,
            last_error,
        )
        return fallback_decision()
