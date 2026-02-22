"""
Post-processing: apply policy rules (e.g. meal cap), normalize decision.
Implements IPostProcessingService; no LLM dependency.
"""

from __future__ import annotations

import logging
from typing import Any

from core.interfaces import IPostProcessingService
from core.exceptions import PostProcessingError

logger = logging.getLogger(__name__)
MEAL_EXPENSE_TYPE = "meal"


def _meal_cap_from_policy(policy: dict[str, Any]) -> float | None:
    ma = policy.get("meal_allowance")
    if not isinstance(ma, dict):
        return None
    limit = ma.get("limit")
    if limit is None:
        return None
    try:
        return float(limit)
    except (TypeError, ValueError):
        return None


def _apply_meal_cap(
    decision: dict[str, Any],
    structured_bill: dict[str, Any],
    policy: dict[str, Any],
    expense_type: str,
    remaining_day_cap: float | None,
) -> dict[str, Any]:
    if (decision.get("decision") or "") != "APPROVED":
        return decision
    approved = decision.get("approved_amount")
    if approved is None:
        return decision
    if (expense_type or "").lower() != MEAL_EXPENSE_TYPE:
        return decision
    cap = _meal_cap_from_policy(policy)
    if cap is None:
        return decision
    effective_cap = min(cap, remaining_day_cap) if remaining_day_cap is not None else cap
    if effective_cap <= 0:
        new_approved = 0.0
    else:
        bill_amt = float(structured_bill.get("amount") or 0)
        new_approved = min(approved, effective_cap, bill_amt)
    if new_approved == approved:
        return decision
    out = dict(decision)
    out["approved_amount"] = new_approved
    return out


class PostProcessingService(IPostProcessingService):
    """Apply meal cap and any other policy-based post-processing."""

    def apply(
        self,
        decision: dict[str, Any],
        structured_bill: dict[str, Any],
        policy: dict[str, Any],
        expense_type: str,
        remaining_day_cap: float | None = None,
    ) -> dict[str, Any]:
        try:
            return _apply_meal_cap(
                decision,
                structured_bill,
                policy,
                expense_type,
                remaining_day_cap,
            )
        except Exception as e:
            raise PostProcessingError(f"Post-processing failed: {e}") from e
