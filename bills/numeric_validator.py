"""
Numeric context validator: guardrails to prevent invoice numbers, IDs, and codes
from being accepted as bill amount. Only amounts that appear in valid payment context
(e.g. near "Total", "Net Payable") are accepted.
"""
from __future__ import annotations

import re
from typing import Sequence

# Accept: value must appear near one of these (case-insensitive)
ACCEPT_PHRASES: Sequence[str] = (
    "Total",
    "Invoice Total",
    "Grand Total",
    "Net Payable",
    "Payment Summary",
    "Payment",
    "UPI",
)

# Reject: value must NOT appear near any of these (case-insensitive)
REJECT_PHRASES: Sequence[str] = (
    "Invoice No",
    "GST",
    "HSN",
    "FSSAI",
    "Order ID",
)

# Window (chars) before/after an occurrence to check for accept/reject phrases
CONTEXT_WINDOW = 80


def _candidate_strings(value: float) -> list[str]:
    """Return possible string representations of the number for searching in text."""
    candidates: list[str] = []
    if value == int(value):
        candidates.append(str(int(value)))
    candidates.append(str(value))
    # Avoid duplicates and add common formats
    if "." in str(value):
        candidates.append(str(int(value)))
    return list(dict.fromkeys(candidates))


def _is_embedded_in_alphanumeric(text: str, start: int, end: int) -> bool:
    """True if the span [start, end) is part of a longer alphanumeric token (no spaces)."""
    if start > 0 and text[start - 1].isalnum():
        return True
    if end < len(text) and text[end].isalnum():
        return True
    return False


def _is_near_phrase(text: str, position: int, phrase: str) -> bool:
    """True if phrase appears within CONTEXT_WINDOW of position (case-insensitive)."""
    low = max(0, position - CONTEXT_WINDOW)
    high = min(len(text), position + CONTEXT_WINDOW)
    window = text[low:high]
    return phrase.lower() in window.lower()


def _line_has_other_amount(text: str, position: int, candidate_value: float) -> bool:
    """True if the line containing position has another number that looks like a real amount (not rupee)."""
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    # Find numbers with 2+ digits or decimal (real amounts); single digit 1-9 is often rupee
    for m in re.finditer(r"\d+(?:\.\d{2})?|\d{2,}", line):
        val_str = m.group(0)
        try:
            v = float(val_str.replace(",", ""))
        except ValueError:
            continue
        if v == candidate_value:
            continue
        if v > 9 or ("." in val_str):
            return True
    return False


def is_valid_amount(raw_text: str, candidate_value: float) -> bool:
    """
    Return True only if the candidate value appears in raw_text in a valid amount context.

    Rules:
    - Accept only if the value appears near one of: Total, Invoice Total, Grand Total,
      Net Payable, Payment Summary, Payment, UPI.
    - Reject if it appears near: Invoice No, GST, HSN, FSSAI, Order ID.
    - Reject if the value is embedded inside an alphanumeric string (e.g. JUIQIT313692S).
    - Reject single-digit (1-9) when the same line has another number that looks like an amount
      (e.g. "Total 8 70.00" → reject 8, accept 70 or 170 from elsewhere).

    When raw_text is empty (e.g. vision-only path), returns False (no context to validate).
    """
    if not (raw_text and raw_text.strip()):
        return False

    text = raw_text.strip()
    found_valid_occurrence = False
    is_single_digit = candidate_value in (1, 2, 3, 4, 5, 6, 7, 8, 9)

    for num_str in _candidate_strings(candidate_value):
        if not num_str:
            continue
        # Find all occurrences; use regex with word-boundary-like check to avoid 500 matching 5000
        pattern = re.escape(num_str)
        for m in re.finditer(pattern, text):
            start, end = m.span()
            # Reject if embedded in alphanumeric (e.g. JUIQIT313692S)
            if _is_embedded_in_alphanumeric(text, start, end):
                continue
            # Reject single-digit when same line has another amount (rupee misread)
            if is_single_digit and _line_has_other_amount(text, start, candidate_value):
                continue
            # Reject if near any reject phrase
            near_reject = any(
                _is_near_phrase(text, start, phrase) for phrase in REJECT_PHRASES
            )
            if near_reject:
                continue
            # Accept only if near at least one accept phrase
            near_accept = any(
                _is_near_phrase(text, start, phrase) for phrase in ACCEPT_PHRASES
            )
            if near_accept:
                found_valid_occurrence = True
                break
        if found_valid_occurrence:
            break

    return found_valid_occurrence
