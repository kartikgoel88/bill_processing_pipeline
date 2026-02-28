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
    "Net Amount",
    "Amount",
    "Payment Summary",
    "Payment",
    "UPI",
    "Selected Price",
    "Price",
    "Fare",
    "Ride",
)

# Reject: value must NOT appear near any of these (case-insensitive)
REJECT_PHRASES: Sequence[str] = (
    "Invoice No",
    "GST",
    "HSN",
    "FSSAI",
    "Order ID",
    "Pincode",
)

# Window (chars) before/after an occurrence to check for accept/reject phrases
# 150 allows amount and "Total"/"Net Payable" to match across a few lines in multi-page OCR
CONTEXT_WINDOW = 150

# Year range: values in this range that appear in date context (e.g. "Nov 14th 2024") are rejected as amounts
YEAR_MIN, YEAR_MAX = 2000, 2030


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


def _get_line_at(text: str, position: int) -> str:
    """Return the line containing position (without leading/trailing newlines)."""
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    if line_end < 0:
        line_end = len(text)
    return text[line_start:line_end]


def _line_contains_any_phrase(line: str, phrases: Sequence[str]) -> bool:
    """True if line (case-insensitive) contains any of the phrases."""
    lower = line.lower()
    return any(p.lower() in lower for p in phrases)


def _line_or_adjacent_has_accept(text: str, position: int) -> bool:
    """True if the line containing position or the line before/after contains an accept phrase."""
    line_at = _get_line_at(text, position)
    if _line_contains_any_phrase(line_at, ACCEPT_PHRASES):
        return True
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    if line_end < 0:
        line_end = len(text)
    # Previous line
    if line_start > 0:
        prev_start = text.rfind("\n", 0, line_start - 1) + 1
        prev_line = text[prev_start:line_start - 1]
        if _line_contains_any_phrase(prev_line, ACCEPT_PHRASES):
            return True
    # Next line
    if line_end < len(text):
        next_end = text.find("\n", line_end + 1)
        if next_end < 0:
            next_end = len(text)
        next_line = text[line_end + 1:next_end]
        if _line_contains_any_phrase(next_line, ACCEPT_PHRASES):
            return True
    return False


def _is_year_in_date_context(text: str, start: int, end: int, value: float) -> bool:
    """True if value is a year (2000-2030) and this occurrence looks like part of a date (e.g. 'Nov 14th 2024')."""
    if value != int(value) or not (YEAR_MIN <= value <= YEAR_MAX):
        return False
    year_str = str(int(value))
    # Window around the occurrence to detect date context
    low = max(0, start - 30)
    high = min(len(text), end + 10)
    window = text[low:high]
    # Common date patterns: "14th 2024", "Nov 14th 2024", "2024,", " 2024 ", "11/14/2024"
    if re.search(r"(?:st|nd|rd|th)\s*" + re.escape(year_str) + r"\b", window, re.IGNORECASE):
        return True
    if re.search(r"\b" + re.escape(year_str) + r"\s*[,:]", window):
        return True
    if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s*" + re.escape(year_str), window, re.IGNORECASE):
        return True
    return False


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
            # Reject year (2000-2030) when it appears in date context (e.g. "Nov 14th 2024" → 2024 is year, not amount)
            if _is_year_in_date_context(text, start, end, candidate_value):
                continue
            # Reject single-digit when same line has another amount (rupee misread)
            if is_single_digit and _line_has_other_amount(text, start, candidate_value):
                continue
            line_at = _get_line_at(text, start)
            # Accept if near an accept phrase, on same line, or on line adjacent to accept phrase (e.g. "Total\n2126.82")
            near_accept = any(
                _is_near_phrase(text, start, phrase) for phrase in ACCEPT_PHRASES
            )
            line_has_accept = _line_contains_any_phrase(line_at, ACCEPT_PHRASES)
            adjacent_accept = _line_or_adjacent_has_accept(text, start)
            # Reject if near any reject phrase (or on same line as reject phrase),
            # but allow accept context to override reject (e.g. "Total" and "GST" in same window).
            near_reject = any(
                _is_near_phrase(text, start, phrase) for phrase in REJECT_PHRASES
            )
            line_has_reject = _line_contains_any_phrase(line_at, REJECT_PHRASES)
            if (near_reject or line_has_reject) and not (near_accept or line_has_accept or adjacent_accept):
                continue
            if near_accept or line_has_accept or adjacent_accept:
                found_valid_occurrence = True
                break
        if found_valid_occurrence:
            break

    return found_valid_occurrence
