"""
Bill parser: parse OCR text or LLM vision output into ReimbursementSchema.
GST/Indian-invoice patterns; employee_id and expense_type from context.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from core.schema import LineItemSchema, ReimbursementSchema
from core.exceptions import BillExtractionError
from extraction.numeric_validator import is_valid_amount

logger = logging.getLogger(__name__)

# Year range: reject as amount when it appears in date context (e.g. "Nov 14th 2024")
YEAR_AMOUNT_MIN, YEAR_AMOUNT_MAX = 2000, 2030


def _text_has_year_in_date_context(raw_text: str, value: float) -> bool:
    """True if value is a year (2000-2030) and appears in raw_text in a date-like pattern."""
    if value != int(value) or not (YEAR_AMOUNT_MIN <= value <= YEAR_AMOUNT_MAX):
        return False
    year_str = str(int(value))
    # "Nov 14th 2024", "14th 2024", "2024,", " 2024 "
    if re.search(r"(?:st|nd|rd|th)\s*" + re.escape(year_str) + r"\b", raw_text, re.IGNORECASE):
        return True
    if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s*" + re.escape(year_str), raw_text, re.IGNORECASE):
        return True
    return False

MONTH_PATTERN = re.compile(r"\b(20\d{2})[-/]?(0[1-9]|1[0-2])\b")
DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%B %d, %Y", "%b %d, %Y",
]

# Indian GST invoice patterns: use last match for Total (subtotal + total)
INVOICE_DATE_PATTERN = re.compile(
    r"Invoice\s+Date\s*:\s*(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
# Total line: allow leading noise (e.g. "Total F158", "Total ₹ 33.60", "Net Amount 78.75")
TOTAL_AMOUNT_PATTERN = re.compile(
    r"(?:Total|Grand\s+Total|Net\s+Amount|Net\s+Payable)\s*[^\d]*(?:₹|Rs?\.?|INR)?\s*(\d+(?:\.\d{2})?)",
    re.IGNORECASE,
)
# Fallback: last amount-like number on a line containing "Total" (handles "Total 8 70.00" OCR)
TOTAL_LINE_LAST_NUMBER = re.compile(r"(\d+(?:\.\d{2})?)")
# Comma-thousands (e.g. "23,400.00" on Grand Total line)
TOTAL_LINE_COMMA_THOUSANDS = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")
# Total/Paytm UPI then optional newlines/whitespace then amount (HungerBox table layout)
TOTAL_AMOUNT_MULTILINE = re.compile(
    r"(?:Total|Paytm\s+UPI|Payment\s+Summary|Net\s+Amount)\s*(?:₹|Rs?\.?|INR)?\s*[\s\n]*(\d+(?:\.\d{2})?)",
    re.IGNORECASE | re.DOTALL,
)
# Payment/UPI lines often have the final amount (e.g. "Paytm UPI % 170.00", "Payment Summary 170.00", "Net Amount 78.75")
# Include ride/commute receipt phrases: Selected Price, Price, Fare
PAYMENT_LINE_KEYWORDS = re.compile(
    r"\b(?:Payment\s+Summary|Payable|UPI|Paid|Total|Net\s+Amount|Net\s+Payable|Selected\s+Price|Price|Fare)\b",
    re.IGNORECASE,
)
# Ride receipt date: "Time of Ride Nov 11th 2024" or "Nov 14th 2024"
RIDE_DATE_PATTERN = re.compile(
    r"(?:Time\s+of\s+Ride|Date)\s*[:\s]*"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\s+(20\d{2})",
    re.IGNORECASE,
)
# Fallback: month name + day + year anywhere (e.g. "Nov 11th 2024")
MONTH_NAME_YEAR_PATTERN = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\s+(20\d{2})\b",
    re.IGNORECASE,
)
# DD/MM/YY or DD/MM/YYYY (common on Indian receipts: "Date : 18/12/25", "12/12/25")
DATE_DDMMYY_PATTERN = re.compile(
    r"(?:Date|Dato|Bill\s+Date)\s*[:\s]*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b",
    re.IGNORECASE,
)
# Standalone: 18/12/2025 or 18/12/25 (2-digit year)
DATE_DDMMYY_STANDALONE = re.compile(
    r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2}|\d{2})\b",
)
# DD-MON-YYYY (e.g. 08-DEC-2025, 19-DEC-2025 on hotel/invoice bills)
DATE_DD_MON_YYYY = re.compile(
    r"\b(\d{1,2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-]?\s*(20\d{2})\b",
    re.IGNORECASE,
)
# Month name + day only (e.g. "Aug 29, 15:00") — use for month when no year on same line
MONTH_DAY_PATTERN = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})\b",
    re.IGNORECASE,
)
# First line after "Bill To:" is typically the merchant/vendor name on GST invoices
ORDERED_FROM_PATTERN = re.compile(
    r"Ordered\s+From\s*:[\s\S]+?Bill\s+To\s*:\s*\n\s*([^\n]+)",
    re.IGNORECASE,
)
# Currency detection (INR)
CURRENCY_INR_PATTERN = re.compile(r"₹|(?:^|\s)(?:Rs\.?|INR)(?:\s|$)", re.IGNORECASE)

# Reasonable max amount per expense type (INR) to avoid OCR picking invoice IDs / barcodes
from core.schema import DEFAULT_EXPENSE_TYPE

REASONABLE_AMOUNT_MAX = {"fuel": 50_000.0, "meal": 10_000.0, "commute": 10_000.0}  # keys from EXPENSE_TYPES
DEFAULT_REASONABLE_MAX = 100_000.0
# Indian pincode range: 6-digit numbers in 100000-599999 (and 6xxxxx) often appear in addresses
PINCODE_AMOUNT_MIN, PINCODE_AMOUNT_MAX = 100_000.0, 599_999.0


def _parse_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def _parse_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    s = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date().isoformat()
        except ValueError:
            continue
    return None


def _find_month_in_text(text: str) -> str | None:
    m = MONTH_PATTERN.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # DD-MON-YYYY (e.g. 08-DEC-2025)
    for m in DATE_DD_MON_YYYY.finditer(text):
        try:
            dt = datetime.strptime(f"{m.group(2)} 1 {m.group(3)}", "%b %d %Y")
            return dt.strftime("%Y-%m")
        except ValueError:
            continue
    # Ride-style: "Time of Ride Nov 11th 2024" or "Nov 14th 2024"
    for pattern in (RIDE_DATE_PATTERN, MONTH_NAME_YEAR_PATTERN):
        ride = pattern.search(text)
        if ride:
            month_name, year = ride.group(1), ride.group(2)
            try:
                dt = datetime.strptime(f"{month_name} 1 {year}", "%b %d %Y")
                return dt.strftime("%Y-%m")
            except ValueError:
                pass
    # "Aug 29, 15:00" — month name + day; infer year from Invoice Date or use current
    year_from_invoice = re.search(r"Invoice\s+Date\s*:\s*20(\d{2})-\d{2}-\d{2}", text, re.IGNORECASE)
    year_str = year_from_invoice.group(1) if year_from_invoice else str(datetime.now().year)[2:]
    year = 2000 + int(year_str) if len(year_str) == 2 else int(year_str)
    for m in MONTH_DAY_PATTERN.finditer(text):
        try:
            dt = datetime.strptime(f"{m.group(1)} 1 {year}", "%b %d %Y")
            return dt.strftime("%Y-%m")
        except ValueError:
            continue
    return None


def _extract_amount_from_payment_lines(raw_text: str, max_amount: float = 1_000_000.0) -> list[float]:
    """Collect amounts from Payment Summary, UPI, Payable, Total lines (e.g. 'Paytm UPI % 170.00' → 170)."""
    candidates: list[float] = []
    for line in raw_text.splitlines():
        if not PAYMENT_LINE_KEYWORDS.search(line):
            continue
        for num_str in TOTAL_LINE_LAST_NUMBER.findall(line):
            val = _parse_float(num_str)
            if 0 < val <= max_amount and val not in (2.0, 7.0):  # skip common ₹ misreads
                candidates.append(val)
    return candidates


def _correct_rupee_read_as_leading_2(raw_text: str, amount: float) -> float:
    """
    OCR/LLM often reads ₹ as '2', so e.g. ₹78.75 becomes 278.75.
    If amount is in [200, 300) and (amount - 200) appears in the text, return the corrected value.
    """
    if not (200 <= amount < 300):
        return amount
    corrected = amount - 200.0
    # Check if the corrected amount appears in text (e.g. 78.75 or 78,75)
    if f"{corrected:.2f}" in raw_text:
        return corrected
    dec_str = f"{corrected:.2f}".rstrip("0").rstrip(".")
    if dec_str in raw_text:
        return corrected
    # Comma as decimal separator
    if f"{corrected:.2f}".replace(".", ",") in raw_text:
        return corrected
    return amount


def _line_looks_like_address_or_trip(line: str) -> bool:
    """True if line looks like address (503/2, 560,) or trip (kilometers, minutes)."""
    lower = line.lower()
    if "kilometer" in lower or "minutes" in lower:
        return True
    if re.search(r"\d+/\d+", line):  # e.g. 503/2
        return True
    if re.search(r",\s*\d{2,3}\s", line) and re.search(r"(Layout|Rd|Road|India|Karnataka|Bengaluru)", line, re.IGNORECASE):
        return True
    return False


def _extract_amount_from_last_total_or_payment_line(raw_text: str, max_amount: float = 1_000_000.0) -> float:
    """Get amount from the last line that contains Total/Payment Summary/Paytm UPI/Selected Price, or the next 1–2 lines."""
    lines = raw_text.splitlines()
    last_idx = None
    last_keyword = None
    for i, line in enumerate(lines):
        m = re.search(r"\b(Total|Payment\s+Summary|Paytm\s+UPI|Net\s+Amount|Payments|Selected\s+Price)\b", line, re.IGNORECASE)
        if m:
            keyword = (m.group(1) or "").strip().lower()
            # Skip footnote line "Selected Price refers to the initial price range" so we use the real header line
            if keyword == "selected price" and ("refers" in line.lower() or "range" in line.lower()):
                continue
            last_idx = i
            last_keyword = keyword
    if last_idx is None:
        return 0.0
    # For "Selected Price", take amount only from next line(s) to avoid time (7:11, 11) on same line
    start = last_idx + 1 if (last_keyword == "selected price") else last_idx
    for idx in range(start, min(last_idx + 3, len(lines))):
        line = lines[idx]
        if _line_looks_like_address_or_trip(line):
            continue
        for m in TOTAL_LINE_COMMA_THOUSANDS.finditer(line):
            val = _parse_float(m.group(1))
            if 0 < val <= max_amount and val > 9:
                return val
        for num_str in TOTAL_LINE_LAST_NUMBER.findall(line):
            val = _parse_float(num_str)
            if PINCODE_AMOUNT_MIN <= val <= PINCODE_AMOUNT_MAX:
                continue
            if 0 < val <= max_amount and val > 9:
                return val
    return 0.0


def _extract_amount_gst_style(raw_text: str) -> float:
    """Extract amount from GST-style text. Prefer last Total/Payment line (invoice total at end); else max of candidates."""
    # Use DEFAULT_REASONABLE_MAX so Grand Total 23,400.00 is found; pincodes (>599999) still rejected in last-line logic
    last_line_amt = _extract_amount_from_last_total_or_payment_line(raw_text, max_amount=DEFAULT_REASONABLE_MAX)
    if last_line_amt > 0:
        return last_line_amt
    # Multiline Total...amount (HungerBox / table layout where Total and amount are on different lines)
    for m in reversed(list(TOTAL_AMOUNT_MULTILINE.finditer(raw_text))):
        val = _parse_float(m.group(1))
        if 0 < val < 1_000_000 and val > 9 and not (PINCODE_AMOUNT_MIN <= val <= PINCODE_AMOUNT_MAX):
            return val
    candidates: list[float] = []

    # 1) TOTAL_AMOUNT_PATTERN: skip single-digit (1-9); collect same-line or captured amount
    matches = list(TOTAL_AMOUNT_PATTERN.finditer(raw_text))
    for m in reversed(matches):
        val = _parse_float(m.group(1))
        if 0 < val < 1_000_000:
            if 1 <= val <= 9:
                line_start = raw_text.rfind("\n", 0, m.start()) + 1
                line_end = raw_text.find("\n", m.end())
                if line_end < 0:
                    line_end = len(raw_text)
                line = raw_text[line_start:line_end]
                for num_str in TOTAL_LINE_LAST_NUMBER.findall(line):
                    v = _parse_float(num_str)
                    if v > 9 and 0 < v < 1_000_000:
                        candidates.append(v)
                continue
            candidates.append(val)
    # 2) Total line fallback — skip single digit
    for line in reversed(raw_text.splitlines()):
        if re.search(r"\bTotal\b", line, re.IGNORECASE):
            for num_str in TOTAL_LINE_LAST_NUMBER.findall(line):
                val = _parse_float(num_str)
                if val > 9 and 0 < val < 1_000_000:
                    candidates.append(val)
            break
    # 3) Payment/UPI lines (e.g. "Paytm UPI % 170.00") — often the final amount
    candidates.extend(_extract_amount_from_payment_lines(raw_text))

    return max(candidates) if candidates else 0.0


def _extract_amount_from_total_lines_with_cap(raw_text: str, max_amount: float) -> float:
    """
    From lines containing 'Total' (or amount-related keywords), collect all numbers in (0, max_amount]
    and return the largest. When the keyword line has no number, check the next 1–2 lines (e.g. "Total\\n123.02").
    Skips address/trip lines (e.g. "560, Marathahalli...") and Rapido footnote so 147 wins over 560.
    """
    if max_amount <= 0:
        return 0.0
    amount_keywords = re.compile(
        r"\b(Total|Grand\s+Total|Net\s+Amount|Net\s+Payable|Amount|Paid|Payable|Payment\s+Summary|Payment|Payments|UPI|₹|Rs\.?|INR|Selected\s+Price|Price|Fare)\b",
        re.IGNORECASE,
    )
    lines = raw_text.splitlines()
    candidates: list[float] = []
    for i, line in enumerate(lines):
        if not amount_keywords.search(line):
            continue
        # Rapido footnote: don't use this line or its next lines (would add 560 from address)
        if re.search(r"Selected\s+Price", line, re.IGNORECASE) and ("refers" in line.lower() or "range" in line.lower()):
            continue
        if _line_looks_like_address_or_trip(line):
            continue
        for m in TOTAL_LINE_COMMA_THOUSANDS.finditer(line):
            val = _parse_float(m.group(1))
            if 0 < val <= max_amount:
                candidates.append(val)
        for num_str in TOTAL_LINE_LAST_NUMBER.findall(line):
            val = _parse_float(num_str)
            if PINCODE_AMOUNT_MIN <= val <= PINCODE_AMOUNT_MAX:
                continue
            if 0 < val <= max_amount:
                candidates.append(val)
        for j in range(i + 1, min(i + 3, len(lines))):
            next_line = lines[j]
            if _line_looks_like_address_or_trip(next_line):
                continue
            for m in TOTAL_LINE_COMMA_THOUSANDS.finditer(next_line):
                val = _parse_float(m.group(1))
                if 0 < val <= max_amount:
                    candidates.append(val)
            for num_str in TOTAL_LINE_LAST_NUMBER.findall(next_line):
                val = _parse_float(num_str)
                if PINCODE_AMOUNT_MIN <= val <= PINCODE_AMOUNT_MAX:
                    continue
                if 0 < val <= max_amount:
                    candidates.append(val)
    for m in TOTAL_AMOUNT_PATTERN.finditer(raw_text):
        line_start = raw_text.rfind("\n", 0, m.start()) + 1
        line_end = raw_text.find("\n", m.end())
        if line_end < 0:
            line_end = len(raw_text)
        line = raw_text[line_start:line_end]
        if _line_looks_like_address_or_trip(line):
            continue
        val = _parse_float(m.group(1))
        if 0 < val <= max_amount:
            candidates.append(val)
    return max(candidates) if candidates else 0.0


def _parse_ddmmyy(day: str, month: str, year: str) -> str | None:
    """Parse DD/MM/YY or DD/MM/YYYY to YYYY-MM-DD. Returns None on failure."""
    try:
        d, m = int(day), int(month)
        y = int(year)
        if y < 100:
            y += 2000 if y < 50 else 1900
        if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
            return datetime(y, m, d).date().isoformat()
    except (ValueError, TypeError):
        pass
    return None


def _extract_date_ddmmyy_style(raw_text: str) -> tuple[str, str]:
    """Extract date from 'Date : DD/MM/YY' or standalone DD/MM/YYYY. Returns (YYYY-MM-DD, YYYY-MM)."""
    # "Date : 18/12/25" or "Dato : 16/12/25" (OCR typo)
    m = DATE_DDMMYY_PATTERN.search(raw_text)
    if m:
        parsed = _parse_ddmmyy(m.group(1), m.group(2), m.group(3))
        if parsed:
            return parsed, parsed[:7]
    # Standalone 18/12/2025 or 18-12-25 (prefer last occurrence as often the bill date)
    for m in reversed(list(DATE_DDMMYY_STANDALONE.finditer(raw_text))):
        parsed = _parse_ddmmyy(m.group(1), m.group(2), m.group(3))
        if parsed:
            return parsed, parsed[:7]
    return "", ""


def _parse_dd_mon_yyyy(day: str, month_name: str, year: str) -> str | None:
    """Parse DD-MON-YYYY to YYYY-MM-DD."""
    try:
        dt = datetime.strptime(f"{month_name[:3]} {day} {year}", "%b %d %Y")
        return dt.date().isoformat()
    except ValueError:
        return None


def _extract_invoice_date_gst_style(raw_text: str) -> str:
    """Extract only date (YYYY-MM-DD) from 'Invoice Date: YYYY-MM-DD'. No invoice number."""
    m = INVOICE_DATE_PATTERN.search(raw_text)
    if m:
        return m.group(1).strip()
    # Fallback: first YYYY-MM-DD in text
    date_candidate = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", raw_text)
    if date_candidate:
        return date_candidate.group(1)
    # DD-MON-YYYY (e.g. 08-DEC-2025, 19-DEC-2025)
    for m in reversed(list(DATE_DD_MON_YYYY.finditer(raw_text))):
        parsed = _parse_dd_mon_yyyy(m.group(1), m.group(2), m.group(3))
        if parsed:
            return parsed
    # DD/MM/YY style (Indian receipts)
    bill_date_str, _ = _extract_date_ddmmyy_style(raw_text)
    return bill_date_str


def _extract_vendor_ordered_from(raw_text: str) -> str:
    """Vendor = first line after 'Bill To:' on GST invoices (merchant name). Do not use 'Invoice issued by'."""
    m = ORDERED_FROM_PATTERN.search(raw_text)
    if not m:
        return ""
    first_line = (m.group(1) or "").strip()
    if re.match(r"^[\dA-Z]{10,}$", first_line):
        return ""
    return first_line[:100]


def _detect_currency(raw_text: str) -> str:
    """Detect INR from ₹ / Rs / INR in text; else default INR for this pipeline."""
    if CURRENCY_INR_PATTERN.search(raw_text):
        return "INR"
    return "INR"


def parse_structured_from_ocr(
    raw_text: str,
    *,
    employee_id: str = "",
    expense_type: str = "",
) -> ReimbursementSchema | None:
    """
    Parse OCR text into ReimbursementSchema. Uses GST/Indian-invoice patterns first
    (amount from Total line, date from Invoice Date, vendor from Ordered From);
    fallback to generic heuristics only for amount when no Total line found.
    Returns None if parsing fails.
    """
    if not (raw_text and raw_text.strip()):
        return None

    text = raw_text.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Amount: GST-style last Total, tolerant to OCR noise
    amount = _extract_amount_gst_style(text)
    # Rupee symbol (₹) often read as leading "2": e.g. ₹78.75 → 278.75; correct when XX.XX appears in text
    amount = _correct_rupee_read_as_leading_2(text, amount)
    # Reject year (2000-2030) when it appears in date context (e.g. "Nov 14th 2024" → 2024 is year, not fare)
    if amount >= YEAR_AMOUNT_MIN and amount <= YEAR_AMOUNT_MAX and _text_has_year_in_date_context(text, amount):
        logger.info(
            "Bill amount %.0f rejected (year in date context); using 0",
            amount,
        )
        amount = 0.0
    # Reject pincode-like amounts (100000-599999) — often from addresses; use Total-line amount instead
    if PINCODE_AMOUNT_MIN <= amount <= PINCODE_AMOUNT_MAX:
        alt = _extract_amount_from_total_lines_with_cap(
            text, REASONABLE_AMOUNT_MAX.get((expense_type or DEFAULT_EXPENSE_TYPE).lower(), DEFAULT_REASONABLE_MAX)
        )
        if alt > 0 and is_valid_amount(text, alt):
            logger.info("Bill amount %.0f rejected (pincode-like); using Total-line amount %.2f", amount, alt)
            amount = alt
        else:
            amount = 0.0
    # Numeric guardrails: reject amounts that appear only near Invoice No, GST, etc. or embedded in IDs
    if amount > 0 and not is_valid_amount(text, amount):
        alt = _extract_amount_from_total_lines_with_cap(
            text, REASONABLE_AMOUNT_MAX.get((expense_type or DEFAULT_EXPENSE_TYPE).lower(), DEFAULT_REASONABLE_MAX)
        )
        if alt > 0 and is_valid_amount(text, alt):
            logger.info(
                "Bill amount rejected by context (e.g. invoice ID); using Total-line amount %.2f",
                alt,
            )
            amount = alt
        else:
            logger.info(
                "Bill amount %.2f rejected by numeric validator (not near Total/Net Payable or embedded in ID); using 0",
                amount,
            )
            amount = 0.0
    # If amount is unreasonably high for expense type, OCR likely picked invoice ID/barcode — re-scan with cap
    max_reasonable = REASONABLE_AMOUNT_MAX.get((expense_type or DEFAULT_EXPENSE_TYPE).lower(), DEFAULT_REASONABLE_MAX)
    if amount > max_reasonable:
        alt = _extract_amount_from_total_lines_with_cap(text, max_reasonable)
        if alt > 0 and is_valid_amount(text, alt):
            logger.info(
                "Bill amount corrected from %s to %s (expense_type=%s, max_reasonable=%.0f)",
                amount, alt, expense_type or DEFAULT_EXPENSE_TYPE, max_reasonable,
            )
            amount = alt
        else:
            logger.warning(
                "Bill amount %s exceeds reasonable max %.0f for %s; no alternative found in Total lines",
                amount, max_reasonable, expense_type or DEFAULT_EXPENSE_TYPE,
            )
    if amount == 0.0:
        for line in lines:
            lower = line.lower()
            if "total" in lower or "grand total" in lower or "net amount" in lower or "net payable" in lower:
                cand = _parse_float(line)
                if 0 < cand < 1_000_000 and is_valid_amount(text, cand):
                    amount = cand
                    break
            if amount == 0.0 and "amount" in lower and ":" in line:
                cand = _parse_float(line.split(":", 1)[-1])
                if cand > 0 and is_valid_amount(text, cand):
                    amount = cand
                    break
            if amount == 0.0 and ("selected price" in lower or ("price" in lower and "fare" in lower)):
                for num_str in TOTAL_LINE_LAST_NUMBER.findall(line):
                    cand = _parse_float(num_str)
                    if 0 < cand < 1_000_000 and is_valid_amount(text, cand):
                        amount = cand
                        break
                if amount > 0:
                    break
            if amount == 0.0 and "price" in lower and ":" in line:
                cand = _parse_float(line.split(":", 1)[-1])
                if cand > 0 and is_valid_amount(text, cand):
                    amount = cand
                    break
        if amount == 0.0:
            for part in re.split(r"[\s]+", text):
                val = _parse_float(part)
                if PINCODE_AMOUNT_MIN <= val <= PINCODE_AMOUNT_MAX:
                    continue
                if 0 < val < 1_000_000 and is_valid_amount(text, val):
                    amount = val
                    break

    # Date: Invoice Date, YYYY-MM-DD, or DD/MM/YY (Indian receipts)
    bill_date_str = _extract_invoice_date_gst_style(text)
    if bill_date_str and not re.match(r"^\d{4}-\d{2}-\d{2}$", bill_date_str):
        parsed = _parse_date(bill_date_str)
        bill_date_str = parsed or ""

    month = _find_month_in_text(text)
    if not month and bill_date_str and len(bill_date_str) >= 7:
        month = bill_date_str[:7]
    if not month:
        _, month = _extract_date_ddmmyy_style(text)

    # Vendor: only from "Ordered From:" block (first line)
    vendor_name = _extract_vendor_ordered_from(text)
    if not vendor_name:
        for line in lines:
            if "vendor" in line.lower() and ":" in line:
                vendor_name = line.split(":", 1)[-1].strip()[:100]
                break

    currency = _detect_currency(text)

    line_items: list[LineItemSchema] = []
    if amount > 0:
        line_items = [LineItemSchema(description="Bill total", amount=amount, quantity=1)]

    data: dict[str, Any] = {
        "employee_id": employee_id,
        "expense_type": expense_type or DEFAULT_EXPENSE_TYPE,
        "amount": amount,
        "month": month or "",
        "bill_date": bill_date_str or "",
        "vendor_name": vendor_name or "Unknown",
        "currency": currency,
        "line_items": line_items,
    }
    try:
        return ReimbursementSchema(**data)
    except PydanticValidationError:
        return None


def _bill_extraction_system_prompt() -> str:
    from prompts import load_prompt
    return load_prompt("system_prompt_bill_extraction.txt")


def _bill_extraction_vision_prompt() -> str:
    """Short prompt that accompanies the image for the vision LLM."""
    from prompts import load_prompt
    try:
        return load_prompt("vision_prompt_bill_extraction.txt")
    except FileNotFoundError:
        return "Extract the reimbursement fields from this receipt/bill image. Return only one valid JSON object, no other text."


def _bill_extraction_from_text_prompt(ocr_text: str) -> str:
    """User prompt for LLM extraction from raw OCR text (no image). Same JSON schema as vision."""
    instruction = (
        "Below is the raw OCR text from a receipt or bill. Extract the reimbursement fields and return "
        "exactly one JSON object with keys: amount, month, bill_date, vendor_name, currency, line_items. "
        "Use the same rules as for images: amount = Grand Total / Net Payable / final total (not invoice/order ID); "
        "month = YYYY-MM; bill_date = YYYY-MM-DD; line_items = array of {description, amount, quantity, code}. "
        "Output only the JSON object, no markdown or explanation.\n\n---\n\n"
    )
    return instruction + (ocr_text or "").strip()


def _fix_invalid_json_escapes(s: str) -> str:
    """
    Fix invalid backslash escapes in JSON strings.
    JSON only allows \\ \" \\/ \\b \\f \\n \\r \\t \\uXXXX. LLMs often produce \\p, \\$, etc.
    """
    def repl(m: re.Match) -> str:
        return "\\\\" + m.group(1)
    return re.sub(r'\\(?!["\\\\/bfnrt]|u[0-9a-fA-F]{4})(.)', repl, s)


def _sanitize_llm_json_string(s: str) -> str:
    """Fix common invalid JSON from vision/LLM: unquoted string values, trailing text, invalid escapes."""
    # Take only first ```json ... ``` block to drop trailing explanation text
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    # Replace unquoted string values (e.g. "id": string or bar code image) with quoted placeholder
    s = re.sub(r':\s*string\s+or\s+[^",}\]\n]+', ': "unknown"', s, flags=re.IGNORECASE)
    # Remove keys that would break JSON: "key": unquoted_identifier (single word)
    s = re.sub(r',\s*"[^"]+"\s*:\s*[a-zA-Z_][a-zA-Z0-9_]*\s*([,}\]])', r'\1', s)
    # Fix invalid backslash escapes (e.g. \p, \$) so json.loads does not raise
    s = _fix_invalid_json_escapes(s)
    return s


def _extract_json_from_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    sanitized = _sanitize_llm_json_string(stripped)
    # Extract first {...} object if still messy
    start = sanitized.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(sanitized)):
            if sanitized[i] == "{":
                depth += 1
            elif sanitized[i] == "}":
                depth -= 1
                if depth == 0:
                    sanitized = sanitized[start : i + 1]
                    break
    return json.loads(sanitized)


def _extract_json_array_or_object_from_response(text: str) -> list[dict[str, Any]]:
    """
    Parse LLM response as either a single bill object {...} or an array of bills [{...}, ...].
    Returns a list of dicts (one element for single object, N for array).
    """
    stripped = text.strip()
    sanitized = _sanitize_llm_json_string(stripped)
    # Prefer array if response starts with [
    start_arr = sanitized.find("[")
    start_obj = sanitized.find("{")
    if start_arr >= 0 and (start_obj < 0 or start_arr < start_obj):
        depth = 0
        for i in range(start_arr, len(sanitized)):
            if sanitized[i] == "[":
                depth += 1
            elif sanitized[i] == "]":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(sanitized[start_arr : i + 1])
                    if not isinstance(parsed, list):
                        return [parsed] if isinstance(parsed, dict) else []
                    return [x for x in parsed if isinstance(x, dict)]
        # Unbalanced brackets: fall through to single-object extraction
    if start_obj >= 0:
        depth = 0
        for i in range(start_obj, len(sanitized)):
            if sanitized[i] == "{":
                depth += 1
            elif sanitized[i] == "}":
                depth -= 1
                if depth == 0:
                    single = json.loads(sanitized[start_obj : i + 1])
                    return [single] if isinstance(single, dict) else []
    return []


def _normalize_llm_bill_json(
    data: dict[str, Any],
    *,
    employee_id: str = "",
    expense_type: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "employee_id": employee_id,
        "expense_type": expense_type or DEFAULT_EXPENSE_TYPE,
        "amount": 0.0,
        "month": "",
        "bill_date": "",
        "vendor_name": "Unknown",
        "currency": "INR",
        "line_items": [],
    }
    out["amount"] = _parse_float(data.get("amount"))

    month_val = data.get("month")
    if month_val and str(month_val).strip():
        s = str(month_val).strip().replace("/", "-")
        m = MONTH_PATTERN.search(s)
        out["month"] = f"{m.group(1)}-{m.group(2)}" if m else (s[:7] if len(s) >= 7 else s)

    date_val = data.get("bill_date")
    if date_val and str(date_val).strip():
        parsed = _parse_date(date_val)
        out["bill_date"] = parsed or str(date_val).strip()

    out["vendor_name"] = (str(data.get("vendor_name") or "").strip() or "Unknown")[:200]
    currency_raw = (str(data.get("currency") or "").strip() or "INR").upper()
    out["currency"] = "INR" if currency_raw == "INR" else (currency_raw or "INR")[:10]

    items = data.get("line_items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                try:
                    q = item.get("quantity")
                    qty = int(float(q)) if q is not None else 1
                except (TypeError, ValueError):
                    qty = 1
                out["line_items"].append(
                    LineItemSchema(
                        description=(str(item.get("description") or "").strip())[:500],
                        amount=_parse_float(item.get("amount")),
                        quantity=max(1, qty) if qty > 0 else 1,
                        code=(str(item.get("code") or "").strip())[:50],
                    )
                )
    if not out["line_items"] and out["amount"] > 0:
        out["line_items"] = [LineItemSchema(description="Bill total", amount=out["amount"], quantity=1)]

    # Ensure line_items are plain dicts so structured_bill is JSON-serializable (e.g. for decision service)
    out["line_items"] = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in out["line_items"]
    ]
    return out


def parse_llm_extraction(
    raw_response: str,
    *,
    employee_id: str = "",
    expense_type: str = "",
) -> ReimbursementSchema:
    """Parse LLM JSON response into ReimbursementSchema."""
    try:
        data = _extract_json_from_response(raw_response)
    except json.JSONDecodeError as e:
        raise BillExtractionError(f"Invalid JSON from bill LLM: {e}") from e
    normalized = _normalize_llm_bill_json(data, employee_id=employee_id, expense_type=expense_type)
    try:
        return ReimbursementSchema(**normalized)
    except PydanticValidationError as e:
        raise BillExtractionError(f"Bill schema validation failed: {e}") from e


def parse_llm_extraction_multi(
    raw_response: str,
    *,
    employee_id: str = "",
    expense_type: str = "",
) -> list[dict[str, Any]]:
    """
    Parse LLM JSON response as one or more bills. Accepts single {...} or array [{...}, ...].
    Returns list of normalized bill dicts (suitable for structured_bill or structured_bills).
    """
    try:
        raw_list = _extract_json_array_or_object_from_response(raw_response)
    except json.JSONDecodeError as e:
        raise BillExtractionError(f"Invalid JSON from bill LLM: {e}") from e
    if not raw_list:
        return []
    normalized_list: list[dict[str, Any]] = []
    for data in raw_list:
        if not isinstance(data, dict):
            continue
        try:
            norm = _normalize_llm_bill_json(
                data, employee_id=employee_id, expense_type=expense_type
            )
            normalized_list.append(norm)
        except Exception:
            continue
    return normalized_list


