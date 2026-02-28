"""
OCR text normalization: fix common misreadings (e.g. ₹ read as 2 or 7) so that
downstream parser and LLM see correct amounts without manual rules.
"""

from __future__ import annotations

import re


# Lines containing these keywords are treated as amount lines; only there we fix "2 500" / "7 500"
_AMOUNT_KEYWORDS = re.compile(
    r"\b(Total|Selected\s+Price|Net\s+Amount|Fare|Price|Payable|Net\s+Payable|"
    r"Payment\s+Summary|Invoice\s+Value|Grand\s+Total|Paytm\s+UPI|Amount)\b",
    re.IGNORECASE,
)

# On amount lines: "2 500" or "7 33.60" where 2/7 is misread ₹. Capture the number part.
_RUPEE_AS_2_OR_7 = re.compile(r"\b([27])\s+(\d{1,6}(?:\.\d{2})?)\b")


def _is_likely_year(num_str: str) -> bool:
    """True if the number looks like a year (2000-2030), so we don't replace it."""
    try:
        val = float(num_str)
        return val == int(val) and 2000 <= val <= 2030
    except ValueError:
        return False


def normalize_rupee_misread(text: str) -> str:
    """
    Fix OCR misreading of ₹ as 2 or 7 on amount lines.
    Only modifies lines that contain amount-related keywords (Total, Selected Price, etc.).
    Replaces "2 500" / "7 33.60" with "500" / "33.60". Skips numbers that look like years (2000-2030).
    """
    if not text or not text.strip():
        return text

    def process_line(line: str) -> str:
        if not _AMOUNT_KEYWORDS.search(line):
            return line

        def repl(m: re.Match) -> str:
            num_str = m.group(2)
            if _is_likely_year(num_str):
                return m.group(0)
            return num_str

        return _RUPEE_AS_2_OR_7.sub(repl, line)

    return "\n".join(process_line(line) for line in text.split("\n"))
