"""Shared JSON parsing helpers for policy LLM responses."""
from __future__ import annotations

import re


def extract_first_json_object(text: str) -> str:
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


def fix_json(s: str) -> str:
    """Remove trailing commas and other common invalid JSON."""
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s
