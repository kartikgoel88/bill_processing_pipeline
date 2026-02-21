"""
Policy extractor (allowances shape): use LLM to extract client_location_allowance,
fuel 2W/4W, meal_allowance from policy text. Prompt-driven; returns raw dict.
Used by the PDF → OCR → LLM → JSON flow with the same shape as admin_billdesk.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from commons.exceptions import PolicyExtractionError
from policy._json_utils import extract_first_json_object, fix_json
from prompts import load_prompt

logger = logging.getLogger(__name__)


def _load_prompt_from_path(path: str | Path) -> str:
    """Load system prompt from file (absolute or cwd-relative path)."""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def extract_allowances_from_text(
    policy_text: str,
    llm_client: Any,
    *,
    system_prompt_path: str | Path | None = None,
    model: str = "llama3.2",
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    Send policy text to LLM and parse response into allowances JSON
    (client_location_allowance, fuel_reimbursement_two_wheeler,
     fuel_reimbursement_four_wheeler, meal_allowance).

    :param policy_text: Raw text from policy PDF (e.g. after OCR).
    :param llm_client: LLM client with chat_with_retry or chat.
    :param system_prompt_path: Optional path to system prompt file; else uses prompts/system_prompt_policy_allowances.txt.
    :param model: Model name.
    :param max_retries: Retries for LLM call.
    :return: Dict with the four allowance keys (prompt-driven shape).
    :raises PolicyExtractionError: On empty text, LLM failure, or invalid JSON.
    """
    if not (policy_text or policy_text.strip()):
        raise PolicyExtractionError("Policy text is empty")

    system_prompt = (
        _load_prompt_from_path(system_prompt_path)
        if system_prompt_path
        else load_prompt("system_prompt_policy_allowances.txt")
    )
    text_snippet = policy_text[:12000].strip() if len(policy_text) > 12000 else policy_text
    user_content = (
        "Extract the allowances and reimbursements from this policy document (OCR text):\n\n"
        + text_snippet
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = ""
    try:
        if hasattr(llm_client, "chat_with_retry"):
            raw = llm_client.chat_with_retry(
                messages, model=model, max_retries=max_retries
            )
        else:
            raw = llm_client.chat(messages, model=model)
    except Exception as e:
        logger.warning("LLM allowances extraction failed: %s", e)
        raise PolicyExtractionError(f"LLM allowances extraction failed: {e}") from e

    stripped = (raw or "").strip()
    if not stripped:
        raise PolicyExtractionError("LLM returned empty response")

    # Strip markdown code block if present
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if m:
        stripped = m.group(1).strip()
    stripped = extract_first_json_object(stripped)
    stripped = fix_json(stripped)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        logger.debug("Allowances LLM raw snippet: %s", stripped[:500])
        raise PolicyExtractionError(f"Invalid JSON from allowances LLM: {e}") from e

    if not isinstance(data, dict):
        raise PolicyExtractionError("LLM response is not a JSON object")

    # Ensure expected keys exist (prompt-driven shape)
    for key in (
        "client_location_allowance",
        "fuel_reimbursement_two_wheeler",
        "fuel_reimbursement_four_wheeler",
        "meal_allowance",
    ):
        if key not in data:
            data[key] = None
        elif data[key] is not None and not isinstance(data[key], dict):
            data[key] = None

    return data
