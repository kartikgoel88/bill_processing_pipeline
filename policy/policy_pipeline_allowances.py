"""
Policy pipeline (allowances shape): PDF → OCR → LLM → JSON.
Same flow as admin_billdesk: read policy PDF (with OCR fallback), extract
client_location_allowance, fuel 2W/4W, meal_allowance via LLM, return/save dict.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from policy.policy_reader import read_policy_pdf
from policy.policy_extractor_allowances import extract_allowances_from_text
from commons.exceptions import PolicyExtractionError

logger = logging.getLogger(__name__)


def run_policy_allowances_pipeline(
    policy_pdf_path: str | Path,
    llm_client: Any,
    *,
    output_json_path: str | Path | None = None,
    system_prompt_path: str | Path | None = None,
    model: str = "llama3.2",
    max_retries: int = 2,
    use_ocr_fallback: bool = True,
    skip_cover_pages: bool = True,
) -> dict[str, Any]:
    """
    Run full flow: PDF → OCR (if needed) → LLM → JSON (allowances shape).

    :param policy_pdf_path: Path to policy PDF.
    :param llm_client: LLM client (chat_with_retry or chat).
    :param output_json_path: If set, write extracted JSON to this path.
    :param system_prompt_path: Optional path to system prompt; else uses prompts/system_prompt_policy_allowances.txt.
    :param model: LLM model name.
    :param max_retries: LLM retries.
    :param use_ocr_fallback: Use OCR when pypdf yields little text.
    :param skip_cover_pages: When using OCR, skip first and last page.
    :return: Dict with client_location_allowance, fuel_reimbursement_*_wheeler, meal_allowance.
    :raises FileNotFoundError: If PDF or prompt file missing.
    :raises PolicyExtractionError: If PDF yields no text or LLM/JSON fails.
    """
    path = Path(policy_pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy PDF not found: {path}")

    logger.info("Policy allowances pipeline: reading PDF %s", path)
    text = read_policy_pdf(
        path,
        use_ocr_fallback=use_ocr_fallback,
        skip_cover_pages=skip_cover_pages,
    )
    if not (text and text.strip()):
        raise PolicyExtractionError("Policy PDF yielded no text")

    logger.info("Policy allowances pipeline: extracting via LLM (model=%s)", model)
    data = extract_allowances_from_text(
        text,
        llm_client,
        system_prompt_path=system_prompt_path,
        model=model,
        max_retries=max_retries,
    )

    if output_json_path:
        out = Path(output_json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Policy allowances JSON written to %s", out)

    return data
