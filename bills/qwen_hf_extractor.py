"""
Hugging Face–based Qwen-VL / Qwen2.5-VL extractor for bill/receipt images.

- VISION_BACKEND=huggingface: run model locally (pip install -e ".[document]").
- VISION_BACKEND=hf_api: use Hugging Face Inference API with HF_TOKEN (no local download).
"""
from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

from commons.schema import ReimbursementSchema

logger = logging.getLogger(__name__)

# Module-level cache: (model_id, device) -> pipeline
_qwen_hf_cache: dict[tuple[str, str], Any] = {}


def _get_prompt() -> str:
    """Same prompt as the API-based vision LLM path."""
    from bills.bill_extractor import (
        _bill_extraction_system_prompt,
        _bill_extraction_vision_prompt,
    )
    return f"{_bill_extraction_system_prompt()}\n\n---\n\n{_bill_extraction_vision_prompt()}"


def _load_qwen_vl(model_id: str, device: str = "auto") -> Any:
    """Load Qwen2-VL or Qwen2.5-VL via image-text-to-text pipeline; cache by (model_id, device)."""
    key = (model_id, device)
    if key in _qwen_hf_cache:
        return _qwen_hf_cache[key]

    import torch
    from transformers import pipeline

    pipe = pipeline(
        task="image-text-to-text",
        model=model_id,
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    _qwen_hf_cache[key] = pipe
    return pipe


def extract_bill_via_qwen_hf(
    image_bytes: bytes,
    *,
    employee_id: str = "",
    expense_type: str = "meal",
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "auto",
    max_new_tokens: int = 512,
) -> tuple[ReimbursementSchema, float]:
    """
    Run Qwen2-VL or Qwen2.5-VL via Hugging Face on a bill image and parse to ReimbursementSchema.

    :param image_bytes: JPEG/PNG image bytes (e.g. first page of PDF rendered to image).
    :param employee_id: From folder context.
    :param expense_type: fuel | meal | commute.
    :param model_id: Hugging Face model id (e.g. Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2-VL-7B-Instruct).
    :param device: "auto", "cuda", or "cpu".
    :param max_new_tokens: Max tokens for generation.
    :return: (ReimbursementSchema, confidence_placeholder).
    """
    from bills.bill_extractor import parse_llm_extraction

    pipe = _load_qwen_vl(model_id, device)
    prompt = _get_prompt()

    # Pipeline chat format expects image as URL. Use a temp file path so the processor can load it.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name
    try:
        # Pipeline accepts image URL (http/https) or local path; use path for temp file
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": tmp_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        out = pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if isinstance(out, list) and len(out) > 0:
        item = out[0]
        text = item.get("generated_text", item) if isinstance(item, dict) else str(item)
    else:
        text = str(out).strip()
    if not text:
        raise ValueError("Qwen HF pipeline returned empty output")

    logger.info("Qwen HF (%s) output length: %s", model_id, len(text))
    schema = parse_llm_extraction(text, employee_id=employee_id, expense_type=expense_type)
    return schema, 0.85


def extract_bill_via_qwen_hf_api(
    image_bytes: bytes,
    *,
    employee_id: str = "",
    expense_type: str = "meal",
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    hf_token: str | None = None,
    max_new_tokens: int = 512,
) -> tuple[ReimbursementSchema, float]:
    """
    Run Qwen2-VL / Qwen2.5-VL via Hugging Face Inference API (hosted; uses HF token).
    No local model download. Requires: pip install huggingface_hub and HF_TOKEN set.

    :param image_bytes: JPEG/PNG image bytes.
    :param employee_id: From folder context.
    :param expense_type: fuel | meal | commute.
    :param model_id: Hugging Face model id (e.g. Qwen/Qwen2.5-VL-7B-Instruct).
    :param hf_token: Hugging Face token (Inference API). If None, uses HF_TOKEN env or saved login.
    :param max_new_tokens: Max tokens for generation.
    :return: (ReimbursementSchema, confidence_placeholder).
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError as e:
        raise ImportError(
            "Hugging Face Inference API requires huggingface_hub. Install with: pip install huggingface_hub"
        ) from e

    from bills.bill_extractor import parse_llm_extraction

    prompt = _get_prompt()
    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    client = InferenceClient(model=model_id, token=hf_token or None)
    out = client.chat.completions.create(
        messages=messages,
        max_tokens=max_new_tokens,
    )
    text = (out.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Hugging Face Inference API returned empty content")

    logger.info("Qwen HF API (%s) output length: %s", model_id, len(text))
    schema = parse_llm_extraction(text, employee_id=employee_id, expense_type=expense_type)
    return schema, 0.85
