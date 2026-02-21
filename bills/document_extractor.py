"""
Document understanding extractors: Donut and LayoutLM (vision/document models)
as alternatives to the vision LLM for bill/receipt extraction.

Donut: OCR-free, image -> structured JSON (e.g. CORD receipt format).
LayoutLM: Optional; uses OCR text + layout for token-level extraction.
"""
from __future__ import annotations

import io
import json
import logging
import re
from typing import Any

from commons.schema import ReimbursementSchema
from commons.exceptions import BillExtractionError

logger = logging.getLogger(__name__)

# Default Donut model for receipt/invoice parsing (CORD-style output)
DEFAULT_DONUT_MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"

# CORD / receipt JSON keys we map to our schema
CORD_TOTAL_KEYS = ("total_price", "total", "grand_total", "amount", "total_price")
CORD_DATE_KEYS = ("date", "bill_date", "invoice_date")
CORD_VENDOR_KEYS = ("vendor", "seller", "merchant", "store_name", "vendor_name")
MONTH_PATTERN = re.compile(r"\b(20\d{2})[-/]?(0[1-9]|1[0-2])\b")


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


def _donut_cord_to_reimbursement_dict(
    cord: dict[str, Any],
    *,
    employee_id: str = "",
    expense_type: str = "",
) -> dict[str, Any]:
    """Map CORD-style Donut output (or similar) to our reimbursement schema shape."""
    out: dict[str, Any] = {
        "employee_id": employee_id,
        "expense_type": expense_type or "meal",
        "amount": 0.0,
        "month": "",
        "bill_date": "",
        "vendor_name": "Unknown",
        "currency": "INR",
        "line_items": [],
    }

    def _get_nested(d: Any, *keys: str) -> Any:
        if not isinstance(d, dict):
            return None
        for k in keys:
            if k in d and d[k] is not None:
                v = d[k]
                if isinstance(v, dict):
                    return _get_nested(v, "total_price", "price", "value") or v
                return v
        return None

    # Amount: total.total_price, total_price, sub_total, menu.price, etc.
    amount = _parse_float(_get_nested(cord, "total", "total_price", "sub_total", "subtotal_price"))
    if amount <= 0:
        total_obj = cord.get("total") or cord.get("sub_total") or {}
        if isinstance(total_obj, dict):
            amount = _parse_float(
                total_obj.get("total_price") or total_obj.get("subtotal_price") or total_obj.get("price")
            )
    if amount <= 0 and "menu" in cord and isinstance(cord["menu"], dict):
        amount = _parse_float(cord["menu"].get("price"))
    if amount <= 0:
        for key in ("total_price", "grand_total", "amount", "total"):
            amount = _parse_float(cord.get(key))
            if amount > 0:
                break
    out["amount"] = amount if amount > 0 else 0.0

    # Month / date
    date_str = None
    for key in CORD_DATE_KEYS:
        v = cord.get(key)
        if v and str(v).strip():
            date_str = str(v).strip()
            break
    if not date_str and "total" in cord and isinstance(cord["total"], dict):
        date_str = (cord["total"].get("date") or cord["total"].get("bill_date")) or ""
    if date_str:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            out["bill_date"] = date_str
            out["month"] = date_str[:7]
        else:
            m = MONTH_PATTERN.search(date_str)
            if m:
                out["month"] = f"{m.group(1)}-{m.group(2)}"
            out["bill_date"] = date_str[:10] if len(date_str) >= 10 else date_str

    # Vendor
    for key in CORD_VENDOR_KEYS:
        v = cord.get(key)
        if v and str(v).strip():
            out["vendor_name"] = (str(v).strip() or "Unknown")[:200]
            break

    # Line items (CORD menu or items array)
    items = cord.get("menu") or cord.get("items") or cord.get("line_items") or []
    if isinstance(items, dict):
        items = [items]
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                desc = str(it.get("nm") or it.get("description") or it.get("name") or "").strip()
                amt = _parse_float(it.get("price") or it.get("amount"))
                cnt = it.get("cnt", 1)
                try:
                    if isinstance(cnt, (int, float)):
                        qty = max(1, int(float(cnt)))
                    else:
                        qty = max(1, int(re.sub(r"[^\d]", "", str(cnt)) or 1))
                except (TypeError, ValueError):
                    qty = 1
                if not desc and amt > 0:
                    desc = "Line item"
                out["line_items"].append({
                    "description": desc[:500],
                    "amount": amt,
                    "quantity": qty,
                    "code": str(it.get("code") or "")[:50],
                })
    if not out["line_items"] and out["amount"] > 0:
        out["line_items"] = [{"description": "Bill total", "amount": out["amount"], "quantity": 1, "code": ""}]

    return out


def _load_donut_model(model_id: str):
    """Lazy-load Donut processor and model. Raises ImportError if transformers/torch missing."""
    try:
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel
    except ImportError as e:
        raise ImportError(
            "Donut extractor requires: pip install transformers torch pillow. "
            "Or install optional deps: pip install bill-processing-pipeline[document]"
        ) from e

    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


# Module-level cache for Donut model (avoid reload per bill)
_donut_cache: dict[str, tuple[Any, Any, str]] = {}


def extract_bill_via_donut(
    image_bytes: bytes,
    *,
    employee_id: str = "",
    expense_type: str = "meal",
    model_id: str = DEFAULT_DONUT_MODEL_ID,
) -> tuple[ReimbursementSchema, float]:
    """
    Run Donut document model on the bill image and map output to ReimbursementSchema.
    Returns (schema, confidence). Uses CORD-style or generic JSON mapping.
    """
    try:
        from PIL import Image
    except ImportError:
        raise BillExtractionError("PIL/Pillow required for Donut extractor") from None

    if model_id not in _donut_cache:
        _donut_cache[model_id] = _load_donut_model(model_id)
    processor, model, device = _donut_cache[model_id]

    import torch

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    task_prompt = ""
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    # Empty prompt can yield (1, 0) and cause "index -1 out of bounds" in generate; ensure at least one token
    if decoder_input_ids.shape[1] == 0:
        start_id = (
            getattr(processor.tokenizer, "decoder_start_token_id", None)
            or processor.tokenizer.bos_token_id
            or processor.tokenizer.pad_token_id
        )
        if start_id is None:
            start_id = 0
        decoder_input_ids = torch.tensor(
            [[start_id]], dtype=decoder_input_ids.dtype
        )
    pixel_values = processor(image, return_tensors="pt").pixel_values

    pixel_values = pixel_values.to(device)
    decoder_input_ids = decoder_input_ids.to(device)

    gen_kwargs = {
        "max_length": model.decoder.config.max_position_embeddings,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    unk_id = getattr(processor.tokenizer, "unk_token_id", None)
    if unk_id is not None:
        gen_kwargs["bad_words_ids"] = [[unk_id]]
    outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, **gen_kwargs)

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = (
        sequence.replace(processor.tokenizer.eos_token or "", "")
        .replace(processor.tokenizer.pad_token or "", "")
        .strip()
    )
    # Remove first task/special token if present (e.g. <s_task>)
    sequence = re.sub(r"^<[^>]+>", "", sequence).strip()

    # Donut CORD models expose token2json; others may return raw JSON string
    if hasattr(processor, "token2json"):
        try:
            cord = processor.token2json(sequence)
        except Exception:
            cord = {}
            try:
                cord = json.loads(sequence) if sequence.strip().startswith("{") else {}
            except json.JSONDecodeError:
                pass
    else:
        cord = {}
        try:
            start = sequence.find("{")
            if start >= 0:
                end = sequence.rfind("}") + 1
                if end > start:
                    cord = json.loads(sequence[start:end])
        except json.JSONDecodeError:
            pass

    if not isinstance(cord, dict):
        cord = {}

    data = _donut_cord_to_reimbursement_dict(
        cord, employee_id=employee_id, expense_type=expense_type
    )
    # line_items already list of dicts; Pydantic will coerce to LineItemSchema
    try:
        schema = ReimbursementSchema(**data)
    except Exception as e:
        raise BillExtractionError(f"Donut output schema validation failed: {e}") from e

    return schema, 0.85


def extract_bill_via_layoutlm(
    image_bytes: bytes,
    raw_text: str,
    *,
    employee_id: str = "",
    expense_type: str = "",
    model_id: str | None = None,
) -> tuple[ReimbursementSchema, float]:
    """
    LayoutLM-based extraction (OCR text + optional image). Currently a stub.
    Use Donut or vision_llm for production.
    """
    raise NotImplementedError(
        "LayoutLM extractor is not implemented. Use vision_extractor=donut or vision_extractor=vision_llm."
    )
