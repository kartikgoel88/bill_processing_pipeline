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
# Default LayoutLMv3 model (CORD token classification)
DEFAULT_LAYOUTLM_MODEL_ID = "nielsr/layoutlmv3-finetuned-cord"

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


# ---------------------------------------------------------------------------
# LayoutLM: words + boxes from OCR, then token classification
# ---------------------------------------------------------------------------

def _ocr_words_and_boxes(image_bytes: bytes) -> tuple[list[str], list[list[int]]]:
    """
    Run Tesseract OCR on image to get word-level text and bounding boxes.
    Returns (words, boxes) with boxes in pixel coords; caller normalizes to 0-1000.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError as e:
        raise BillExtractionError(
            "LayoutLM extractor requires PIL and pytesseract for word-level OCR. "
            "Install: pip install pillow pytesseract (and Tesseract binary)."
        ) from e
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: list[str] = []
    boxes: list[list[int]] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        words.append(text)
        boxes.append([x, y, x + w, y + h])
    if not words:
        words = [""]
        boxes = [[0, 0, 100, 100]]
    return words, boxes


def _normalize_boxes_to_1000(boxes: list[list[int]], width: int, height: int) -> list[list[int]]:
    """Scale pixel boxes to 0-1000 coordinate system for LayoutLM."""
    if not width or not height:
        return [[0, 0, 1000, 1000]] * len(boxes)
    out = []
    for (x0, y0, x1, y1) in boxes:
        nx0 = max(0, min(1000, int(1000 * x0 / width)))
        ny0 = max(0, min(1000, int(1000 * y0 / height)))
        nx1 = max(0, min(1000, int(1000 * x1 / width)))
        ny1 = max(0, min(1000, int(1000 * y1 / height)))
        if nx0 == nx1:
            nx1 = nx0 + 1
        if ny0 == ny1:
            ny1 = ny0 + 1
        out.append([nx0, ny0, nx1, ny1])
    return out


# Base model for processor (ensures 224x224 image config); fine-tuned CORD model may have no processor_config
_LAYOUTLM_BASE_FOR_PROCESSOR = "microsoft/layoutlmv3-base"


def _load_layoutlm_model(model_id: str):
    """Lazy-load LayoutLMv3 processor and token classification model."""
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "LayoutLM extractor requires: pip install transformers torch pillow. "
            "Or install optional deps: pip install bill-processing-pipeline[document]"
        ) from e
    # Load processor from base model so image size is 224x224 (fine-tuned repo may lack processor_config)
    processor = AutoProcessor.from_pretrained(_LAYOUTLM_BASE_FOR_PROCESSOR, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


_layoutlm_cache: dict[str, tuple[Any, Any, str]] = {}


def _layoutlm_entities_to_cord(
    entity_list: list[tuple[str, str]],
) -> dict[str, Any]:
    """
    Map list of (label, text) from LayoutLM CORD token classification to CORD-style dict
    for _donut_cord_to_reimbursement_dict. CORD labels: total.total_price, total.date,
    menu.nm, menu.price, menu.cnt, vendor, etc.
    """
    cord: dict[str, Any] = {}
    # Group by label (strip B- / I- prefix and take the entity type)
    by_label: dict[str, list[str]] = {}
    for label, text in entity_list:
        if not text or not label or label in ("O", "o"):
            continue
        # Normalize B-total.total_price / I-total.total_price -> total.total_price
        key = label.replace("B-", "").replace("I-", "").strip()
        if key not in by_label:
            by_label[key] = []
        by_label[key].append(text.strip())

    def _join(values: list[str]) -> str:
        return " ".join(values).strip() if values else ""

    # Total block
    total_price = _join(by_label.get("total.total_price", []))
    if not total_price:
        total_price = _join(by_label.get("total_price", []))
    total_date = _join(by_label.get("total.date", []) or by_label.get("date", []))
    subtotal = _join(by_label.get("total.subtotal_price", []) or by_label.get("subtotal_price", []))
    if total_price or total_date or subtotal:
        cord["total"] = {
            "total_price": total_price or None,
            "date": total_date or None,
            "subtotal_price": subtotal or None,
        }

    # Vendor
    vendor = _join(by_label.get("vendor", []) or by_label.get("vendor_name", []) or by_label.get("seller", []))
    if vendor:
        cord["vendor"] = vendor

    # Date at top level
    if not total_date:
        top_date = _join(by_label.get("total.date", []) or by_label.get("date", []))
        if top_date:
            cord["date"] = top_date

    # Menu / line items: collect menu.nm, menu.price, menu.cnt and pair by index
    menu_nm = by_label.get("menu.nm", []) or by_label.get("menu.name", [])
    menu_price = by_label.get("menu.price", []) or by_label.get("menu_price", [])
    menu_cnt = by_label.get("menu.cnt", [])
    if menu_nm or menu_price:
        items = []
        max_len = max(len(menu_nm), len(menu_price), 1)
        for i in range(max_len):
            nm = menu_nm[i] if i < len(menu_nm) else ""
            pr = menu_price[i] if i < len(menu_price) else ""
            cnt = menu_cnt[i] if i < len(menu_cnt) else "1"
            if nm or pr:
                items.append({"nm": nm or "Item", "price": pr or "0", "cnt": cnt})
        if items:
            cord["menu"] = items

    return cord


def extract_bill_via_layoutlm(
    image_bytes: bytes,
    raw_text: str,
    *,
    employee_id: str = "",
    expense_type: str = "",
    model_id: str | None = None,
) -> tuple[ReimbursementSchema, float]:
    """
    LayoutLM-based extraction: word-level OCR + LayoutLMv3 token classification (CORD),
    then map entities to ReimbursementSchema. Uses same CORD mapping as Donut.
    """
    model_id = model_id or DEFAULT_LAYOUTLM_MODEL_ID
    try:
        from PIL import Image
    except ImportError:
        raise BillExtractionError("PIL/Pillow required for LayoutLM extractor") from None

    words, boxes = _ocr_words_and_boxes(image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    boxes_1000 = _normalize_boxes_to_1000(boxes, width, height)

    # LayoutLMv3 image encoder expects a fixed patch grid; resize to 224x224 (model default) so patch count matches.
    # Boxes are already in 0-1000 normalized coords so they remain valid after resize.
    if width != 224 or height != 224:
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        width, height = 224, 224

    if model_id not in _layoutlm_cache:
        _layoutlm_cache[model_id] = _load_layoutlm_model(model_id)
    processor, model, device = _layoutlm_cache[model_id]

    import torch

    # LayoutLMv3Processor(images, text=words, boxes=boxes) for single sample
    encoding = processor(
        image,
        text=words,
        boxes=boxes_1000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    # Preserve word_ids for token->word mapping (BatchEncoding method) before we build model inputs
    raw_encoding = encoding
    # Build model inputs: ensure pixel_values is (1, 3, 224, 224) so patch grid is 14x14
    model_inputs = {}
    for k, v in encoding.items():
        if hasattr(v, "to"):
            v = v.to(device)
        if k == "pixel_values":
            if isinstance(v, list):
                if len(v) == 1:
                    t = v[0]
                    v = t.unsqueeze(0).to(device) if hasattr(t, "unsqueeze") else torch.tensor(t, device=device).unsqueeze(0)
                else:
                    v = torch.stack([x.to(device) if hasattr(x, "to") else torch.tensor(x, device=device) for x in v])
            # Ensure batch dim and 224x224 so model gets 196 patches (14x14)
            if hasattr(v, "dim") and v.dim() == 3:
                v = v.unsqueeze(0)
            if hasattr(v, "shape") and len(v.shape) == 4 and (v.shape[2] != 224 or v.shape[3] != 224):
                # Resize to 224x224 via interpolate (C, H, W) or (B, C, H, W)
                v = torch.nn.functional.interpolate(
                    v, size=(224, 224), mode="bilinear", align_corners=False
                )
        model_inputs[k] = v

    model.eval()
    with torch.no_grad():
        outputs = model(**model_inputs)
    logits = outputs.logits  # (1, seq_len, num_labels)
    pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
    id2label = getattr(model.config, "id2label", {})
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}

    # Map subword predictions back to words using word_ids (method on BatchEncoding for Fast tokenizers)
    word_id_list: list[int | None]
    if hasattr(raw_encoding, "word_ids") and callable(raw_encoding.word_ids):
        try:
            word_id_list = raw_encoding.word_ids(batch_index=0) or []
        except TypeError:
            word_id_list = raw_encoding.word_ids(0) or []
    else:
        word_id_list = raw_encoding.get("word_ids")
        if isinstance(word_id_list, list) and word_id_list and isinstance(word_id_list[0], list):
            word_id_list = word_id_list[0]
        elif hasattr(word_id_list, "cpu"):
            word_id_list = word_id_list[0].cpu().tolist() if word_id_list is not None else []
        else:
            word_id_list = list(range(len(words)))

    # Build (label, word) per word (take prediction of first subword of each word)
    entity_list: list[tuple[str, str]] = []
    seen_word_idx: set[int | None] = set()
    for idx in range(min(len(pred_ids), len(word_id_list) if word_id_list else 0)):
        wid = word_id_list[idx] if idx < len(word_id_list) else None
        if wid is None or wid in seen_word_idx or wid < 0 or wid >= len(words):
            continue
        seen_word_idx.add(wid)
        label = id2label.get(pred_ids[idx], "O")
        if label and label != "O":
            entity_list.append((label, words[wid]))

    cord = _layoutlm_entities_to_cord(entity_list)
    data = _donut_cord_to_reimbursement_dict(
        cord, employee_id=employee_id, expense_type=expense_type
    )
    try:
        schema = ReimbursementSchema(**data)
    except Exception as e:
        raise BillExtractionError(f"LayoutLM output schema validation failed: {e}") from e
    return schema, 0.82
