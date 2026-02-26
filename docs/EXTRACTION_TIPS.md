# Improving Bill Extraction (Fewer Rejections)

When many meal (or other) bills are **REJECTED** with "Critical validation failed: amount is zero or negative; month missing", the **extraction** step is not parsing amount/month from the receipt. The decision LLM never runs for those bills.

---

## Why OCR Fails on Some PDFs (e.g. 31st_Lunch_1.pdf / HungerBox)

**Example: HungerBox-style tax invoices** (e.g. `test_input/meal/.../31st_Lunch_1.pdf`)

- **What works**: Header and footer text (Tax Invoice, Ordered From, Bill To, vendor name, GST text) are read correctly by Tesseract.
- **What fails**: The **middle section** (line items + Total / Net Amount in a table or formatted block) is often read as garbage, e.g.  
  `1 R4000—S*abi&F~SSi*iR LOR 100 fko00 R200` instead of real numbers and "Total" / "Net Amount".

**Reasons:**

1. **Table/cell layout** – Tesseract confuses table lines and cell boundaries with characters (especially with `--psm 11`).
2. **Small or low-contrast text** in the amount area.
3. **PDF rendering** – Default 300 DPI may be too low for small receipt text.

So the parser never sees a clear "Total" or "Net Amount" with a number, and gets `amount: 0`, `month: ""` → critical validation fails.

## What Was Fixed in Code

- **OCR parser** (`extraction/parser.py`): Added support for **Net Amount** / **Net Payable** lines and for **DD/MM/YY** and **Date : DD/MM/YY** (Indian receipts). Month is now derived from these dates when present.
- **Numeric validator** (`extraction/numeric_validator.py`): **Net Amount** is now an accepted phrase so amounts near "Net Amount" are not rejected.
- **Decision prompt**: Clarified to prefer APPROVED when the bill is valid and within policy; REJECT only for clear policy violation.

## Digital (non-image) PDFs: native text is used first

For **PDFs that contain embedded text** (digital/vector invoices, not scans), the pipeline now **extracts text directly from the PDF** (via pypdf) before running OCR. If the PDF has enough extractable text (≥80 characters), that text is used for parsing and **OCR is skipped**. So:

- **Digital PDFs** (e.g. many HungerBox/tax invoices): you get clean text and no OCR noise; the same parser runs on it.
- **Scanned/image-only PDFs**: native extraction returns little or no text, so the pipeline falls back to render-to-image + OCR as before.

No config change needed; this is automatic for all PDFs. If your PDFs are still failing, they may be image-based (scanned) — use **fusion** or **vision_first** with a vision model for those.

---

## Best Way to Extract This PDF (31st_Lunch_1 / HungerBox-style)

**Recommended: use a vision LLM so the image is read directly.**

1. In `config.yaml` set:
   ```yaml
   extraction:
     strategy: fusion   # or vision_first
   ```
2. Ensure a vision model is configured (e.g. `vision_model: qwen2.5vl:latest`) and the vision LLM is running (Ollama or your provider).
3. The pipeline will run OCR first; if you use `vision_first` it uses the vision model for extraction. With `fusion`, it can use vision when OCR confidence is low or when you tune `fallback_threshold`.

The vision model sees the same page image and can read "Total", "Net Amount", and the date from layout and context even when Tesseract output is garbled in the table area.

**Optional tweaks if you stay OCR-only:**

- **Higher DPI** for PDF→image: in `config.yaml` under `ocr:` set `dpi: 400` (or set env `OCR_DPI=400`). This can help small text.
- **Try EasyOCR**: set `ocr.engine: easyocr` (or `OCR_ENGINE=easyocr`); install with `pip install easyocr` or `.[ocr]`. Sometimes better on receipts.

---

## Other Ways to Improve Results

1. **Use vision extraction for poor OCR**  
   In `config.yaml`, set `extraction.strategy` to `fusion` or `vision_first` so a vision LLM is used to extract from the image. This is the most reliable fix for table-heavy or HungerBox-style receipts.

2. **OCR DPI**  
   Under `ocr:` set `dpi: 400` (or `OCR_DPI=400`) to render PDFs at higher resolution; can help small text.

3. **Improve scan quality**  
   Low-resolution or skewed PDFs hurt OCR. Use clear, upright scans or photos.

4. **Tune fallback**  
   With `fusion`, lowering `fallback_threshold` (e.g. to 0.5) makes it more likely to use the vision result instead of OCR when vision is uncertain.

5. **Policy limits**  
   Ensure `policy_path` includes the right allowances (e.g. `meal_allowance`) so valid bills are not rejected.

6. **Re-run after parser changes**  
   After parser/validator improvements, re-run the batch; many previously rejected meal bills may now get amount and month from OCR when the receipt text is clear.
