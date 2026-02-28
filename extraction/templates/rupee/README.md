# Rupee (₹) templates for OCR preprocessing

When `ocr.mask_rupee_symbol: true`, the pipeline detects the ₹ symbol using OpenCV template matching and masks it (white) before OCR to reduce misreads as "2" or "7".

- **Default**: Templates are generated from the "₹" character using a system font (DejaVu, Liberation, Arial, etc.) if available.
- **Optional**: Add your own PNG images here (e.g. `rupee_bold.png`, `rupee_light.png`) to improve detection on invoices with different fonts or rotations. Each file should be a small grayscale or RGB crop of the ₹ symbol on a light background. They will be loaded and used in addition to the generated templates.
