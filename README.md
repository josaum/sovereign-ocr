# `sovereign-ocr` — Sovereign OCR & Neuro-Symbolic Document Intelligence Engine

`sovereign-ocr` is a 100% air-gapped, domain-agnostic, zero-cloud-token document intelligence and OCR engine combining zero-copy Rust format parsers, geometry- & DPI-aware visual preprocessing, strongly-typed Pydantic v2 to GLiNER 2.5 schema compilation, active runtime ontology grounding, and deterministic cryptographic provenance receipts.

---

## Five-Stage Architecture

1. **Stage 1 — Multi-Format Native Unpacking & Candidate Routing (`sovereign_ocr.ingest`):**
   - Unpacks `.msg` / `.eml` (including raw OLE CFB `__attach_version1.0_*` binary streams via `email_fast`), `.pdf` (`pdf_fast`, `pdf_fast_write`), `.docx` (`docx_fast`), and `.xlsx` / `.xlsm` (`xlsx_fast`).
   - Scores multi-attachment packets (`classify_document_unit`) to separate primary actionable commercial documents (invoices, credit memos, statements) from supporting backup artifacts (weight/scale tickets, quotes with `TBD` dates, POS receipts, and split spreadsheets).
   - Splits multi-invoice PDFs zero-copy via `detect_and_split_multi_invoice_pdf`.

2. **Stage 2 — Self-Healing Font-Corruption Gate + 5-Step Visual Preprocessing + Spatial OCR (`sovereign_ocr.ocr_layout`):**
   - **Self-Healing PDF Gate (`is_corrupted_or_scanned_pdf_page`):** Detects scanned pages (`< 80` chars) and broken Identity-H / Type0 `ToUnicode` CMap font tables (NUL bytes, control chars, `cmap_fallback_chars` ratio, vocabulary check).
   - **5-Step Visual Preprocessing (`preprocess_document_image`):**
     1. `compute_adaptive_dpi`: Computes target DPI (`120..300 DPI`) from physical PDF dimensions (`pdf_fast.parse_pdf_page_dimensions`).
     2. `enhance_contrast_and_denoise`: Stretches compressed luminance (`[p1..p95] -> [0..255]`) *before* orientation/skew estimation.
     3. Midpoint foreground binarization (`0.5 * (p5 + p80)`) so faded gray backgrounds are never misclassified as ink.
     4. `detect_coarse_rotation_90_180_270`: Combines PDF `/Rotate` metadata with horizontal vs. vertical 1D ink projection-profile variance (`V_var > 2.2 * H_var`).
     5. `estimate_and_correct_skew_angle` (`-12°..+12°` two-stage projection-profile variance deskew) + `dpi_aware_resize_image` (Lanczos resize into `[900 px short edge .. 2560 px long edge]`).
   - **Spatial Reading Order (`reconstruct_text_from_ocr_runs`):** Reconstructs multi-column and tabular rows via vertical line clustering and `layout_fast.order_blocks`.

3. **Stage 3 — Strongly-Typed Pydantic v2 $\rightarrow$ Local GLiNER 2.5 Schema Compilation (`sovereign_ocr.schemas`):**
   - Compiles any annotated Pydantic v2 model into a `gliner2.Schema` (`StructureInput`, `FieldInput`, `ClassificationInput`, `RegexValidator`) for `fastino/gliner2.5-multi-v1` and `gliner_fast_py` ONNX span extraction in strict offline mode (`HF_HUB_OFFLINE=1`).

4. **Stage 4 — Active Runtime Ontology Grounding (`sovereign_ocr.ontology_engine`):**
   - Grounds extracted entities against any pluggable enterprise ontology / master entity graph using `olaya` (`OWL2RLReasoner`, `CandidateRouter`), `ontology_fast`, `owl_fast`, and `align_fast_py`, enforcing ERP accounting invariants (e.g., excluding reconciliation control accounts, aligning cost/profit centers to entity prefixes, and distinguishing billed sales tax `I1` from self-assessed consumer use tax `U1`).

5. **Stage 5 — Deterministic Cryptographic Provenance (`quad_fast`):**
   - Seals every extracted fact and ontology grounding edge into an Arrow IPC / RDF quad graph (`build_deterministic_quad_receipt`) with reproducible SHA-256 receipt hashes.
