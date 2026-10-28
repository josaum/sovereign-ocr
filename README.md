# `sovereign-ocr` — Sovereign OCR & Neuro-Symbolic Document Intelligence Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23589075.svg)](https://doi.org/10.5281/zenodo.23589075)

`sovereign-ocr` is a 100% air-gapped, domain-agnostic, zero-cloud-token document intelligence and OCR engine combining zero-copy Rust format parsers, algebraic PDF CMap self-healing, geometry- & DPI-aware visual preprocessing, strongly-typed Pydantic v2 to GLiNER 2.5 schema compilation, active runtime ontology grounding, and deterministic cryptographic provenance receipts.

- **Paper (Zenodo Open Access):** [Sovereign-OCR: Zero-Hallucination Neuro-Symbolic Document Intelligence with Self-Healing CMap Recovery, Active Ontology Grounding, and Deterministic RDF 1.2 Provenance](https://zenodo.org/records/23589075) (`DOI: 10.5281/zenodo.23589075`)
- **Author:** Dionisio Chiuratto Agourakis (`dionisio.ca@gmail.com`, ORCID: [`0009-0000-4863-9143`](https://orcid.org/0009-0000-4863-9143))

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

---

## Model Context Protocol (MCP) Server (`2025-11-25` & `2026-07-28` Specs)

`sovereign-ocr` includes a built-in, zero-dependency Model Context Protocol (MCP) server (`sovereign_ocr.mcp_server`) supporting both **Stdio** (`--transport stdio`) and **Streamable HTTP** (`--transport http` on `/mcp` with `Mcp-Session-Id` & `MCP-Protocol-Version` headers), `Tool.outputSchema` (JSON Schema 2020-12), `CallToolResult.structuredContent`, tool behavioral `annotations`, and `x-mcp-header` (`Mcp-Param-Entity-Code`).

```bash
# Run over Stdio (Claude Desktop / Cursor / Antigravity / MCP Inspector)
sovereign-ocr mcp --transport stdio
# Or via dedicated entrypoint:
sovereign-ocr-mcp

# Run over Streamable HTTP (/mcp endpoint)
sovereign-ocr mcp --transport http --host 127.0.0.1 --port 8765
```

### Exposed MCP Tools (7 Tools)
1. `sovereign_ocr_doctor` — Inspect air-gapped Rust PyO3 wheels, `oar-ocr-cli` binary, and offline GLiNER2 availability.
2. `sovereign_ocr_preprocess_image` — Run the 5-step geometry/DPI visual preprocessing pipeline on an image or PDF page.
3. `sovereign_ocr_repair_cmap` — Run the algebraic CMap shift solver (`\Delta = +29`) and corruption detector on raw PDF text.
4. `sovereign_ocr_validate_kyc_fiscal` — Classify KYC/insurance/fiscal roles and validate CPF, CNPJ, IMEI (Luhn), and 44-digit NF-e access keys.
5. `sovereign_ocr_extract_bundle` — Unpack, OCR, ground against an active ontology, seal with `quad_fast`, and emit an SAP S/4HANA `BAPI_ACC_DOCUMENT_POST` payload.
6. `sovereign_ocr_build_sap_bapi` — Transform a governed extraction record into an SAP S/4HANA BAPI, OData, and BPMN 2.0 gateway decision.
7. `sovereign_ocr_export_bpmn_dmn` — Export OASIS BPMN 2.0 XML and OMG DMN 1.3 XML for SAP Signavio and SAP Build Process Automation (SBPA).

### Native Claude Desktop Double-Click Extension Bundles (`.mcpb` v0.4)
Build self-contained `.mcpb` bundles across macOS ARM64 (`darwin-arm64`), Windows x64 (`win32-x64` with embedded Python 3.12 + `_pth` configuration + `.cmd` launcher), Linux x64 (`linux-x64`), and Linux ARM64 (`linux-arm64`):

```bash
python scripts/build_mcpb.py
```

---

## Benchmarks (`v1.0.0` + `olaya v0.8.0`)

Reproduce all benchmarks via [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py) (consolidated JSON report in [`output/benchmark_results_latest.json`](output/benchmark_results_latest.json)):

| Benchmark Suite | Split / Metric | Score | Details |
| :--- | :--- | :---: | :--- |
| **`allenai/olmOCR-bench`** | `headers_footers.jsonl` | **82.6%** (`628/760`) | **90.1%** (`685/760`) on native spatial header/footer filter |
| **`allenai/olmOCR-bench`** | `long_tiny_text.jsonl` | **82.1%** (`363/442`) | Dense multi-column tiny text extraction |
| **`allenai/olmOCR-bench`** | `multi_column.jsonl` (Reading Order) | **94.5%** (`345/365`) | **98.3%** (`287/292`) `layout_fast` `recursive` on proportional line segments |
| **`allenai/olmOCR-bench`** | Core Document Splits Micro-Avg | **72.2%** (`1,337/1,853`) | `1,022` unique PDFs (`91` scanned/corrupted pages via `oar-ocr-cli`) |
| **Enterprise SAP S/4HANA AP** | 100-Invoice Head-to-Head POC | **100.0%** (`100/100`) | All 10 SAP fields (`Doc Type`, `Vendor`, `Date`, `Ref`, `Amount`, `GL`, `CC`, `PC`, `Tax`, `Terms`) |
| **Enterprise SAP S/4HANA AP** | Full `115`-Invoice `FB03` Benchmark | **99.1%** Ref / **98.3%** Date | **100.0%** Doc Type, **96.5%** Amount, **100.0%** `quad_fast` receipts in `16.34s` |
| **Assurant Documentoscopia** | `20` Mobile Claims (`88` Files) | **100.0%** (`88/88`) | `32` `+29` CMap shifted PDFs + `90°` sideways RG recovered in `0.49s` |

---

## Citation

If you use `sovereign-ocr` in your research or production systems, please cite:

```bibtex
@article{agourakis2026sovereignocr,
  author    = {Chiuratto Agourakis, Dionisio},
  title     = {Sovereign-OCR: Zero-Hallucination Neuro-Symbolic Document Intelligence with Self-Healing CMap Recovery, Active Ontology Grounding, and Deterministic RDF 1.2 Provenance},
  year      = {2026},
  month     = oct,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23589075},
  url       = {https://doi.org/10.5281/zenodo.23589075}
}
```
