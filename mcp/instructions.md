# `sovereign-ocr` MCP Server Instructions

100% air-gapped Sovereign-OCR & Neuro-Symbolic Document Intelligence MCP server (DOI: 10.5281/zenodo.23589075). Use `sovereign_ocr_extract_bundle` to unpack and extract structured fields + deterministic RDF 1.2 `quad_fast` receipts from `.pdf`, `.msg`, `.eml`, `.docx`, `.xlsx`, or image bundles; `sovereign_ocr_preprocess_image` for 5-step geometry/DPI visual preprocessing; `sovereign_ocr_repair_cmap` for algebraic Type0/Identity-H CMap shift recovery; `sovereign_ocr_validate_kyc_fiscal` for CPF/CNPJ/IMEI/NF-e check-digit verification; and `sovereign_ocr_build_sap_bapi` / `sovereign_ocr_export_bpmn_dmn` for SAP S/4HANA `BAPI_ACC_DOCUMENT_POST`, OData, BPMN 2.0, and DMN 1.3 integration.

## Supported MCP Specification Versions
- `2026-07-28` (Stateless `server/discover`, `outputSchema`, `structuredContent`, `x-mcp-header`)
- `2025-11-25` / `2025-06-18` (Streamable HTTP `/mcp`, `Tool.outputSchema`, `CallToolResult.structuredContent`, `annotations`)
- `2025-03-26` / `2024-11-05` (JSON-RPC 2.0 over `stdio` and Streamable HTTP)
