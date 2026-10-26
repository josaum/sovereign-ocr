# Sovereign-OCR — Native Claude Desktop Plugin (`.mcpb`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23589075.svg)](https://doi.org/10.5281/zenodo.23589075)

Native **Model Context Protocol Bundle (`.mcpb` v0.4)** for one-click / double-click installation in **Claude Desktop** across:
- **macOS Apple Silicon (`darwin-arm64`)**: `dist/sovereign-ocr-darwin-arm64.mcpb` (and `dist/sovereign-ocr.mcpb`)
- **Windows x64 (`win32-x64`)**: `dist/sovereign-ocr-win32-x64.mcpb` *(includes embedded CPython 3.12 runtime `runtime-python/python.exe` + native `win_amd64` Rust wheels)*
- **Linux x64 (`linux-x64`)**: `dist/sovereign-ocr-linux-x64.mcpb` *(includes native `manylinux_x86_64` Rust wheels)*
- **Linux ARM64 (`linux-arm64`)**: `dist/sovereign-ocr-linux-arm64.mcpb` *(includes native `manylinux_aarch64` Rust wheels)*

## Double-Click Installation

1. Double-click the `.mcpb` file matching your operating system (or drag-and-drop it into **Claude Desktop $\rightarrow$ Settings $\rightarrow$ Extensions**).
2. Claude Desktop verifies the `manifest.json` (`manifest_version: "0.4"`), prompts for optional configuration (`Default SAP Company Code`, `Active Enterprise Ontology JSON`, `Document Workspace Directory`), and installs the extension locally.
3. All 7 MCP tools (`sovereign_ocr_doctor`, `sovereign_ocr_preprocess_image`, `sovereign_ocr_repair_cmap`, `sovereign_ocr_validate_kyc_fiscal`, `sovereign_ocr_extract_bundle`, `sovereign_ocr_build_sap_bapi`, `sovereign_ocr_export_bpmn_dmn`), 4 resources, and 2 prompts become immediately available offline.
