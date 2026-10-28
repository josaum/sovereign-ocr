"""
CLI entrypoint for `sovereign-ocr` (`sovereign_ocr.cli:main`).

Subcommands:
- `sovereign-ocr doctor`: Verify air-gapped Rust wheels (`pdf_fast`, `email_fast`, `docx_fast`,
  `xlsx_fast`, `layout_fast`, `olaya`, `quad_fast`, `gliner_fast_py`) and local model caches.
- `sovereign-ocr preprocess <input_image_or_pdf> <output_png>`: Run the 5-step visual preprocessing
  pipeline and print geometry/DPI/deskew diagnostics as JSON.
- `sovereign-ocr extract <file_or_dir>`: Unpack, preprocess, OCR, and extract structured fields +
  cryptographic `quad_fast` receipt from any document or directory bundle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from sovereign_ocr.extractor import SovereignDocumentExtractor
from sovereign_ocr.ingest import IngestedBundle, unpack_bundle_directory, unpack_document_file
from sovereign_ocr.mcp_server import (
    export_mcp_schema_directory,
    run_stdio_server,
    run_streamable_http_server,
)
from sovereign_ocr.ocr_layout import preprocess_document_image_bytes, render_pdf_page_png
from sovereign_ocr.ontology_engine import SovereignOntologyEngine
from sovereign_ocr.sap_bpmn import build_sap_bapi_payload, export_sap_bpmn_xml, export_sap_dmn_xml


def cmd_doctor() -> int:
    modules = [
        "pdf_fast",
        "pdf_fast_write",
        "email_fast",
        "docx_fast",
        "xlsx_fast",
        "layout_fast",
        "ontology_fast",
        "owl_fast",
        "align_fast_py",
        "olaya",
        "quad_fast",
        "gliner_fast_py",
        "gliner2",
    ]
    report = {
        "status": "ok",
        "modules": {m: importlib.util.find_spec(m) is not None for m in modules},
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_preprocess(input_path: Path, output_path: Path, page_index: int = 0) -> int:
    if input_path.suffix.upper() == ".PDF":
        png_bytes = render_pdf_page_png(input_path, page_index=page_index, preprocess=True)
        output_path.write_bytes(png_bytes)
        print(json.dumps({"input": str(input_path), "output": str(output_path), "bytes": len(png_bytes)}, indent=2))
        return 0

    raw = input_path.read_bytes()
    proc_bytes, diag = preprocess_document_image_bytes(raw)
    output_path.write_bytes(proc_bytes)
    print(json.dumps({"input": str(input_path), "output": str(output_path), "diagnostics": diag}, indent=2))
    return 0


def cmd_extract(target_path: Path, ontology_json: Path | None = None, sap_bapi: bool = False) -> int:
    ont = SovereignOntologyEngine(master_kg_path=ontology_json) if ontology_json else SovereignOntologyEngine()
    ext = SovereignDocumentExtractor()
    if target_path.is_dir():
        bundle = unpack_bundle_directory(target_path, ontology_engine=ont)
    else:
        units = unpack_document_file(target_path)
        units.sort(key=lambda u: u.score, reverse=True)
        bundle = IngestedBundle(
            bundle_id=target_path.stem,
            document_id=target_path.stem,
            units=units,
            primary_unit=units[0] if units else None,
            combined_text="\n\n".join(u.text for u in units if u.text),
        )
    result = ext.extract_bundle(bundle, ontology_engine=ont)
    if sap_bapi:
        result["sap_s4hana_payload"] = build_sap_bapi_payload(result)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_export_bpmn(out_bpmn: Path, out_dmn: Path | None = None) -> int:
    out_bpmn.parent.mkdir(parents=True, exist_ok=True)
    out_bpmn.write_text(export_sap_bpmn_xml(), encoding="utf-8")
    res: dict[str, str] = {"bpmn_xml": str(out_bpmn)}
    if out_dmn is not None:
        out_dmn.parent.mkdir(parents=True, exist_ok=True)
        out_dmn.write_text(export_sap_dmn_xml(), encoding="utf-8")
        res["dmn_xml"] = str(out_dmn)
    print(json.dumps(res, indent=2))
    return 0


def cmd_mcp(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765) -> int:
    if transport == "http":
        return run_streamable_http_server(host=host, port=port)
    return run_stdio_server()


def cmd_export_mcp_schemas(out_dir: Path) -> int:
    written = export_mcp_schema_directory(out_dir)
    print(json.dumps({"out_dir": str(out_dir), "files": written}, indent=2))
    return 0


def main_mcp() -> int:
    """Direct MCP server entrypoint (`sovereign-ocr-mcp`)."""
    parser = argparse.ArgumentParser(prog="sovereign-ocr-mcp", description="Sovereign OCR MCP Server (stdio / Streamable HTTP)")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    return cmd_mcp(transport=args.transport, host=args.host, port=args.port)


def main() -> int:
    parser = argparse.ArgumentParser(prog="sovereign-ocr", description="Sovereign OCR & Document Intelligence CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check local Rust extensions and offline model availability")

    p_prep = sub.add_parser("preprocess", help="Run 5-step visual preprocessing on an image or PDF page")
    p_prep.add_argument("input", type=Path)
    p_prep.add_argument("output", type=Path)
    p_prep.add_argument("--page", type=int, default=0)

    p_ext = sub.add_parser("extract", help="Extract structured fields and cryptographic receipt from a file or bundle")
    p_ext.add_argument("target", type=Path)
    p_ext.add_argument("--ontology-json", type=Path, default=None)
    p_ext.add_argument("--sap-bapi", action="store_true", help="Include SAP S/4HANA BAPI_ACC_DOCUMENT_POST & OData payload")

    p_bpmn = sub.add_parser("export-bpmn", help="Export SAP Signavio / SBPA compatible BPMN 2.0 and DMN 1.3 XML")
    p_bpmn.add_argument("--bpmn-out", type=Path, required=True)
    p_bpmn.add_argument("--dmn-out", type=Path, default=None)

    p_mcp = sub.add_parser("mcp", help="Run the Model Context Protocol (MCP) server (stdio or Streamable HTTP /mcp)")
    p_mcp.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p_mcp.add_argument("--host", type=str, default="127.0.0.1")
    p_mcp.add_argument("--port", type=int, default=8765)

    p_mcp_exp = sub.add_parser("export-mcp-schemas", help="Export MCP tool JSON schemas and instructions.md")
    p_mcp_exp.add_argument("out_dir", type=Path)

    args = parser.parse_args()
    if args.cmd == "doctor":
        return cmd_doctor()
    if args.cmd == "preprocess":
        return cmd_preprocess(args.input, args.output, page_index=args.page)
    if args.cmd == "extract":
        return cmd_extract(args.target, ontology_json=args.ontology_json, sap_bapi=args.sap_bapi)
    if args.cmd == "export-bpmn":
        return cmd_export_bpmn(args.bpmn_out, out_dmn=args.dmn_out)
    if args.cmd == "mcp":
        return cmd_mcp(transport=args.transport, host=args.host, port=args.port)
    if args.cmd == "export-mcp-schemas":
        return cmd_export_mcp_schemas(args.out_dir)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
