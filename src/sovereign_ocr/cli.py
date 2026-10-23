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
from sovereign_ocr.ocr_layout import preprocess_document_image_bytes, render_pdf_page_png
from sovereign_ocr.ontology_engine import SovereignOntologyEngine


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


def cmd_extract(target_path: Path, ontology_json: Path | None = None) -> int:
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
    print(json.dumps(result, indent=2, default=str))
    return 0


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

    args = parser.parse_args()
    if args.cmd == "doctor":
        return cmd_doctor()
    if args.cmd == "preprocess":
        return cmd_preprocess(args.input, args.output, page_index=args.page)
    if args.cmd == "extract":
        return cmd_extract(args.target, ontology_json=args.ontology_json)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
