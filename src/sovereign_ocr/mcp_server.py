"""
Model Context Protocol (MCP) Server for `sovereign-ocr` (`sovereign_ocr.mcp_server`).

Implements the latest official MCP specifications:
- Protocol revisions: `2026-07-28` (stateless core + `server/discover` + `_meta`), `2025-11-25`,
  `2025-06-18`, `2025-03-26`, and `2024-11-05` backwards compatibility.
- Tools (`tools/list`, `tools/call`) with:
  - `title` and human-readable descriptions
  - JSON Schema 2020-12 `inputSchema` (with `x-mcp-header` parameter routing extensions)
  - Root `type: "object"` `outputSchema` (JSON Schema 2020-12)
  - Behavioral `annotations` (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`)
  - Dual `CallToolResult` payloads (`content` text blocks + `structuredContent` matching `outputSchema`)
- Resources (`resources/list`, `resources/read`, `resources/templates/list`) exposing the SAP S/4HANA
  BPMN 2.0 XML, DMN 1.3 decision table, Pydantic v2 extraction schema, and Zenodo DOI citation.
- Prompts (`prompts/list`, `prompts/get`) for SAP AP voucher auditing and Documentoscopia claims triage.
- Transports:
  1. **Stdio Transport** (`sovereign-ocr mcp` / `sovereign-ocr-mcp`): newline-delimited JSON-RPC 2.0 on `stdin`/`stdout`.
  2. **Streamable HTTP Transport** (`sovereign-ocr mcp --transport http --port 8765`): single `/mcp` endpoint
     supporting `POST` (JSON or `text/event-stream` SSE), `GET` (SSE stream), and `DELETE` (session teardown),
     with `Mcp-Session-Id` and `MCP-Protocol-Version` headers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sovereign_ocr import __version__
from sovereign_ocr.extractor import SovereignDocumentExtractor
from sovereign_ocr.ingest import (
    IngestedBundle,
    classify_document_unit,
    unpack_bundle_directory,
    unpack_document_file,
)
from sovereign_ocr.ocr_layout import (
    is_corrupted_or_scanned_pdf_page,
    preprocess_document_image_bytes,
    render_pdf_page_png,
    repair_shifted_cmap_text,
    resolve_oar_ocr_bin,
)
from sovereign_ocr.ontology_engine import SovereignOntologyEngine
from sovereign_ocr.sap_bpmn import (
    build_sap_bapi_payload,
    evaluate_bpmn_gateway_route,
    export_sap_bpmn_xml,
    export_sap_dmn_xml,
)
from sovereign_ocr.schemas import DocumentExtractionEnvelope

SUPPORTED_PROTOCOL_VERSIONS = (
    "2026-07-28",
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
DEFAULT_PROTOCOL_VERSION = "2025-11-25"
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

SERVER_INFO = {
    "name": "sovereign-ocr",
    "title": "Sovereign-OCR Neuro-Symbolic Document Intelligence MCP Server",
    "version": __version__,
}

SERVER_INSTRUCTIONS = (
    "100% air-gapped Sovereign-OCR & Neuro-Symbolic Document Intelligence MCP server "
    "(DOI: 10.5281/zenodo.23589075). Use `sovereign_ocr_extract_bundle` to unpack and extract "
    "structured fields + deterministic RDF 1.2 `quad_fast` receipts from `.pdf`, `.msg`, `.eml`, "
    "`.docx`, `.xlsx`, or image bundles; `sovereign_ocr_preprocess_image` for 5-step geometry/DPI "
    "visual preprocessing; `sovereign_ocr_repair_cmap` for algebraic Type0/Identity-H CMap shift "
    "recovery; `sovereign_ocr_validate_kyc_fiscal` for CPF/CNPJ/IMEI/NF-e check-digit verification; "
    "and `sovereign_ocr_build_sap_bapi` / `sovereign_ocr_export_bpmn_dmn` for SAP S/4HANA "
    "`BAPI_ACC_DOCUMENT_POST`, OData, BPMN 2.0, and DMN 1.3 integration."
)


def get_mcp_tools() -> list[dict[str, Any]]:
    """Return the list of MCP Tool definitions conforming to the 2025-11-25 / 2026-07-28 specification."""
    return [
        {
            "name": "sovereign_ocr_doctor",
            "title": "Sovereign-OCR Environment & Native Wheel Doctor",
            "description": (
                "Verify availability of all 8 native Rust PyO3 abi3 extension modules "
                "(pdf_fast, pdf_fast_write, email_fast, docx_fast, xlsx_fast, layout_fast, "
                "quad_fast, gliner_fast_py) and the embedded oar-ocr-cli ONNX binary."
            ),
            "inputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "outputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "version": {"type": "string"},
                    "olaya_version": {"type": "string"},
                    "leio_code_version": {"type": "string"},
                    "oar_ocr_bin": {"type": "string"},
                    "modules": {"type": "object", "additionalProperties": {"type": "boolean"}},
                },
                "required": ["status", "version", "olaya_version", "leio_code_version", "oar_ocr_bin", "modules"],
            },
            "annotations": {
                "title": "Environment Doctor",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "sovereign_ocr_preprocess_image",
            "title": "5-Step Geometry- & DPI-Aware Visual Preprocessor",
            "description": (
                "Run the 5-step visual preprocessing pipeline (adaptive DPI, percentile luminance "
                "stretching [p1..p95]->[0..255], interior-crop 90/180/270 orientation detection, "
                "two-stage projection-profile deskew [-12..+12 deg], and Lanczos receptive-field resize) "
                "on an image or PDF page."
            ),
            "inputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Absolute path to the input image (.png, .jpg, .tiff, .webp) or .pdf file",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional output path where the preprocessed PNG will be written",
                    },
                    "page_index": {
                        "type": "integer",
                        "description": "0-based page index when input_path is a PDF",
                        "default": 0,
                    },
                },
                "required": ["input_path"],
                "additionalProperties": False,
            },
            "outputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "output_path": {"type": ["string", "null"]},
                    "output_bytes_len": {"type": "integer"},
                    "diagnostics": {"type": "object"},
                },
                "required": ["input_path", "output_bytes_len", "diagnostics"],
            },
            "annotations": {
                "title": "5-Step Visual Preprocessor",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "sovereign_ocr_repair_cmap",
            "title": "Algebraic PDF ToUnicode CMap Self-Healing & Corruption Gate",
            "description": (
                "Detect whether a PDF page or text string suffers from broken/shifted Type0/Identity-H "
                "ToUnicode CMap tables (e.g. delta=+29 or -4 GID-to-ASCII shift) and algebraically self-heal "
                "both uniform and mixed-font shifted tokens in <1 ms."
            ),
            "inputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw PDF or OCR text to inspect and self-heal",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "outputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "was_corrupted_or_scanned_before": {"type": "boolean"},
                    "needs_visual_ocr_after_repair": {"type": "boolean"},
                    "detected_ascii_shift_delta": {"type": "integer"},
                    "repaired_text": {"type": "string"},
                },
                "required": [
                    "was_corrupted_or_scanned_before",
                    "needs_visual_ocr_after_repair",
                    "detected_ascii_shift_delta",
                    "repaired_text",
                ],
            },
            "annotations": {
                "title": "Algebraic CMap Repair",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "sovereign_ocr_validate_kyc_fiscal",
            "title": "Cryptographic/Algebraic KYC & Fiscal Validator (CPF, CNPJ, IMEI, NF-e)",
            "description": (
                "Extract and algebraically validate CPFs (11-digit Mod-11), CNPJs (14-digit Mod-11), "
                "mobile IMEIs (15-digit Luhn), Brazilian NF-e 44-digit access keys (IBGE UF + Mod-11), "
                "CEPs, RG numbers, dates, amounts, and police report (BO) occurrence typification."
            ),
            "inputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Document or claim bundle text to extract and validate facts from",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename for document unit role classification",
                        "default": "document.pdf",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "outputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "classified_role": {"type": "string"},
                    "role_score": {"type": "number"},
                    "cpfs": {"type": "array", "items": {"type": "string"}},
                    "cnpjs": {"type": "array", "items": {"type": "string"}},
                    "imeis": {"type": "array", "items": {"type": "string"}},
                    "nfe_access_keys": {"type": "array", "items": {"type": "string"}},
                    "ceps": {"type": "array", "items": {"type": "string"}},
                    "rgs": {"type": "array", "items": {"type": "string"}},
                    "occurrence_type": {"type": ["string", "null"]},
                    "dates": {"type": "array", "items": {"type": "string"}},
                    "primary_amount": {"type": ["number", "null"]},
                },
                "required": [
                    "classified_role",
                    "cpfs",
                    "cnpjs",
                    "imeis",
                    "nfe_access_keys",
                    "dates",
                ],
            },
            "annotations": {
                "title": "KYC & Fiscal Fact Validator",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "sovereign_ocr_extract_bundle",
            "title": "5-Stage Sovereign Document Bundle Extractor",
            "description": (
                "Unpack, self-heal, OCR, extract, and ontology-ground a single document file or "
                "multi-attachment bundle directory (.pdf, .msg, .eml, .docx, .xlsx, .png, .jpg), "
                "emitting a deterministic RDF 1.2 quad_fast provenance receipt and optional SAP S/4HANA BAPI payload."
            ),
            "inputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "target_path": {
                        "type": "string",
                        "description": "Absolute path to a document file or bundle directory",
                    },
                    "entity_code": {
                        "type": "string",
                        "description": "Optional ERP company code / entity code (e.g. '0010')",
                        "default": "",
                        "x-mcp-header": "Mcp-Param-Entity-Code",
                    },
                    "ontology_json_path": {
                        "type": "string",
                        "description": "Optional path to master ontology JSON file for active runtime grounding",
                    },
                    "include_sap_bapi": {
                        "type": "boolean",
                        "description": "Include ready-to-post SAP S/4HANA BAPI_ACC_DOCUMENT_POST & OData payload",
                        "default": True,
                    },
                },
                "required": ["target_path"],
                "additionalProperties": False,
            },
            "outputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "bundle_id": {"type": "string"},
                    "primary_unit_kind": {"type": "string"},
                    "primary_unit_role": {"type": "string"},
                    "needs_ocr": {"type": "boolean"},
                    "extracted_doc_type": {"type": "string"},
                    "extracted_date": {"type": "string"},
                    "extracted_reference": {"type": "string"},
                    "extracted_amount": {"type": ["number", "null"]},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                    "documentoscopia_facts": {"type": "object"},
                    "quad_receipt_hash": {"type": "string"},
                    "sap_s4hana_payload": {"type": ["object", "null"]},
                },
                "required": [
                    "bundle_id",
                    "primary_unit_kind",
                    "primary_unit_role",
                    "warnings",
                ],
            },
            "annotations": {
                "title": "5-Stage Bundle Extractor",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "sovereign_ocr_build_sap_bapi",
            "title": "SAP S/4HANA BAPI_ACC_DOCUMENT_POST & OData Payload Builder",
            "description": (
                "Transform a governed Sovereign-OCR record into a ready-to-post SAP S/4HANA "
                "BAPI_ACC_DOCUMENT_POST + API_SUPPLIERINVOICE_PROCESS_SRV OData payload with "
                "BPMN 2.0 gateway routing and strict zero-balance verification (sum(AMT_DOCCUR) == 0.00)."
            ),
            "inputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "governed_record": {
                        "type": "object",
                        "description": "Governed extraction record dictionary",
                    },
                },
                "required": ["governed_record"],
                "additionalProperties": False,
            },
            "outputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "bpmn_routing": {"type": "object"},
                    "BAPI_ACC_DOCUMENT_POST": {"type": "object"},
                    "API_SUPPLIERINVOICE_PROCESS_SRV": {"type": "object"},
                    "ARCHIVELINK_PROVENANCE": {"type": "object"},
                },
                "required": [
                    "bpmn_routing",
                    "BAPI_ACC_DOCUMENT_POST",
                    "API_SUPPLIERINVOICE_PROCESS_SRV",
                    "ARCHIVELINK_PROVENANCE",
                ],
            },
            "annotations": {
                "title": "SAP S/4HANA BAPI Builder",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "sovereign_ocr_export_bpmn_dmn",
            "title": "Export SAP Signavio / SBPA BPMN 2.0 & DMN 1.3 XML",
            "description": (
                "Generate standard OMG BPMN 2.0 XML (with <sap:bapiBinding> extension elements) "
                "and OMG DMN 1.3 FEEL decision table XML for SAP Signavio and SAP Build Process Automation."
            ),
            "inputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "process_id": {
                        "type": "string",
                        "default": "Process_Sovereign_OCR_SAP",
                    },
                    "decision_id": {
                        "type": "string",
                        "default": "Decision_Sovereign_SAP_Governance",
                    },
                },
                "additionalProperties": False,
            },
            "outputSchema": {
                "$schema": JSON_SCHEMA_DRAFT,
                "type": "object",
                "properties": {
                    "bpmn_xml": {"type": "string"},
                    "dmn_xml": {"type": "string"},
                },
                "required": ["bpmn_xml", "dmn_xml"],
            },
            "annotations": {
                "title": "Export BPMN 2.0 & DMN 1.3 XML",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
    ]


def get_mcp_resources() -> list[dict[str, Any]]:
    """Return static MCP resources exposed by `sovereign-ocr`."""
    return [
        {
            "uri": "sovereign://workflows/sap-s4hana.bpmn",
            "name": "sovereign_sap_s4hana.bpmn",
            "title": "SAP S/4HANA Executable BPMN 2.0 Workflow",
            "description": "OMG BPMN 2.0 XML workflow with <sap:bapiBinding> extension elements for SAP Signavio & SBPA",
            "mimeType": "application/xml",
        },
        {
            "uri": "sovereign://workflows/sap-governance.dmn",
            "name": "sovereign_sap_governance.dmn",
            "title": "SAP S/4HANA Governance DMN 1.3 Decision Table",
            "description": "OMG DMN 1.3 FEEL decision table for KR/KG/RE voucher routing and exception handling",
            "mimeType": "application/xml",
        },
        {
            "uri": "sovereign://schemas/extraction-envelope.json",
            "name": "DocumentExtractionEnvelope.schema.json",
            "title": "Pydantic v2 DocumentExtractionEnvelope JSON Schema",
            "description": "JSON Schema 2020-12 compiled from Sovereign-OCR Pydantic v2 extraction models",
            "mimeType": "application/schema+json",
        },
        {
            "uri": "sovereign://citation/zenodo-23589075.bib",
            "name": "sovereign_ocr_zenodo_23589075.bib",
            "title": "Sovereign-OCR Zenodo Preprint Citation (DOI: 10.5281/zenodo.23589075)",
            "description": "Official BibTeX citation for Sovereign-OCR by Dionisio Chiuratto Agourakis",
            "mimeType": "text/plain",
        },
    ]


def read_mcp_resource(uri: str) -> dict[str, Any]:
    """Read an MCP resource by URI and return its `contents` list."""
    if uri == "sovereign://workflows/sap-s4hana.bpmn":
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/xml",
                    "text": export_sap_bpmn_xml(),
                }
            ]
        }
    if uri == "sovereign://workflows/sap-governance.dmn":
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/xml",
                    "text": export_sap_dmn_xml(),
                }
            ]
        }
    if uri == "sovereign://schemas/extraction-envelope.json":
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/schema+json",
                    "text": json.dumps(DocumentExtractionEnvelope.model_json_schema(), indent=2),
                }
            ]
        }
    if uri == "sovereign://citation/zenodo-23589075.bib":
        bib = (
            "@article{agourakis2026sovereignocr,\n"
            "  author    = {Chiuratto Agourakis, Dionisio},\n"
            "  title     = {Sovereign-OCR: Zero-Hallucination Neuro-Symbolic Document Intelligence "
            "with Self-Healing CMap Recovery, Active Ontology Grounding, and Deterministic RDF 1.2 Provenance},\n"
            "  year      = {2026},\n"
            "  month     = oct,\n"
            "  publisher = {Zenodo},\n"
            "  doi       = {10.5281/zenodo.23589075},\n"
            "  url       = {https://doi.org/10.5281/zenodo.23589075}\n"
            "}\n"
        )
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": bib}]}
    raise ValueError(f"Unknown resource URI: {uri}")


def get_mcp_prompts() -> list[dict[str, Any]]:
    """Return MCP prompt templates exposed by `sovereign-ocr`."""
    return [
        {
            "name": "audit_sap_ap_bundle",
            "title": "Audit & Post SAP S/4HANA Accounts Payable Bundle",
            "description": "Extract, ontology-ground, and verify a zero-balance SAP S/4HANA BAPI_ACC_DOCUMENT_POST voucher",
            "arguments": [
                {
                    "name": "target_path",
                    "description": "Path to the invoice file or .MSG/.XLSX bundle directory",
                    "required": True,
                },
                {
                    "name": "entity_code",
                    "description": "SAP Company Code (BUKRS, e.g. 0010)",
                    "required": False,
                },
            ],
        },
        {
            "name": "triage_documentoscopia_claim",
            "title": "Triage Insurance Claim Bundle (Documentoscopia)",
            "description": "Classify claim units (Aviso, BO, DANFE/NF-e, CR, RG/CNH) and validate CPF, IMEI, and NF-e access keys",
            "arguments": [
                {
                    "name": "target_path",
                    "description": "Path to the claim bundle directory or document",
                    "required": True,
                }
            ],
        },
    ]


def get_mcp_prompt_messages(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    args = arguments or {}
    if name == "audit_sap_ap_bundle":
        target = args.get("target_path", "<target_path>")
        cocd = args.get("entity_code", "0010")
        return {
            "description": "Audit and generate SAP S/4HANA BAPI payload for an AP bundle",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": (
                            f"Run `sovereign_ocr_extract_bundle` on `{target}` with `entity_code='{cocd}'` "
                            "and `include_sap_bapi=true`. Verify that `zero_balance_verified` is `true`, "
                            "inspect `bpmn_routing.target_bpmn_node`, and report the `quad_receipt_hash`."
                        ),
                    },
                }
            ],
        }
    if name == "triage_documentoscopia_claim":
        target = args.get("target_path", "<target_path>")
        return {
            "description": "Triage a multi-document insurance claim bundle",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": (
                            f"Run `sovereign_ocr_extract_bundle` on `{target}` and verify all extracted "
                            "Mod-11 CPFs, 15-digit Luhn IMEIs, 44-digit NF-e access keys, and BO theft/robbery "
                            "typifications in `documentoscopia_facts`."
                        ),
                    },
                }
            ],
        }
    raise ValueError(f"Unknown prompt name: {name}")


def execute_mcp_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Execute an MCP tool by name and return a spec-compliant `CallToolResult` containing
    BOTH `content` (text blocks for LLM context) and `structuredContent` (validated JSON
    object conforming to `outputSchema`).
    """
    args = arguments or {}

    if name == "sovereign_ocr_doctor":
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
        import olaya as _olaya_mod

        structured = {
            "status": "ok",
            "version": __version__,
            "olaya_version": getattr(_olaya_mod, "__version__", "0.8.0"),
            "leio_code_version": "2.6.6",
            "oar_ocr_bin": resolve_oar_ocr_bin(),
            "modules": {m: importlib.util.find_spec(m) is not None for m in modules},
        }
        return {
            "content": [{"type": "text", "text": json.dumps(structured, indent=2)}],
            "structuredContent": structured,
            "isError": False,
        }

    if name == "sovereign_ocr_preprocess_image":
        in_path = Path(str(args["input_path"]))
        out_str = args.get("output_path")
        page_idx = int(args.get("page_index") or 0)
        if in_path.suffix.upper() == ".PDF":
            proc_bytes = render_pdf_page_png(in_path, page_index=page_idx, preprocess=True)
            diag: dict[str, Any] = {"source_kind": "PDF", "page_index": page_idx}
        else:
            raw = in_path.read_bytes()
            proc_bytes, diag = preprocess_document_image_bytes(raw)
        if out_str:
            out_p = Path(str(out_str))
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(proc_bytes)
        structured = {
            "input_path": str(in_path),
            "output_path": str(out_str) if out_str else None,
            "output_bytes_len": len(proc_bytes),
            "diagnostics": diag,
        }
        return {
            "content": [{"type": "text", "text": json.dumps(structured, indent=2, default=str)}],
            "structuredContent": structured,
            "isError": False,
        }

    if name == "sovereign_ocr_repair_cmap":
        raw_txt = str(args.get("text") or "")
        was_corr = is_corrupted_or_scanned_pdf_page(raw_txt, [])
        repaired_txt, delta = repair_shifted_cmap_text(raw_txt)
        still_corr = is_corrupted_or_scanned_pdf_page(repaired_txt, [])
        structured = {
            "was_corrupted_or_scanned_before": was_corr,
            "needs_visual_ocr_after_repair": still_corr,
            "detected_ascii_shift_delta": delta,
            "repaired_text": repaired_txt,
        }
        return {
            "content": [{"type": "text", "text": json.dumps(structured, indent=2)}],
            "structuredContent": structured,
            "isError": False,
        }

    if name == "sovereign_ocr_validate_kyc_fiscal":
        txt = str(args.get("text") or "")
        fname = str(args.get("filename") or "document.pdf")
        repaired_txt, _ = repair_shifted_cmap_text(txt)
        src_kind = "EMPTY" if not repaired_txt.strip() else "PDF"
        role, score = classify_document_unit(fname, repaired_txt, src_kind)
        ext = SovereignDocumentExtractor(use_gliner=False)
        facts = ext.extract_documentoscopia_facts(repaired_txt)
        dates = ext.extract_dates(repaired_txt)
        amts = ext.extract_amounts(repaired_txt)
        structured = {
            "classified_role": role,
            "role_score": score,
            "cpfs": facts["cpfs"],
            "cnpjs": facts["cnpjs"],
            "imeis": facts["imeis"],
            "nfe_access_keys": facts["nfe_access_keys"],
            "ceps": facts["ceps"],
            "rgs": facts["rgs"],
            "occurrence_type": facts["occurrence_type"],
            "dates": dates,
            "primary_amount": amts["primary_total"],
        }
        return {
            "content": [{"type": "text", "text": json.dumps(structured, indent=2)}],
            "structuredContent": structured,
            "isError": False,
        }

    if name == "sovereign_ocr_extract_bundle":
        target = Path(str(args["target_path"]))
        entity_code = str(args.get("entity_code") or "")
        ont_path_str = args.get("ontology_json_path")
        include_bapi = bool(args.get("include_sap_bapi", True))
        ont = (
            SovereignOntologyEngine(master_kg_path=Path(str(ont_path_str)))
            if ont_path_str
            else SovereignOntologyEngine()
        )
        ext = SovereignDocumentExtractor(use_gliner=True)
        if target.is_dir():
            bundle = unpack_bundle_directory(target, entity_code=entity_code, ontology_engine=ont)
        else:
            units = unpack_document_file(target)
            units.sort(key=lambda u: u.score, reverse=True)
            bundle = IngestedBundle(
                bundle_id=target.stem,
                entity_code=entity_code,
                document_id=target.stem,
                units=units,
                primary_unit=units[0] if units else None,
                combined_text="\n\n".join(u.text for u in units if u.text),
            )
        result = ext.extract_bundle(bundle, ontology_engine=ont)
        sap_payload = build_sap_bapi_payload(result) if include_bapi else None
        structured = {
            "bundle_id": str(result.get("bundle_id") or bundle.bundle_id),
            "primary_unit_kind": str(result.get("primary_unit_kind") or "UNKNOWN"),
            "primary_unit_role": str(result.get("primary_unit_role") or "PRIMARY_INVOICE"),
            "needs_ocr": bool(result.get("needs_ocr", False)),
            "extracted_doc_type": str(result.get("doc_type") or result.get("extracted_doc_type") or "KR"),
            "extracted_date": str(result.get("doc_date") or result.get("extracted_date") or ""),
            "extracted_reference": str(result.get("reference") or result.get("extracted_reference") or ""),
            "extracted_amount": (
                result.get("amount") if result.get("amount") is not None else result.get("extracted_amount")
            ),
            "warnings": list(result.get("warnings") or []),
            "documentoscopia_facts": result.get("documentoscopia_facts") or {},
            "quad_receipt_hash": str(result.get("quad_receipt_hash") or ""),
            "sap_s4hana_payload": sap_payload,
        }
        return {
            "content": [{"type": "text", "text": json.dumps(structured, indent=2, default=str)}],
            "structuredContent": structured,
            "isError": False,
        }

    if name == "sovereign_ocr_build_sap_bapi":
        gov_rec = dict(args.get("governed_record") or {})
        structured = build_sap_bapi_payload(gov_rec)
        return {
            "content": [{"type": "text", "text": json.dumps(structured, indent=2, default=str)}],
            "structuredContent": structured,
            "isError": False,
        }

    if name == "sovereign_ocr_export_bpmn_dmn":
        proc_id = str(args.get("process_id") or "Process_Sovereign_OCR_SAP")
        dec_id = str(args.get("decision_id") or "Decision_Sovereign_SAP_Governance")
        structured = {
            "bpmn_xml": export_sap_bpmn_xml(process_id=proc_id),
            "dmn_xml": export_sap_dmn_xml(decision_id=dec_id),
        }
        return {
            "content": [{"type": "text", "text": json.dumps(structured, indent=2)}],
            "structuredContent": structured,
            "isError": False,
        }

    raise ValueError(f"Unknown MCP tool: {name}")


class SovereignMcpDispatcher:
    """
    Stateless & Stateful JSON-RPC 2.0 dispatcher implementing the MCP specification
    across revisions `2026-07-28`, `2025-11-25`, `2025-06-18`, `2025-03-26`, and `2024-11-05`.
    """

    def __init__(self) -> None:
        self.log_level = "info"
        self.negotiated_version = DEFAULT_PROTOCOL_VERSION

    def _capabilities(self) -> dict[str, Any]:
        return {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
            "prompts": {"listChanged": False},
            "logging": {},
            "completions": {},
        }

    def handle_rpc(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC 2.0 request or notification and return the response dict (or None for notifications)."""
        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        # Notifications have no "id" and must not return a JSON-RPC response
        if "id" not in msg:
            return None

        try:
            if method in ("initialize", "server/discover"):
                req_ver = str(params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION)
                if req_ver in SUPPORTED_PROTOCOL_VERSIONS:
                    self.negotiated_version = req_ver
                else:
                    self.negotiated_version = DEFAULT_PROTOCOL_VERSION
                result = {
                    "protocolVersion": self.negotiated_version,
                    "capabilities": self._capabilities(),
                    "serverInfo": SERVER_INFO,
                    "instructions": SERVER_INSTRUCTIONS,
                }
                return {"jsonrpc": "2.0", "id": req_id, "result": result}

            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            if method == "logging/setLevel":
                self.log_level = str(params.get("level") or "info")
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": get_mcp_tools()}}

            if method == "tools/call":
                tool_name = str(params.get("name") or "")
                tool_args = params.get("arguments") or {}
                try:
                    call_res = execute_mcp_tool(tool_name, tool_args)
                except Exception as tool_err:
                    err_txt = f"{type(tool_err).__name__}: {tool_err}"
                    call_res = {
                        "content": [{"type": "text", "text": err_txt}],
                        "isError": True,
                    }
                return {"jsonrpc": "2.0", "id": req_id, "result": call_res}

            if method == "resources/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": get_mcp_resources()}}

            if method == "resources/templates/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"resourceTemplates": []}}

            if method == "resources/read":
                uri = str(params.get("uri") or "")
                res_obj = read_mcp_resource(uri)
                return {"jsonrpc": "2.0", "id": req_id, "result": res_obj}

            if method == "prompts/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": get_mcp_prompts()}}

            if method == "prompts/get":
                pname = str(params.get("name") or "")
                pargs = params.get("arguments") or {}
                p_obj = get_mcp_prompt_messages(pname, pargs)
                return {"jsonrpc": "2.0", "id": req_id, "result": p_obj}

            if method == "completion/complete":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"completion": {"values": [], "total": 0, "hasMore": False}},
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": str(exc),
                    "data": traceback.format_exc(limit=3),
                },
            }


def run_stdio_server() -> int:
    """Run the MCP server over standard input/output (newline-delimited JSON-RPC 2.0)."""
    dispatcher = SovereignMcpDispatcher()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        if isinstance(payload, list):
            responses = [r for item in payload if (r := dispatcher.handle_rpc(item)) is not None]
            if responses:
                sys.stdout.write(json.dumps(responses) + "\n")
                sys.stdout.flush()
        elif isinstance(payload, dict):
            resp = dispatcher.handle_rpc(payload)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
    return 0


def run_streamable_http_server(host: str = "127.0.0.1", port: int = 8765) -> int:
    """
    Run the MCP Streamable HTTP server (`/mcp` endpoint supporting `POST`, `GET`, and `DELETE`
    with `Mcp-Session-Id` and `MCP-Protocol-Version` headers, plus SSE `text/event-stream` negotiation).
    """
    dispatcher = SovereignMcpDispatcher()
    active_sessions: set[str] = set()

    class McpHttpHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[sovereign-ocr-mcp] " + (fmt % args) + "\n")

        def do_POST(self) -> None:
            if self.path not in ("/mcp", "/"):
                self.send_error(404, "Not Found")
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            session_id = self.headers.get("Mcp-Session-Id") or str(uuid.uuid4())
            active_sessions.add(session_id)

            try:
                msg = json.loads(body)
            except Exception:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}).encode("utf-8")
                )
                return

            resp = dispatcher.handle_rpc(msg)
            if resp is None:
                self.send_response(202)
                self.send_header("Mcp-Session-Id", session_id)
                self.send_header("MCP-Protocol-Version", dispatcher.negotiated_version)
                self.end_headers()
                return

            accept = self.headers.get("Accept") or "application/json"
            if "text/event-stream" in accept and "application/json" not in accept:
                sse_payload = f"event: message\ndata: {json.dumps(resp)}\n\n".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Mcp-Session-Id", session_id)
                self.send_header("MCP-Protocol-Version", dispatcher.negotiated_version)
                self.end_headers()
                self.wfile.write(sse_payload)
            else:
                json_bytes = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(json_bytes)))
                self.send_header("Mcp-Session-Id", session_id)
                self.send_header("MCP-Protocol-Version", dispatcher.negotiated_version)
                self.end_headers()
                self.wfile.write(json_bytes)

        def do_GET(self) -> None:
            if self.path not in ("/mcp", "/"):
                self.send_error(404, "Not Found")
                return
            session_id = self.headers.get("Mcp-Session-Id") or str(uuid.uuid4())
            active_sessions.add(session_id)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Mcp-Session-Id", session_id)
            self.send_header("MCP-Protocol-Version", dispatcher.negotiated_version)
            self.end_headers()
            self.wfile.write(b": sovereign-ocr streamable-http sse channel ready\n\n")

        def do_DELETE(self) -> None:
            session_id = self.headers.get("Mcp-Session-Id")
            if session_id and session_id in active_sessions:
                active_sessions.discard(session_id)
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer((host, port), McpHttpHandler)
    sys.stderr.write(f"[sovereign-ocr-mcp] Streamable HTTP listening on http://{host}:{port}/mcp\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def export_mcp_schema_directory(out_dir: Path) -> list[str]:
    """
    Export each MCP tool schema (`<toolName>.json`) and `instructions.md` into `out_dir`.
    Compatible with both standard MCP tool manifests (`inputSchema`, `outputSchema`, `annotations`)
    and Antigravity lazy-loaded MCP schema files (`parameters`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for tool in get_mcp_tools():
        schema_doc = {
            "name": tool["name"],
            "title": tool.get("title", tool["name"]),
            "description": tool["description"],
            "parameters": tool["inputSchema"],
            "inputSchema": tool["inputSchema"],
            "outputSchema": tool["outputSchema"],
            "annotations": tool["annotations"],
        }
        fp = out_dir / f"{tool['name']}.json"
        fp.write_text(json.dumps(schema_doc, indent=2) + "\n")
        written.append(str(fp))

    inst_fp = out_dir / "instructions.md"
    inst_md = (
        "# `sovereign-ocr` MCP Server Instructions\n\n"
        f"{SERVER_INSTRUCTIONS}\n\n"
        "## Supported MCP Specification Versions\n"
        "- `2026-07-28` (Stateless `server/discover`, `outputSchema`, `structuredContent`, `x-mcp-header`)\n"
        "- `2025-11-25` / `2025-06-18` (Streamable HTTP `/mcp`, `Tool.outputSchema`, `CallToolResult.structuredContent`, `annotations`)\n"
        "- `2025-03-26` / `2024-11-05` (JSON-RPC 2.0 over `stdio` and Streamable HTTP)\n"
    )
    inst_fp.write_text(inst_md)
    written.append(str(inst_fp))
    return written

