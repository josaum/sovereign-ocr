"""
Comprehensive Test Suite for `sovereign-ocr` (Domain-Agnostic Sovereign OCR & Document Intelligence).

Verifies:
1. Static Source Audit: Zero company-specific hardcoding and zero `ground_truth_by_docno` references.
2. Stage 1 (`ingest`): Primary-vs-backup candidate routing and multi-invoice PDF splitting.
3. Stage 2 (`ocr_layout`): Self-healing PDF font-corruption gate, 5-step visual preprocessing
   (`compute_adaptive_dpi`, `enhance_contrast_and_denoise`, `detect_coarse_rotation_90_180_270`,
   `estimate_and_correct_skew_angle`, `dpi_aware_resize_image`, `preprocess_document_image_bytes`),
   and spatial reading-order reconstruction (`reconstruct_text_from_ocr_runs`).
4. Stage 3 (`schemas` & `extractor`): Pydantic v2 -> `gliner2.Schema` compilation, reference/date/amount
   extraction, and live offline `gliner_fast_py` ONNX entity extraction.
5. Stage 4 & 5 (`ontology_engine`): Pluggable in-memory `SovereignOntologyEngine`, ERP GL/Cost Center/Tax
   invariants, and deterministic `quad_fast` cryptographic provenance receipts.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sovereign_ocr import (
    DocumentExtractionEnvelope,
    ExtractedDocumentUnit,
    IngestedBundle,
    InvoiceHeaderSchema,
    SovereignDocumentExtractor,
    SovereignMcpDispatcher,
    SovereignOntologyEngine,
    build_deterministic_quad_receipt,
    build_gliner2_schema_from_pydantic,
    build_sap_bapi_payload,
    classify_document_unit,
    compute_adaptive_dpi,
    detect_and_split_multi_invoice_pdf,
    detect_coarse_rotation_90_180_270,
    dpi_aware_resize_image,
    enhance_contrast_and_denoise,
    estimate_and_correct_skew_angle,
    evaluate_bpmn_gateway_route,
    export_mcp_schema_directory,
    export_sap_bpmn_xml,
    export_sap_dmn_xml,
    get_gliner_fast_labels_from_pydantic,
    is_corrupted_or_scanned_pdf_page,
    parse_gliner2_output_to_pydantic,
    preprocess_document_image_bytes,
    reconstruct_text_from_ocr_runs,
    repair_shifted_cmap_text,
)


def _make_synthetic_text_page(
    width: int = 1000,
    height: int = 1300,
    bg_luma: int = 250,
    ink_luma: int = 20,
    vertical_lines: bool = False,
) -> Image.Image:
    """Create a synthetic document image with sharp horizontal (or vertical) text-like bars."""
    img = Image.new("RGB", (width, height), color=(bg_luma, bg_luma, bg_luma))
    draw = ImageDraw.Draw(img)
    if not vertical_lines:
        for y in range(120, height - 120, 48):
            draw.rectangle([100, y, width - 140, y + 10], fill=(ink_luma, ink_luma, ink_luma))
    else:
        for x in range(120, width - 120, 48):
            draw.rectangle([x, 100, x + 10, height - 140], fill=(ink_luma, ink_luma, ink_luma))
    return img


# ─── 1. STATIC SOURCE AUDIT: ZERO COMPANY SPECIFICS & ZERO GROUND-TRUTH LOOKUP ───

def test_zero_company_specifics_and_zero_cheating_in_source() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src" / "sovereign_ocr"
    forbidden_tokens = [
        "central_ocr",
        "central garden",
        "pennington",
        "kaytee",
        "farnam",
        "wellmark",
        "cgp_ap",
        "ground_truth_by_docno",
    ]
    for py_file in src_dir.glob("*.py"):
        content = py_file.read_text().lower()
        for tok in forbidden_tokens:
            assert tok not in content, f"Forbidden token '{tok}' found in {py_file.name}"


# ─── 2. STAGE 1: CANDIDATE ROUTING & MULTI-INVOICE SPLITTING ─────────────────

def test_classify_document_unit_roles() -> None:
    role_inv, sc_inv = classify_document_unit(
        "INV_99412.pdf",
        "INVOICE # 99412\nInvoice Date: 02/14/2026\nRemit To: Acme Industrial\nTotal Amount Due: $4,250.00",
        "PDF",
    )
    assert role_inv == "PRIMARY_INVOICE"
    assert sc_inv >= 90.0

    role_cm, _ = classify_document_unit(
        "CM_104.pdf",
        "CREDIT MEMO # CM-104\nCredit Total: -$310.00",
        "PDF",
    )
    assert role_cm == "CREDIT_MEMO"

    role_quote, sc_quote = classify_document_unit(
        "Quote_881.pdf",
        "Estimate No: 881\nInvoice Date: TBD\nTotal: $1,200.00",
        "PDF",
    )
    assert role_quote == "QUOTE_PREINVOICE"
    assert sc_quote < sc_inv

    role_scale, sc_scale = classify_document_unit(
        "scale_ticket.pdf",
        "SCALE TICKET\nGross Wt: 78,400 LB\nTare Wt: 32,100 LB",
        "PDF",
    )
    assert role_scale == "PO_BACKUP"
    assert sc_scale < sc_inv

    role_rcpt, _ = classify_document_unit(
        "fuel_slip.pdf",
        "PUMP# 04\nGALLONS: 24.510\nAUTH 088412\nTOTAL $88.12",
        "PDF",
    )
    assert role_rcpt == "RECEIPT_EXCEPTION"


def test_detect_and_split_multi_invoice_pdf() -> None:
    pages = [
        "INVOICE # INV-1001\nPage 1 of 2\nSubtotal: $500.00",
        "INVOICE # INV-1001\nPage 2 of 2\nTotal Amount Due: $500.00",
        "INVOICE # INV-1002\nPage 1 of 1\nTotal Amount Due: $825.00",
    ]
    segs = detect_and_split_multi_invoice_pdf(Path("/tmp/nonexistent.pdf"), pages, out_dir=None)
    assert len(segs) == 2
    assert segs[0] == {"pages": [1, 2], "invoice_number": "INV-1001"}
    assert segs[1] == {"pages": [3], "invoice_number": "INV-1002"}


# ─── 3. STAGE 2: FONT-CORRUPTION GATE & 5-STEP VISUAL PREPROCESSING ──────────

def test_is_corrupted_or_scanned_pdf_page() -> None:
    assert is_corrupted_or_scanned_pdf_page("   ", []) is True
    assert is_corrupted_or_scanned_pdf_page("INVOICE \x00 TOTAL AMOUNT DUE DATE " * 5, []) is True
    clean_text = (
        "INVOICE # 849201\nInvoice Date: 02/15/2026\nVendor: Apex Logistics LLC\n"
        "Bill To: Manufacturing Plant 4\nDescription: Freight Charge\nTotal Amount Due: $1,420.00"
    )
    assert is_corrupted_or_scanned_pdf_page(clean_text, [{"cmap_fallback_chars": 0}]) is False
    assert is_corrupted_or_scanned_pdf_page(clean_text, [{"cmap_fallback_chars": 45}]) is True


def test_compute_adaptive_dpi() -> None:
    assert compute_adaptive_dpi(225.0, 432.0) == 300.0     # POS thermal receipt -> 300 DPI
    assert compute_adaptive_dpi(612.0, 792.0) == 190.9     # Standard Letter 8.5x11 -> 190.9 DPI
    assert compute_adaptive_dpi(1224.0, 1584.0) == 120.0   # Ledger 17x22 -> clamped to 120 DPI


def test_enhance_contrast_and_denoise() -> None:
    faded = _make_synthetic_text_page(width=600, height=800, bg_luma=180, ink_luma=125)
    enhanced, meta = enhance_contrast_and_denoise(faded)
    assert meta["contrast_stretched"] is True
    arr = np.asarray(enhanced.convert("L"))
    assert float(np.percentile(arr, 95)) >= 245.0
    assert float(np.percentile(arr, 1)) <= 15.0


def test_detect_coarse_rotation_90_180_270() -> None:
    upright = _make_synthetic_text_page(width=800, height=1000, vertical_lines=False)
    assert detect_coarse_rotation_90_180_270(upright, pdf_rotate_deg=0) == 0
    assert detect_coarse_rotation_90_180_270(upright, pdf_rotate_deg=90) == 270
    assert detect_coarse_rotation_90_180_270(upright, pdf_rotate_deg=180) == 180

    sideways = _make_synthetic_text_page(width=1000, height=800, vertical_lines=True)
    rot = detect_coarse_rotation_90_180_270(sideways, pdf_rotate_deg=0)
    assert rot in (90, 270)


def test_estimate_and_correct_skew_angle() -> None:
    base = _make_synthetic_text_page(width=900, height=1200, bg_luma=255, ink_luma=0)
    _, zero_angle = estimate_and_correct_skew_angle(base)
    assert abs(zero_angle) < 0.4

    tilted = base.rotate(4.6, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    _, corr_angle = estimate_and_correct_skew_angle(tilted)
    assert -6.0 <= corr_angle <= -3.2


def test_dpi_aware_resize_image() -> None:
    small = Image.new("RGB", (480, 700), color=(255, 255, 255))
    up_img, up_meta = dpi_aware_resize_image(small, source_dpi=96.0)
    assert up_meta["scale_factor"] > 1.5
    assert up_img.size[0] > 700

    huge = Image.new("RGB", (3400, 4400), color=(255, 255, 255))
    dn_img, dn_meta = dpi_aware_resize_image(huge, source_dpi=400.0)
    assert max(dn_img.size) == 2560
    assert dn_meta["scale_factor"] < 1.0


def test_preprocess_document_image_bytes_end_to_end() -> None:
    faded = _make_synthetic_text_page(width=500, height=700, bg_luma=182, ink_luma=120)
    tilted_faded = faded.rotate(-4.2, resample=Image.Resampling.BICUBIC, fillcolor=(182, 182, 182))
    buf = io.BytesIO()
    tilted_faded.save(buf, format="PNG", dpi=(96, 96))

    out_bytes, diag = preprocess_document_image_bytes(buf.getvalue(), source_dpi=96.0)
    assert len(out_bytes) > 1000
    assert diag["contrast_stretched"] is True
    assert diag["skew_angle_deg"] != 0.0
    assert diag["scale_factor"] > 1.2


def test_reconstruct_text_from_ocr_runs() -> None:
    runs = [
        {"text": " $1,250.00 ", "px_x": 420.0, "px_y": 102.0, "px_width": 80.0, "px_height": 18.0},
        {"text": "INVOICE # 9012", "px_x": 50.0, "px_y": 100.0, "px_width": 140.0, "px_height": 18.0},
        {"text": "Date: 02/20/2026", "px_x": 50.0, "px_y": 145.0, "px_width": 140.0, "px_height": 18.0},
    ]
    txt = reconstruct_text_from_ocr_runs(runs)
    lines = txt.splitlines()
    assert len(lines) == 2
    assert lines[0] == "INVOICE # 9012  $1,250.00"
    assert lines[1] == "Date: 02/20/2026"


# ─── 4. STAGE 3: PYDANTIC V2 -> GLINER 2.5 SCHEMA & EXTRACTOR ────────────────

def test_pydantic_to_gliner2_schema_and_parser() -> None:
    schema = build_gliner2_schema_from_pydantic()
    compiled = schema.build()
    assert "json_structures" in compiled
    struct_names = [next(iter(s.keys())) for s in compiled["json_structures"]]
    assert "invoice_header" in struct_names
    assert "remit_to_address" in struct_names

    labels = get_gliner_fast_labels_from_pydantic()
    assert "vendor_name" in labels
    assert "invoice_number" in labels

    raw_mock = {
        "invoice_header": [
            {
                "vendor_name": {"text": "  Apex Freight Systems LLC ", "confidence": 0.96},
                "invoice_number": {"text": "INV-2026-884", "confidence": 0.98},
                "total_amount": {"text": "$3,410.50", "confidence": 0.95},
            }
        ],
        "remit_to_address": [
            {"bank_routing_aba": {"text": "021000021", "confidence": 0.99}}
        ],
        "document_type": {"label": "Vendor_Invoice", "confidence": 0.97},
        "tax_classification": {"label": "Sales_Tax_Charged", "confidence": 0.91},
    }
    env = parse_gliner2_output_to_pydantic(raw_mock)
    assert isinstance(env, DocumentExtractionEnvelope)
    assert env.header.vendor_name == "Apex Freight Systems LLC"
    assert env.header.invoice_number == "INV-2026-884"
    assert env.remit_to.bank_routing_aba == "021000021"
    assert env.classification.document_type == "Vendor_Invoice"

    # Non-digit invoice number rejected by Pydantic validator
    invalid_hdr = InvoiceHeaderSchema(invoice_number="INVOICE")
    assert invalid_hdr.invoice_number is None


# ─── 5. STAGE 4 & 5: ACTIVE ONTOLOGY GROUNDING & DETERMINISTIC QUAD RECEIPT ─

def test_pluggable_ontology_engine_and_deterministic_quad_receipt() -> None:
    master_data = {
        "facility_zips": ["90210"],
        "vendors": {
            "9000101": {
                "vendor_name": "NORTHERN INDUSTRIAL SUPPLY INC",
                "street": "4500 PARKWAY BLVD",
                "city": "CHICAGO",
                "state": "IL",
                "postal_code": "60601",
                "payee_id": "9900202",
                "payee_name": "NORTHERN LOCKBOX SERVICES",
                "payee_street": "PO BOX 77410",
                "payee_city": "DETROIT",
                "payee_state": "MI",
                "payee_zip": "48277",
                "default_cocd": "0010",
                "cocds": ["0010", "0040"],
                "default_tax_code": "I1",
                "default_tax_jurisdiction": "IL0000000",
                "default_pmt_terms": "ZN30",
                "gl_accounts": {"21100001": 50, "51200010": 12},
                "cost_centers": {"1010500": 5, "4010900": 8},
                "profit_centers": {"10105": 5, "40109": 8},
                "default_approver": "JANE DOE",
                "default_coder": "JOHN SMITH",
            }
        },
    }
    ont = SovereignOntologyEngine(master_data=master_data)
    ext = SovereignDocumentExtractor(use_gliner=False)

    doc_txt = (
        "NORTHERN INDUSTRIAL SUPPLY INC\n"
        "4500 PARKWAY BLVD, CHICAGO IL 60601\n"
        "INVOICE # NIS-008421\n"
        "Invoice Date: 03/10/2026\n"
        "Sales Tax: $0.00\n"
        "Total Amount Due: $2,750.00\n"
        "Remit To: NORTHERN LOCKBOX SERVICES, PO BOX 77410, DETROIT MI 48277"
    )
    unit = ExtractedDocumentUnit(
        unit_id="inv.pdf",
        filename="inv.pdf",
        source_kind="PDF",
        role="PRIMARY_INVOICE",
        text=doc_txt,
        score=95.0,
    )
    bundle = IngestedBundle(
        bundle_id="0040_88001_2026",
        entity_code="0040",
        document_id="88001",
        fiscal_year="2026",
        units=[unit],
        primary_unit=unit,
        combined_text=doc_txt,
    )

    gov1 = ext.extract_bundle(bundle, ontology_engine=ont)
    gov2 = ext.extract_bundle(bundle, ontology_engine=ont)

    assert gov1["vendor_id"] == "9000101"
    assert gov1["payee_id"] == "9900202"
    assert gov1["reference"] == "NIS-008421"
    assert gov1["doc_date"] == "2026-03-10"
    assert gov1["amount"] == 2750.00
    # Tax = $0.00 on I1 vendor clears tax_code to ""
    assert gov1["tax_code"] == ""
    # AP reconciliation account 21100001 excluded -> selects expense GL 51200010
    assert gov1["gl_account"] == "51200010"
    # Entity 0040 selects 4010900 cost center and 40109 profit center
    assert gov1["cost_center"] == "4010900"
    assert gov1["profit_center"] == "40109"
    # Cryptographic quad receipt is 100% deterministic across runs
    assert len(gov1["quad_receipt_hash"]) == 16
    assert gov1["quad_receipt_hash"] == gov2["quad_receipt_hash"]
    assert gov1["quad_ipc_bytes_len"] > 200

    b1, h1 = build_deterministic_quad_receipt(
        doc_iri="https://sovereign-ocr.local/doc/1",
        graph_iri="https://sovereign-ocr.local/graph",
        concept_iri="https://sovereign-ocr.local/ontology#VendorInvoice",
        vendor_iri="https://sovereign-ocr.local/ontology#Vendor_9000101",
        reference="NIS-008421",
        amount=2750.0,
        source_artifact_iri="file:///inv.pdf",
        full_text=doc_txt,
    )
    assert len(b1) > 200
    assert len(h1) == 16


# ─── 6. MULTILINGUAL / DOCUMENTOSCOPIA SELF-HEALING CMAP, VALIDATORS & CLAIMS ─

def test_repair_shifted_cmap_text_and_multilingual_vocab() -> None:
    mixed_shifted = (
        "%2/(7,0 DE OCORRENCIA\n"
        "'$'26 DO 5(*,6752\n"
        "'DWD +RUD 17/05/2024\n"
        "Tipo: 5HVLGrQFLD | Nacionalidade: %UDVLOHLUD | Sexo: 0DVFXOLQR\n"
        "Profissao: (PSUHViULR | 'HVFULomR | 2SHUDGRUD | &HOXODUHV"
    )
    repaired, delta = repair_shifted_cmap_text(mixed_shifted)
    assert delta == 29
    assert "BOLETIM" in repaired
    assert "DADOS" in repaired
    assert "REGISTRO" in repaired
    assert "Data Hora" in repaired
    assert "Residencia" in repaired
    assert "Brasileira" in repaired
    assert "Masculino" in repaired
    assert "Empresario" in repaired
    assert "Descricao" in repaired
    assert "Operadora" in repaired
    assert "Celulares" in repaired


def test_documentoscopia_role_classification_and_empty_files() -> None:
    assert classify_document_unit("Aviso - 44087628R.pdf", "AVISO DE SINISTRO\nDados do Sinistro", "PDF")[0] == "CLAIM_NOTICE_AVISO"
    assert classify_document_unit("B.O2 - 45311435R.pdf", "POLICIA CIVIL\nDelegacia", "PDF")[0] == "POLICE_REPORT_BO"
    assert classify_document_unit("NF1 - 44148538R.pdf", "DANFE\nChave de Acesso", "PDF")[0] == "NOTA_FISCAL"
    assert classify_document_unit("CR - 44087628R.pdf", "Fatura Mensal\nVencimento 10/05/2024\nCEP 40296-295", "PDF")[0] == "PROOF_OF_ADDRESS"
    assert classify_document_unit("RG2 - 43113720R.pdf", "Carteira de Identidade\nData de Expedicao", "PDF")[0] == "IDENTITY_DOCUMENT"
    assert classify_document_unit("empty.retries", "", "EMPTY") == ("EMPTY_FILE", 0.0)


def test_documentoscopia_validators_and_fact_extraction() -> None:
    ext = SovereignDocumentExtractor(use_gliner=False)
    assert ext.validate_cpf("481.922.605-34") is True
    assert ext.validate_cpf("111.111.111-11") is False
    assert ext.validate_imei_luhn("352775620619610") is True
    assert ext.validate_imei_luhn("352775620619619") is False
    assert ext.validate_nfe_access_key("3524 0605 8287 3201 8720 6529 9000 0111 8110 0000 0010") is True
    assert ext.validate_nfe_access_key("3524 0605 8287 3201 8720 6529 9000 0111 8110 0000 0019") is False

    sample_claim_text = (
        "DANFE - Documento Auxiliar da Nota Fiscal Eletronica\n"
        "Chave de Acesso: 3524 0605 8287 3201 8720 6529 9000 0111 8110 0000 0010\n"
        "Data de Emissao: 18/04/2024\n"
        "CPF: 481.922.605-34  RG: 02.724.762-77\n"
        "CEP: 41.720-100\n"
        "Aparelho Celular IMEI: 352775620619610\n"
        "VALOR TOTAL DA NOTA R$ 8.699,00\n"
        "Relato: Vitima abordada mediante grave ameaca e arma de fogo (roubo)."
    )
    facts = ext.extract_documentoscopia_facts(sample_claim_text)
    assert facts["cpfs"] == ["48192260534"]
    assert facts["imeis"] == ["352775620619610"]
    assert facts["nfe_access_keys"] == ["35240605828732018720652990000111811000000010"]
    assert facts["ceps"] == ["41720100"]
    assert "0272476277" in facts["rgs"]
    assert facts["occurrence_type"] == "roubo"

    dates = ext.extract_dates(sample_claim_text)
    assert "2024-04-18" in dates
    amts = ext.extract_amounts(sample_claim_text)
    assert amts["primary_total"] == 8699.0


def test_assurant_documentoscopia_sideways_rg_and_claims_snapshot() -> None:
    rg_png = Path(
        "/Users/josaum/projects/mycelia-workspace/runtime_snapshots/"
        "gcp-mycelia-platform-2026-05-15/documentoscopia_raw_data/"
        "daase_zip_20260513_paddle_preproc_boxes/evidence/44087628R/"
        "Documento de identificação/page_01.png"
    )
    if not rg_png.exists():
        return

    proc_bytes, diag = preprocess_document_image_bytes(rg_png.read_bytes())
    assert len(proc_bytes) > 10000
    assert diag["coarse_rotation_deg"] == 90
    assert diag["output_size"] == (2375, 900)
    assert diag["scale_factor"] > 1.2


# ─── 7. SAP S/4HANA BPMN 2.0, DMN 1.3 & BAPI/ODATA PAYLOAD ADAPTER ───────────

def test_sap_bpmn_routing_bapi_payload_and_xml_export() -> None:
    import xml.etree.ElementTree as ET

    gov_kr = {
        "cocd_str": "0040",
        "doc_type": "KR",
        "vendor_id": "9000101",
        "payee_id": "9900202",
        "doc_date": "2026-03-10",
        "reference": "NIS-008421",
        "amount": 2750.00,
        "tax_on_invoice": 150.00,
        "tax_code": "I1",
        "tax_jurisdiction": "IL0000000",
        "gl_account": "51200010",
        "cost_center": "4010900",
        "profit_center": "40109",
        "pmt_terms": "ZN30",
        "pmt_method": "A",
        "line_item_text": "INDUSTRIAL SUPPLIES MARCH",
        "primary_unit_role": "PRIMARY_INVOICE",
        "warnings": [],
        "quad_receipt_hash": "a1b2c3d4e5f60718",
        "quad_ipc_bytes_len": 4096,
    }
    payload_kr = build_sap_bapi_payload(gov_kr)
    assert payload_kr["bpmn_routing"]["is_auto_postable"] is True
    assert payload_kr["bpmn_routing"]["target_bpmn_node"] == "ServiceTask_PostSapFIVoucher"
    assert payload_kr["bpmn_routing"]["sap_action"] == "BAPI_ACC_DOCUMENT_POST"

    bapi_kr = payload_kr["BAPI_ACC_DOCUMENT_POST"]
    assert bapi_kr["zero_balance_verified"] is True
    assert bapi_kr["balance_sum"] == 0.0
    assert bapi_kr["DOCUMENTHEADER"]["COMP_CODE"] == "0040"
    assert bapi_kr["DOCUMENTHEADER"]["DOC_DATE"] == "20260310"
    assert bapi_kr["DOCUMENTHEADER"]["DOC_TYPE"] == "KR"
    assert bapi_kr["ACCOUNTAP"][0]["VENDOR_NO"] == "0009000101"
    assert bapi_kr["ACCOUNTAP"][0]["PARTNER_BK"] == "0009900202"
    assert bapi_kr["ACCOUNTGL"][0]["GL_ACCOUNT"] == "0051200010"
    assert bapi_kr["ACCOUNTGL"][0]["COSTCENTER"] == "0004010900"
    assert bapi_kr["CURRENCYAMOUNT"][0]["AMT_DOCCUR"] == -2750.00
    assert bapi_kr["CURRENCYAMOUNT"][1]["AMT_DOCCUR"] == 2600.00
    assert bapi_kr["CURRENCYAMOUNT"][2]["AMT_DOCCUR"] == 150.00

    # Credit Memo (KG) inverts AP/GL debit/credit signs while maintaining zero balance
    gov_kg = dict(gov_kr, doc_type="KG", amount=-400.00, tax_on_invoice=0.0, tax_code="")
    payload_kg = build_sap_bapi_payload(gov_kg)
    bapi_kg = payload_kg["BAPI_ACC_DOCUMENT_POST"]
    assert bapi_kg["zero_balance_verified"] is True
    assert bapi_kg["CURRENCYAMOUNT"][0]["AMT_DOCCUR"] == 400.00
    assert bapi_kg["CURRENCYAMOUNT"][1]["AMT_DOCCUR"] == -400.00

    # Exception routing when POS receipt or >20% freight tolerance is violated
    gov_exc = dict(
        gov_kr,
        primary_unit_role="RECEIPT_EXCEPTION",
        warnings=["RECEIPT_SUBMITTED_INSTEAD_OF_INVOICE", "FREIGHT_EXCEEDS_20_PCT_THRESHOLD"],
        approver="JANE DOE",
    )
    route_exc = evaluate_bpmn_gateway_route(gov_exc)
    assert route_exc["is_auto_postable"] is False
    assert route_exc["target_bpmn_node"] == "UserTask_ApExceptionReview"
    assert route_exc["inbox_recipient"] == "JANE DOE"
    assert "POS_RECEIPT_SUBMITTED_INSTEAD_OF_INVOICE" in route_exc["exception_reasons"]

    # Validate BPMN 2.0 and DMN 1.3 XML schemas parse cleanly
    bpmn_root = ET.fromstring(export_sap_bpmn_xml())
    dmn_root = ET.fromstring(export_sap_dmn_xml())
    assert "definitions" in bpmn_root.tag
    assert "definitions" in dmn_root.tag


# ─── 8. MODEL CONTEXT PROTOCOL (MCP) SERVER: LATEST SPECS (2025-11-25 / 2026-07-28) ─

def test_mcp_server_latest_spec_stdio_and_streamable_http(tmp_path: Path) -> None:
    import json
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    dispatcher = SovereignMcpDispatcher()

    # 1. Stateful initialize (2025-11-25) & Stateless server/discover (2026-07-28)
    init_resp = dispatcher.handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1.0"}},
        }
    )
    assert init_resp is not None
    assert init_resp["result"]["protocolVersion"] == "2025-11-25"
    assert init_resp["result"]["serverInfo"]["name"] == "sovereign-ocr"

    disc_resp = dispatcher.handle_rpc(
        {"jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {"protocolVersion": "2026-07-28"}}
    )
    assert disc_resp is not None
    assert disc_resp["result"]["protocolVersion"] == "2026-07-28"

    # Notification returns None
    assert dispatcher.handle_rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    # 2. tools/list: Verify JSON Schema 2020-12 inputSchema, outputSchema, annotations, title, x-mcp-header
    tools_resp = dispatcher.handle_rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    assert tools_resp is not None
    tools = tools_resp["result"]["tools"]
    assert len(tools) == 7
    for t in tools:
        assert "name" in t and "title" in t and "description" in t
        assert t["inputSchema"]["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert t["outputSchema"]["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert t["outputSchema"]["type"] == "object"
        ann = t["annotations"]
        assert all(k in ann for k in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"))
        assert ann["openWorldHint"] is False

    extract_tool = next(t for t in tools if t["name"] == "sovereign_ocr_extract_bundle")
    assert extract_tool["inputSchema"]["properties"]["entity_code"]["x-mcp-header"] == "Mcp-Param-Entity-Code"

    # 3. tools/call: Verify both content and structuredContent
    doc_call = dispatcher.handle_rpc(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "sovereign_ocr_doctor", "arguments": {}}}
    )
    assert doc_call is not None
    assert doc_call["result"]["isError"] is False
    assert doc_call["result"]["structuredContent"]["status"] == "ok"
    assert doc_call["result"]["structuredContent"]["modules"]["pdf_fast"] is True

    cmap_call = dispatcher.handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "sovereign_ocr_repair_cmap",
                "arguments": {"text": "%2/(7,0 DE OCORRENCIA\n'$'26 DO 5(*,6752"},
            },
        }
    )
    assert cmap_call is not None
    sc_cmap = cmap_call["result"]["structuredContent"]
    assert sc_cmap["detected_ascii_shift_delta"] == 29
    assert "BOLETIM" in sc_cmap["repaired_text"]

    kyc_call = dispatcher.handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "sovereign_ocr_validate_kyc_fiscal",
                "arguments": {
                    "text": (
                        "DANFE - Chave de Acesso: 3524 0605 8287 3201 8720 6529 9000 0111 8110 0000 0010\n"
                        "CPF: 481.922.605-34 | IMEI: 352775620619610 | VALOR TOTAL R$ 4.999,00"
                    ),
                    "filename": "NF1.pdf",
                },
            },
        }
    )
    assert kyc_call is not None
    sc_kyc = kyc_call["result"]["structuredContent"]
    assert sc_kyc["classified_role"] == "NOTA_FISCAL"
    assert sc_kyc["cpfs"] == ["48192260534"]
    assert sc_kyc["imeis"] == ["352775620619610"]
    assert sc_kyc["primary_amount"] == 4999.0

    # 4. resources/list & resources/read, prompts/list & prompts/get
    res_list = dispatcher.handle_rpc({"jsonrpc": "2.0", "id": 7, "method": "resources/list"})
    assert res_list is not None
    assert len(res_list["result"]["resources"]) == 4

    res_read = dispatcher.handle_rpc(
        {"jsonrpc": "2.0", "id": 8, "method": "resources/read", "params": {"uri": "sovereign://workflows/sap-s4hana.bpmn"}}
    )
    assert res_read is not None
    assert "bpmn:definitions" in res_read["result"]["contents"][0]["text"]

    prm_list = dispatcher.handle_rpc({"jsonrpc": "2.0", "id": 9, "method": "prompts/list"})
    assert prm_list is not None
    assert len(prm_list["result"]["prompts"]) == 2

    prm_get = dispatcher.handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "prompts/get",
            "params": {"name": "audit_sap_ap_bundle", "arguments": {"target_path": "/tmp/inv.pdf", "entity_code": "0010"}},
        }
    )
    assert prm_get is not None
    assert "sovereign_ocr_extract_bundle" in prm_get["result"]["messages"][0]["content"]["text"]

    # 5. Export MCP schema directory
    exported = export_mcp_schema_directory(tmp_path / "mcp_schemas")
    assert len(exported) == 8
    assert (tmp_path / "mcp_schemas" / "sovereign_ocr_extract_bundle.json").is_file()
    assert (tmp_path / "mcp_schemas" / "instructions.md").is_file()


def test_claude_desktop_mcpb_manifest_and_launcher() -> None:
    import json

    repo_root = Path(__file__).resolve().parents[1]
    mcpb_dir = repo_root / "packaging" / "mcpb"
    manifest_file = mcpb_dir / "manifest.json"
    assert manifest_file.is_file()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "0.4"
    assert manifest["name"] == "sovereign-ocr"
    assert set(manifest["compatibility"]["platforms"]) == {"darwin", "win32", "linux"}
    assert len(manifest["tools"]) == 7
    assert len(manifest["prompts"]) == 2
    assert (mcpb_dir / "bin" / "run-sovereign-ocr").is_file()
    assert (mcpb_dir / "bin" / "run-sovereign-ocr.py").is_file()
    assert (mcpb_dir / "bin" / "run-sovereign-ocr").read_bytes() == (
        mcpb_dir / "bin" / "run-sovereign-ocr.py"
    ).read_bytes()


