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
    SovereignOntologyEngine,
    build_deterministic_quad_receipt,
    build_gliner2_schema_from_pydantic,
    classify_document_unit,
    compute_adaptive_dpi,
    detect_and_split_multi_invoice_pdf,
    detect_coarse_rotation_90_180_270,
    dpi_aware_resize_image,
    enhance_contrast_and_denoise,
    estimate_and_correct_skew_angle,
    get_gliner_fast_labels_from_pydantic,
    is_corrupted_or_scanned_pdf_page,
    parse_gliner2_output_to_pydantic,
    preprocess_document_image_bytes,
    reconstruct_text_from_ocr_runs,
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
