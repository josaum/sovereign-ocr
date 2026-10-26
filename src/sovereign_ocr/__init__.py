"""
`sovereign_ocr`: 100% Air-Gapped Sovereign OCR & Neuro-Symbolic Document Intelligence Engine.

Exposes the 5-stage Sovereign Document Intelligence pipeline:
1. `ingest`: Multi-format native unpacking (`email_fast`, `pdf_fast`, `docx_fast`, `xlsx_fast`),
   primary-vs-backup candidate routing, and zero-copy multi-document PDF splitting.
2. `ocr_layout`: Self-healing PDF font-corruption gate, 5-step geometry/DPI-aware visual preprocessing
   (adaptive DPI, contrast stretching, 90/180/270 orientation, projection-profile deskew, Lanczos resize),
   and spatial reading-order reconstruction (`layout_fast`).
3. `schemas`: Strongly-typed Pydantic v2 models and generic compiler to `gliner2.Schema` & `gliner_fast_py`.
4. `extractor`: Hybrid neuro-symbolic document field extractor (`SovereignDocumentExtractor`).
5. `ontology_engine`: Pluggable active runtime ontology grounding (`SovereignOntologyEngine`) and
   deterministic `quad_fast` cryptographic provenance receipts.
"""

from sovereign_ocr.extractor import SovereignDocumentExtractor
from sovereign_ocr.ingest import (
    ExtractedDocumentUnit,
    IngestedBundle,
    classify_document_unit,
    detect_and_split_multi_invoice_pdf,
    extract_docx_text,
    extract_xlsx_text,
    unpack_bundle_directory,
    unpack_document_file,
)
from sovereign_ocr.ocr_layout import (
    DEFAULT_DOCUMENT_VOCABULARY,
    compute_adaptive_dpi,
    detect_coarse_rotation_90_180_270,
    dpi_aware_resize_image,
    enhance_contrast_and_denoise,
    estimate_and_correct_skew_angle,
    is_corrupted_or_scanned_pdf_page,
    preprocess_document_image,
    preprocess_document_image_bytes,
    reconstruct_text_from_ocr_runs,
    render_pdf_page_png,
    repair_shifted_cmap_text,
    run_batch_ocr,
)
from sovereign_ocr.ontology_engine import (
    SovereignOntologyEngine,
    build_deterministic_quad_receipt,
    norm_str,
)
from sovereign_ocr.sap_bpmn import (
    build_sap_bapi_payload,
    evaluate_bpmn_gateway_route,
    export_sap_bpmn_xml,
    export_sap_dmn_xml,
)
from sovereign_ocr.schemas import (
    DocumentClassificationSchema,
    DocumentExtractionEnvelope,
    InvoiceHeaderSchema,
    RemitToPayeeSchema,
    build_gliner2_schema_from_pydantic,
    configure_offline_model_env,
    get_gliner_fast_labels_from_pydantic,
    parse_gliner2_output_to_pydantic,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_DOCUMENT_VOCABULARY",
    "DocumentClassificationSchema",
    "DocumentExtractionEnvelope",
    "ExtractedDocumentUnit",
    "IngestedBundle",
    "InvoiceHeaderSchema",
    "RemitToPayeeSchema",
    "SovereignDocumentExtractor",
    "SovereignOntologyEngine",
    "build_deterministic_quad_receipt",
    "build_gliner2_schema_from_pydantic",
    "build_sap_bapi_payload",
    "classify_document_unit",
    "compute_adaptive_dpi",
    "configure_offline_model_env",
    "detect_and_split_multi_invoice_pdf",
    "detect_coarse_rotation_90_180_270",
    "dpi_aware_resize_image",
    "enhance_contrast_and_denoise",
    "estimate_and_correct_skew_angle",
    "evaluate_bpmn_gateway_route",
    "export_sap_bpmn_xml",
    "export_sap_dmn_xml",
    "extract_docx_text",
    "extract_xlsx_text",
    "get_gliner_fast_labels_from_pydantic",
    "is_corrupted_or_scanned_pdf_page",
    "norm_str",
    "parse_gliner2_output_to_pydantic",
    "preprocess_document_image",
    "preprocess_document_image_bytes",
    "reconstruct_text_from_ocr_runs",
    "render_pdf_page_png",
    "repair_shifted_cmap_text",
    "run_batch_ocr",
    "unpack_bundle_directory",
    "unpack_document_file",
]
