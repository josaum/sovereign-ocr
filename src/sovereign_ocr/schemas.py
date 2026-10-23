"""
Pydantic v2 Schemas & Local Air-Gapped GLiNER 2.5 Schema Compiler (`sovereign_ocr.schemas`).

Defines domain-agnostic, strongly-typed Pydantic v2 models that compile directly into:
1. `gliner2.Schema` (`SchemaInput`, `StructureInput`, `FieldInput`, `ClassificationInput`,
   and `RegexValidator`) for `fastino/gliner2.5-multi-v1` (`BoundaryExtractor`).
2. `gliner_fast_py.GlinerEngine` (`gliner_multi_v2.1` FP32 ONNX) zero-shot entity labels and
   post-extraction Pydantic validators.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_EXTERNAL_HF_HUB = "/Volumes/SAMSUNG_9100/MacBook/huggingface-hub"
DEFAULT_GLINER25_SNAPSHOT_DIR = (
    "/Volumes/SAMSUNG_9100/MacBook/huggingface-hub/"
    "models--fastino--gliner2.5-multi-v1/snapshots/a221b77a8baf4a613b8f8652661d41fa10a5641e"
)
DEFAULT_GLINER_ONNX_MODEL_DIR = "/Users/josaum/projects/mycelia-workspace/mycelia-gateway/models/gliner_multi_v2.1"


def configure_offline_model_env(hf_hub_dir: str | None = None) -> dict[str, str]:
    """Configure HuggingFace / Transformers in 100% air-gapped offline mode (`HF_HUB_OFFLINE=1`)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    hub_dir = hf_hub_dir or os.environ.get("SOVEREIGN_HF_HUB", DEFAULT_EXTERNAL_HF_HUB)
    if hub_dir and os.path.isdir(hub_dir):
        os.environ.setdefault("HF_HOME", hub_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_dir)
    return {
        "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
        "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
    }


class InvoiceHeaderSchema(BaseModel):
    """
    Domain-agnostic Commercial Invoice / Document Header structure extracted by
    GLiNER 2.5 (`BoundaryExtractor`) and `gliner_fast_py`.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "gliner_structure_name": "invoice_header",
            "gliner_mode": "anchorless",
        },
    )

    vendor_name: str | None = Field(
        default=None,
        description=(
            "Legal name or DBA of the vendor, supplier, carrier, utility, or organization "
            "issuing the invoice, credit memo, or payment request"
        ),
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.35,
            "regex_pattern": r"^[A-Za-z0-9][A-Za-z0-9\s&\.,\-\'/()]{2,80}$",
        },
    )
    invoice_number: str | None = Field(
        default=None,
        description=(
            "Vendor invoice number, bill number, credit memo number, statement reference number, "
            "or service invoice identifier"
        ),
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.35,
            "regex_pattern": r"^[A-Za-z0-9][A-Za-z0-9\-\._/:\s]{0,22}$",
        },
    )
    invoice_date: str | None = Field(
        default=None,
        description="Invoice date, bill date, credit memo date, statement date, or request date",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.35,
            "regex_pattern": r"^(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+20\d{2})$",
        },
    )
    total_amount: str | None = Field(
        default=None,
        description="Total invoice amount due, current charges, net credit amount, or total payment amount",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.35,
            "regex_pattern": r"^[\$\(\-\s]*\d[\d,]*\.?\d*[\)\-\sA-Z]*$",
        },
    )
    sales_tax_amount: str | None = Field(
        default=None,
        description="Sales tax, state tax, VAT, or county/local tax amount charged on the document",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.40,
            "regex_pattern": r"^[\$\(\-\s]*\d[\d,]*\.\d{2}[\)\-\s]*$",
        },
    )
    freight_amount: str | None = Field(
        default=None,
        description="Freight, shipping, handling, or delivery charge amount on the document",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.40,
            "regex_pattern": r"^[\$\(\-\s]*\d[\d,]*\.\d{2}[\)\-\sA-Z]*$",
        },
    )
    purchase_order_number: str | None = Field(
        default=None,
        description="Customer purchase order number or ERP PO reference number",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.40,
            "regex_pattern": r"^[A-Za-z0-9][A-Za-z0-9\-]{2,18}$",
        },
    )
    customer_account_number: str | None = Field(
        default=None,
        description="Customer account number, utility account number, or bill-to account ID",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.40,
        },
    )
    payment_terms: str | None = Field(
        default=None,
        description="Payment terms such as Net 30, Net 15, Net 60, 1% 10 Net 30, or Due on Receipt",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.40,
        },
    )

    @field_validator("*", mode="before")
    @classmethod
    def normalize_whitespace(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = re.sub(r"\s+", " ", v).strip()
            return cleaned if cleaned else None
        return v

    @field_validator("invoice_number")
    @classmethod
    def clean_invoice_number(cls, v: str | None) -> str | None:
        if not v:
            return None
        v = v.strip()
        return v if any(ch.isdigit() for ch in v) else None


class RemitToPayeeSchema(BaseModel):
    """Permitted Payee / Remit-To Address structure extracted from the remittance section."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "gliner_structure_name": "remit_to_address",
            "gliner_mode": "anchorless",
        },
    )

    remit_to_name: str | None = Field(
        default=None,
        description="Remit-to payee name, factoring company, or lockbox entity to whom payment is sent",
        json_schema_extra={"dtype": "str", "threshold": 0.40},
    )
    remit_to_street_or_pobox: str | None = Field(
        default=None,
        description="Remit-to street address, lockbox PO Box, or department number",
        json_schema_extra={"dtype": "str", "threshold": 0.40},
    )
    remit_to_city_state_zip: str | None = Field(
        default=None,
        description="Remit-to city, state/province code, and postal/ZIP code",
        json_schema_extra={"dtype": "str", "threshold": 0.40},
    )
    bank_routing_aba: str | None = Field(
        default=None,
        description="9-digit ABA bank routing number for ACH or wire remittance",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.45,
            "regex_pattern": r"^\d{9}$",
        },
    )
    bank_account_number: str | None = Field(
        default=None,
        description="Bank account number for ACH or electronic wire remittance",
        json_schema_extra={
            "dtype": "str",
            "threshold": 0.45,
            "regex_pattern": r"^\d{6,17}$",
        },
    )

    @field_validator("*", mode="before")
    @classmethod
    def normalize_whitespace(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = re.sub(r"\s+", " ", v).strip()
            return cleaned if cleaned else None
        return v


class DocumentClassificationSchema(BaseModel):
    """Document-level commercial classifications driven by GLiNER 2.5 `ClassificationInput`."""

    document_type: Literal["Vendor_Invoice", "Vendor_Credit_Memo"] = Field(
        default="Vendor_Invoice",
        description="Commercial document type: Vendor_Invoice for standard invoice/bill, Vendor_Credit_Memo for credit memo",
        json_schema_extra={
            "label_descriptions": {
                "Vendor_Invoice": "Standard vendor invoice, utility bill, freight bill, or payment request asking for payment",
                "Vendor_Credit_Memo": "Credit memo, credit note, return merchandise credit, or negative balance credit adjustment",
            }
        },
    )
    tax_classification: Literal["Sales_Tax_Charged", "Tax_Exempt_Zero_Tax"] = Field(
        default="Tax_Exempt_Zero_Tax",
        description="Tax classification: Sales_Tax_Charged when positive tax > 0 is billed, Tax_Exempt_Zero_Tax when $0.00 tax",
        json_schema_extra={
            "label_descriptions": {
                "Sales_Tax_Charged": "Document charges a positive state, county, VAT, or local sales tax amount",
                "Tax_Exempt_Zero_Tax": "Document has zero sales tax ($0.00), is tax-exempt, or does not charge sales tax",
            }
        },
    )


class DocumentExtractionEnvelope(BaseModel):
    """Composite Pydantic v2 model representing the structured extraction output for a document bundle."""

    header: InvoiceHeaderSchema = Field(default_factory=InvoiceHeaderSchema)
    remit_to: RemitToPayeeSchema = Field(default_factory=RemitToPayeeSchema)
    classification: DocumentClassificationSchema = Field(default_factory=DocumentClassificationSchema)
    field_confidences: dict[str, float] = Field(default_factory=dict)


def build_gliner2_schema_from_pydantic(
    header_cls: type[BaseModel] = InvoiceHeaderSchema,
    remit_cls: type[BaseModel] = RemitToPayeeSchema,
    classification_cls: type[BaseModel] = DocumentClassificationSchema,
) -> Any:
    """
    Compile strongly-typed Pydantic v2 models directly into a validated `gliner2.Schema`
    (with `StructureBuilder`, `RegexValidator`, and `ClassificationInput`).
    """
    from gliner2 import ClassificationInput, FieldInput, RegexValidator, Schema, SchemaInput, StructureInput

    structures_dict: dict[str, StructureInput] = {}
    validator_map: dict[tuple[str, str], tuple[float | None, list[RegexValidator]]] = {}

    for model_cls in (header_cls, remit_cls):
        cfg_extra = model_cls.model_config.get("json_schema_extra") or {}
        struct_name = str(cfg_extra.get("gliner_structure_name", model_cls.__name__.lower()))
        mode = cfg_extra.get("gliner_mode")
        field_inputs: list[FieldInput] = []

        for fname, finfo in model_cls.model_fields.items():
            fextra = finfo.json_schema_extra if isinstance(finfo.json_schema_extra, dict) else {}
            dtype = fextra.get("dtype", "str")
            threshold = fextra.get("threshold")
            regex_pat = fextra.get("regex_pattern")
            validators: list[RegexValidator] = []
            if regex_pat:
                validators.append(RegexValidator(regex_pat, mode="full"))
            validator_map[(struct_name, fname)] = (threshold, validators)
            field_inputs.append(
                FieldInput(
                    name=fname,
                    dtype=dtype,
                    description=finfo.description,
                )
            )
        structures_dict[struct_name] = StructureInput(fields=field_inputs, mode=mode)

    cls_inputs: list[ClassificationInput] = []
    cls_desc_map: dict[str, dict[str, str]] = {}
    for cname, cinfo in classification_cls.model_fields.items():
        cextra = cinfo.json_schema_extra if isinstance(cinfo.json_schema_extra, dict) else {}
        ann = cinfo.annotation
        labels = list(get_args(ann)) if get_origin(ann) is Literal else []
        if len(labels) >= 2:
            cls_inputs.append(ClassificationInput(task=cname, labels=labels, multi_label=False))
            if "label_descriptions" in cextra:
                cls_desc_map[cname] = cextra["label_descriptions"]

    validated_input = SchemaInput(
        structures=structures_dict,
        classifications=cls_inputs,
    )

    schema = Schema()
    for struct_name, struct_in in (validated_input.structures or {}).items():
        builder = schema.structure(struct_name, mode=struct_in.mode)
        for f_in in struct_in.fields:
            thr, vals = validator_map.get((struct_name, f_in.name), (None, []))
            builder.field(
                name=f_in.name,
                dtype=f_in.dtype,
                choices=f_in.choices,
                description=f_in.description,
                threshold=thr,
                validators=vals or None,
            )
        builder._auto_finish()

    for cls_in in validated_input.classifications or []:
        labels_arg: Any = cls_desc_map.get(cls_in.task, cls_in.labels)
        schema.classification(
            task=cls_in.task,
            labels=labels_arg,
            multi_label=cls_in.multi_label,
        )

    return schema


def get_gliner_fast_labels_from_pydantic(
    header_cls: type[BaseModel] = InvoiceHeaderSchema,
    remit_cls: type[BaseModel] = RemitToPayeeSchema,
) -> list[str]:
    """Derive flat entity label list from the Pydantic models for `gliner_fast_py` ONNX span extraction."""
    labels: list[str] = []
    for model_cls in (header_cls, remit_cls):
        for fname in model_cls.model_fields:
            if fname not in labels:
                labels.append(fname)
    return labels


def parse_gliner2_output_to_pydantic(raw_result: dict[str, Any]) -> DocumentExtractionEnvelope:
    """
    Parse raw `gliner2` (`BoundaryExtractor`) output dictionary into a validated
    `DocumentExtractionEnvelope` Pydantic model, preserving per-field confidence scores.
    """
    confidences: dict[str, float] = {}

    def unpack_struct(struct_key: str) -> dict[str, Any]:
        rows = raw_result.get(struct_key)
        if isinstance(rows, dict):
            first = rows
        elif isinstance(rows, list) and rows and isinstance(rows[0], dict):
            first = rows[0]
        else:
            return {}
        out: dict[str, Any] = {}
        for k, v in first.items():
            if isinstance(v, dict) and "text" in v:
                txt = (v.get("text") or "").strip()
                if txt:
                    out[k] = txt
                    confidences[f"{struct_key}.{k}"] = round(float(v.get("confidence", 0.0)), 4)
            elif isinstance(v, str) and v.strip():
                out[k] = v.strip()
        return out

    header_data = unpack_struct("invoice_header")
    remit_data = unpack_struct("remit_to_address")

    cls_data: dict[str, Any] = {}
    for task_name in ("document_type", "tax_classification"):
        c_obj = raw_result.get(task_name)
        if isinstance(c_obj, dict) and c_obj.get("label"):
            cls_data[task_name] = c_obj["label"]
            confidences[task_name] = round(float(c_obj.get("confidence", 0.0)), 4)
        elif isinstance(c_obj, str) and c_obj.strip():
            cls_data[task_name] = c_obj.strip()

    return DocumentExtractionEnvelope(
        header=InvoiceHeaderSchema(**header_data),
        remit_to=RemitToPayeeSchema(**remit_data),
        classification=DocumentClassificationSchema(**cls_data),
        field_confidences=confidences,
    )
