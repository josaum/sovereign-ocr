"""
SAP S/4HANA BPMN 2.0, DMN 1.3 & BAPI/OData Payload Adapter (`sovereign_ocr.sap_bpmn`).

Bridges `sovereign-ocr` governed extraction bundles directly into SAP's native workflow
and posting standards without requiring a custom proprietary workflow DSL:
1. `evaluate_bpmn_gateway_route`: Evaluates BPMN 2.0 exclusive gateway conditions (`RECEIPT_EXCEPTION`,
   `EMPTY_FILE`, `FREIGHT_EXCEEDS_20_PCT_THRESHOLD`, PO-backed `RE` vs. Non-PO `KR`/`KG`) and routes
   either to automated SAP FI/MM posting or to `UserTask_ApExceptionReview` (SAP My Inbox / xSuite).
2. `build_sap_bapi_payload`: Maps a governed bundle into:
   - `BAPI_ACC_DOCUMENT_POST` (`DOCUMENTHEADER`, `ACCOUNTAP`, `ACCOUNTGL`, `ACCOUNTTAX`, `CURRENCYAMOUNT`
     with strict zero-balance $\\sum \\text{AMT\\_DOCCUR} = 0.00$ verification).
   - `API_SUPPLIERINVOICE_PROCESS_SRV` (`A_SupplierInvoice` OData v2/v4 payload for SAP S/4HANA).
   - `ARCHIVELINK_PROVENANCE` (`quad_fast` RDF 1.2 SHA-256 receipt binding).
3. `export_sap_bpmn_xml` & `export_sap_dmn_xml`: Emits standard OMG BPMN 2.0 XML (with
   `<bpmn:extensionElements>` SAP BAPI bindings) and OMG DMN 1.3 XML decision tables ready for
   import into SAP Signavio and SAP Build Process Automation (SBPA).
"""

from __future__ import annotations

import re
from typing import Any


def _pad_sap_alpha_num(val: Any, length: int) -> str:
    """Zero-pad numeric SAP identifiers (e.g. LIFNR=10, HKONT=10, KOSTL=10, BUKRS=4) to `length`."""
    s = str(val or "").strip()
    if not s:
        return ""
    if s.isdigit() and len(s) < length:
        return s.zfill(length)
    return s[:length]


def _format_sap_date_yyyymmdd(iso_date: str) -> str:
    """Convert ISO `YYYY-MM-DD` date string to SAP `YYYYMMDD` (`DATS`) format."""
    s = str(iso_date or "").strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        return digits
    return ""


def evaluate_bpmn_gateway_route(governed_record: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate BPMN 2.0 exclusive gateway conditions on a governed extraction record and
    return the target BPMN node, SAP API action, and exception queue metadata.
    """
    warnings = list(governed_record.get("warnings") or [])
    role = str(governed_record.get("primary_unit_role") or "PRIMARY_INVOICE")
    vendor_id = str(governed_record.get("vendor_id") or "").strip()
    ref = str(governed_record.get("reference") or governed_record.get("extracted_reference") or "").strip()
    doc_date = str(governed_record.get("doc_date") or governed_record.get("extracted_date") or "").strip()
    amount = governed_record.get("amount")
    if amount is None:
        amount = governed_record.get("extracted_amount")
    po_num = str(governed_record.get("purchase_order") or governed_record.get("extracted_po") or "").strip()
    doc_type = str(governed_record.get("doc_type") or governed_record.get("extracted_doc_type") or "KR").strip()

    exception_reasons: list[str] = []
    if role == "EMPTY_FILE" or any(w.startswith("EMPTY_FILE_DETECTED") for w in warnings):
        exception_reasons.append("EMPTY_DOCUMENT_UPLOAD")
    if role == "RECEIPT_EXCEPTION" or "RECEIPT_SUBMITTED_INSTEAD_OF_INVOICE" in warnings:
        exception_reasons.append("POS_RECEIPT_SUBMITTED_INSTEAD_OF_INVOICE")
    if "FREIGHT_EXCEEDS_20_PCT_THRESHOLD" in warnings:
        exception_reasons.append("FREIGHT_EXCEEDS_20_PCT_TOLERANCE")
    if not vendor_id:
        exception_reasons.append("UNRESOLVED_SAP_VENDOR_LIFNR")
    if not ref:
        exception_reasons.append("MISSING_INVOICE_REFERENCE_XBLNR")
    if amount is None:
        exception_reasons.append("MISSING_DOCUMENT_AMOUNT_WRBTR")

    if exception_reasons:
        return {
            "is_auto_postable": False,
            "target_bpmn_node": "UserTask_ApExceptionReview",
            "sap_action": "PARK_TO_SAP_INBOX_EXCEPTION_QUEUE",
            "sap_odata_service": "API_SUPPLIERINVOICE_PROCESS_SRV",
            "sap_doc_type": doc_type,
            "inbox_recipient": (
                str(governed_record.get("approver") or "").strip()
                or str(governed_record.get("coder") or "").strip()
                or "SAP_AP_EXCEPTION_SPECIALIST"
            ),
            "exception_reasons": exception_reasons,
        }

    if po_num and doc_type == "RE":
        return {
            "is_auto_postable": True,
            "target_bpmn_node": "ServiceTask_PostSapMMInvoice",
            "sap_action": "BAPI_INVOICE_CREATE",
            "sap_odata_service": "API_SUPPLIERINVOICE_PROCESS_SRV",
            "sap_doc_type": "RE",
            "inbox_recipient": "",
            "exception_reasons": [],
        }

    return {
        "is_auto_postable": True,
        "target_bpmn_node": "ServiceTask_PostSapFIVoucher",
        "sap_action": "BAPI_ACC_DOCUMENT_POST",
        "sap_odata_service": "API_JOURNALENTRYITEMBASIC_SRV",
        "sap_doc_type": "KG" if doc_type == "KG" else "KR",
        "inbox_recipient": "",
        "exception_reasons": [],
    }


def build_sap_bapi_payload(governed_record: dict[str, Any]) -> dict[str, Any]:
    """
    Transform a governed `sovereign-ocr` record into a ready-to-post SAP S/4HANA
    `BAPI_ACC_DOCUMENT_POST` + `API_SUPPLIERINVOICE_PROCESS_SRV` OData payload with
    strict zero-balance currency items (`sum(AMT_DOCCUR) == 0.00`).
    """
    routing = evaluate_bpmn_gateway_route(governed_record)

    cocd = _pad_sap_alpha_num(
        governed_record.get("cocd_str") or governed_record.get("cocd") or "0010",
        4,
    )
    doc_date_iso = str(governed_record.get("doc_date") or governed_record.get("extracted_date") or "")
    pstng_date_iso = str(governed_record.get("pstng_date") or doc_date_iso)
    bldat = _format_sap_date_yyyymmdd(doc_date_iso)
    budat = _format_sap_date_yyyymmdd(pstng_date_iso) or bldat
    fisc_year = str(governed_record.get("year") or (bldat[:4] if len(bldat) >= 4 else "2026"))
    doc_type = routing["sap_doc_type"]

    ref = str(governed_record.get("reference") or governed_record.get("extracted_reference") or "").strip()[:16]
    sgtxt = str(governed_record.get("line_item_text") or "").strip()[:50]
    bktxt = sgtxt[:25] if sgtxt else f"SOVEREIGN {doc_type} {ref}"[:25]
    currency = str(governed_record.get("currency") or "USD").strip() or "USD"

    lifnr = _pad_sap_alpha_num(governed_record.get("vendor_id"), 10)
    empfb = _pad_sap_alpha_num(governed_record.get("payee_id"), 10)
    if empfb == lifnr:
        empfb = ""

    hkont = _pad_sap_alpha_num(governed_record.get("gl_account") or "0050000000", 10)
    kostl = _pad_sap_alpha_num(governed_record.get("cost_center"), 10)
    prctr = _pad_sap_alpha_num(governed_record.get("profit_center"), 10)
    zterm = str(governed_record.get("pmt_terms") or "").strip()[:4]
    zlsch = str(governed_record.get("pmt_method") or "").strip()[:1]
    mwskz = str(governed_record.get("tax_code") or "").strip()[:2]
    txjcd = str(governed_record.get("tax_jurisdiction") or "").strip()[:15]
    ebeln = _pad_sap_alpha_num(
        governed_record.get("purchase_order") or governed_record.get("extracted_po"),
        10,
    )
    ebelp = _pad_sap_alpha_num(governed_record.get("po_item") or ("00010" if ebeln else ""), 5)

    raw_amt = governed_record.get("amount")
    if raw_amt is None:
        raw_amt = governed_record.get("extracted_amount") or 0.0
    gross_abs = round(abs(float(raw_amt)), 2)

    raw_tax = governed_record.get("tax_on_invoice")
    if raw_tax is None:
        raw_tax = governed_record.get("extracted_tax") or 0.0
    tax_abs = round(abs(float(raw_tax or 0.0)), 2)
    if tax_abs > gross_abs:
        tax_abs = 0.0
    net_abs = round(gross_abs - tax_abs, 2)

    # In SAP BAPI_ACC_DOCUMENT_POST:
    # - For a Vendor Invoice (KR): Vendor AP line is a Credit (-gross_abs), GL/Tax lines are Debits (+net_abs, +tax_abs)
    # - For a Credit Memo (KG): Vendor AP line is a Debit (+gross_abs), GL/Tax lines are Credits (-net_abs, -tax_abs)
    is_credit_memo = doc_type == "KG" or float(raw_amt or 0.0) < 0.0
    ap_signed = +gross_abs if is_credit_memo else -gross_abs
    gl_signed = -net_abs if is_credit_memo else +net_abs
    tax_signed = -tax_abs if is_credit_memo else +tax_abs

    doc_header = {
        "OBJ_TYPE": "BKPFF",
        "OBJ_KEY": "$",
        "USERNAME": str(governed_record.get("user_name") or "SOVEREIGN_AI")[:12],
        "HEADER_TXT": bktxt,
        "COMP_CODE": cocd,
        "DOC_DATE": bldat,
        "PSTNG_DATE": budat,
        "FISC_YEAR": fisc_year,
        "DOC_TYPE": doc_type,
        "REF_DOC_NO": ref,
    }

    account_ap = [
        {
            "ITEMNO_ACC": "0000000001",
            "VENDOR_NO": lifnr,
            "PARTNER_BK": empfb,
            "COMP_CODE": cocd,
            "PMNTTRMS": zterm,
            "PYMT_METH": zlsch,
            "ITEM_TEXT": sgtxt,
        }
    ]

    account_gl = [
        {
            "ITEMNO_ACC": "0000000002",
            "GL_ACCOUNT": hkont,
            "COMP_CODE": cocd,
            "COSTCENTER": kostl,
            "PROFIT_CTR": prctr,
            "TAX_CODE": mwskz,
            "TAXJURCODE": txjcd,
            "ITEM_TEXT": sgtxt,
        }
    ]

    account_tax: list[dict[str, Any]] = []
    currency_amount = [
        {
            "ITEMNO_ACC": "0000000001",
            "CURR_TYPE": "00",
            "CURRENCY": currency,
            "AMT_DOCCUR": ap_signed,
        },
        {
            "ITEMNO_ACC": "0000000002",
            "CURR_TYPE": "00",
            "CURRENCY": currency,
            "AMT_DOCCUR": gl_signed,
        },
    ]

    if tax_abs > 0.0 or mwskz:
        account_tax.append(
            {
                "ITEMNO_ACC": "0000000003",
                "TAX_CODE": mwskz or "I1",
                "TAXJURCODE": txjcd,
            }
        )
        currency_amount.append(
            {
                "ITEMNO_ACC": "0000000003",
                "CURR_TYPE": "00",
                "CURRENCY": currency,
                "AMT_DOCCUR": tax_signed,
                "AMT_BASE": gl_signed,
            }
        )

    balance_sum = round(sum(float(item["AMT_DOCCUR"]) for item in currency_amount), 2)

    odata_supplier_invoice: dict[str, Any] = {
        "CompanyCode": cocd,
        "DocumentDate": doc_date_iso,
        "PostingDate": pstng_date_iso,
        "FiscalYear": fisc_year,
        "InvoicingParty": lifnr,
        "AlternativePayee": empfb,
        "SupplierInvoiceIDByInvcgParty": ref,
        "DocumentCurrency": currency,
        "InvoiceGrossAmount": f"{gross_abs:.2f}",
        "SupplierInvoiceIsCreditMemo": is_credit_memo,
        "PaymentTerms": zterm,
        "PaymentMethod": zlsch,
        "AccountingDocumentType": doc_type,
        "DocumentHeaderText": bktxt,
        "to_SuplrInvcItemPurOrdRef": (
            [
                {
                    "SupplierInvoiceItem": "000001",
                    "PurchaseOrder": ebeln,
                    "PurchaseOrderItem": ebelp,
                    "TaxCode": mwskz,
                    "TaxJurisdiction": txjcd,
                    "DocumentCurrency": currency,
                    "SupplierInvoiceItemAmount": f"{net_abs:.2f}",
                    "SupplierInvoiceItemText": sgtxt,
                }
            ]
            if ebeln
            else []
        ),
    }

    return {
        "bpmn_routing": routing,
        "BAPI_ACC_DOCUMENT_POST": {
            "DOCUMENTHEADER": doc_header,
            "ACCOUNTAP": account_ap,
            "ACCOUNTGL": account_gl,
            "ACCOUNTTAX": account_tax,
            "CURRENCYAMOUNT": currency_amount,
            "zero_balance_verified": balance_sum == 0.0,
            "balance_sum": balance_sum,
        },
        "API_SUPPLIERINVOICE_PROCESS_SRV": odata_supplier_invoice,
        "ARCHIVELINK_PROVENANCE": {
            "quad_receipt_hash": governed_record.get("quad_receipt_hash", ""),
            "quad_ipc_bytes_len": int(governed_record.get("quad_ipc_bytes_len") or 0),
        },
    }


def export_sap_dmn_xml(decision_id: str = "Decision_Sovereign_SAP_Governance") -> str:
    """
    Export a standard OMG DMN 1.3 XML decision table encoding the deterministic
    SAP S/4HANA posting & exception-routing rules in FEEL syntax (importable into
    SAP Signavio and SAP Build Process Automation).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/"
             xmlns:dmndi="https://www.omg.org/spec/DMN/20191111/DMNDI/"
             id="Definitions_{decision_id}"
             name="Sovereign-OCR SAP S/4HANA Governance DMN 1.3"
             namespace="https://sovereign-ocr.local/dmn/sap-governance">
  <decision id="{decision_id}" name="SAP S/4HANA Voucher &amp; Exception Routing">
    <decisionTable id="DecisionTable_SAP_Routing" hitPolicy="FIRST">
      <input id="Input_UnitRole" label="Primary Unit Role">
        <inputExpression id="Expr_UnitRole" typeRef="string">
          <text>primary_unit_role</text>
        </inputExpression>
      </input>
      <input id="Input_FreightPct" label="Freight Ratio">
        <inputExpression id="Expr_FreightPct" typeRef="number">
          <text>freight_ratio</text>
        </inputExpression>
      </input>
      <input id="Input_HasPO" label="Has Purchase Order">
        <inputExpression id="Expr_HasPO" typeRef="boolean">
          <text>has_purchase_order</text>
        </inputExpression>
      </input>
      <output id="Output_SapDocType" label="SAP Document Type (BLART)" name="sap_doc_type" typeRef="string" />
      <output id="Output_BpmnNode" label="Target BPMN Node" name="target_bpmn_node" typeRef="string" />
      <output id="Output_SapBapi" label="SAP BAPI / OData Action" name="sap_action" typeRef="string" />
      <rule id="Rule_EmptyFile">
        <inputEntry id="IE_Empty_1"><text>"EMPTY_FILE"</text></inputEntry>
        <inputEntry id="IE_Empty_2"><text>-</text></inputEntry>
        <inputEntry id="IE_Empty_3"><text>-</text></inputEntry>
        <outputEntry id="OE_Empty_1"><text>"EXCEPTION"</text></outputEntry>
        <outputEntry id="OE_Empty_2"><text>"UserTask_ApExceptionReview"</text></outputEntry>
        <outputEntry id="OE_Empty_3"><text>"PARK_TO_SAP_INBOX_EXCEPTION_QUEUE"</text></outputEntry>
      </rule>
      <rule id="Rule_ReceiptException">
        <inputEntry id="IE_Rcpt_1"><text>"RECEIPT_EXCEPTION"</text></inputEntry>
        <inputEntry id="IE_Rcpt_2"><text>-</text></inputEntry>
        <inputEntry id="IE_Rcpt_3"><text>-</text></inputEntry>
        <outputEntry id="OE_Rcpt_1"><text>"EXCEPTION"</text></outputEntry>
        <outputEntry id="OE_Rcpt_2"><text>"UserTask_ApExceptionReview"</text></outputEntry>
        <outputEntry id="OE_Rcpt_3"><text>"PARK_TO_SAP_INBOX_EXCEPTION_QUEUE"</text></outputEntry>
      </rule>
      <rule id="Rule_FreightExceeds20Pct">
        <inputEntry id="IE_Frt_1"><text>-</text></inputEntry>
        <inputEntry id="IE_Frt_2"><text>&gt; 0.20</text></inputEntry>
        <inputEntry id="IE_Frt_3"><text>-</text></inputEntry>
        <outputEntry id="OE_Frt_1"><text>"EXCEPTION"</text></outputEntry>
        <outputEntry id="OE_Frt_2"><text>"UserTask_ApExceptionReview"</text></outputEntry>
        <outputEntry id="OE_Frt_3"><text>"PARK_TO_SAP_INBOX_EXCEPTION_QUEUE"</text></outputEntry>
      </rule>
      <rule id="Rule_CreditMemoKG">
        <inputEntry id="IE_KG_1"><text>"CREDIT_MEMO"</text></inputEntry>
        <inputEntry id="IE_KG_2"><text>&lt;= 0.20</text></inputEntry>
        <inputEntry id="IE_KG_3"><text>-</text></inputEntry>
        <outputEntry id="OE_KG_1"><text>"KG"</text></outputEntry>
        <outputEntry id="OE_KG_2"><text>"ServiceTask_PostSapFIVoucher"</text></outputEntry>
        <outputEntry id="OE_KG_3"><text>"BAPI_ACC_DOCUMENT_POST"</text></outputEntry>
      </rule>
      <rule id="Rule_POInvoiceRE">
        <inputEntry id="IE_RE_1"><text>"PRIMARY_INVOICE"</text></inputEntry>
        <inputEntry id="IE_RE_2"><text>&lt;= 0.20</text></inputEntry>
        <inputEntry id="IE_RE_3"><text>true</text></inputEntry>
        <outputEntry id="OE_RE_1"><text>"RE"</text></outputEntry>
        <outputEntry id="OE_RE_2"><text>"ServiceTask_PostSapMMInvoice"</text></outputEntry>
        <outputEntry id="OE_RE_3"><text>"BAPI_INVOICE_CREATE"</text></outputEntry>
      </rule>
      <rule id="Rule_VendorInvoiceKR">
        <inputEntry id="IE_KR_1"><text>"PRIMARY_INVOICE", "STATEMENT"</text></inputEntry>
        <inputEntry id="IE_KR_2"><text>&lt;= 0.20</text></inputEntry>
        <inputEntry id="IE_KR_3"><text>false</text></inputEntry>
        <outputEntry id="OE_KR_1"><text>"KR"</text></outputEntry>
        <outputEntry id="OE_KR_2"><text>"ServiceTask_PostSapFIVoucher"</text></outputEntry>
        <outputEntry id="OE_KR_3"><text>"BAPI_ACC_DOCUMENT_POST"</text></outputEntry>
      </rule>
    </decisionTable>
  </decision>
</definitions>
"""


def export_sap_bpmn_xml(
    process_id: str = "Process_Sovereign_OCR_SAP",
    process_name: str = "Sovereign-OCR SAP S/4HANA Executable Workflow",
) -> str:
    """
    Export a standard OMG BPMN 2.0 XML workflow definition with `<bpmn:extensionElements>`
    declaring the SAP S/4HANA `BAPI_ACC_DOCUMENT_POST` and `API_SUPPLIERINVOICE_PROCESS_SRV`
    bindings, compatible with SAP Signavio and SAP Build Process Automation.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
    xmlns:sap="https://sovereign-ocr.local/schema/sap-bapi-extension"
    id="Definitions_{process_id}"
    targetNamespace="https://sovereign-ocr.local/bpmn/sap-s4hana"
    exporter="Sovereign-OCR Engine"
    exporterVersion="1.0.0">
  <bpmn:process id="{process_id}" name="{process_name}" isExecutable="true">
    <bpmn:startEvent id="StartEvent_BundleReceived" name="Document Bundle Received">
      <bpmn:outgoing>Flow_ToIngest</bpmn:outgoing>
    </bpmn:startEvent>
    <bpmn:serviceTask id="Task_MultiFormatIngest" name="Stage 1: Native Multi-Format Unpacking &amp; Candidate Routing">
      <bpmn:incoming>Flow_ToIngest</bpmn:incoming>
      <bpmn:outgoing>Flow_ToOcrGate</bpmn:outgoing>
    </bpmn:serviceTask>
    <bpmn:serviceTask id="Task_SelfHealingOcrAndPreproc" name="Stage 2: Algebraic CMap Repair &amp; 5-Step Geometry/DPI Visual OCR">
      <bpmn:incoming>Flow_ToOcrGate</bpmn:incoming>
      <bpmn:outgoing>Flow_ToSpanExtract</bpmn:outgoing>
    </bpmn:serviceTask>
    <bpmn:serviceTask id="Task_PydanticGlinerExtract" name="Stage 3: Zero-Hallucination Pydantic-to-GLiNER 2.5 Span Extraction">
      <bpmn:incoming>Flow_ToSpanExtract</bpmn:incoming>
      <bpmn:outgoing>Flow_ToOntology</bpmn:outgoing>
    </bpmn:serviceTask>
    <bpmn:businessRuleTask id="Task_ActiveOntologyGovernance" name="Stage 4: Active Runtime Ontology Grounding &amp; DMN 1.3 Governance">
      <bpmn:extensionElements>
        <sap:dmnDecisionRef decisionId="Decision_Sovereign_SAP_Governance" />
      </bpmn:extensionElements>
      <bpmn:incoming>Flow_ToOntology</bpmn:incoming>
      <bpmn:outgoing>Flow_ToPostingGateway</bpmn:outgoing>
    </bpmn:businessRuleTask>
    <bpmn:exclusiveGateway id="Gateway_SapPostingRoute" name="Auto-Postable or Exception?">
      <bpmn:incoming>Flow_ToPostingGateway</bpmn:incoming>
      <bpmn:outgoing>Flow_ToSapFI</bpmn:outgoing>
      <bpmn:outgoing>Flow_ToSapMM</bpmn:outgoing>
      <bpmn:outgoing>Flow_ToExceptionQueue</bpmn:outgoing>
    </bpmn:exclusiveGateway>
    <bpmn:serviceTask id="ServiceTask_PostSapFIVoucher" name="Stage 5A: Post SAP FI Voucher (KR/KG) + ArchiveLink Quad Receipt">
      <bpmn:extensionElements>
        <sap:bapiBinding bapiName="BAPI_ACC_DOCUMENT_POST" odataService="API_JOURNALENTRYITEMBASIC_SRV">
          <sap:field mapFrom="cocd_str" sapTable="DOCUMENTHEADER" sapField="COMP_CODE" />
          <sap:field mapFrom="doc_date" sapTable="DOCUMENTHEADER" sapField="DOC_DATE" />
          <sap:field mapFrom="doc_type" sapTable="DOCUMENTHEADER" sapField="DOC_TYPE" />
          <sap:field mapFrom="reference" sapTable="DOCUMENTHEADER" sapField="REF_DOC_NO" />
          <sap:field mapFrom="vendor_id" sapTable="ACCOUNTAP" sapField="VENDOR_NO" />
          <sap:field mapFrom="payee_id" sapTable="ACCOUNTAP" sapField="PARTNER_BK" />
          <sap:field mapFrom="gl_account" sapTable="ACCOUNTGL" sapField="GL_ACCOUNT" />
          <sap:field mapFrom="cost_center" sapTable="ACCOUNTGL" sapField="COSTCENTER" />
          <sap:field mapFrom="profit_center" sapTable="ACCOUNTGL" sapField="PROFIT_CTR" />
          <sap:field mapFrom="tax_code" sapTable="ACCOUNTGL" sapField="TAX_CODE" />
          <sap:field mapFrom="amount" sapTable="CURRENCYAMOUNT" sapField="AMT_DOCCUR" />
          <sap:field mapFrom="quad_receipt_hash" sapTable="ARCHIVELINK" sapField="ARCHIVE_DOC_ID" />
        </sap:bapiBinding>
      </bpmn:extensionElements>
      <bpmn:incoming>Flow_ToSapFI</bpmn:incoming>
      <bpmn:outgoing>Flow_FIToEnd</bpmn:outgoing>
    </bpmn:serviceTask>
    <bpmn:serviceTask id="ServiceTask_PostSapMMInvoice" name="Stage 5B: Post SAP MM PO Invoice (RE) + ArchiveLink Quad Receipt">
      <bpmn:extensionElements>
        <sap:bapiBinding bapiName="BAPI_INVOICE_CREATE" odataService="API_SUPPLIERINVOICE_PROCESS_SRV" />
      </bpmn:extensionElements>
      <bpmn:incoming>Flow_ToSapMM</bpmn:incoming>
      <bpmn:outgoing>Flow_MMToEnd</bpmn:outgoing>
    </bpmn:serviceTask>
    <bpmn:userTask id="UserTask_ApExceptionReview" name="Stage 5C: SAP My Inbox / xSuite Exception Review Queue">
      <bpmn:incoming>Flow_ToExceptionQueue</bpmn:incoming>
      <bpmn:outgoing>Flow_ExceptionToEnd</bpmn:outgoing>
    </bpmn:userTask>
    <bpmn:endEvent id="EndEvent_Posted" name="SAP S/4HANA Voucher Posted">
      <bpmn:incoming>Flow_FIToEnd</bpmn:incoming>
      <bpmn:incoming>Flow_MMToEnd</bpmn:incoming>
    </bpmn:endEvent>
    <bpmn:endEvent id="EndEvent_ParkedForReview" name="Parked in Exception Queue">
      <bpmn:incoming>Flow_ExceptionToEnd</bpmn:incoming>
    </bpmn:endEvent>
    <bpmn:sequenceFlow id="Flow_ToIngest" sourceRef="StartEvent_BundleReceived" targetRef="Task_MultiFormatIngest" />
    <bpmn:sequenceFlow id="Flow_ToOcrGate" sourceRef="Task_MultiFormatIngest" targetRef="Task_SelfHealingOcrAndPreproc" />
    <bpmn:sequenceFlow id="Flow_ToSpanExtract" sourceRef="Task_SelfHealingOcrAndPreproc" targetRef="Task_PydanticGlinerExtract" />
    <bpmn:sequenceFlow id="Flow_ToOntology" sourceRef="Task_PydanticGlinerExtract" targetRef="Task_ActiveOntologyGovernance" />
    <bpmn:sequenceFlow id="Flow_ToPostingGateway" sourceRef="Task_ActiveOntologyGovernance" targetRef="Gateway_SapPostingRoute" />
    <bpmn:sequenceFlow id="Flow_ToSapFI" sourceRef="Gateway_SapPostingRoute" targetRef="ServiceTask_PostSapFIVoucher">
      <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">${{is_auto_postable == true &amp;&amp; sap_doc_type != 'RE'}}</bpmn:conditionExpression>
    </bpmn:sequenceFlow>
    <bpmn:sequenceFlow id="Flow_ToSapMM" sourceRef="Gateway_SapPostingRoute" targetRef="ServiceTask_PostSapMMInvoice">
      <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">${{is_auto_postable == true &amp;&amp; sap_doc_type == 'RE'}}</bpmn:conditionExpression>
    </bpmn:sequenceFlow>
    <bpmn:sequenceFlow id="Flow_ToExceptionQueue" sourceRef="Gateway_SapPostingRoute" targetRef="UserTask_ApExceptionReview">
      <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">${{is_auto_postable == false}}</bpmn:conditionExpression>
    </bpmn:sequenceFlow>
    <bpmn:sequenceFlow id="Flow_FIToEnd" sourceRef="ServiceTask_PostSapFIVoucher" targetRef="EndEvent_Posted" />
    <bpmn:sequenceFlow id="Flow_MMToEnd" sourceRef="ServiceTask_PostSapMMInvoice" targetRef="EndEvent_Posted" />
    <bpmn:sequenceFlow id="Flow_ExceptionToEnd" sourceRef="UserTask_ApExceptionReview" targetRef="EndEvent_ParkedForReview" />
  </bpmn:process>
</bpmn:definitions>
"""
