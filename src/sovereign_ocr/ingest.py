"""
Multi-Format Document Ingestion, Recursive `.MSG`/`.EML` Attachment Unpacking,
Primary-vs-Backup Document Classification, and Multi-Document PDF Splitting.

Powered by native zero-copy Rust extensions:
- `email_fast` (v0.2.6+ with OLE CFB `__attach_version1.0_*` & binary stream support)
- `pdf_fast` & `pdf_fast_write` (`extract_pages` for zero-copy multi-invoice PDF splitting)
- `docx_fast` & `xlsx_fast`
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docx_fast
import email_fast
import pdf_fast
import xlsx_fast

try:
    import pdf_fast_write
except ImportError:
    pdf_fast_write = None  # type: ignore[assignment]

from sovereign_ocr.ocr_layout import (
    extract_ordered_pdf_page_text,
    is_corrupted_or_scanned_pdf_page,
    preprocess_document_image_bytes,
    reconstruct_text_from_ocr_runs,
    render_pdf_page_png,
    repair_shifted_cmap_text,
    run_batch_ocr,
)


@dataclass
class ExtractedDocumentUnit:
    """Represents a single logical document or attachment within an ingested document bundle."""

    unit_id: str
    filename: str
    source_kind: str  # "PDF", "MSG_BODY", "MSG_ATTACHMENT", "DOCX", "XLSX", "RTF", "TXT"
    role: str         # "PRIMARY_INVOICE", "CREDIT_MEMO", "STATEMENT", "SPLIT_BACKUP", "PO_BACKUP", "RECEIPT_EXCEPTION", "EMAIL_APPROVAL"
    text: str
    page_count: int = 1
    needs_ocr: bool = False
    email_meta: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class IngestedBundle:
    """Represents an ingested multi-file or multi-attachment document bundle."""

    bundle_id: str
    entity_code: str = ""
    document_id: str = ""
    fiscal_year: str = ""
    units: list[ExtractedDocumentUnit] = field(default_factory=list)
    primary_unit: ExtractedDocumentUnit | None = None
    combined_text: str = ""
    multi_invoice_segments: list[dict[str, Any]] = field(default_factory=list)


def _walk_docx_blocks(blocks: list[Any]) -> list[str]:
    out: list[str] = []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "paragraph":
            t = (b.get("text") or "").strip()
            if not t and b.get("runs"):
                t = "".join(r.get("text", "") for r in b["runs"] if isinstance(r, dict)).strip()
            if t:
                out.append(t)
        elif btype == "table":
            for row in b.get("rows") or []:
                cell_strs = []
                cells = row.get("cells") if isinstance(row, dict) else row
                for cell in cells or []:
                    if isinstance(cell, dict):
                        c_lines = _walk_docx_blocks(cell.get("blocks") or [])
                        if not c_lines and cell.get("text"):
                            c_lines = [cell["text"].strip()]
                        if c_lines:
                            cell_strs.append(" ".join(c_lines))
                if cell_strs:
                    out.append(" | ".join(cell_strs))
    return out


def extract_docx_text(path: Path) -> str:
    """Extract structured text and tables from `.docx` via `docx_fast`."""
    d = docx_fast.parse_docx(str(path))
    if isinstance(d, str):
        return d.strip()
    lines: list[str] = []
    for hdr in d.get("headers") or []:
        if isinstance(hdr, dict):
            lines.extend(_walk_docx_blocks(hdr.get("blocks") or []))
    lines.extend(_walk_docx_blocks(d.get("blocks") or []))
    for ftr in d.get("footers") or []:
        if isinstance(ftr, dict):
            lines.extend(_walk_docx_blocks(ftr.get("blocks") or []))
    return "\n".join(lines)


def extract_xlsx_text(path: Path) -> str:
    """Extract cell coordinates and values (including ISO dates) from `.xlsx`/`.xlsm` via `xlsx_fast`."""
    raw = json.loads(xlsx_fast.parse_xlsx_json(str(path)))
    lines: list[str] = []
    for s in raw.get("sheets", []):
        lines.append(f"=== Sheet: {s.get('name')} ===")
        cells = s.get("cells") or []
        by_row: dict[int, list[tuple[int, str]]] = {}
        for c in cells:
            vobj = c.get("value")
            if isinstance(vobj, dict):
                val = vobj.get("v") if "v" in vobj else (str(vobj.get("iso"))[:10] if "iso" in vobj else None)
            else:
                val = vobj
            if val is not None and str(val).strip() != "":
                by_row.setdefault(c["row"], []).append((c["col"], str(val).strip()))
        for r_idx in sorted(by_row.keys()):
            row_str = " | ".join(f"C{col}:{val}" for col, val in sorted(by_row[r_idx]))
            lines.append(f"R{r_idx}: {row_str}")
    return "\n".join(lines)


def classify_document_unit(filename: str, text: str, source_kind: str) -> tuple[str, float]:
    """
    Score and classify a document or attachment unit to distinguish:
    - Primary commercial documents (`PRIMARY_INVOICE`, `CREDIT_MEMO`, `STATEMENT`)
    - Supporting / backup commercial documents (`SPLIT_BACKUP`, `PO_BACKUP`, `RECEIPT_EXCEPTION`,
      `RECEIPT_BACKUP`, `QUOTE_PREINVOICE`, `EMAIL_APPROVAL`)
    - Multilingual / Insurance / Documentoscopia claim units (`CLAIM_NOTICE_AVISO`,
      `POLICE_REPORT_BO`, `NOTA_FISCAL`, `PROOF_OF_ADDRESS`, `IDENTITY_DOCUMENT`, `EMPTY_FILE`).
    """
    if source_kind == "EMPTY":
        return ("EMPTY_FILE", 0.0)

    low = text.lower()
    fname_low = filename.lower()
    if source_kind == "MSG_BODY":
        if "invoice" in low and ("total" in low or "amount due" in low or "$" in low):
            return ("EMAIL_APPROVAL", 35.0)
        return ("EMAIL_APPROVAL", 15.0)

    # POS fuel/fleet card terminal receipt exception
    if ("gas service" in low or "pump#" in low or "gallons" in low or "fuel" in low) and "auth " in low:
        return ("RECEIPT_EXCEPTION", 25.0)

    # Expense backup receipts ("This is not an invoice", "Receipt for ...")
    if "this is not an invoice" in low or "receipt for " in low or "receipt" in fname_low:
        return ("RECEIPT_BACKUP", 18.0)

    # Quotes, Estimates, Pre-Invoices, or draft documents with "Invoice Date: TBD"
    if (
        "pre-invoice" in fname_low
        or "quote" in fname_low
        or "opportunity to quote" in low
        or "estimate no" in low
        or bool(re.search(r"invoice\s+date\s*[:\n]\s*tbd\b", low))
    ):
        return ("QUOTE_PREINVOICE", 16.0)

    if source_kind in ("XLSX", "XLSM") and (
        "intercompany" in low or "reconcile" in low or "split" in low or "allocation" in low
    ):
        return ("SPLIT_BACKUP", 30.0)

    # Documentoscopia / Insurance Claims & KYC Document Roles
    if (
        "aviso de sinistro" in low
        or "dados do sinistro" in low
        or "tipo de sinistro" in low
        or "aviso" in fname_low
    ):
        return ("CLAIM_NOTICE_AVISO", 90.0)

    if (
        "boletim de ocorrencia" in low
        or "boletim de ocorrência" in low
        or "policia civil" in low
        or "polícia civil" in low
        or "delegacia" in low
        or "secretaria da seguranca" in low
        or "secretaria de seguranca" in low
        or bool(re.search(r"(?:^|[\s_\-])b\.?o\.?\d*(?:$|[\s_\-\.])", fname_low))
        or "boletim" in fname_low
        or "ocorrencia" in fname_low
        or "ocorrência" in fname_low
    ):
        return ("POLICE_REPORT_BO", 85.0)

    if (
        "comprovante de residencia" in low
        or "comprovante de residência" in low
        or "conta de energia" in low
        or "conta de luz" in low
        or "energia eletrica" in low
        or "energia elétrica" in low
        or ("residencia" in fname_low or "residência" in fname_low)
        or bool(re.search(r"(?:^|[\s_\-])cr\d*(?:$|[\s_\-\.])", fname_low))
        or (
            ("fatura" in low or "vencimento" in low)
            and ("cep" in low or "endereco" in low or "endereço" in low or "kwh" in low or "telecom" in low)
        )
    ):
        return ("PROOF_OF_ADDRESS", 72.0)

    if (
        "danfe" in low
        or "documento auxiliar da nota fiscal" in low
        or "chave de acesso" in low
        or "nota fiscal eletronica" in low
        or "nota fiscal eletrônica" in low
        or bool(re.search(r"(?:^|[\s_\-])(?:nf|nfe|danfe)\d*(?:$|[\s_\-\.])", fname_low))
        or "nota fiscal" in fname_low
    ):
        return ("NOTA_FISCAL", 88.0)

    if (
        "carteira de identidade" in low
        or "registro geral" in low
        or "carteira nacional de habilitacao" in low
        or "carteira nacional de habilitação" in low
        or "data de expedicao" in low
        or "data de expedição" in low
        or "filiacao" in low
        or "filiação" in low
        or "naturalidade" in low
        or "identifica" in fname_low
        or bool(re.search(r"(?:^|[\s_\-])(?:rg|cnh|id)\d*(?:$|[\s_\-\.])", fname_low))
    ):
        return ("IDENTITY_DOCUMENT", 75.0)

    score = 0.0
    if "invoice" in fname_low or fname_low.startswith("in") or "inv" in fname_low:
        score += 25.0
    if re.search(r"(?<!or\s)(?<!for\s)\b(?:credit\s+memo|credit\s+note)\b", low):
        score += 45.0
        return ("CREDIT_MEMO", score)
    if re.search(r"\binvoice\s*(?:#|no|number|nbr|date|total)\b", low):
        score += 40.0
    if re.search(r"\b(?:total\s+amount\s+due|amount\s+due|total\s+due|balance\s+due|invoice\s+total)\b", low):
        score += 30.0
    if re.search(r"\bremit\s+to\b|\bplease\s+remit\b|\bmake\s+checks\s+payable\b", low):
        score += 20.0
    if "scale ticket" in low or "gross wt" in low or "tare wt" in low or ("bill of lading" in low and score < 40.0):
        return ("PO_BACKUP", max(score - 25.0, 10.0))
    if "statement date" in low or "account summary" in low or "statement for open invoices" in low:
        return ("STATEMENT", max(score, 50.0))

    return ("PRIMARY_INVOICE", max(score, 40.0))


def detect_and_split_multi_invoice_pdf(
    pdf_path: Path,
    page_texts: list[str],
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Detect when a multi-page PDF contains multiple distinct invoices and optionally
    split them into individual PDFs via `pdf_fast_write.extract_pages`.
    """
    if len(page_texts) <= 1:
        return [{"pages": [1], "invoice_number": None}]

    page_inv_nums: list[str | None] = []
    for ptxt in page_texts:
        m = re.search(
            r"(?:Invoice\s*(?:#|No\.?|Number)\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\-_]{2,20})",
            ptxt,
            re.IGNORECASE,
        )
        inv_no = m.group(1).strip() if m else None
        if inv_no and inv_no.upper() in ("DATE", "NUMBER", "TOTAL", "PAGE", "SUMMARY", "AMOUNT"):
            inv_no = None
        page_inv_nums.append(inv_no)

    segments: list[dict[str, Any]] = []
    curr_pages: list[int] = [1]
    curr_inv: str | None = page_inv_nums[0]

    for idx in range(1, len(page_texts)):
        pno = idx + 1
        inv = page_inv_nums[idx]
        if inv and curr_inv and inv != curr_inv:
            segments.append({"pages": curr_pages, "invoice_number": curr_inv})
            curr_pages = [pno]
            curr_inv = inv
        else:
            curr_pages.append(pno)
            if not curr_inv and inv:
                curr_inv = inv
    segments.append({"pages": curr_pages, "invoice_number": curr_inv})

    if len(segments) > 1 and out_dir is not None and pdf_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        src_bytes = pdf_path.read_bytes()
        for s_idx, seg in enumerate(segments):
            try:
                split_bytes = pdf_fast_write.extract_pages(src_bytes, seg["pages"])
                split_file = out_dir / f"{pdf_path.stem}__split_{s_idx+1}.pdf"
                split_file.write_bytes(bytes(split_bytes))
                seg["split_path"] = str(split_file)
            except Exception:
                pass

    return segments


def _extract_pdf_pages_with_self_healing_ocr(
    pdf_path: Path,
    oar_bin: str | None = None,
    models_dir: str | None = None,
    strip_headers_footers: bool = False,
) -> tuple[str, int, bool]:
    """
    Extract native text from a PDF via `pdf_fast`, repair any shifted Identity-H / Type0
    CMap ASCII offsets (`repair_shifted_cmap_text`), and automatically route any scanned or
    font-corrupted pages through 5-step visual preprocessing (`render_pdf_page_png`) + `oar-ocr-cli`.
    """
    resolved = pdf_path.resolve()
    pages_text: list[str] = []
    pages_runs: list[list[dict[str, Any]]] = []

    pcount = 0
    try:
        scanner = pdf_fast.PdfScanner(str(resolved))
        pc_attr = getattr(scanner, "page_count", 0)
        pcount = int(pc_attr() if callable(pc_attr) else pc_attr)
    except Exception:
        pcount = 0

    if pcount > 0:
        for idx in range(pcount):
            ptxt, pruns = extract_ordered_pdf_page_text(
                resolved,
                page_index=idx,
                strip_headers_footers=strip_headers_footers,
            )
            pages_text.append(ptxt)
            pages_runs.append(pruns)
    else:
        try:
            full_txt = pdf_fast.parse_pdf_text(str(resolved))
            for p_str in full_txt.split("\f"):
                repaired, _ = repair_shifted_cmap_text(p_str)
                pages_text.append(repaired)
                pages_runs.append([])
        except Exception:
            try:
                parsed_json = json.loads(pdf_fast.parse_pdf_json(str(resolved)))
                for p_obj in parsed_json.get("pages") or []:
                    repaired, _ = repair_shifted_cmap_text(p_obj.get("text") or "")
                    pages_text.append(repaired)
                    pages_runs.append([])
            except Exception:
                pages_text = [""]
                pages_runs = [[]]

    page_count = max(len(pages_text), 1)
    needs_ocr_any = False
    ocr_jobs: list[dict[str, str]] = []
    final_pages: list[str] = []

    with tempfile.TemporaryDirectory(prefix="sovereign_ocr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx in range(len(pages_text)):
            ptxt = pages_text[idx] or ""
            pruns = pages_runs[idx] if idx < len(pages_runs) else []
            if is_corrupted_or_scanned_pdf_page(ptxt, pruns):
                needs_ocr_any = True
                png_bytes = render_pdf_page_png(resolved, idx, preprocess=True)
                if png_bytes:
                    img_file = tmp_path / f"page_{idx}.png"
                    img_file.write_bytes(png_bytes)
                    ocr_jobs.append({"id": str(idx), "image_path": str(img_file), "path": str(img_file)})
            final_pages.append(ptxt)

        if ocr_jobs:
            ocr_out = run_batch_ocr(ocr_jobs, oar_bin=oar_bin, models_dir=models_dir, cache_dir=tmp_path)
            for job in ocr_jobs:
                p_idx = int(job["id"])
                runs = ocr_out.get(job["id"]) or []
                reconstructed = reconstruct_text_from_ocr_runs(
                    runs,
                    strip_headers_footers=strip_headers_footers,
                )
                if reconstructed.strip():
                    final_pages[p_idx] = reconstructed

    return "\n\n".join(p for p in final_pages if p.strip()), page_count, needs_ocr_any


def unpack_document_file(
    path: Path,
    oar_bin: str | None = None,
    models_dir: str | None = None,
) -> list[ExtractedDocumentUnit]:
    """
    Unpack any single document file (`.pdf`, `.msg`, `.eml`, `.docx`, `.xlsx`, `.xlsm`,
    `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`, `.txt`)
    into one or more classified `ExtractedDocumentUnit` objects.
    """
    suffix = path.suffix.upper()
    fname = path.name
    units: list[ExtractedDocumentUnit] = []

    try:
        if path.exists() and path.stat().st_size == 0:
            return [
                ExtractedDocumentUnit(
                    unit_id=fname,
                    filename=fname,
                    source_kind="EMPTY",
                    role="EMPTY_FILE",
                    text="",
                    page_count=0,
                    needs_ocr=False,
                    score=0.0,
                )
            ]
    except Exception:
        pass

    if suffix in (".MSG", ".EML"):
        parsed = json.loads(email_fast.parse_email_file_json(str(path)))
        subj = (parsed.get("subject") or "").strip()
        body = (parsed.get("body_text") or "").strip()
        sender = (parsed.get("from") or "").strip()
        recip = (parsed.get("to") or "").strip()
        full_email_txt = f"Subject: {subj}\nFrom: {sender}\nTo: {recip}\n{body}".strip()
        if full_email_txt:
            role, score = classify_document_unit(fname, full_email_txt, "MSG_BODY")
            units.append(
                ExtractedDocumentUnit(
                    unit_id=f"{fname}::body",
                    filename=fname,
                    source_kind="MSG_BODY",
                    role=role,
                    text=full_email_txt,
                    email_meta=parsed,
                    score=score,
                )
            )
        with tempfile.TemporaryDirectory(prefix="sovereign_att_") as tmp_dir:
            for att in parsed.get("attachments") or []:
                aname = att.get("filename") or "attachment.bin"
                data_bytes = bytes(att.get("data") or [])
                if not data_bytes:
                    continue
                att_path = Path(tmp_dir) / aname
                att_path.write_bytes(data_bytes)
                sub_units = unpack_document_file(att_path, oar_bin=oar_bin, models_dir=models_dir)
                for su in sub_units:
                    su.unit_id = f"{fname}::{su.filename}"
                    su.source_kind = f"MSG_{su.source_kind}"
                    su.email_meta = parsed
                    units.append(su)
        return units

    if suffix == ".PDF":
        txt, pcount, needed_ocr = _extract_pdf_pages_with_self_healing_ocr(
            path, oar_bin=oar_bin, models_dir=models_dir
        )
        role, score = classify_document_unit(fname, txt, "PDF")
        return [
            ExtractedDocumentUnit(
                unit_id=fname,
                filename=fname,
                source_kind="PDF",
                role=role,
                text=txt,
                page_count=pcount,
                needs_ocr=needed_ocr,
                score=score,
            )
        ]

    if suffix in (".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF", ".WEBP"):
        raw_bytes = path.read_bytes()
        proc_png, _ = preprocess_document_image_bytes(raw_bytes)
        txt = ""
        with tempfile.TemporaryDirectory(prefix="sovereign_img_ocr_") as tmp_dir:
            tmp_img = Path(tmp_dir) / "preprocessed.png"
            tmp_img.write_bytes(proc_png if proc_png else raw_bytes)
            ocr_out = run_batch_ocr(
                [{"id": "0", "image_path": str(tmp_img), "path": str(tmp_img)}],
                oar_bin=oar_bin,
                models_dir=models_dir,
                cache_dir=Path(tmp_dir),
            )
            runs = ocr_out.get("0") or []
            txt = reconstruct_text_from_ocr_runs(runs)
        role, score = classify_document_unit(fname, txt, "IMAGE")
        return [
            ExtractedDocumentUnit(
                unit_id=fname,
                filename=fname,
                source_kind="IMAGE",
                role=role,
                text=txt,
                page_count=1,
                needs_ocr=True,
                score=score,
            )
        ]

    if suffix == ".DOCX":
        txt = extract_docx_text(path)
        role, score = classify_document_unit(fname, txt, "DOCX")
        return [
            ExtractedDocumentUnit(
                unit_id=fname,
                filename=fname,
                source_kind="DOCX",
                role=role,
                text=txt,
                score=score,
            )
        ]

    if suffix in (".XLSX", ".XLSM"):
        txt = extract_xlsx_text(path)
        kind = suffix.lstrip(".")
        role, score = classify_document_unit(fname, txt, kind)
        return [
            ExtractedDocumentUnit(
                unit_id=fname,
                filename=fname,
                source_kind=kind,
                role=role,
                text=txt,
                score=score,
            )
        ]

    txt = path.read_text(errors="ignore")
    kind = suffix.lstrip(".") or "TXT"
    role, score = classify_document_unit(fname, txt, kind)
    return [
        ExtractedDocumentUnit(
            unit_id=fname,
            filename=fname,
            source_kind=kind,
            role=role,
            text=txt,
            score=score,
        )
    ]


def unpack_bundle_directory(
    bundle_dir: Path,
    bundle_id: str | None = None,
    entity_code: str = "",
    ontology_engine: Any = None,
    oar_bin: str | None = None,
    models_dir: str | None = None,
) -> IngestedBundle:
    """
    Unpack all documents inside `bundle_dir` into a scored `IngestedBundle`, routing
    primary vs. supporting documents with optional active `ontology_engine` boosting.
    """
    bid = bundle_id or bundle_dir.name
    bundle = IngestedBundle(bundle_id=bid, entity_code=entity_code)

    for p in sorted(bundle_dir.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            bundle.units.extend(unpack_document_file(p, oar_bin=oar_bin, models_dir=models_dir))

    if ontology_engine is not None:
        for u in bundle.units:
            if u.role in ("PRIMARY_INVOICE", "CREDIT_MEMO", "STATEMENT") and u.text:
                vm = ontology_engine.match_vendor(u.text, u.text, entity_code, primary_role=u.role)
                u.score += float(vm.get("confidence", 0.0)) * 20.0

    if bundle.units:
        bundle.units.sort(key=lambda u: u.score, reverse=True)
        bundle.primary_unit = bundle.units[0]
        bundle.combined_text = "\n\n=== DOCUMENT UNIT BREAK ===\n\n".join(
            f"[{u.source_kind} | {u.role} | {u.filename}]\n{u.text}" for u in bundle.units if u.text
        )
    return bundle
