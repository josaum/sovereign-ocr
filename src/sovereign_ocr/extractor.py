"""
Hybrid Neuro-Symbolic Document Field Extractor (`gliner2` + `gliner_fast_py` + Active Ontology + Structural Rules).

Combines:
1. Strongly-typed Pydantic v2 schemas (`InvoiceHeaderSchema`, `RemitToPayeeSchema`,
   `DocumentClassificationSchema`, `DocumentExtractionEnvelope`) compiled into:
   - `gliner2.Schema` (`StructureInput`, `FieldInput`, `ClassificationInput`, `RegexValidator`)
     for `fastino/gliner2.5-multi-v1` (`BoundaryExtractor`).
   - `gliner_fast_py.GlinerEngine` (`gliner_multi_v2.1` FP32 ONNX) zero-shot neural entity extraction.
2. Active runtime Ontology resolution (`SovereignOntologyEngine`) to disambiguate customer account
   numbers vs. invoice references, multi-attachment `.MSG` approvals, and fleet/utility statements.
3. Domain-agnostic commercial normalization rules:
   - Space/underscore/slash/colon stripping, pure-digit leading-zero stripping, and OCR
     zero/O/I/l correction in invoice references.
   - Utility / Statement `<AccountNo> <MMYY>` and Payment Request `<MDDYY>` reference derivation.
   - Credit Memo (`KG`) vs. Credit Invoice (`KR` with negative amount) vs. Vendor Invoice (`KR`).
   - Current / New Charges vs. Balance Forward discrimination.
   - Sales Tax (`I1`) vs. Consumer Use Tax (`U1`) vs. Tax-Exempt (`""`) classification.
   - Freight > 20% tolerance check.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any

try:
    import gliner_fast_py
except ImportError:
    gliner_fast_py = None  # type: ignore[assignment]

from sovereign_ocr.ingest import IngestedBundle
from sovereign_ocr.schemas import (
    DEFAULT_GLINER25_SNAPSHOT_DIR,
    DEFAULT_GLINER_ONNX_MODEL_DIR,
    DocumentExtractionEnvelope,
    build_gliner2_schema_from_pydantic,
    configure_offline_model_env,
    get_gliner_fast_labels_from_pydantic,
    parse_gliner2_output_to_pydantic,
)

GLINER_LABELS = get_gliner_fast_labels_from_pydantic()

BAD_INV_TOKENS = {
    "DATE", "TOTAL", "PAGE", "NUMBER", "AMOUNT", "TERMS", "SUMMARY",
    "NULL", "ORIGINAL", "CUSTOMER", "ACCOUNT", "BILL", "SHIP", "ORDER",
    "INVOICE", "INVOICENO", "INVOICENUMBER", "INCLUDED", "ENCLOSED",
    "PAYMENT", "DUE", "NET", "NET30", "NET15", "NET60", "USD", "CAD",
    "TBD", "NONE", "NA", "N/A", "FAX", "PHONE", "TEL", "EMAIL", "TO",
    "FROM", "SOLD", "REMIT", "SERVICE", "LOCATION", "DESCRIPTION",
}


class SovereignDocumentExtractor:
    """
    Domain-agnostic neuro-symbolic document extractor driven by Pydantic v2 models
    (`InvoiceHeaderSchema`, `RemitToPayeeSchema`, `DocumentClassificationSchema`)
    with `gliner2` (`fastino/gliner2.5-multi-v1`) and `gliner_fast_py` (`gliner_multi_v2.1` FP32 ONNX),
    actively grounded against any pluggable `SovereignOntologyEngine`.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_GLINER_ONNX_MODEL_DIR,
        gliner25_dir: str = DEFAULT_GLINER25_SNAPSHOT_DIR,
        use_gliner: bool = True,
        use_gliner25_pydantic: bool = False,
    ) -> None:
        configure_offline_model_env()
        self.use_gliner = use_gliner
        self.use_gliner25_pydantic = use_gliner25_pydantic
        self.gliner25_dir = gliner25_dir
        self._engine: Any = None
        self._gliner25_extractor: Any = None
        self._gliner2_schema: Any = None

        try:
            self._gliner2_schema = build_gliner2_schema_from_pydantic()
        except Exception:
            self._gliner2_schema = None

        if use_gliner:
            try:
                self._engine = gliner_fast_py.GlinerEngine(model_dir, variant="fp32")
            except Exception:
                self._engine = None

        if use_gliner25_pydantic and os.path.isdir(gliner25_dir):
            self._ensure_gliner25_loaded()

    def _ensure_gliner25_loaded(self) -> Any:
        """Lazily load `fastino/gliner2.5-multi-v1` (`BoundaryExtractor`) in local offline mode."""
        if self._gliner25_extractor is not None:
            return self._gliner25_extractor
        if not os.path.isdir(self.gliner25_dir):
            return None
        try:
            import torch
            from gliner2 import AutoExtractor

            configure_offline_model_env()
            ext = AutoExtractor.from_pretrained(self.gliner25_dir, local_files_only=True)
            if torch.backends.mps.is_available():
                ext = ext.to("mps")
            self._gliner25_extractor = ext
        except Exception:
            self._gliner25_extractor = None
        return self._gliner25_extractor

    def extract_pydantic_gliner25_batch(
        self,
        texts: list[str],
        threshold: float = 0.35,
        batch_size: int = 8,
    ) -> list[DocumentExtractionEnvelope]:
        """
        Run `gliner2` (`fastino/gliner2.5-multi-v1` `BoundaryExtractor`) over a batch of
        document texts using the compiled Pydantic `gliner2.Schema`, returning validated
        `DocumentExtractionEnvelope` Pydantic instances.
        """
        ext = self._ensure_gliner25_loaded()
        if ext is None or self._gliner2_schema is None or not texts:
            return [DocumentExtractionEnvelope() for _ in texts]
        snippets = [t[:2200] if t.strip() else "INVOICE" for t in texts]
        raw_results = ext.batch_extract(
            snippets,
            self._gliner2_schema,
            batch_size=batch_size,
            threshold=threshold,
            include_confidence=True,
        )
        return [parse_gliner2_output_to_pydantic(r) for r in raw_results]

    def extract_gliner_entities(self, text: str) -> dict[str, list[dict[str, Any]]]:
        """
        Run `gliner_fast_py` (`predict_entities`) over the primary document text header/footer
        using the Pydantic model field labels (`GLINER_LABELS`).
        """
        if not self._engine or not text.strip():
            return {}
        snippet = text[:1100]
        try:
            spans = self._engine.predict_entities(snippet, GLINER_LABELS[:9], threshold=0.35)
        except Exception:
            return {}
        by_label: dict[str, list[dict[str, Any]]] = {}
        for s in spans:
            lbl = s.get("label", "")
            by_label.setdefault(lbl, []).append({
                "text": s.get("text", "").strip(),
                "score": float(s.get("score", 0.0)),
                "start": int(s.get("start", 0)),
                "end": int(s.get("end", 0)),
            })
        for lbl in by_label:
            by_label[lbl].sort(key=lambda x: x["score"], reverse=True)
        return by_label

    @staticmethod
    def normalize_invoice_reference(raw_ref: str, strip_leading_zeros: bool = True) -> str:
        """
        Normalize an extracted invoice reference / document number:
        - Preserve single space only if utility `<9-16 digits> <4 digits MMYY>` format.
        - Strip spaces, underscores, slashes, and colons.
        - Strip trailing column-header artifacts (`ORDER`, `FAX`, `DATE`, `INVOICE`, `PAGE`, `TERMS`).
        - Fix OCR O/0 and l/I confusion inside structured prefixes (`CRO25100` -> `CR025100`,
          `POS 534` -> `P0S534`, `034892l-1` -> `034892I-1`).
        - Optionally strip leading zeros on pure-digit invoice numbers (`071557` -> `71557`).
        """
        s = (raw_ref or "").strip().upper()
        if not s:
            return ""
        if re.match(r"^\d[\d\s]{8,16}\s+\d{4}$", s):
            parts = s.split()
            mmyy = parts[-1]
            acct = "".join(parts[:-1])
            if acct.isdigit() and mmyy.isdigit() and len(acct) >= 9:
                return f"{acct} {mmyy}"
        s = re.sub(r"[\s_/:]+", "", s)
        for trail in ("ORDER", "FAX", "DATE", "INVOICE", "PAGE", "TERMS"):
            if s.endswith(trail) and len(s) > len(trail) + 2:
                s = s[: -len(trail)]
        if s.startswith("CRO") and s[3:].isdigit():
            s = "CR0" + s[3:]
        if s.startswith("POS") and s[3:].isdigit():
            s = "P0S" + s[3:]
        if re.match(r"^\d+L-\d+$", s):
            s = s.replace("L-", "I-")
        if strip_leading_zeros and s.isdigit():
            s = s.lstrip("0") or "0"
        return s

    @staticmethod
    def validate_cpf(raw: str) -> bool:
        """Validate an 11-digit Brazilian CPF using standard Mod-11 check digits."""
        d = re.sub(r"\D", "", raw or "")
        if len(d) != 11 or len(set(d)) == 1:
            return False
        s1 = sum(int(d[i]) * (10 - i) for i in range(9))
        r1 = (s1 * 10) % 11
        if (0 if r1 == 10 else r1) != int(d[9]):
            return False
        s2 = sum(int(d[i]) * (11 - i) for i in range(10))
        r2 = (s2 * 10) % 11
        return (0 if r2 == 10 else r2) == int(d[10])

    @staticmethod
    def validate_cnpj(raw: str) -> bool:
        """Validate a 14-digit Brazilian CNPJ using standard Mod-11 weights."""
        d = re.sub(r"\D", "", raw or "")
        if len(d) != 14 or len(set(d)) == 1:
            return False
        w1 = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
        w2 = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
        r1 = sum(int(d[i]) * w1[i] for i in range(12)) % 11
        d1 = 0 if r1 < 2 else 11 - r1
        if d1 != int(d[12]):
            return False
        r2 = sum(int(d[i]) * w2[i] for i in range(13)) % 11
        d2 = 0 if r2 < 2 else 11 - r2
        return d2 == int(d[13])

    @staticmethod
    def validate_imei_luhn(raw: str) -> bool:
        """Validate a 15-digit mobile device IMEI using the Luhn check digit."""
        d = re.sub(r"\D", "", raw or "")
        if len(d) != 15 or len(set(d)) == 1:
            return False
        total = 0
        for i, ch in enumerate(d):
            n = int(ch)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    @staticmethod
    def validate_nfe_access_key(raw: str) -> bool:
        """Validate a 44-digit Brazilian NF-e Chave de Acesso (IBGE UF prefix + Mod-11 DV)."""
        d = re.sub(r"\D", "", raw or "")
        if len(d) != 44:
            return False
        valid_ufs = {
            "11", "12", "13", "14", "15", "16", "17",
            "21", "22", "23", "24", "25", "26", "27", "28", "29",
            "31", "32", "33", "35", "41", "42", "43",
            "50", "51", "52", "53",
        }
        if d[:2] not in valid_ufs:
            return False
        weights = [2, 3, 4, 5, 6, 7, 8, 9]
        total = 0
        for idx, ch in enumerate(reversed(d[:43])):
            total += int(ch) * weights[idx % 8]
        rem = total % 11
        dv = 0 if rem in (0, 1) else 11 - rem
        return dv == int(d[43])

    @classmethod
    def extract_documentoscopia_facts(cls, text: str) -> dict[str, Any]:
        """
        Extract and cryptographically/algebraically validate multilingual KYC, fiscal,
        and insurance claim entities (CPFs, CNPJs, IMEIs, 44-digit NF-e access keys,
        CEPs, RG numbers, and BO theft/robbery occurrence typification).
        """
        if not text:
            return {
                "cpfs": [],
                "cnpjs": [],
                "imeis": [],
                "nfe_access_keys": [],
                "ceps": [],
                "rgs": [],
                "occurrence_type": None,
            }

        cpfs: list[str] = []
        for m in re.finditer(r"\b(\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[\-\s/\.]\d{2})\b", text):
            digits = re.sub(r"\D", "", m.group(1))
            if cls.validate_cpf(digits) and digits not in cpfs:
                cpfs.append(digits)
        for m in re.finditer(r"(?:CPF|C\.P\.F\.?)[\s:\-]*(\d{11})\b", text, re.IGNORECASE):
            digits = m.group(1)
            if cls.validate_cpf(digits) and digits not in cpfs:
                cpfs.append(digits)

        cnpjs: list[str] = []
        for m in re.finditer(r"\b(\d{2}\.?\d{3}\.?\d{3}/\d{4}\-?\d{2})\b", text):
            digits = re.sub(r"\D", "", m.group(1))
            if cls.validate_cnpj(digits) and digits not in cnpjs:
                cnpjs.append(digits)
        for m in re.finditer(r"(?:CNPJ|C\.N\.P\.J\.?)[\s:\-]*(\d{14})\b", text, re.IGNORECASE):
            digits = m.group(1)
            if cls.validate_cnpj(digits) and digits not in cnpjs:
                cnpjs.append(digits)

        imeis: list[str] = []
        for m in re.finditer(r"(?:IMEI|N[úu]mero\s+de\s+S[ée]rie)\s*[:\-]?\s*(\d[\d\s\.\-]{13,18}\d)", text, re.IGNORECASE):
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) == 15 and digits not in imeis:
                imeis.append(digits)
        for m in re.finditer(r"\b(35\d{13}|86\d{13})\b", text):
            digits = m.group(1)
            if cls.validate_imei_luhn(digits) and digits not in imeis:
                imeis.append(digits)

        nfe_keys: list[str] = []
        for m in re.finditer(r"(?:\b\d{4}[\s\.\-]*){10}\d{4}\b", text):
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) == 44 and cls.validate_nfe_access_key(digits) and digits not in nfe_keys:
                nfe_keys.append(digits)

        ceps: list[str] = []
        for m in re.finditer(r"\b(\d{2}\.?\d{3}\-\d{3})\b", text):
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) == 8 and digits not in ceps:
                ceps.append(digits)

        rgs: list[str] = []
        for m in re.finditer(
            r"(?:Registro\s+Geral|\bRG\b)\s*[:\-]?\s*([0-9]{1,2}\.?[0-9]{3}\.?[0-9]{3}[\-\s]?[0-9X]|[0-9]{7,11})",
            text,
            re.IGNORECASE,
        ):
            val = re.sub(r"[^0-9X]", "", m.group(1).upper())
            if 6 <= len(val) <= 11 and val not in rgs and val not in cpfs:
                rgs.append(val)
        for m in re.finditer(r"\b(\d{2}\.\d{3}\.\d{3}-[0-9X]{1,2})\b", text, re.IGNORECASE):
            val = re.sub(r"[^0-9X]", "", m.group(1).upper())
            if 6 <= len(val) <= 11 and val not in rgs and val not in cpfs:
                rgs.append(val)

        low = text.lower()
        occ_type: str | None = None
        if any(
            k in low
            for k in (
                "roubo",
                "assalto",
                "arma de fogo",
                "grave ameaca",
                "grave ameaça",
                "mediante violencia",
                "mediante violência",
                "anunciou o assalto",
            )
        ):
            occ_type = "roubo"
        elif any(
            k in low
            for k in (
                "furto qualificado",
                "abuso de confianca",
                "abuso de confiança",
                "destruicao ou rompimento",
                "destruição ou rompimento",
                "escalada",
                "mediante fraude",
            )
        ):
            occ_type = "furto_qualificado"
        elif "furto" in low or "subtraiu" in low or "subtraido" in low or "subtraído" in low:
            occ_type = "furto_simples"

        return {
            "cpfs": cpfs,
            "cnpjs": cnpjs,
            "imeis": imeis,
            "nfe_access_keys": nfe_keys,
            "ceps": ceps,
            "rgs": rgs,
            "occurrence_type": occ_type,
        }

    @staticmethod
    def extract_dates(text: str) -> list[str]:
        """Extract candidate dates normalized to ISO `YYYY-MM-DD` (supporting both MM/DD/YYYY and DD/MM/YYYY)."""
        out: list[str] = []
        norm_txt = re.sub(r"\xa0", " ", text)
        low_txt = norm_txt.lower()
        is_dmy_locale = any(
            w in low_txt
            for w in (
                "emissão", "emissao", "expedição", "expedicao", "nascimento",
                "vencimento", "sinistro", "ocorrência", "ocorrencia", "boletim",
                "nota fiscal", "danfe", "cpf", "cnpj", "r$",
            )
        )

        def _resolve_slash_date(a: int, b: int, yy: str, force_dmy: bool = False) -> str | None:
            yr = (2000 + int(yy) if int(yy) < 50 else 1900 + int(yy)) if len(yy) == 2 else int(yy)
            if not (1920 <= yr <= 2035):
                return None
            if a > 12 and 1 <= b <= 12 and a <= 31:
                mm, dd = b, a
            elif b > 12 and 1 <= a <= 12 and b <= 31:
                mm, dd = a, b
            elif 1 <= a <= 12 and 1 <= b <= 12:
                mm, dd = (b, a) if (force_dmy or is_dmy_locale) else (a, b)
            else:
                return None
            return f"{yr:04d}-{mm:02d}-{dd:02d}"

        labeled_pats = [
            (
                r"(?:Invoice\s+Date|Inv\.?\s+Date|Bill\s+Date|Statement\s+Date|Request\s+Date|Credit\s+Memo\s+Date|DATE)\s*[:\|]?\s*(\d{1,2})\s*[/\-\.]+\s*(\d{1,2})\s*[/\-\.]+\s*(\d{2,4})\b",
                False,
            ),
            (
                r"(?:Data\s+de\s+Emiss[ãa]o|Data\s+de\s+Expedi[çc][ãa]o|Data\s+de\s+Nascimento|Data\s+do\s+Fato|Data\s+Hora|Vencimento|Emiss[ãa]o|DATA)\s*[:\|]?\s*(\d{1,2})\s*[/\-\.]+\s*(\d{1,2})\s*[/\-\.]+\s*(\d{2,4})\b",
                True,
            ),
        ]
        for pat, force_dmy in labeled_pats:
            for m in re.finditer(pat, norm_txt, re.IGNORECASE):
                iso = _resolve_slash_date(int(m.group(1)), int(m.group(2)), m.group(3), force_dmy=force_dmy)
                if iso and iso not in out:
                    out.append(iso)

        en_month_pat = (
            r"(?:Invoice\s+Date|Inv\.?\s+Date|Bill\s+Date|DATE)?\s*[:\n\s]*\b"
            r"(January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(20\d{2})\b"
        )
        for m in re.finditer(en_month_pat, norm_txt, re.IGNORECASE):
            try:
                dt = datetime.strptime(f"{m.group(1)[:3]} {int(m.group(2))} {m.group(3)}", "%b %d %Y")
                iso = dt.strftime("%Y-%m-%d")
                if iso not in out:
                    out.append(iso)
            except Exception:
                pass

        for m in re.finditer(r"\b(\d{1,2})\s*[/\-\.]+\s*(\d{1,2})\s*[/\-\.]+\s*(\d{2,4})\b", norm_txt):
            iso = _resolve_slash_date(int(m.group(1)), int(m.group(2)), m.group(3))
            if iso and iso not in out:
                out.append(iso)

        for m in re.finditer(r"\b(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", norm_txt):
            iso = m.group(0)
            if iso not in out:
                out.append(iso)
        return out

    @staticmethod
    def extract_amounts(text: str) -> dict[str, Any]:
        """
        Extract total amount, current charges (ignoring prior balance forward),
        sales tax, freight, and discount from document text (supports both USD `$1,250.00`
        and BRL/European `R$ 8.699,00` formats).
        """
        current_charge_pats = [
            r"(?:Current\s+Charges|New\s+Charges|Total\s+Current\s+Charges|Current\s+Bill|Current\s+Amount\s+Due)\s*[:\$]*\s*\(?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2}|[0-9]+\.[0-9]{2})\)?",
            r"(?:Total\s+Invoice\s+Amount|Invoice\s+Total|Total\s+Amount\s+Due|Amount\s+Due|Balance\s+Due|Total\s+Due|Net\s+Amount|Pay\s+This\s+Amount|Total\s+Check\s+Request|Credit\s+Total|Total\s+Credit|TOTAL)\s*[:\$]*\s*(-?\(?\$?\s*(?:[0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2}|[0-9]+\.[0-9]{2})\)?)",
        ]
        candidates: list[float] = []
        for pat in current_charge_pats:
            for m in re.finditer(pat, text, re.IGNORECASE):
                raw = m.group(1).strip()
                is_neg = "(" in raw or "-" in raw
                clean = re.sub(r"[^0-9.]", "", raw)
                try:
                    val = float(clean)
                    if is_neg:
                        val = -abs(val)
                    candidates.append(val)
                except Exception:
                    pass

        # BRL / European comma-decimal labeled totals (e.g. VALOR TOTAL DA NOTA 8.699,00 or R$ 145,90)
        brl_pats = [
            r"(?:VALOR\s+TOTAL\s+DA\s+NOTA|VALOR\s+TOTAL\s+DOS\s+PRODUTOS|Total\s+a\s+Pagar|Valor\s+Total|Valor\s+do\s+Documento)\s*[:\sR\$]*\s*([0-9]{1,3}(?:\.[0-9]{3})*,[0-9]{2}|[0-9]+,[0-9]{2})",
            r"R\$\s*([0-9]{1,3}(?:\.[0-9]{3})*,[0-9]{2}|[0-9]+,[0-9]{2})",
        ]
        for pat in brl_pats:
            for m in re.finditer(pat, text, re.IGNORECASE):
                raw_brl = m.group(1).strip().replace(".", "").replace(",", ".")
                try:
                    val = float(raw_brl)
                    if val > 0 and val not in candidates:
                        candidates.append(val)
                except Exception:
                    pass

        if not candidates:
            for m in re.finditer(r"(?<!R)\$\s*(-?[0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2})", text):
                try:
                    candidates.append(float(m.group(1).replace(",", "")))
                except Exception:
                    pass

        tax_val: float | None = None
        for m in re.finditer(
            r"(?:Sales\s+Tax|State\s+Tax|Total\s+Tax|Tax\s+Amount|\bTax\b)\s*[:\$]*\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2})",
            text,
            re.IGNORECASE,
        ):
            try:
                tax_val = float(m.group(1).replace(",", ""))
                break
            except Exception:
                pass

        freight_val: float | None = None
        for m in re.finditer(
            r"(?:Freight|Shipping\s*&\s*Handling|Shipping|Delivery\s+Charge)\s*[:\$]*\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2})",
            text,
            re.IGNORECASE,
        ):
            try:
                fv = float(m.group(1).replace(",", ""))
                if fv > 0:
                    freight_val = fv
                    break
            except Exception:
                pass

        discount_val: float | None = None
        for m in re.finditer(
            r"(?:Discount|Less\s+Discount)\s*[:\$]*\s*-?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2})",
            text,
            re.IGNORECASE,
        ):
            try:
                dv = float(m.group(1).replace(",", ""))
                if dv > 0:
                    discount_val = dv
                    break
            except Exception:
                pass

        return {
            "total_candidates": candidates,
            "primary_total": candidates[0] if candidates else None,
            "tax": tax_val,
            "freight": freight_val,
            "discount": discount_val,
        }

    @classmethod
    def extract_invoice_reference(
        cls,
        primary_text: str,
        full_text: str,
        extracted_date: str = "",
        gliner_spans: dict[str, list[dict[str, Any]]] | None = None,
    ) -> tuple[str, str]:
        """Extract and normalize the invoice/document reference number from text and GLiNER spans."""
        pats = [
            r"(?:Invoice\s*(?:#|No\.?|Number|Nbr)|Credit\s+Memo\s*(?:#|No\.?|Number)|Bill\s*(?:#|No\.?|Number)|Reference\s*(?:#|No\.?))\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\._/]{1,22})",
            r"\bINV(?:OICE)?\s*[:#\-]\s*([A-Z0-9][A-Z0-9\-\._/]{2,20})",
        ]
        for scope_name, txt in (("PRIMARY_REGEX", primary_text), ("BUNDLE_REGEX", full_text)):
            for pat in pats:
                for m in re.finditer(pat, txt, re.IGNORECASE):
                    raw_cand = m.group(1).strip()
                    norm_cand = cls.normalize_invoice_reference(raw_cand)
                    if (
                        norm_cand
                        and norm_cand.upper() not in BAD_INV_TOKENS
                        and any(c.isdigit() for c in norm_cand)
                        and len(norm_cand) >= 2
                    ):
                        return norm_cand, scope_name

        if gliner_spans:
            for sp in gliner_spans.get("invoice_number") or []:
                norm_cand = cls.normalize_invoice_reference(sp.get("text", ""))
                if (
                    norm_cand
                    and norm_cand.upper() not in BAD_INV_TOKENS
                    and any(c.isdigit() for c in norm_cand)
                ):
                    return norm_cand, "GLINER_NEURAL_SPAN"

        if extracted_date and re.match(r"^20\d{2}-\d{2}-\d{2}$", extracted_date):
            yy = extracted_date[2:4]
            mm = int(extracted_date[5:7])
            dd = extracted_date[8:10]
            return f"{mm}{dd}{yy}", "PAYMENT_REQUEST_MDDYY_DERIVED"

        return "", "UNRESOLVED"

    def extract_bundle(
        self,
        bundle: IngestedBundle,
        ontology_engine: Any = None,
    ) -> dict[str, Any]:
        """
        Run hybrid neuro-symbolic extraction over an `IngestedBundle` and optionally
        ground against `ontology_engine`.
        """
        p_unit = bundle.primary_unit
        p_text = p_unit.text if p_unit else bundle.combined_text
        f_text = bundle.combined_text or p_text

        gliner_entities = self.extract_gliner_entities(p_text)
        dates = self.extract_dates(p_text) or self.extract_dates(f_text)
        extracted_date = dates[0] if dates else ""

        amts = self.extract_amounts(p_text)
        if amts["primary_total"] is None and f_text != p_text:
            amts = self.extract_amounts(f_text)

        ref, ref_source = self.extract_invoice_reference(
            p_text,
            f_text,
            extracted_date=extracted_date,
            gliner_spans=gliner_entities,
        )

        # Document type classification: Credit Memo (KG) vs Invoice (KR)
        p_role = p_unit.role if p_unit else "PRIMARY_INVOICE"
        is_credit_memo = p_role == "CREDIT_MEMO" or bool(
            re.search(r"(?<!or\s)(?<!for\s)\b(?:credit\s+memo|credit\s+note)\b", p_text, re.IGNORECASE)
        )
        doc_type = "KG" if is_credit_memo else "KR"

        total_val = amts["primary_total"]
        if doc_type == "KG" and total_val is not None and total_val > 0:
            total_val = -abs(total_val)

        po_match = re.search(
            r"(?:P\.?O\.?\s*(?:#|No\.?|Number)?|Purchase\s+Order)\s*[:\-]?\s*\b(4[57]\d{8}|[A-Z0-9\-]{4,16})\b",
            f_text,
            re.IGNORECASE,
        )
        po_number = po_match.group(1).strip() if po_match else ""
        if po_number.upper() in BAD_INV_TOKENS or not any(c.isdigit() for c in po_number):
            po_number = ""

        warnings: list[str] = []
        for u in bundle.units:
            if u.role == "EMPTY_FILE":
                warnings.append(f"EMPTY_FILE_DETECTED:{u.filename}")
        if p_role == "RECEIPT_EXCEPTION":
            warnings.append("RECEIPT_SUBMITTED_INSTEAD_OF_INVOICE")
        if len(bundle.multi_invoice_segments) > 1:
            warnings.append(f"MULTI_INVOICE_PDF_DETECTED({len(bundle.multi_invoice_segments)})")
        if amts["freight"] and total_val and abs(total_val) > 0:
            if float(amts["freight"]) / abs(float(total_val)) > 0.20:
                warnings.append("FREIGHT_EXCEEDS_20_PCT_THRESHOLD")

        doc_facts = self.extract_documentoscopia_facts(f_text)

        vendor_match = None
        if ontology_engine is not None:
            vendor_match = ontology_engine.match_vendor(
                p_text,
                f_text,
                bundle.entity_code,
                primary_role=p_role,
                gliner_spans=gliner_entities,
            )

        extracted = {
            "bundle_id": bundle.bundle_id,
            "doc_key": bundle.bundle_id,
            "docno": bundle.document_id or bundle.bundle_id,
            "cocd": bundle.entity_code,
            "year": bundle.fiscal_year,
            "extracted_doc_type": doc_type,
            "extracted_date": extracted_date,
            "extracted_reference": ref,
            "reference_source": ref_source,
            "extracted_amount": total_val,
            "extracted_tax": amts["tax"],
            "extracted_freight": amts["freight"],
            "extracted_discount": amts["discount"],
            "extracted_po": po_number,
            "primary_unit_kind": p_unit.source_kind if p_unit else "UNKNOWN",
            "primary_unit_role": p_role,
            "needs_ocr": any(u.needs_ocr for u in bundle.units),
            "warnings": warnings,
            "gliner_entities": gliner_entities,
            "documentoscopia_facts": doc_facts,
            "vendor_match": vendor_match,
        }

        if ontology_engine is not None:
            return ontology_engine.resolve_and_govern_bundle(extracted, f_text, primary_text=p_text)
        return extracted
