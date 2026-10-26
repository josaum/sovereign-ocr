"""
Pluggable Ontology-Grounded Resolution, OWL 2 RL Deductive Reasoning (`olaya`),
Formal Concept Analysis (`fca_fast`), Ontology Alignment (`align_fast_py`),
and Deterministic Cryptographic RDF Quad Provenance (`quad_fast`).

Domain-agnostic active runtime ontology grounding engine:
- Accepts any Turtle (`.ttl`), O-Laya JSON (`.json`), or in-memory master entity dictionary
  (`vendors`, `payees`, `facility_zips`, `base_iri`).
- Enforces universal ERP accounting invariants:
  1. Excludes AP reconciliation control accounts (`21100001`, `KOART='K'`) when selecting expense GL accounts.
  2. Aligns multi-entity Cost Centers (`KOSTL`) and Profit Centers (`PRCTR`) with the resolved Entity/Company Code prefix.
  3. Distinguishes vendor-billed Sales Tax (`I1`, cleared to `""` when `Tax = $0.00`) from self-assessed Consumer Use Tax (`U1`, retained when `Tax = $0.00`).
- Seals extracted facts and ontology edges into deterministic Arrow IPC / RDF quads via `quad_fast`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import olaya
import ontology_fast

try:
    import align_fast_py
except ImportError:
    align_fast_py = None  # type: ignore[assignment]
try:
    import owl_fast
except ImportError:
    owl_fast = None  # type: ignore[assignment]
try:
    import quad_fast
except ImportError:
    quad_fast = None  # type: ignore[assignment]


def norm_str(s: str) -> str:
    """Normalize organization names and addresses for multi-signal ontology matching."""
    s = (s or "").upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"\b(INC|LLC|LTD|CORP|CORPORATION|COMPANY|COMPANIES|CO|LP|LLP|PLC|HOLDINGS)\b\.?", " ", s)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


GENERIC_SINGLE_WORDS = {
    "LIFT", "MAIN", "STATES", "ABLE", "CENTRAL", "CITY", "SOUTHERN",
    "MATERIAL", "PREMIER", "NATIONAL", "AMERICAN", "UNITED", "FIRST",
    "GENERAL", "STANDARD", "QUALITY", "TOTAL", "BEST", "GREAT", "ROYAL",
    "ALL", "FILL", "PACKAGING", "SUPPLY", "PRODUCTS", "GROUP", "DIRECT",
    "SALES", "SERVICE", "TRUCK", "PARTS", "EQUIPMENT", "REPAIR",
}


def build_deterministic_quad_receipt(
    doc_iri: str,
    graph_iri: str,
    concept_iri: str,
    vendor_iri: str,
    reference: str,
    amount: float,
    source_artifact_iri: str,
    full_text: str,
    observed_at_micros: int = 1771545600_000_000,
    predicate_ns: str = "https://sovereign-ocr.local/ontology#",
) -> tuple[bytes, str]:
    """
    Seal extracted document facts into a deterministic Arrow IPC / RDF quad graph via
    `quad_fast.build_quad_ipc` and return `(quad_ipc_bytes, receipt_sha256_hex16)`.
    """
    content_sha = hashlib.sha256(full_text.encode("utf-8", errors="ignore")).digest()
    prov = {
        "source_artifact_iri": source_artifact_iri,
        "source_content_sha256": content_sha,
        "source_offset_start": 0,
        "source_offset_end": len(full_text),
        "observed_at_micros": int(observed_at_micros),
    }
    quads = [
        (
            ("iri", doc_iri),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            ("iri", concept_iri, None, None),
            ("iri", graph_iri),
            prov,
        ),
        (
            ("iri", doc_iri),
            f"{predicate_ns}hasVendor",
            ("iri", vendor_iri, None, None),
            ("iri", graph_iri),
            prov,
        ),
        (
            ("iri", doc_iri),
            f"{predicate_ns}invoiceReference",
            ("literal", str(reference), "http://www.w3.org/2001/XMLSchema#string", None),
            ("iri", graph_iri),
            prov,
        ),
        (
            ("iri", doc_iri),
            f"{predicate_ns}amountInLocalCurrency",
            ("literal", f"{float(amount):.2f}", "http://www.w3.org/2001/XMLSchema#decimal", None),
            ("iri", graph_iri),
            prov,
        ),
    ]
    if quad_fast is not None:
        quad_ipc_bytes = quad_fast.build_quad_ipc(quads)
    else:
        quad_ipc_bytes = json.dumps(
            quads,
            sort_keys=True,
            default=lambda b: b.hex() if isinstance(b, bytes) else str(b),
        ).encode("utf-8")
    receipt_hash = hashlib.sha256(quad_ipc_bytes).hexdigest()[:16]
    return quad_ipc_bytes, receipt_hash


class SovereignOntologyEngine:
    """
    Domain-agnostic neuro-symbolic ontology engine combining O-Laya OWL 2 RL reasoning,
    O-Laya System 1 `CandidateRouter`, `ontology_fast`, `owl_fast`, `align_fast_py`,
    and deterministic `quad_fast` provenance.

    Can be initialized from files (`ttl_path`, `olaya_json_path`, `master_kg_path`) OR
    directly in-memory via `master_data={"vendors": {...}, "payees": {...}, "facility_zips": [...]}`.
    """

    def __init__(
        self,
        ttl_path: Path | None = None,
        olaya_json_path: Path | None = None,
        master_kg_path: Path | None = None,
        master_data: dict[str, Any] | None = None,
        base_iri: str = "https://sovereign-ocr.local/ontology",
        reconciliation_gl_accounts: set[str] | None = None,
    ) -> None:
        self.base_iri = base_iri.rstrip("#/")
        self.reconciliation_gl_accounts = reconciliation_gl_accounts or {"21100001"}
        self.aligner = align_fast_py.Aligner() if align_fast_py is not None else None

        # 1. Load optional RDF/OWL Turtle ontology
        if ttl_path is not None and ttl_path.exists():
            ttl_bytes = ttl_path.read_bytes()
            raw_triples = ontology_fast.parse_triples(ttl_bytes)
            self.triples = json.loads(raw_triples) if isinstance(raw_triples, str) else raw_triples
            if owl_fast is not None:
                raw_chunks = owl_fast.parse_ontology_chunks(ttl_path.read_text(), "turtle")
                self.owl_chunks = json.loads(raw_chunks) if isinstance(raw_chunks, str) else raw_chunks
            else:
                self.owl_chunks = []
        else:
            self.triples = []
            self.owl_chunks = []

        # 2. Load master entity knowledge graph (from file or in-memory dict)
        kg: dict[str, Any] = {}
        if master_kg_path is not None and master_kg_path.exists():
            kg = json.loads(master_kg_path.read_text())
        elif master_data is not None:
            kg = dict(master_data)

        self.vendors: dict[str, dict[str, Any]] = kg.get("vendors", {})
        self.payees: dict[str, dict[str, Any]] = kg.get("payees", {})
        self.facility_zips: set[str] = set(kg.get("facility_zips") or [])
        self.fleet_card_vendor_id: str = str(kg.get("fleet_card_vendor_id") or "")

        # 3. Initialize O-Laya Ontology, OWL2RLReasoner & CandidateRouter
        if olaya_json_path is not None and olaya_json_path.exists():
            self.olaya_ontology = olaya.Ontology.from_json(olaya_json_path.read_text())
        else:
            self.olaya_ontology = self._build_olaya_ontology_from_vendors(self.vendors)

        self.reasoner = olaya.OWL2RLReasoner(self.olaya_ontology)
        self.closure = self.reasoner.compute_closure()
        self.router = olaya.CandidateRouter(self.olaya_ontology, max_candidates=12)

        # 4. Pre-index vendors for sub-millisecond multi-signal scoring
        self._vendor_index: list[dict[str, Any]] = []
        for vid, v in self.vendors.items():
            vname = v.get("vendor_name", "")
            nname = norm_str(vname)
            if nname.startswith("THE "):
                nname = nname[4:].strip()
            pname = norm_str(v.get("payee_name", ""))
            if pname.startswith("THE "):
                pname = pname[4:].strip()
            tokens = [t for t in nname.split() if len(t) >= 2]
            if not tokens:
                continue
            is_generic_single = len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_WORDS
            sig_tokens = [
                t
                for t in tokens
                if len(t) >= 3
                and t not in GENERIC_SINGLE_WORDS
                and t not in ("AND", "THE", "FOR", "SVCS", "SERVICES", "MFG", "ENTERPRISES", "SOLUTIONS", "SYSTEMS")
            ]
            v_zip5 = (v.get("postal_code") or "")[:5]
            p_zip5 = (v.get("payee_zip") or "")[:5]
            v_street = norm_str(v.get("street") or "")
            p_street = norm_str(v.get("payee_street") or "")
            v_city = norm_str(v.get("city") or "")

            street_prefixes: list[str] = []
            for st in (v_street, p_street):
                if st and len(st) >= 5:
                    st_toks = st.split()
                    if st_toks[:2] == ["PO", "BOX"] and len(st_toks) >= 3:
                        street_prefixes.append(" ".join(st_toks[:3]))
                    elif len(st_toks) >= 2 and st_toks[0] != "PO":
                        p2 = " ".join(st_toks[:2])
                        if len(p2) >= 5:
                            street_prefixes.append(p2)

            cust_accts = [
                str(ca).strip().upper()
                for ca in (v.get("customer_accounts") or [])
                if len(str(ca).strip()) >= 4
            ]
            approver_norm = norm_str(v.get("default_approver") or "")
            self._vendor_index.append(
                {
                    "vid": vid,
                    "vname": vname,
                    "nname": nname,
                    "nname_ns": nname.replace(" ", ""),
                    "pname": pname,
                    "pname_ns": pname.replace(" ", ""),
                    "is_generic_single": is_generic_single,
                    "sig_tokens": sig_tokens,
                    "v_zip5": v_zip5 if len(v_zip5) == 5 else "",
                    "p_zip5": p_zip5 if len(p_zip5) == 5 else "",
                    "v_city": v_city,
                    "street_prefixes": street_prefixes,
                    "cust_accts": cust_accts,
                    "approver_norm": approver_norm if len(approver_norm) >= 5 else "",
                    "b_aba": v.get("bank_routing") or "",
                    "b_acct": v.get("bank_account") or "",
                    "default_cocd": v.get("default_cocd") or "",
                    "cocds": set(v.get("cocds") or []),
                }
            )

    def _build_olaya_ontology_from_vendors(self, vendors: dict[str, dict[str, Any]]) -> Any:
        """Synthesize an in-memory `olaya.Ontology` from a master vendor dictionary."""
        concepts = [
            {
                "iri": f"{self.base_iri}#VendorInvoice",
                "curie": "ap:VendorInvoice",
                "label": "Vendor Invoice",
                "synonyms": ["invoice", "bill", "amount due", "remit to"],
                "definition": "Commercial vendor invoice requesting payment",
            },
            {
                "iri": f"{self.base_iri}#VendorCreditMemo",
                "curie": "ap:VendorCreditMemo",
                "label": "Vendor Credit Memo",
                "synonyms": ["credit memo", "credit note"],
                "definition": "Commercial credit memo reducing payable balance",
            },
        ]
        for vid, v in vendors.items():
            vname = v.get("vendor_name") or f"Vendor {vid}"
            syns = [s for s in [v.get("payee_name"), v.get("city"), v.get("postal_code")] if s]
            concepts.append({
                "iri": f"{self.base_iri}#Vendor_{vid}",
                "curie": f"Vendor_{vid}",
                "label": vname,
                "synonyms": syns,
                "definition": f"Master Vendor {vid} ({vname})",
            })
        payload = {
            "iri": self.base_iri,
            "version": "1.0.0",
            "concepts": concepts,
            "object_properties": [],
            "axioms": [],
        }
        return olaya.Ontology.from_json(json.dumps(payload))

    def match_vendor(
        self,
        primary_text: str,
        full_text: str,
        cocd: str = "",
        primary_role: str = "",
        gliner_spans: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """
        Resolve Vendor ID, Vendor Name, and Permitted Payee at runtime using the active Ontology:
        1. Optional Fleet Card Receipt Axiom (`RECEIPT_EXCEPTION`)
        2. Explicit Vendor ID on internal payment request / upload template
        3. O-Laya `CandidateRouter` + `gliner_fast_py` neural spans + Multi-Signal Ontology Verification
           (Street Address, 5-digit ZIP, City, Permitted Payee, Customer Account, ABA Routing/Account, Entity Code).
        """
        if primary_role == "RECEIPT_EXCEPTION" and self.fleet_card_vendor_id and self.fleet_card_vendor_id in self.vendors:
            v = self.vendors[self.fleet_card_vendor_id]
            return {
                "vendor_id": self.fleet_card_vendor_id,
                "vendor_name": v["vendor_name"],
                "match_method": "OWL2RL_FLEET_CARD_AXIOM",
                "confidence": 0.99,
                "vendor_record": v,
            }

        for pat in [
            r"C2:Vendor\s*\|.*?R2:[^\n]*?C2:(\d{5,10})(?:\.0)?\b",
            r"\bVendor\s*#\s*(\d{5,10})\b",
        ]:
            m = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
            if m and m.group(1) in self.vendors:
                vid = m.group(1)
                v = self.vendors[vid]
                return {
                    "vendor_id": vid,
                    "vendor_name": v["vendor_name"],
                    "match_method": "ONTOLOGY_EXPLICIT_VENDOR_ID",
                    "confidence": 1.0,
                    "vendor_record": v,
                }

        gliner_vendor_txts: list[str] = []
        if gliner_spans:
            for lbl in ("vendor_name", "remit_to_name", "customer_account_number"):
                for sp in (gliner_spans.get(lbl) or [])[:3]:
                    gliner_vendor_txts.append(sp.get("text", ""))
        query_str = " ".join(gliner_vendor_txts) + " " + primary_text[:800]
        olaya_routed = self.router.route_candidates(query_str, top_k=10)
        olaya_vendor_boost: dict[str, float] = {}
        for concept, r_score in olaya_routed:
            if concept.curie.startswith("Vendor_"):
                r_vid = concept.curie.split("_", 1)[1]
                olaya_vendor_boost[r_vid] = min(float(r_score) * 0.08, 0.45)

        p_norm = f" {norm_str(primary_text)} "
        p_head_norm = f" {norm_str(primary_text[:350])} "
        f_norm = f" {norm_str(full_text)} "
        p_nospace = p_norm.replace(" ", "")
        f_nospace = f_norm.replace(" ", "")
        up_full = full_text.upper()
        gliner_norm = f" {norm_str(' '.join(gliner_vendor_txts))} " if gliner_vendor_txts else ""

        best_vid = ""
        best_score = -1.0

        for item in self._vendor_index:
            vid = item["vid"]
            score = olaya_vendor_boost.get(vid, 0.0)
            nname = item["nname"]
            pname = item["pname"]

            if nname and len(nname) >= 4 and not item["is_generic_single"]:
                if f" {nname} " in p_norm:
                    score += 1.25
                    if f" {nname} " in p_head_norm:
                        score += 0.45
                    if p_norm.strip().startswith(nname):
                        score += 0.50
                elif f" {nname} " in f_norm:
                    score += 0.95
                elif len(nname) >= 8 and item["nname_ns"] in p_nospace:
                    score += 1.05
                elif len(nname) >= 8 and item["nname_ns"] in f_nospace:
                    score += 0.85
                if gliner_norm and f" {nname} " in gliner_norm:
                    score += 0.35

            if pname and len(pname) >= 4 and pname != nname:
                if f" {pname} " in p_norm or item["pname_ns"] in p_nospace:
                    score += 1.15

            sig = item["sig_tokens"]
            if len(sig) >= 2:
                matched_sig = sum(1 for t in sig if f" {t} " in f_norm)
                if matched_sig == len(sig):
                    score += 0.90
                elif matched_sig >= 2 and matched_sig >= len(sig) - 1:
                    score += 0.60

            v_zip5 = item["v_zip5"]
            p_zip5 = item["p_zip5"]
            st_matched = any(f" {sp} " in f_norm for sp in item["street_prefixes"])
            if st_matched:
                score += 0.85

            if v_zip5 and v_zip5 in up_full:
                if v_zip5 not in self.facility_zips or st_matched or score > 0.5:
                    score += 0.45
                    if item["v_city"] and f" {item['v_city']} " in f_norm:
                        score += 0.30
            elif p_zip5 and p_zip5 in up_full:
                if p_zip5 not in self.facility_zips or st_matched or score > 0.5:
                    score += 0.45

            for ca in item["cust_accts"]:
                if ca in up_full:
                    score += 0.65

            if item.get("approver_norm") and f" {item['approver_norm']} " in f_norm and score >= 0.40:
                score += 0.35

            if item["b_aba"] and len(item["b_aba"]) == 9 and item["b_aba"] in up_full:
                score += 0.65
            if item["b_acct"] and len(item["b_acct"]) >= 6 and item["b_acct"] in up_full:
                score += 0.65

            if score >= 0.25 and cocd:
                if item["default_cocd"] == cocd:
                    score += 0.12
                elif cocd in item["cocds"]:
                    score += 0.08

            if score > best_score:
                best_score = score
                best_vid = vid

        if best_vid and best_score > 0.0:
            v = self.vendors[best_vid]
            return {
                "vendor_id": best_vid,
                "vendor_name": v["vendor_name"],
                "match_method": "OLAYA_ROUTER_ONTOLOGY_GROUNDED",
                "confidence": min(round(best_score / 2.5, 3), 0.99),
                "vendor_record": v,
            }

        return {
            "vendor_id": "",
            "vendor_name": "",
            "match_method": "UNRESOLVED",
            "confidence": 0.0,
            "vendor_record": {},
        }

    def resolve_and_govern_bundle(
        self,
        extracted: dict[str, Any],
        full_text: str,
        primary_text: str = "",
    ) -> dict[str, Any]:
        """
        Ground an extracted document bundle against the active Ontology, Vendor/Payee Master,
        and OWL 2 RL axioms, sealing the result with a deterministic `quad_fast` receipt.
        """
        docno = str(extracted.get("docno") or extracted.get("bundle_id") or "DOC")
        cocd = str(extracted.get("cocd") or "")
        year = str(extracted.get("year") or "2026")
        doc_key = str(extracted.get("doc_key") or f"{cocd}_{docno}_{year}".strip("_"))
        p_txt = primary_text or full_text

        vmatch = extracted.get("vendor_match") or self.match_vendor(
            p_txt,
            full_text,
            cocd,
            primary_role=extracted.get("primary_unit_role", ""),
            gliner_spans=extracted.get("gliner_entities"),
        )
        vrec = vmatch["vendor_record"]

        routed = self.router.route_candidates(p_txt[:1500], top_k=5)
        olaya_concepts = [{"curie": c.curie, "label": c.label, "score": round(s, 3)} for c, s in routed]

        doc_type = extracted.get("extracted_doc_type") or vrec.get("default_doc_type") or "KR"
        vendor_id = vmatch["vendor_id"]
        vendor_name = vmatch["vendor_name"]
        doc_date = extracted.get("extracted_date") or ""
        reference = extracted.get("extracted_reference") or ""
        ref_source = extracted.get("reference_source") or "EXTRACTED_ONTOLOGY_GROUNDED"

        raw_amt = extracted.get("extracted_amount")
        amt_val = float(raw_amt) if raw_amt is not None else 0.0

        # Tax Code semantics: I1 (Vendor-billed Sales Tax, cleared when Tax == 0.0) vs U1 (Self-Assessed Use Tax, kept when Tax == 0.0)
        raw_tax = extracted.get("extracted_tax")
        try:
            tax_val = float(raw_tax) if raw_tax not in (None, "") else 0.0
        except Exception:
            tax_val = 0.0
        v_tax = vrec.get("default_tax_code", "")
        if v_tax == "I1" and raw_tax is not None and raw_tax != "" and abs(tax_val) == 0.0:
            tax_code = ""
        else:
            tax_code = v_tax
        tax_jur = vrec.get("default_tax_jurisdiction", "")

        # Expense GL Account: exclude reconciliation control accounts (`21100001`, `KOART='K'`)
        gl_map: dict[str, int] = {
            str(k): int(v)
            for k, v in (vrec.get("gl_accounts") or {}).items()
            if str(k) not in self.reconciliation_gl_accounts
        }
        if not gl_map:
            def_gl = str(vrec.get("default_gl_account") or "")
            gl_account = "" if def_gl in self.reconciliation_gl_accounts else def_gl
        else:
            gl_account = max(gl_map.items(), key=lambda kv: kv[1])[0]

        # Cost Center & Profit Center aligned with Entity Code (`cocd`) prefix
        cc_map: dict[str, int] = {str(k): int(v) for k, v in (vrec.get("cost_centers") or {}).items() if str(k)}
        cost_center = vrec.get("default_cost_center", "")
        if len(cc_map) > 1 and cocd:
            prefix = cocd.lstrip("0") or cocd[-2:]
            matching_ccs = [cc for cc in cc_map if cc.startswith(prefix) or cc.startswith(cocd[-2:])]
            if matching_ccs:
                cost_center = matching_ccs[0]

        pc_map: dict[str, int] = {str(k): int(v) for k, v in (vrec.get("profit_centers") or {}).items() if str(k)}
        profit_center = vrec.get("default_profit_center", "")
        if cost_center and len(cost_center) >= 5 and cost_center[:5] in pc_map:
            profit_center = cost_center[:5]

        pmt_terms = "ZNET" if doc_type == "KG" else vrec.get("default_pmt_terms", "ZNET")

        doc_iri = f"{self.base_iri}/doc/{doc_key}"
        graph_iri = f"{self.base_iri}/graphs/governed"
        concept_iri = (
            f"{self.base_iri}#VendorCreditMemo"
            if doc_type == "KG"
            else f"{self.base_iri}#VendorInvoice"
        )
        vendor_iri = f"{self.base_iri}#Vendor_{vendor_id or 'UNKNOWN'}"
        quad_ipc_bytes, receipt_hash = build_deterministic_quad_receipt(
            doc_iri=doc_iri,
            graph_iri=graph_iri,
            concept_iri=concept_iri,
            vendor_iri=vendor_iri,
            reference=reference,
            amount=amt_val,
            source_artifact_iri=f"file:///bundle/{doc_key}",
            full_text=full_text,
            observed_at_micros=int(extracted.get("observed_at_micros") or 1771545600_000_000),
            predicate_ns=f"{self.base_iri}#",
        )

        return {
            "doc_key": doc_key,
            "docno": docno,
            "cocd": cocd,
            "doc_type": doc_type,
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "doc_date": doc_date,
            "reference": reference,
            "reference_source": ref_source,
            "amount": amt_val,
            "pmt_terms": pmt_terms,
            "tax_code": tax_code,
            "tax_jurisdiction": tax_jur,
            "tax_on_invoice": tax_val,
            "freight": extracted.get("extracted_freight") or "",
            "discount": extracted.get("extracted_discount") or "",
            "vendor_street": vrec.get("street", ""),
            "vendor_city": vrec.get("city", ""),
            "vendor_state": vrec.get("state", ""),
            "vendor_zip": vrec.get("postal_code", ""),
            "payee_id": vrec.get("payee_id", ""),
            "payee_name": vrec.get("payee_name", ""),
            "payee_street": vrec.get("payee_street", ""),
            "payee_city": vrec.get("payee_city", ""),
            "payee_state": vrec.get("payee_state", ""),
            "payee_zip": vrec.get("payee_zip", ""),
            "purchase_order": extracted.get("extracted_po", ""),
            "bank_routing": vrec.get("bank_routing", ""),
            "bank_account": vrec.get("bank_account", ""),
            "gl_account": gl_account,
            "cost_center": cost_center,
            "profit_center": profit_center,
            "approver": vrec.get("default_approver", ""),
            "coder": vrec.get("default_coder", ""),
            "primary_unit_kind": extracted.get("primary_unit_kind", "UNKNOWN"),
            "primary_unit_role": extracted.get("primary_unit_role", "PRIMARY_INVOICE"),
            "needs_ocr": extracted.get("needs_ocr", False),
            "warnings": extracted.get("warnings", []),
            "olaya_concepts": olaya_concepts,
            "quad_receipt_hash": receipt_hash,
            "quad_ipc_bytes_len": len(quad_ipc_bytes),
        }
