"""
Tiered PDF, Visual Preprocessing & OCR Layout Engine (`pdf_fast` + `PIL`/`numpy` + `oar-ocr-cli` + `layout_fast`).

Handles:
1. Sub-millisecond native PDF semantic text & bounding-box extraction via `pdf_fast`.
2. Automatic detection of scanned image PDFs AND broken/shifted ToUnicode CMap font tables
   (`is_corrupted_or_scanned_pdf_page`).
3. Geometry- & DPI-Aware Visual Preprocessing Pipeline prior to ONNX OCR:
   - `compute_adaptive_dpi`: Computes optimal rasterization DPI from PDF page physical dimensions
     (`pdf_fast.parse_pdf_page_dimensions`), upscaling small thermal POS receipts (`300 DPI`),
     targeting `~190.9 DPI` for standard Letter/A4 pages, and clamping oversized sheets (`120 DPI`).
   - `enhance_contrast_and_denoise`: Adaptive percentile luminance stretching (`[p1..p95] -> [0..255]`)
     and background whitening applied BEFORE rotation and deskew.
   - `detect_coarse_rotation_90_180_270`: Combines PDF `/Rotate` metadata and horizontal-vs-vertical
     projection-profile variance (`H_var` vs `V_var` with adaptive midpoint threshold `0.5 * (p5 + p80)`)
     to detect sideways (`90°`/`270°`) or inverted (`180°`) pages.
   - `estimate_and_correct_skew_angle`: Two-stage vectorized NumPy projection-profile variance deskew
     (`-12°` to `+12°` coarse + fine sweep) that straightens tilted feeder scans (`|angle| >= 0.4°`)
     before tabular line clustering.
   - `dpi_aware_resize_image`: Rescales low-DPI/small images (`short_edge < min_short_edge_px`) or oversized
     400-600 DPI scans (`long_edge > max_long_edge_px`) with Lanczos interpolation into the PP-OCRv6 receptive-field sweet spot.
4. High-speed native Rust PDF-to-PNG rasterization (`pdf_fast.PdfScanner.page_rasterized_png_interpreted`)
   paired with batched ONNX PP-OCRv6 visual OCR (`oar-ocr-cli`).
5. Spectral reading-order reconstruction via `layout_fast.order_blocks`.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pdf_fast
from PIL import Image, ImageOps

try:
    import layout_fast
except ImportError:
    layout_fast = None  # type: ignore[assignment]

DEFAULT_DOCUMENT_VOCABULARY = {
    # Commercial / Accounts Payable (EN)
    "invoice", "date", "total", "amount", "bill", "ship", "sold", "remit",
    "account", "balance", "due", "tax", "payment", "service", "page",
    "description", "terms", "net", "order", "customer", "credit", "statement",
    "receipt", "charge", "from", "vendor", "supplier", "number", "subtotal",
    "item", "quantity", "price", "currency", "contract", "reference",
    # Multilingual / Fiscal / Insurance / Documentoscopia (PT/ES)
    "nota", "fiscal", "danfe", "chave", "acesso", "natureza", "operacao",
    "destinatario", "remetente", "valor", "fatura", "vencimento", "comprovante",
    "residencia", "endereco", "cliente", "boletim", "ocorrencia", "delegacia",
    "policia", "sinistro", "aviso", "segurado", "titular", "vitima", "relato",
    "roubo", "furto", "identidade", "expedicao", "nascimento", "filiacao",
    "naturalidade", "cpf", "cnpj", "cep", "imei", "aparelho", "celular",
    "documento", "estado", "secretaria", "seguranca", "registro", "dados",
}


def _shift_ascii_char_plus29(ch: str) -> str:
    """Shift a single character by +29 in ASCII printable space (Identity-H GID-to-ASCII repair)."""
    o = ord(ch)
    if 35 <= o <= 93:
        return chr(o + 29)
    return ch


def _repair_plus29_token(tok: str) -> str:
    """
    Detect and repair a single mixed-font token affected by +29 Identity-H GID-to-ASCII shift
    (e.g. `'%2/(7,0'` -> `'BOLETIM'`, `'5HVLGrQFLD'` -> `'Residencia'`, `''HVFULomR'` -> `'Descricao'`,
    `'2SHUDGRUD'` -> `'Operadora'`, `'%UDVLOHLUD'` -> `'Brasileira'`, `'0DVFXOLQR'` -> `'Masculino'`).
    """
    if len(tok) < 4:
        return tok

    # Never touch pure numbers, currency amounts ($1,250.00), or dates (17/05/2024)
    if re.match(r"^[\$\-\(\)\d,\./:]+$", tok):
        return tok
    # Never touch normal pure-alphabetic uppercase words (e.g. INVOICE, TOTAL)
    if tok.isalpha() and tok.isupper():
        return tok

    # Case 1: All-caps shifted token like %2/(7,0 -> BOLETIM or '$'26 -> DADOS or 5(*,6752 -> REGISTRO
    if any(c in tok for c in "%$'*,/()") and all(35 <= ord(c) <= 61 for c in tok):
        shifted = "".join(chr(ord(c) + 29) for c in tok)
        if shifted.isalpha() and shifted.lower() in DEFAULT_DOCUMENT_VOCABULARY:
            return shifted

    # Case 2: Title-case or lowercase shifted token where lowercase a-z was stored at [D-]] (ord 68..93)
    # and uppercase A-Z was stored at [$-=] (ord 36..61)
    has_mid_upper = any("D" <= c <= "Z" for c in tok[1:])
    has_weird_lead = tok[0] in "$%&'()*+,-./0123456789:;<=>"
    if not (has_mid_upper or has_weird_lead):
        return tok

    # Pre-normalize common Latin-1 accented glyph artifacts inside +29 shifted words
    norm_tok = tok.replace("omR", "cao").replace("rQ", "en").replace("iU", "ar")
    out_chars: list[str] = []
    for idx, ch in enumerate(norm_tok):
        o = ord(ch)
        if idx == 0 and 36 <= o <= 61:
            out_chars.append(chr(o + 29))
        elif 68 <= o <= 93:
            out_chars.append(chr(o + 29))
        elif ch in ("a", "c", "e", "n", "o", "r") and norm_tok != tok:
            out_chars.append(ch)
        else:
            return tok
    candidate = "".join(out_chars)
    if candidate.isalpha() and len(candidate) >= 4:
        low_cand = candidate.lower()
        vowels = sum(1 for c in low_cand if c in "aeiou")
        if 0.25 <= vowels / len(low_cand) <= 0.75:
            return candidate
    return tok


def repair_shifted_cmap_text(
    text: str,
    vocabulary: set[str] | None = None,
) -> tuple[str, int]:
    """
    Detect and self-heal broken Type0 / Identity-H `ToUnicode` CMap ASCII shifts in PDF text
    or OCR output (both uniform page-level shifts `delta in [-35..+35]` and mixed-font `+29`
    token-level shifts).
    Returns `(repaired_text, detected_delta)`.
    """
    if not text or len(text.strip()) < 10:
        return text, 0

    vocab = vocabulary if vocabulary is not None else DEFAULT_DOCUMENT_VOCABULARY
    orig_words = set(re.findall(r"[a-z]{3,}", text.lower()))
    orig_hits = len(orig_words.intersection(vocab))

    best_delta = 0
    best_hits = orig_hits
    best_text = text

    # 1. Check uniform ASCII shift across the text (common deltas: +29, -4, +3, etc.)
    sample = text[:4000]
    for delta in (29, -29, -4, 4, 3, -3, 1, -1, 28, 30):
        shifted_chars = []
        for ch in sample:
            o = ord(ch)
            if 32 <= o <= 126:
                no = o + delta
                if 32 <= no <= 126:
                    shifted_chars.append(chr(no))
                else:
                    shifted_chars.append(ch)
            else:
                shifted_chars.append(ch)
        cand_sample = "".join(shifted_chars)
        cand_words = set(re.findall(r"[a-z]{3,}", cand_sample.lower()))
        hits = len(cand_words.intersection(vocab))
        if hits > best_hits and hits >= max(3, orig_hits * 2 + 2):
            best_hits = hits
            best_delta = delta

    if best_delta != 0:
        full_shifted = []
        for ch in text:
            o = ord(ch)
            if 32 <= o <= 126:
                no = o + best_delta
                full_shifted.append(chr(no) if 32 <= no <= 126 else ch)
            else:
                full_shifted.append(ch)
        best_text = "".join(full_shifted)

    # 2. Token-level mixed-font +29 shift repair (e.g. pages with normal headers + shifted body tokens)
    repaired_tokens_count = 0

    def _replace_match(m: re.Match[str]) -> str:
        nonlocal repaired_tokens_count
        raw_tok = m.group(0)
        fixed = _repair_plus29_token(raw_tok)
        if fixed != raw_tok:
            repaired_tokens_count += 1
        return fixed

    token_repaired = re.sub(r"[!-~]{3,}", _replace_match, best_text)
    if repaired_tokens_count > 0:
        best_text = token_repaired
        if best_delta == 0:
            best_delta = 29

    return best_text, best_delta


_GENERAL_PROSE_VOCABULARY = {
    "the", "and", "for", "with", "that", "this", "are", "was", "were", "which",
    "have", "has", "had", "not", "all", "can", "also", "into", "more", "other",
    "such", "than", "between", "where", "when", "under", "over", "after", "before",
    "during", "within", "without", "through", "study", "system", "data", "model",
    "results", "method", "analysis", "research", "university", "college", "department",
    "journal", "article", "published", "copyright", "theorem", "lemma", "proof",
    "equation", "matrix", "function", "section", "figure", "table", "chapter",
    "volume", "del", "las", "los", "por", "para", "con", "una", "que", "como",
    "este", "esta", "sobre", "entre", "der", "die", "das", "und", "von", "mit",
    "des", "les", "dans", "pour", "avec", "sur",
}


def is_corrupted_or_scanned_pdf_page(
    page_txt: str,
    page_runs: list[dict[str, Any]],
    vocabulary: set[str] | None = None,
) -> bool:
    """
    Detect whether a PDF page is either a scanned image (<80 chars) OR has a
    corrupted/shifted ToUnicode CMap font table (producing NUL bytes, control chars,
    high cmap_fallback_chars, or <2 recognizable domain/prose vocabulary words).
    """
    stripped = page_txt.strip()
    if len(stripped) < 80:
        return True
    if "\x00" in stripped:
        return True
    ctrl_count = sum(1 for c in stripped if ord(c) < 32 and c not in ("\n", "\r", "\t"))
    if ctrl_count > 3:
        return True
    fallback_chars = sum(int(r.get("cmap_fallback_chars") or 0) for r in page_runs)
    total_chars = max(len(stripped), 1)
    if fallback_chars >= 4 and fallback_chars * 5 >= total_chars:
        return True
    vocab = (
        vocabulary
        if vocabulary is not None
        else (DEFAULT_DOCUMENT_VOCABULARY | _GENERAL_PROSE_VOCABULARY)
    )
    words = set(re.findall(r"[a-z]{3,}", stripped.lower()))
    if len(words.intersection(vocab)) < 2:
        return True
    return False


# ─── VISUAL PREPROCESSING: ADAPTIVE DPI, ROTATION, DESKEW & CONTRAST ─────────

def compute_adaptive_dpi(
    width_pt: float,
    height_pt: float,
    target_long_edge_px: int = 2100,
    min_dpi: float = 120.0,
    max_dpi: float = 300.0,
) -> float:
    """
    Compute the optimal rasterization DPI from a PDF page's physical dimensions in points
    (`72 pt = 1.0 inch`) so that the rendered long edge lands near `target_long_edge_px`
    (`~2100 px`, corresponding to ~28-34 px glyph height for standard 10-pt document text):
    - Small POS thermal receipts (e.g. `225 x 432 pt` / `3.1" x 6"`): scales up to `300.0 DPI`
    - Standard Letter/A4 pages (`612 x 792 pt` / `8.5" x 11"`): computes `~190.9 DPI`
    - Oversized ledger/engineering pages (`1224 x 1584 pt` / `17" x 22"`): clamps to `120.0 DPI`
    """
    long_pt = max(float(width_pt or 612.0), float(height_pt or 792.0), 72.0)
    long_inches = long_pt / 72.0
    ideal_dpi = target_long_edge_px / long_inches
    clamped = max(min_dpi, min(max_dpi, ideal_dpi))
    return round(clamped, 1)


def _ink_projection_variance(ink_mask: np.ndarray, axis: int) -> float:
    """Compute normalized variance of 1D foreground ink projection along `axis` (1=horizontal rows, 0=vertical cols)."""
    if ink_mask.size == 0:
        return 0.0
    proj = np.mean(ink_mask, axis=axis, dtype=np.float32)
    return float(np.var(proj))


def detect_coarse_rotation_90_180_270(
    img: Image.Image,
    pdf_rotate_deg: int = 0,
    sideways_variance_ratio: float = 2.2,
) -> int:
    """
    Detect coarse `0 / 90 / 180 / 270` degree page rotation required to bring text lines upright:
    1. If `pdf_rotate_deg` (`90`, `180`, `270`) is explicitly provided from unapplied metadata,
       returns `(360 - pdf_rotate_deg) % 360`.
    2. Otherwise compares horizontal row-projection variance (`H_var`) vs. vertical column-projection
       variance (`V_var`) on a fast downsampled binary ink mask (`gray < 0.5 * (p5 + p80)`).
       When text lines run vertically (`V_var > sideways_variance_ratio * H_var`), the page is sideways.
    """
    norm_meta = int(pdf_rotate_deg or 0) % 360
    if norm_meta in (90, 180, 270):
        return (360 - norm_meta) % 360

    gray = img.convert("L")
    w, h = gray.size
    if w < 32 or h < 32:
        return 0

    scale = min(1.0, 512.0 / max(w, h))
    if scale < 1.0:
        gray = gray.resize((max(16, int(w * scale)), max(16, int(h * scale))), Image.Resampling.BILINEAR)

    arr = np.asarray(gray, dtype=np.uint8)
    p5, p80 = np.percentile(arr, [5.0, 80.0])
    if p80 - p5 < 15.0:
        return 0
    thresh = 0.5 * (float(p5) + float(p80))
    ink = (arr < thresh).astype(np.float32)
    if float(np.mean(ink)) < 0.002:
        return 0

    h_var = _ink_projection_variance(ink, axis=1)
    v_var = _ink_projection_variance(ink, axis=0)

    if v_var > sideways_variance_ratio * max(h_var, 1e-6):
        ih, iw = ink.shape
        sub = ink[max(1, int(ih * 0.15)) : max(2, int(ih * 0.85)), max(1, int(iw * 0.15)) : max(2, int(iw * 0.85))]
        if sub.shape[1] >= 6:
            mid = sub.shape[1] // 2
            left_ink = float(np.mean(sub[:, :mid]))
            right_ink = float(np.mean(sub[:, mid:]))
            return 270 if left_ink > right_ink else 90
        left_ink = float(np.mean(ink[:, : ink.shape[1] // 3]))
        right_ink = float(np.mean(ink[:, (2 * ink.shape[1]) // 3 :]))
        return 270 if left_ink > right_ink else 90

    return 0


def estimate_and_correct_skew_angle(
    img: Image.Image,
    max_angle_deg: float = 12.0,
    min_correction_deg: float = 0.4,
) -> tuple[Image.Image, float]:
    """
    Estimate fine document skew angle in `[-max_angle_deg, +max_angle_deg]` using two-stage
    Horizontal Projection-Profile Variance maximization (coarse `1.0°` step + fine `0.2°` step),
    and deskew the image with bicubic interpolation when `|angle| >= min_correction_deg`.
    """
    gray = img.convert("L")
    w, h = gray.size
    if w < 64 or h < 64:
        return img, 0.0

    work_scale = min(1.0, 480.0 / max(w, h))
    work_w = max(32, int(w * work_scale))
    work_h = max(32, int(h * work_scale))
    work_gray = gray.resize((work_w, work_h), Image.Resampling.BILINEAR)
    arr = np.asarray(work_gray, dtype=np.uint8)

    p5, p80 = np.percentile(arr, [5.0, 80.0])
    if p80 - p5 < 15.0:
        return img, 0.0
    thresh = 0.5 * (float(p5) + float(p80))
    ink_uint8 = ((arr < thresh).astype(np.uint8)) * 255
    if int(np.count_nonzero(ink_uint8)) < 40:
        return img, 0.0

    ink_pil = Image.fromarray(ink_uint8, mode="L")

    def score_angle(deg: float) -> float:
        rot = ink_pil.rotate(deg, resample=Image.Resampling.NEAREST, fillcolor=0)
        rot_arr = np.asarray(rot, dtype=np.float32) / 255.0
        row_sums = np.sum(rot_arr, axis=1)
        diffs = np.diff(row_sums)
        return float(np.var(row_sums) + 0.5 * np.mean(diffs * diffs))

    coarse_angles = np.arange(-max_angle_deg, max_angle_deg + 0.01, 1.0)
    best_coarse = 0.0
    best_score = score_angle(0.0)
    zero_score = best_score
    for a in coarse_angles:
        if abs(a) < 1e-4:
            continue
        sc = score_angle(float(a))
        if sc > best_score:
            best_score = sc
            best_coarse = float(a)

    fine_angles = np.arange(
        max(-max_angle_deg, best_coarse - 1.0),
        min(max_angle_deg, best_coarse + 1.0) + 0.01,
        0.2,
    )
    best_angle = best_coarse
    for a in fine_angles:
        sc = score_angle(float(a))
        if sc > best_score:
            best_score = sc
            best_angle = float(a)

    best_angle = round(best_angle, 2)
    if abs(best_angle) < min_correction_deg or best_score <= zero_score * 1.03:
        return img, 0.0

    fill_bg: Any = (255, 255, 255) if img.mode == "RGB" else 255
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
        fill_bg = (255, 255, 255)
    corrected = img.rotate(
        best_angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill_bg,
    )
    return corrected, best_angle


def dpi_aware_resize_image(
    img: Image.Image,
    target_long_edge_px: int = 2100,
    min_short_edge_px: int = 900,
    max_long_edge_px: int = 2560,
    source_dpi: float | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Perform DPI-aware and resolution-aware adaptive resizing so glyph heights fall into
    the optimal PP-OCRv6 detection window:
    - Inspects embedded `img.info.get("dpi")` or explicit `source_dpi` (if `< 130 DPI`, upscales toward `200 DPI`).
    - Ensures the short edge is at least `min_short_edge_px` without exceeding `max_long_edge_px`.
    - Clamps oversized high-DPI images (`long_edge > max_long_edge_px`) down to `max_long_edge_px` using Lanczos.
    """
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return img, {"orig_size": (orig_w, orig_h), "new_size": (orig_w, orig_h), "scale_factor": 1.0}

    embedded_dpi = source_dpi
    if embedded_dpi is None:
        dpi_tuple = img.info.get("dpi")
        if isinstance(dpi_tuple, (tuple, list)) and len(dpi_tuple) >= 1 and float(dpi_tuple[0] or 0) > 10:
            embedded_dpi = float(dpi_tuple[0])

    long_edge = max(orig_w, orig_h)
    short_edge = min(orig_w, orig_h)
    scale = 1.0
    reason = "within_optimal_bounds"

    if embedded_dpi is not None and 20.0 <= embedded_dpi < 130.0:
        dpi_scale = min(200.0 / embedded_dpi, max_long_edge_px / max(long_edge, 1))
        if dpi_scale > 1.08:
            scale = dpi_scale
            reason = f"low_dpi_upscale({embedded_dpi:.0f}->200)"

    if scale == 1.0:
        if long_edge > max_long_edge_px:
            scale = max_long_edge_px / float(long_edge)
            reason = f"oversized_downscale({long_edge}->{max_long_edge_px})"
        elif short_edge < min_short_edge_px and long_edge < target_long_edge_px:
            need_short = min_short_edge_px / float(max(short_edge, 1))
            max_allowed = max_long_edge_px / float(max(long_edge, 1))
            cand_scale = min(need_short, max_allowed, 2.5)
            if cand_scale > 1.08:
                scale = cand_scale
                reason = f"small_receipt_or_lowres_upscale(short={short_edge})"

    if abs(scale - 1.0) < 0.05:
        return img, {
            "orig_size": (orig_w, orig_h),
            "new_size": (orig_w, orig_h),
            "scale_factor": 1.0,
            "source_dpi": embedded_dpi,
            "effective_dpi": embedded_dpi or 180.0,
            "resize_reason": "unchanged",
        }

    new_w = max(32, int(round(orig_w * scale)))
    new_h = max(32, int(round(orig_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    eff_dpi = round((embedded_dpi or 150.0) * scale, 1)
    return resized, {
        "orig_size": (orig_w, orig_h),
        "new_size": (new_w, new_h),
        "scale_factor": round(scale, 4),
        "source_dpi": embedded_dpi,
        "effective_dpi": eff_dpi,
        "resize_reason": reason,
    }


def enhance_contrast_and_denoise(img: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """
    Apply adaptive percentile luminance normalization (`[p1..p95] -> [0..255]`) and light background
    whitening when an image has low contrast (`p99 - p1 < 185`, e.g. faded carbon-copy
    tickets or gray-background scans), preserving anti-aliased glyph edges.
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return img, {"contrast_stretched": False, "dynamic_range": 0.0}

    p1, p95, p99 = np.percentile(arr, [1.0, 95.0, 99.0])
    dyn_range = float(p99 - p1)
    if dyn_range >= 185.0 and p95 >= 225.0:
        return img, {"contrast_stretched": False, "dynamic_range": round(dyn_range, 1)}

    hi = max(float(p95), float(p1) + 25.0)
    stretched = np.clip((arr - float(p1)) * (255.0 / (hi - float(p1))), 0.0, 255.0).astype(np.uint8)
    out_img = Image.fromarray(stretched, mode="L").convert("RGB")
    return out_img, {"contrast_stretched": True, "dynamic_range": round(dyn_range, 1)}


def preprocess_document_image(
    img: Image.Image,
    pdf_rotate_deg: int = 0,
    source_dpi: float | None = None,
    enable_rotation: bool = True,
    enable_deskew: bool = True,
    enable_dpi_resize: bool = True,
    enable_contrast: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Execute the complete visual document preprocessing pipeline on a `PIL.Image`:
    1. EXIF orientation transpose (`ImageOps.exif_transpose`)
    2. Adaptive percentile contrast stretching (`enhance_contrast_and_denoise`)
    3. Coarse `90° / 180° / 270°` orientation detection & correction
    4. Fine projection-profile variance deskew (`-12° .. +12°`)
    5. DPI-aware & geometry-aware Lanczos resizing (`dpi_aware_resize_image`)
    """
    try:
        work = ImageOps.exif_transpose(img)
    except Exception:
        work = img

    if work.mode not in ("RGB", "L"):
        work = work.convert("RGB")

    diag: dict[str, Any] = {
        "input_size": work.size,
        "coarse_rotation_deg": 0,
        "skew_angle_deg": 0.0,
        "scale_factor": 1.0,
        "contrast_stretched": False,
    }

    if enable_contrast:
        work, contrast_meta = enhance_contrast_and_denoise(work)
        diag.update(contrast_meta)

    if enable_rotation:
        coarse_deg = detect_coarse_rotation_90_180_270(work, pdf_rotate_deg=pdf_rotate_deg)
        if coarse_deg in (90, 180, 270):
            work = work.rotate(coarse_deg, expand=True)
            diag["coarse_rotation_deg"] = coarse_deg

    if enable_deskew:
        work, skew_deg = estimate_and_correct_skew_angle(work)
        diag["skew_angle_deg"] = skew_deg

    if enable_dpi_resize:
        work, resize_meta = dpi_aware_resize_image(work, source_dpi=source_dpi)
        diag.update(resize_meta)

    diag["output_size"] = work.size
    return work, diag


def preprocess_document_image_bytes(
    png_bytes: bytes,
    pdf_rotate_deg: int = 0,
    source_dpi: float | None = None,
    enable_rotation: bool = True,
    enable_deskew: bool = True,
    enable_dpi_resize: bool = True,
    enable_contrast: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Run `preprocess_document_image` on raw PNG/JPEG bytes and return encoded PNG bytes + diagnostics."""
    if not png_bytes:
        return b"", {"error": "empty_input_bytes"}
    with Image.open(io.BytesIO(png_bytes)) as im:
        im.load()
        proc_img, diag = preprocess_document_image(
            im,
            pdf_rotate_deg=pdf_rotate_deg,
            source_dpi=source_dpi,
            enable_rotation=enable_rotation,
            enable_deskew=enable_deskew,
            enable_dpi_resize=enable_dpi_resize,
            enable_contrast=enable_contrast,
        )
        buf = io.BytesIO()
        eff_dpi = int(round(float(diag.get("effective_dpi") or source_dpi or 180.0)))
        proc_img.save(buf, format="PNG", dpi=(eff_dpi, eff_dpi))
        return buf.getvalue(), diag


def render_pdf_page_png(
    pdf_path: Path,
    page_index: int,
    preprocess: bool = True,
) -> bytes:
    """
    Rasterize a single PDF page to PNG bytes using an adaptive DPI computed from
    `pdf_fast.parse_pdf_page_dimensions`, then run `preprocess_document_image_bytes`.
    """
    width_pt, height_pt = 612.0, 792.0
    try:
        dims = pdf_fast.parse_pdf_page_dimensions(str(pdf_path))
        if dims and 0 <= page_index < len(dims):
            d = dims[page_index]
            width_pt = float(getattr(d, "width", 612.0) or 612.0)
            height_pt = float(getattr(d, "height", 792.0) or 792.0)
    except Exception:
        pass

    adaptive_dpi = compute_adaptive_dpi(width_pt, height_pt)
    candidate_dpis = [adaptive_dpi]
    for fallback_dpi in (150.0, 120.0, 100.0):
        if abs(fallback_dpi - adaptive_dpi) > 10.0:
            candidate_dpis.append(fallback_dpi)

    raw_png = b""
    used_dpi = adaptive_dpi
    for dpi in candidate_dpis:
        try:
            scanner = pdf_fast.PdfScanner(str(pdf_path))
            png_bytes = scanner.page_rasterized_png_interpreted(page_index, dpi)
            if png_bytes:
                raw_png = png_bytes
                used_dpi = dpi
                break
        except Exception:
            pass
        try:
            png_bytes = pdf_fast.parse_pdf_page_rasterized_png(str(pdf_path), page_index, dpi)
            if png_bytes:
                raw_png = png_bytes
                used_dpi = dpi
                break
        except Exception:
            pass

    if not raw_png or not preprocess:
        return raw_png

    try:
        proc_png, _ = preprocess_document_image_bytes(
            raw_png,
            pdf_rotate_deg=0,
            source_dpi=used_dpi,
        )
        return proc_png if proc_png else raw_png
    except Exception:
        return raw_png


_WIDE_CHARS = set("mwMW@%&QOGD")
_NARROW_CHARS = set("ilI1!.,:;'|jft()[]{}")
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00a0": " ",
}


def _clean_ligatures(text: str) -> str:
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    return text


def _proportional_text_width(text: str, fs: float) -> float:
    if not text:
        return 0.0
    total = 0.0
    for ch in text:
        if ch in _WIDE_CHARS:
            total += 0.78 * fs
        elif ch in _NARROW_CHARS:
            total += 0.28 * fs
        elif ch == " ":
            total += 0.28 * fs
        elif ch.isupper():
            total += 0.62 * fs
        else:
            total += 0.46 * fs
    return max(total, 0.25 * fs)


def reconstruct_text_from_ocr_runs(
    runs: list[dict[str, Any]],
    engine: str = "routed",
    multi_column: bool | None = None,
    strip_headers_footers: bool = False,
) -> str:
    """
    Reconstruct clean multi-column / tabular reading order from `oar-ocr-cli`
    bounding-box runs using `layout_fast.order_blocks` and vertical line clustering.
    """
    if not runs:
        return ""

    valid_runs = [r for r in runs if (r.get("text") or "").strip()]
    if not valid_runs:
        return ""

    if strip_headers_footers and len(valid_runs) >= 6:
        max_y = max(float(r.get("px_y", 0.0)) + float(r.get("px_height", 20.0)) for r in valid_runs)
        if max_y > 200.0:
            filtered = [
                r
                for r in valid_runs
                if 0.125 * max_y <= float(r.get("px_y", 0.0)) <= 0.895 * max_y
            ]
            if filtered:
                valid_runs = filtered

    use_multi_column = (len(valid_runs) >= 35) if multi_column is None else bool(multi_column)

    sorted_y = sorted(valid_runs, key=lambda r: (float(r.get("px_y", 0.0)), float(r.get("px_x", 0.0))))
    lines: list[list[dict[str, Any]]] = []
    curr_line: list[dict[str, Any]] = []
    curr_y: float | None = None
    curr_h: float = 20.0

    for r in sorted_y:
        ry = float(r.get("px_y", 0.0))
        rh = max(float(r.get("px_height", 20.0)), 8.0)
        if curr_y is None or abs(ry - curr_y) <= 0.52 * max(curr_h, rh):
            curr_line.append(r)
            if curr_y is None:
                curr_y = ry
                curr_h = rh
            else:
                curr_y = 0.8 * curr_y + 0.2 * ry
                curr_h = max(curr_h, rh)
        else:
            curr_line.sort(key=lambda x: float(x.get("px_x", 0.0)))
            lines.append(curr_line)
            curr_line = [r]
            curr_y = ry
            curr_h = rh

    if curr_line:
        curr_line.sort(key=lambda x: float(x.get("px_x", 0.0)))
        lines.append(curr_line)

    blocks: list[dict[str, Any]] = []
    if not use_multi_column:
        for idx, line_runs in enumerate(lines):
            x0 = min(float(x.get("px_x", 0.0)) for x in line_runs)
            y0 = min(float(x.get("px_y", 0.0)) for x in line_runs)
            x1 = max(float(x.get("px_x", 0.0)) + float(x.get("px_width", 10.0)) for x in line_runs)
            y1 = max(float(x.get("px_y", 0.0)) + float(x.get("px_height", 10.0)) for x in line_runs)
            line_txt = "  ".join(_clean_ligatures(x["text"].strip()) for x in line_runs)
            blocks.append({
                "id": idx,
                "bbox": [x0, y0, x1, y1],
                "font_size": max(y1 - y0, 8.0),
                "font_id": 0,
                "text": line_txt,
                "source": "ocr",
                "kind": "text",
            })
    else:
        for line_runs in lines:
            seg: list[dict[str, Any]] = []
            seg_x1: float | None = None
            seg_h: float = 20.0
            for r in line_runs:
                rx = float(r.get("px_x", 0.0))
                rw = max(float(r.get("px_width", 10.0)), 4.0)
                rh = max(float(r.get("px_height", 20.0)), 8.0)
                if seg_x1 is not None and (rx - seg_x1) > 1.2 * max(seg_h, rh):
                    x0 = min(float(x.get("px_x", 0.0)) for x in seg)
                    y0 = min(float(x.get("px_y", 0.0)) for x in seg)
                    x1 = max(float(x.get("px_x", 0.0)) + float(x.get("px_width", 10.0)) for x in seg)
                    y1 = max(float(x.get("px_y", 0.0)) + float(x.get("px_height", 10.0)) for x in seg)
                    blocks.append({
                        "id": len(blocks),
                        "bbox": [x0, y0, x1, y1],
                        "font_size": max(y1 - y0, 8.0),
                        "font_id": 0,
                        "text": " ".join(_clean_ligatures(x["text"].strip()) for x in seg),
                        "source": "ocr",
                        "kind": "text",
                    })
                    seg = [r]
                    seg_x1 = rx + rw
                    seg_h = rh
                else:
                    seg.append(r)
                    seg_x1 = max(seg_x1 or rx, rx + rw)
                    seg_h = max(seg_h, rh)
            if seg:
                x0 = min(float(x.get("px_x", 0.0)) for x in seg)
                y0 = min(float(x.get("px_y", 0.0)) for x in seg)
                x1 = max(float(x.get("px_x", 0.0)) + float(x.get("px_width", 10.0)) for x in seg)
                y1 = max(float(x.get("px_y", 0.0)) + float(x.get("px_height", 10.0)) for x in seg)
                blocks.append({
                    "id": len(blocks),
                    "bbox": [x0, y0, x1, y1],
                    "font_size": max(y1 - y0, 8.0),
                    "font_id": 0,
                    "text": " ".join(_clean_ligatures(x["text"].strip()) for x in seg),
                    "source": "ocr",
                    "kind": "text",
                })
        if layout_fast is not None and blocks:
            try:
                order = layout_fast.order_blocks(blocks, engine=engine)
                if isinstance(order, list) and len(order) == len(blocks):
                    blocks = [blocks[i] for i in order]
            except Exception:
                pass

    joined = "\n".join(b["text"] for b in blocks)
    if use_multi_column:
        joined = re.sub(r"-\s*\n\s*([a-zA-Z])", r"\1", joined)
    repaired, _ = repair_shifted_cmap_text(joined)
    return repaired


def _join_pdf_segment_runs(seg_runs: list[dict[str, Any]], has_explicit_space: bool) -> str:
    parts: list[str] = []
    prev_end: float | None = None
    prev_fs: float = 9.0
    for r in seg_runs:
        t = _clean_ligatures(r.get("text") or "")
        if not t:
            continue
        rx = float(r.get("_x", 0.0))
        rfs = float(r.get("_fs", 9.0))
        adj_w = float(r.get("_w", 4.0))
        if not parts:
            parts.append(t)
        else:
            gap = rx - (prev_end or rx)
            thresh = 0.28 * max(prev_fs, rfs) if has_explicit_space else 0.14 * max(prev_fs, rfs)
            if gap > thresh and not parts[-1].endswith(" ") and not t.startswith(" "):
                parts.append(" " + t)
            else:
                parts.append(t)
        prev_end = rx + adj_w
        prev_fs = rfs
    return re.sub(r"\s+", " ", "".join(parts))


def extract_ordered_pdf_page_text(
    pdf_path: Path,
    page_index: int = 0,
    engine: str = "recursive",
    strip_headers_footers: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Extract native text and positioned semantic runs for a single PDF page (`page_index`),
    routing single-character kerned multi-column streams through proportional glyph width
    clustering + `layout_fast.order_blocks`, and optionally stripping running headers/footers.
    """
    resolved = pdf_path.resolve()
    page_runs: list[dict[str, Any]] = []
    page_h, page_w = 792.0, 612.0

    try:
        scanner = pdf_fast.PdfScanner(str(resolved))
        dims = json.loads(scanner.page_dimensions_json())
        for d in dims:
            if int(d.get("page_index", 0)) == page_index:
                page_h = float(d.get("height", 792.0)) or 792.0
                page_w = float(d.get("width", 612.0)) or 612.0
                break
        all_runs = json.loads(scanner.semantic_runs_json())
        if isinstance(all_runs, list):
            page_runs = [
                r for r in all_runs if int(r.get("page_index", r.get("page", 1) - 1)) == page_index
            ]
    except Exception:
        page_runs = []

    raw_text = ""
    avg_len = sum(len(r.get("text") or "") for r in page_runs) / max(len(page_runs), 1)
    if len(page_runs) > 400 and avg_len < 2.2 and layout_fast is not None:
        work_runs: list[dict[str, Any]] = []
        for r in page_runs:
            raw_fs = float(r.get("font_size") or 0.0)
            txt = _clean_ligatures(r.get("text") or "")
            eff_fs = 9.0 if raw_fs <= 3.0 else raw_fs
            x = float(r.get("x") or 0.0)
            y = float(r.get("y") or 0.0)
            if strip_headers_footers and (
                y < page_h * 0.13
                or (y + min(eff_fs, 14.0)) > page_h * 0.85
                or x < page_w * 0.055
                or x > page_w * 0.925
            ):
                continue
            rc = dict(r)
            rc["_x"] = x
            rc["_y"] = y
            rc["_fs"] = eff_fs
            rc["_w"] = _proportional_text_width(txt, eff_fs)
            work_runs.append(rc)

        if work_runs:
            by_y = sorted(work_runs, key=lambda r: -r["_y"])
            rows: list[list[dict[str, Any]]] = []
            curr_row: list[dict[str, Any]] = []
            curr_y: float | None = None
            curr_fs: float = 9.0
            for r in by_y:
                ry, rfs = r["_y"], r["_fs"]
                if curr_y is None or abs(ry - curr_y) <= 0.42 * max(curr_fs, rfs):
                    curr_row.append(r)
                    curr_y = ry if curr_y is None else (0.85 * curr_y + 0.15 * ry)
                    curr_fs = max(curr_fs, rfs)
                else:
                    rows.append(curr_row)
                    curr_row, curr_y, curr_fs = [r], ry, rfs
            if curr_row:
                rows.append(curr_row)

            line_blocks: list[dict[str, Any]] = []
            line_texts: list[str] = []
            for row in rows:
                row.sort(key=lambda r: r["_x"])
                has_explicit_space = any((r.get("text") == " ") for r in row)
                seg_runs: list[dict[str, Any]] = []
                seg_x0: float | None = None
                seg_x1: float | None = None
                seg_y: float = row[0]["_y"]
                seg_fs: float = row[0]["_fs"]
                for r in row:
                    rx, rw, rfs = r["_x"], r["_w"], r["_fs"]
                    if seg_x1 is not None and (rx - seg_x1) > max(1.35 * max(seg_fs, rfs), 10.0):
                        txt = _join_pdf_segment_runs(seg_runs, has_explicit_space)
                        if txt.strip():
                            line_blocks.append({
                                "id": len(line_blocks),
                                "bbox": [seg_x0, seg_y, seg_x1, seg_y + seg_fs],
                                "font_size": seg_fs,
                                "font_id": 0,
                                "source": "pdf",
                            })
                            line_texts.append(txt.strip())
                        seg_runs, seg_x0, seg_x1, seg_y, seg_fs = [r], rx, rx + rw, r["_y"], rfs
                    else:
                        seg_runs.append(r)
                        if seg_x0 is None:
                            seg_x0 = rx
                        seg_x1 = max(seg_x1 or rx, rx + rw)
                        seg_fs = max(seg_fs, rfs)
                if seg_runs:
                    txt = _join_pdf_segment_runs(seg_runs, has_explicit_space)
                    if txt.strip():
                        line_blocks.append({
                            "id": len(line_blocks),
                            "bbox": [seg_x0, seg_y, seg_x1, seg_y + seg_fs],
                            "font_size": seg_fs,
                            "font_id": 0,
                            "source": "pdf",
                        })
                        line_texts.append(txt.strip())

            if line_blocks:
                try:
                    order = layout_fast.order_blocks(line_blocks, engine=engine)
                    if isinstance(order, list) and len(order) == len(line_blocks):
                        line_texts = [line_texts[i] for i in order]
                except Exception:
                    pass
                raw_text = "\n".join(line_texts)

    if not raw_text:
        try:
            full_doc_text = pdf_fast.parse_pdf_text(str(resolved))
            pages_split = full_doc_text.split("\f")
            raw_text = pages_split[page_index] if page_index < len(pages_split) else full_doc_text
        except Exception:
            if page_runs:
                raw_text = " ".join((r.get("text") or "").strip() for r in page_runs if (r.get("text") or "").strip())

        if strip_headers_footers and raw_text and page_runs:
            margin_runs: list[str] = []
            body_runs: list[str] = []
            for r in page_runs:
                x = float(r.get("x") or 0.0)
                y = float(r.get("y") or 0.0)
                fs = float(r.get("font_size") or 8.0)
                t = (r.get("text") or "").strip()
                if not t:
                    continue
                if (
                    y < page_h * 0.13
                    or (y + min(fs, 14.0)) > page_h * 0.85
                    or x < page_w * 0.055
                    or x > page_w * 0.925
                ):
                    margin_runs.append(t)
                else:
                    body_runs.append(t)
            if body_runs:
                lines_list = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
                margin_blob = _clean_ligatures(" ".join(margin_runs)).lower()
                body_blob = _clean_ligatures(" ".join(body_runs)).lower()
                kept: list[str] = []
                for idx, ln in enumerate(lines_list):
                    ln_low = _clean_ligatures(ln).lower()
                    if not ln_low:
                        continue
                    if (idx < 4 or idx >= len(lines_list) - 4) and (
                        len(ln_low) <= 6 or re.match(r"^(?:page\s*)?\d+(?:\s*of\s*\d+)?$", ln_low)
                    ):
                        continue
                    if ln_low in margin_blob and (
                        ln_low not in body_blob or idx < 5 or idx >= len(lines_list) - 5
                    ):
                        continue
                    kept.append(ln)
                if kept:
                    raw_text = "\n".join(kept)

    raw_text = _clean_ligatures(raw_text)
    raw_text = re.sub(r"-\s*\n\s*([a-zA-Z])", r"\1", raw_text)
    repaired, _ = repair_shifted_cmap_text(raw_text)

    # If the page has substantial readable prose (>=120 chars and >=4 vocabulary words)
    # and only minor math/symbol control codes (<5% of text), strip stray control chars
    # so LaTeX/academic pages with embedded math symbol fonts are not misflagged as corrupted scans.
    if len(repaired) >= 120:
        ctrl_n = sum(1 for c in repaired if ord(c) < 32 and c not in ("\n", "\r", "\t"))
        if ctrl_n * 20 < len(repaired):
            words_found = set(re.findall(r"[a-z]{3,}", repaired.lower()))
            if len(words_found.intersection(DEFAULT_DOCUMENT_VOCABULARY | _GENERAL_PROSE_VOCABULARY)) >= 4:
                repaired = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", repaired)
                page_runs = [{"cmap_fallback_chars": 0}]

    return repaired, page_runs


def resolve_oar_ocr_bin(explicit_bin: str | None = None) -> str:
    """
    Resolve the `oar-ocr-cli` executable path:
    1. Explicit `explicit_bin` argument
    2. `SOVEREIGN_OAR_OCR_BIN` environment variable
    3. Bundled `sovereign_ocr/bin/oar-ocr-cli` or repo `bin/oar-ocr-cli`
    4. System `PATH` fallback (`"oar-ocr-cli"`)
    """
    if explicit_bin:
        return explicit_bin
    env_bin = os.environ.get("SOVEREIGN_OAR_OCR_BIN")
    if env_bin:
        return env_bin
    candidates = [
        Path(__file__).resolve().parent / "bin" / "oar-ocr-cli",
        Path(__file__).resolve().parents[2] / "bin" / "oar-ocr-cli",
    ]
    for cand in candidates:
        if cand.is_file():
            try:
                if not os.access(cand, os.X_OK):
                    cand.chmod(cand.stat().st_mode | 0o755)
            except Exception:
                pass
            return str(cand)
    return "oar-ocr-cli"


def resolve_oar_models_dir(explicit_dir: str | None = None) -> str:
    """
    Resolve the local ONNX OCR models directory:
    1. Explicit `explicit_dir` argument
    2. `MYCELIA_MODELS_DIR` environment variable
    3. Bundled or repository `models/ocr` directory
    """
    if explicit_dir:
        return explicit_dir
    env_dir = os.environ.get("MYCELIA_MODELS_DIR")
    if env_dir:
        return env_dir
    candidates = [
        Path(__file__).resolve().parent / "models" / "ocr",
        Path(__file__).resolve().parents[2] / "models" / "ocr",
        Path.cwd() / "models" / "ocr",
    ]
    for cand in candidates:
        if cand.is_dir():
            return str(cand.resolve())
    return ""


def run_batch_ocr(
    jobs: list[dict[str, str]],
    oar_bin: str | None = None,
    models_dir: str | None = None,
    cache_dir: Path | None = None,
    chunk_size: int = 16,
) -> dict[str, list[dict[str, Any]]]:
    """
    Execute `oar-ocr-cli --batch-manifest <manifest.json> basic` in chunks of `chunk_size`
    images and return a mapping from `job_id -> list[ocr_run]`.
    """
    if not jobs:
        return {}
    resolved_bin = resolve_oar_ocr_bin(oar_bin)
    resolved_models = resolve_oar_models_dir(models_dir)
    work_dir = cache_dir or Path("/tmp/sovereign_ocr_manifests")
    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if resolved_models:
        env["MYCELIA_MODELS_DIR"] = resolved_models

    normalized_jobs: list[dict[str, str]] = []
    for j in jobs:
        img_p = j.get("image_path") or j.get("path") or ""
        normalized_jobs.append({
            "id": str(j["id"]),
            "image_path": str(img_p),
            "path": str(img_p),
        })

    results: dict[str, list[dict[str, Any]]] = {}
    for i in range(0, len(normalized_jobs), chunk_size):
        chunk = normalized_jobs[i : i + chunk_size]
        manifest_path = work_dir / f"ocr_manifest_{i}.json"
        manifest_path.write_text(json.dumps(chunk))
        proc = subprocess.run(
            [resolved_bin, "--batch-manifest", str(manifest_path), "basic"],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            batch_out = json.loads(proc.stdout)
            for item in batch_out:
                results[item["id"]] = item.get("runs") or []
    return results
