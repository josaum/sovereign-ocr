#!/usr/bin/env python3
"""
Build a single self-contained publishable Fat Wheel (`sovereign_ocr-1.0.0-cp38-abi3-macosx_11_0_arm64.whl`)
and a single standalone executable (`dist/sovereign-ocr`) bundling:
1. `sovereign_ocr` Python package (`ingest`, `ocr_layout`, `schemas`, `extractor`, `ontology_engine`, `cli`)
2. All 8 native Rust PyO3 `abi3` extensions (`email_fast`, `pdf_fast`, `pdf_fast_write`,
   `docx_fast`, `xlsx_fast`, `layout_fast`, `quad_fast`, `gliner_fast_py`)
3. Embedded `oar-ocr-cli` native Rust ONNX OCR binary (`sovereign_ocr/bin/oar-ocr-cli`)
"""

from __future__ import annotations

import base64
import hashlib
import os
import stat
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PKG_DIR = REPO_ROOT / "src" / "sovereign_ocr"
DIST_DIR = REPO_ROOT / "dist"

DEFAULT_VENDOR_WHEELS_DIR = Path("/Users/josaum/projects/central-ocr/vendor/wheels")
DEFAULT_OAR_BIN = Path("/Users/josaum/projects/central-ocr/bin/oar-ocr-cli")

BUNDLED_WHEEL_PREFIXES = (
    "email_fast-",
    "pdf_fast-",
    "pdf_fast_write-",
    "docx_fast-",
    "xlsx_fast-",
    "layout_fast-",
    "quad_fast-",
    "gliner_fast-",
)

PKG_NAME = "sovereign_ocr"
PKG_VERSION = "1.0.0"
WHEEL_TAG = "cp38-abi3-macosx_11_0_arm64"


def _sha256_b64_nopad(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build_fat_wheel(
    include_ocr_bin: bool = True,
    vendor_wheels_dir: Path = DEFAULT_VENDOR_WHEELS_DIR,
    oar_bin_path: Path = DEFAULT_OAR_BIN,
) -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    wheel_filename = f"{PKG_NAME}-{PKG_VERSION}-{WHEEL_TAG}.whl"
    wheel_path = DIST_DIR / wheel_filename
    dist_info = f"{PKG_NAME}-{PKG_VERSION}.dist-info"

    records: list[tuple[str, str, int]] = []

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        def add_bytes(arcname: str, data: bytes, executable: bool = False) -> None:
            zi = zipfile.ZipInfo(arcname)
            zi.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if executable else 0o644
            zi.external_attr = (stat.S_IFREG | mode) << 16
            zf.writestr(zi, data)
            records.append((arcname, f"sha256={_sha256_b64_nopad(data)}", len(data)))

        # 1. Add sovereign_ocr Python sources
        for py_file in sorted(SRC_PKG_DIR.rglob("*.py")):
            rel = py_file.relative_to(REPO_ROOT / "src").as_posix()
            add_bytes(rel, py_file.read_bytes())

        # 2. Bundle native Rust PyO3 abi3 extensions from vendor wheels
        if vendor_wheels_dir.is_dir():
            for whl in sorted(vendor_wheels_dir.glob("*.whl")):
                if not any(whl.name.startswith(pref) for pref in BUNDLED_WHEEL_PREFIXES):
                    continue
                with zipfile.ZipFile(whl, "r") as vz:
                    for item in vz.infolist():
                        if item.is_dir() or ".dist-info/" in item.filename:
                            continue
                        data = vz.read(item.filename)
                        is_exec = item.filename.endswith(".so") or item.filename.endswith(".dylib")
                        add_bytes(item.filename, data, executable=is_exec)

        # 3. Embed oar-ocr-cli binary inside sovereign_ocr/bin/oar-ocr-cli
        if include_ocr_bin and oar_bin_path.is_file():
            add_bytes("sovereign_ocr/bin/oar-ocr-cli", oar_bin_path.read_bytes(), executable=True)

        # 4. Write PEP-427 .dist-info metadata
        readme_txt = (REPO_ROOT / "README.md").read_text() if (REPO_ROOT / "README.md").is_file() else ""
        metadata_content = (
            "Metadata-Version: 2.1\n"
            "Name: sovereign-ocr\n"
            f"Version: {PKG_VERSION}\n"
            "Summary: 100% Air-Gapped Sovereign OCR & Neuro-Symbolic Document Intelligence Engine\n"
            "Requires-Python: >=3.10\n"
            "Requires-Dist: numpy>=1.24.0\n"
            "Requires-Dist: Pillow>=10.0.0\n"
            "Requires-Dist: pydantic>=2.5.0\n"
            "Description-Content-Type: text/markdown\n\n"
            f"{readme_txt}\n"
        ).encode("utf-8")
        add_bytes(f"{dist_info}/METADATA", metadata_content)

        wheel_meta = (
            "Wheel-Version: 1.0\n"
            "Generator: sovereign-ocr-fat-wheel-builder\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {WHEEL_TAG}\n"
        ).encode("utf-8")
        add_bytes(f"{dist_info}/WHEEL", wheel_meta)

        entry_points = (
            "[console_scripts]\n"
            "sovereign-ocr = sovereign_ocr.cli:main\n"
            "sovereign-ocr-mcp = sovereign_ocr.cli:main_mcp\n"
        ).encode("utf-8")
        add_bytes(f"{dist_info}/entry_points.txt", entry_points)

        record_lines = [f"{path},{digest},{size}" for path, digest, size in records]
        record_lines.append(f"{dist_info}/RECORD,,")
        record_bytes = ("\n".join(record_lines) + "\n").encode("utf-8")
        zf.writestr(f"{dist_info}/RECORD", record_bytes)

    return wheel_path


def build_single_executable(wheel_path: Path) -> Path:
    """
    Create a single self-extracting standalone PEP-441 executable `dist/sovereign-ocr` that
    unpacks its content-addressed native wheel cache on first run and executes `sovereign_ocr.cli:main`.
    """
    bin_path = DIST_DIR / "sovereign-ocr"
    whl_bytes = wheel_path.read_bytes()
    whl_hash = hashlib.sha256(whl_bytes).hexdigest()[:16]

    main_py = f"""import os, sys, stat, zipfile
from pathlib import Path

CACHE_HASH = "{whl_hash}"
CACHE_DIR = Path(os.environ.get("SOVEREIGN_OCR_CACHE_DIR", Path.home() / ".cache" / "sovereign-ocr" / CACHE_HASH))
MARKER = CACHE_DIR / ".extracted"
ARCHIVE_PATH = Path(__file__).resolve().parent

if not MARKER.is_file():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        zf.extractall(CACHE_DIR)
    oar_bin = CACHE_DIR / "sovereign_ocr" / "bin" / "oar-ocr-cli"
    if oar_bin.is_file():
        oar_bin.chmod(oar_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    MARKER.write_text(CACHE_HASH)

sys.path.insert(0, str(CACHE_DIR))
from sovereign_ocr.cli import main
if __name__ == "__main__":
    main()
""".encode("utf-8")

    import io
    zip_buf = io.BytesIO(whl_bytes)
    with zipfile.ZipFile(zip_buf, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("__main__.py", main_py)

    shebang = b"#!/usr/bin/env python3\n"
    with open(bin_path, "wb") as f:
        f.write(shebang)
        f.write(zip_buf.getvalue())

    bin_path.chmod(bin_path.stat().st_mode | 0o755)
    return bin_path


if __name__ == "__main__":
    whl = build_fat_wheel(include_ocr_bin=True)
    exe = build_single_executable(whl)
    print(f"Built fat wheel:        {whl} ({whl.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Built single binary:    {exe} ({exe.stat().st_size / (1024*1024):.2f} MB)")
