#!/usr/bin/env python3
"""
Build Native Claude Desktop Double-Click Extension Bundles (`.mcpb` v0.4) for `sovereign-ocr`
across macOS ARM64 (`darwin-arm64`), Windows x64 (`win32-x64`), Linux x64 (`linux-x64`),
and Linux ARM64 (`linux-arm64`).

Outputs in `dist/`:
  - `dist/sovereign-ocr-darwin-arm64.mcpb` (and `dist/sovereign-ocr.mcpb`)
  - `dist/sovereign-ocr-win32-x64.mcpb`
  - `dist/sovereign-ocr-linux-x64.mcpb`
  - `dist/sovereign-ocr-linux-arm64.mcpb`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ZIP_DATE = (1980, 1, 1, 0, 0, 0)

REPO_ROOT = Path(__file__).resolve().parents[1]
MCPB_DIR = REPO_ROOT / "packaging" / "mcpb"
DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build" / "mcpb"

WHEEL_FIND_DIRS = [
    Path("/Users/josaum/projects/central-ocr/vendor/wheels"),
    Path("/Users/josaum/projects/health-audit/vendor/parsers-rs"),
    Path("/Users/josaum/projects/health-audit/vendor/wheels"),
    Path("/Users/josaum/projects/olaya/dist"),
]

WIN_PYTHON_EMBED_ZIP = Path("/Users/josaum/projects/central-ma-toolkit/vendor/python-embed/python-3.12.10-embed-amd64.zip")
MACOS_OAR_BIN = Path("/Users/josaum/projects/central-ocr/bin/oar-ocr-cli")

CORE_PARSERS = [
    "pdf-fast==0.2.4",
    "email-fast==0.2.6",
    "docx-fast==0.2.5",
    "xlsx-fast==0.2.5",
    "ontology-fast==0.1.0",
    "leio-decision==0.8.0",
]

PLATFORMS: dict[str, dict[str, Any]] = {
    "darwin-arm64": {
        "mcpb_platform": "darwin",
        "uv_python_platform": "aarch64-apple-darwin",
        "uv_python_version": "3.12",
        "python_embed": False,
        "include_oar_bin": True,
        "extra_packages": [
            "pdf-fast-write==0.2.0",
            "layout-fast==0.2.0",
            "align-fast==0.2.0",
            "owl-fast==0.2.0",
            "quad-fast==0.1.0",
            "gliner-fast==0.2.0",
        ],
    },
    "win32-x64": {
        "mcpb_platform": "win32",
        "uv_python_platform": "x86_64-pc-windows-msvc",
        "uv_python_version": "3.12",
        "python_embed": True,
        "include_oar_bin": False,
        "extra_packages": [],
    },
    "linux-x64": {
        "mcpb_platform": "linux",
        "uv_python_platform": "x86_64-manylinux_2_34",
        "uv_python_version": "3.12",
        "python_embed": False,
        "include_oar_bin": False,
        "extra_packages": [
            "pdf-fast-write==0.2.0",
            "align-fast==0.2.0",
        ],
    },
    "linux-arm64": {
        "mcpb_platform": "linux",
        "uv_python_platform": "aarch64-manylinux_2_34",
        "uv_python_version": "3.12",
        "python_embed": False,
        "include_oar_bin": False,
        "extra_packages": [
            "pdf-fast-write==0.2.0",
            "align-fast==0.2.0",
        ],
    },
}


def ensure_icon_png(icon_path: Path) -> None:
    """Generate a crisp 512x512 Sovereign-OCR icon for Claude Desktop if not already present."""
    if icon_path.is_file() and icon_path.stat().st_size > 1024:
        return
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (512, 512), (13, 17, 28, 255))
    draw = ImageDraw.Draw(img)

    # Outer rounded border frame
    draw.rounded_rectangle([24, 24, 488, 488], radius=84, fill=(20, 27, 45, 255), outline=(56, 189, 248, 255), width=6)

    # Document sheet emblem
    draw.rounded_rectangle([120, 88, 392, 424], radius=24, fill=(30, 41, 69, 255), outline=(16, 185, 129, 255), width=5)

    # Horizontal text scan lines inside document
    for idx, y in enumerate([148, 196, 244, 292]):
        color = (56, 189, 248, 255) if idx % 2 == 0 else (16, 185, 129, 255)
        draw.rounded_rectangle([156, y, 356, y + 18], radius=8, fill=color)

    # Laser OCR scan beam across center
    draw.line([(88, 268), (424, 268)], fill=(250, 204, 21, 255), width=8)

    # RDF 1.2 Quad Graph Provenance nodes at bottom of document
    nodes = [(180, 368), (256, 344), (332, 368)]
    draw.line([nodes[0], nodes[1], nodes[2]], fill=(250, 204, 21, 255), width=4)
    for nx, ny in nodes:
        draw.ellipse([nx - 14, ny - 14, nx + 14, ny + 14], fill=(16, 185, 129, 255), outline=(250, 204, 21, 255), width=4)

    img.save(icon_path, format="PNG")


def ensure_runtime_libs(platform_key: str) -> Path:
    """Resolve and stage platform-specific binary wheels into `build/mcpb/runtime-libs-<platform_key>`."""
    spec = PLATFORMS[platform_key]
    libs_dir = BUILD_DIR / f"runtime-libs-{platform_key}"
    base_reqs = [
        "pydantic>=2.10",
        "numpy>=1.26",
        "Pillow>=10.0",
    ]
    native_reqs = [
        *CORE_PARSERS,
        *spec["extra_packages"],
    ]
    stamp_payload = json.dumps(
        {"platform": platform_key, "base": base_reqs, "native": native_reqs, "v": 2},
        sort_keys=True,
    )
    stamp = hashlib.sha256(stamp_payload.encode("utf-8")).hexdigest()
    stamp_file = libs_dir / ".build-stamp"
    if libs_dir.is_dir() and stamp_file.is_file() and stamp_file.read_text(encoding="utf-8").strip() == stamp:
        return libs_dir

    if libs_dir.exists():
        shutil.rmtree(libs_dir)
    libs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Install closed PyPI binary wheels (pydantic, numpy, Pillow)
    cmd_base = [
        "uv",
        "pip",
        "install",
        "--quiet",
        "--target",
        str(libs_dir),
        "--python-platform",
        spec["uv_python_platform"],
        "--python-version",
        spec["uv_python_version"],
        "--only-binary",
        ":all:",
        *base_reqs,
    ]
    print(f"[{platform_key}] Staging PyPI runtime-libs via uv ({', '.join(base_reqs)})...")
    subprocess.run(cmd_base, check=True)

    # 2. Unpack vendored Rust PyO3 abi3 wheels + leio-decision (olaya) with --no-deps
    #    For darwin-arm64, prioritize central-ocr/vendor/wheels so full binary wheels are used.
    find_dirs = (
        [
            Path("/Users/josaum/projects/central-ocr/vendor/wheels"),
            Path("/Users/josaum/projects/olaya/dist"),
        ]
        if platform_key == "darwin-arm64"
        else WHEEL_FIND_DIRS
    )
    cmd_native = [
        "uv",
        "pip",
        "install",
        "--quiet",
        "--no-deps",
        "--target",
        str(libs_dir),
        "--python-platform",
        spec["uv_python_platform"],
        "--python-version",
        spec["uv_python_version"],
        "--only-binary",
        ":all:",
    ]
    for wdir in find_dirs:
        if wdir.is_dir():
            cmd_native.extend(["--find-links", str(wdir)])
    cmd_native.extend(native_reqs)

    print(f"[{platform_key}] Staging sovereign Rust abi3 wheels ({len(native_reqs)} crates)...")
    subprocess.run(cmd_native, check=True)

    stamp_file.write_text(stamp + "\n", encoding="utf-8")
    return libs_dir


def _configure_windows_pth_files(py_dir: Path) -> None:
    """
    Configure `python312._pth` inside the Windows embeddable Python directory so that:
    1. `../src` and `../runtime-libs` are natively on `sys.path` even in Windows isolated `_pth` mode.
    2. `import site` is enabled so `.pth` files and standard site initialization work cleanly.
    """
    desired_entries = ["python312.zip", ".", "../src", "../runtime-libs", "import site"]
    pth_files = list(py_dir.glob("python3*._pth")) or [py_dir / "python312._pth"]
    for pth in pth_files:
        existing = (
            [ln.strip() for ln in pth.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
            if pth.is_file()
            else []
        )
        zip_entry = next((ln for ln in existing if ln.endswith(".zip")), "python312.zip")
        final_lines = [zip_entry, ".", "../src", "../runtime-libs", "import site"]
        pth.write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def ensure_windows_python_embed() -> Path:
    """Unpack the embeddable Windows x64 Python 3.12 runtime for zero-prerequisite Windows double-click installs."""
    py_dir = BUILD_DIR / "runtime-python-win32-x64"
    if not (py_dir.is_dir() and (py_dir / "python.exe").is_file()):
        if not WIN_PYTHON_EMBED_ZIP.is_file():
            raise FileNotFoundError(f"Missing Windows embeddable Python archive: {WIN_PYTHON_EMBED_ZIP}")
        if py_dir.exists():
            shutil.rmtree(py_dir)
        py_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(WIN_PYTHON_EMBED_ZIP, "r") as zf:
            zf.extractall(py_dir)
    _configure_windows_pth_files(py_dir)
    return py_dir


def materialize_manifest(platform_key: str) -> Path:
    """Materialize platform-specific `manifest.json` alongside `icon.png` for `@anthropic-ai/mcpb validate`."""
    spec = PLATFORMS[platform_key]
    manifest = json.loads((MCPB_DIR / "manifest.json").read_text(encoding="utf-8"))
    cfg = dict(manifest["server"]["mcp_config"])
    if spec["python_embed"]:
        cfg["command"] = "${__dirname}/runtime-python/python.exe"
        cfg["args"] = ["${__dirname}/bin/run-sovereign-ocr.py"]
        cfg["platform_overrides"] = {
            "win32": {
                "command": "${__dirname}/runtime-python/python.exe",
                "args": ["${__dirname}/bin/run-sovereign-ocr.py"],
            }
        }
    else:
        cfg["command"] = "python3"
        cfg["args"] = ["${__dirname}/bin/run-sovereign-ocr"]
        cfg.pop("platform_overrides", None)
    manifest["server"] = {**manifest["server"], "mcp_config": cfg}
    manifest["compatibility"] = {**manifest["compatibility"], "platforms": [spec["mcpb_platform"]]}

    out_dir = BUILD_DIR / f"manifest-{platform_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_icon_png(MCPB_DIR / "icon.png")
    shutil.copy2(MCPB_DIR / "icon.png", out_dir / "icon.png")
    out_manifest = out_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_manifest


def collect_bundle_entries(platform_key: str) -> list[tuple[Path, str, int]]:
    """Collect `(source_path, archive_rel_path, unix_mode)` for a platform's `.mcpb` archive."""
    spec = PLATFORMS[platform_key]
    entries: list[tuple[Path, str, int]] = []

    manifest_path = materialize_manifest(platform_key)
    entries.append((manifest_path, "manifest.json", 0o644))
    entries.append((MCPB_DIR / "icon.png", "icon.png", 0o644))
    entries.append((MCPB_DIR / "README.md", "README.md", 0o644))

    # Launcher scripts (POSIX + Python + Windows .cmd)
    entries.append((MCPB_DIR / "bin" / "run-sovereign-ocr", "bin/run-sovereign-ocr", 0o755))
    entries.append((MCPB_DIR / "bin" / "run-sovereign-ocr.py", "bin/run-sovereign-ocr.py", 0o755))
    if (MCPB_DIR / "bin" / "run-sovereign-ocr.cmd").is_file():
        entries.append((MCPB_DIR / "bin" / "run-sovereign-ocr.cmd", "bin/run-sovereign-ocr.cmd", 0o755))

    # Sovereign-OCR Python package sources
    src_pkg = REPO_ROOT / "src" / "sovereign_ocr"
    for p in sorted(src_pkg.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(REPO_ROOT).as_posix()
        entries.append((p, rel, 0o644))

    # SAP BPMN 2.0 & DMN 1.3 workflow definitions
    wf_dir = REPO_ROOT / "workflows"
    if wf_dir.is_dir():
        for p in sorted(wf_dir.glob("*")):
            if p.is_file() and not p.name.startswith("."):
                entries.append((p, f"workflows/{p.name}", 0o644))

    # Native `oar-ocr-cli` binary (when available for target)
    if spec["include_oar_bin"] and MACOS_OAR_BIN.is_file():
        entries.append((MACOS_OAR_BIN, "runtime-bin/oar-ocr-cli", 0o755))

    # Vendored `runtime-libs`
    libs_dir = ensure_runtime_libs(platform_key)
    for p in sorted(libs_dir.rglob("*")):
        rel = p.relative_to(libs_dir)
        if not p.is_file() or "__pycache__" in p.parts or rel.parts[0] == "bin":
            continue
        if "tests" in rel.parts[1:] or rel.suffix in (".pyi", ".pyc") or rel.name == ".build-stamp":
            continue
        is_native = p.suffix in (".so", ".dylib", ".pyd", ".dll")
        entries.append((p, f"runtime-libs/{rel.as_posix()}", 0o755 if is_native else 0o644))

    # Embedded Windows Python 3.12 runtime
    if spec["python_embed"]:
        py_dir = ensure_windows_python_embed()
        for p in sorted(py_dir.rglob("*")):
            if p.is_file():
                mode = 0o755 if p.suffix.lower() in (".exe", ".dll", ".pyd") else 0o644
                entries.append((p, f"runtime-python/{p.relative_to(py_dir).as_posix()}", mode))

    return entries


def build_platform_mcpb(platform_key: str) -> Path:
    """Build, validate (`@anthropic-ai/mcpb`), and output `dist/sovereign-ocr-<platform_key>.mcpb`."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_mcpb = DIST_DIR / f"sovereign-ocr-{platform_key}.mcpb"
    tmp_mcpb = out_mcpb.with_suffix(".mcpb.tmp")

    entries = collect_bundle_entries(platform_key)
    hashes: dict[str, str] = {}
    seen: set[str] = set()

    for src, rel, _ in entries:
        if rel in seen:
            raise ValueError(f"Duplicate archive entry in {platform_key}: {rel}")
        seen.add(rel)
        hashes[rel] = hashlib.sha256(src.read_bytes()).hexdigest()

    release_doc = {
        "product": "sovereign-ocr",
        "version": "1.0.0",
        "manifest_version": "0.4",
        "platform": platform_key,
        "mcpb_platform": PLATFORMS[platform_key]["mcpb_platform"],
        "doi": "10.5281/zenodo.23589075",
        "author": "Dionisio Chiuratto Agourakis",
        "sha256": hashes,
    }
    release_bytes = (json.dumps(release_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with zipfile.ZipFile(tmp_mcpb, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, rel, mode in entries:
            info = zipfile.ZipInfo(rel, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | mode) << 16
            zf.writestr(info, src.read_bytes())

        r_info = zipfile.ZipInfo("release.json", date_time=ZIP_DATE)
        r_info.compress_type = zipfile.ZIP_DEFLATED
        r_info.external_attr = (stat.S_IFREG | 0o644) << 16
        zf.writestr(r_info, release_bytes)

    if out_mcpb.exists():
        out_mcpb.unlink()
    tmp_mcpb.rename(out_mcpb)

    if platform_key == "darwin-arm64":
        canonical_mcpb = DIST_DIR / "sovereign-ocr.mcpb"
        shutil.copy2(out_mcpb, canonical_mcpb)

    # Validate manifest & bundle with official @anthropic-ai/mcpb CLI
    manifest_path = BUILD_DIR / f"manifest-{platform_key}" / "manifest.json"
    val_res = subprocess.run(["npx", "@anthropic-ai/mcpb", "validate", str(manifest_path)], capture_output=True, text=True)
    if val_res.returncode != 0:
        raise RuntimeError(f"mcpb validate failed for {platform_key}:\n{val_res.stdout}\n{val_res.stderr}")

    size_mb = out_mcpb.stat().st_size / (1024 * 1024)
    sha = hashlib.sha256(out_mcpb.read_bytes()).hexdigest()[:16]
    print(f"[{platform_key}] Built & validated: {out_mcpb.name} ({size_mb:.2f} MB, sha256={sha}..., files={len(entries)})")
    return out_mcpb


def verify_unpacked_darwin_mcpb(mcpb_path: Path) -> None:
    """Extract `sovereign-ocr-darwin-arm64.mcpb` to an isolated temp dir and run a live JSON-RPC 2.0 test."""
    with tempfile.TemporaryDirectory(prefix="sovereign-mcpb-test-") as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(mcpb_path, "r") as zf:
            zf.extractall(tmp)
        launcher = tmp / "bin" / "run-sovereign-ocr"
        launcher.chmod(0o755)
        rpc_input = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "claude-desktop-test", "version": "1.0.0"},
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "sovereign_ocr_doctor", "arguments": {}},
                }
            )
            + "\n"
        )
        proc = subprocess.run(
            ["python3", str(launcher)],
            input=rpc_input,
            capture_output=True,
            text=True,
            cwd=str(tmp),
            timeout=30,
            check=True,
        )
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        assert len(lines) == 2, f"Expected 2 JSON-RPC responses, got {len(lines)}: {proc.stdout}"
        init_resp = json.loads(lines[0])
        doc_resp = json.loads(lines[1])
        assert init_resp["result"]["serverInfo"]["name"] == "sovereign-ocr"
        assert doc_resp["result"]["structuredContent"]["status"] == "ok"
        assert doc_resp["result"]["structuredContent"]["modules"]["pdf_fast"] is True
        print(f"[darwin-arm64] Live unpacked Claude Desktop JSON-RPC handshake & doctor tool verified!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Claude Desktop .mcpb bundles for sovereign-ocr")
    parser.add_argument(
        "--platform",
        choices=["all", *PLATFORMS.keys()],
        default="all",
        help="Target platform to build (default: all)",
    )
    args = parser.parse_args()
    ensure_icon_png(MCPB_DIR / "icon.png")

    targets = list(PLATFORMS.keys()) if args.platform == "all" else [args.platform]
    for tgt in targets:
        mcpb_path = build_platform_mcpb(tgt)
        if tgt == "darwin-arm64" and sys.platform == "darwin":
            verify_unpacked_darwin_mcpb(mcpb_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
