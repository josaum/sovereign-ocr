#!/usr/bin/env python3
"""
Cross-Platform Claude Desktop (.mcpb) Launcher for Sovereign-OCR.
Supports macOS ARM64 (darwin-arm64), Windows x64 (win32-x64), and Linux (linux-x64 / linux-arm64).
Executes 100% locally and offline (Air-Gapped).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent


def _is_windows_store_stub(path_str: str) -> bool:
    return "windowsapps" in path_str.replace("\\", "/").lower()


def _select_compatible_python(bundle_root: Path) -> str | None:
    import re
    import shutil

    is_win = os.name == "nt"
    if is_win:
        bundled_win_py = bundle_root / "runtime-python" / "python.exe"
        if bundled_win_py.is_file():
            try:
                if Path(sys.executable).resolve() != bundled_win_py.resolve():
                    return str(bundled_win_py)
            except Exception:
                return str(bundled_win_py)
            return None

    libs = bundle_root / "runtime-libs"
    if not libs.is_dir():
        return None
    tags: set[int] = set()
    for p in libs.rglob("*"):
        if p.suffix in (".so", ".pyd", ".dylib"):
            m = re.search(r"\.(?:cpython-3|cp3)(\d+)-", p.name)
            if m:
                tags.add(int(m.group(1)))

    if not tags or (sys.version_info.major == 3 and sys.version_info.minor in tags):
        return None

    for minor in sorted(tags, reverse=True):
        if is_win:
            win_roots = [
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / f"Python3{minor}",
                Path(f"C:/Python3{minor}"),
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / f"Python3{minor}",
            ]
            for wr in win_roots:
                cand = wr / "python.exe"
                if cand.is_file() and not _is_windows_store_stub(str(cand)):
                    return str(cand)
            for env_key in ("APPDATA", "LOCALAPPDATA"):
                base = os.environ.get(env_key)
                if base:
                    uv_win_dir = Path(base) / "uv" / "python"
                    if uv_win_dir.is_dir():
                        for match in sorted(uv_win_dir.glob(f"cpython-3.{minor}*"), reverse=True):
                            cand = match / "python.exe"
                            if cand.is_file():
                                return str(cand)
        else:
            exe_name = f"python3.{minor}"
            found = shutil.which(exe_name)
            if found and Path(found).is_file() and not _is_windows_store_stub(found):
                return found
            for prefix in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
                candidate = Path(prefix) / exe_name
                if candidate.is_file():
                    return str(candidate)
            uv_py_dir = Path.home() / ".local" / "share" / "uv" / "python"
            if uv_py_dir.is_dir():
                for match in sorted(uv_py_dir.glob(f"cpython-3.{minor}*"), reverse=True):
                    cand = match / "bin" / "python3"
                    if cand.is_file():
                        return str(cand)
    return None


override = os.environ.get("SOVEREIGN_OCR_PYTHON") or _select_compatible_python(BUNDLE_ROOT)
if override and os.environ.get("SOVEREIGN_VENV_LAUNCHED") != "1" and sys.executable != override:
    if Path(override).is_file() and not _is_windows_store_stub(override):
        os.environ["SOVEREIGN_VENV_LAUNCHED"] = "1"
        if os.name == "nt":
            import subprocess

            raise SystemExit(subprocess.run([override, str(Path(__file__).resolve()), *sys.argv[1:]]).returncode)
        os.execv(override, [override, str(Path(__file__).resolve()), *sys.argv[1:]])

SRC_DIR = BUNDLE_ROOT / "src"
LIBS_DIR = BUNDLE_ROOT / "runtime-libs"
BIN_DIR = BUNDLE_ROOT / "runtime-bin"

if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if LIBS_DIR.is_dir() and str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

repo_src = BUNDLE_ROOT.parent.parent / "src"
if repo_src.is_dir() and str(repo_src) not in sys.path:
    sys.path.append(str(repo_src))

oar_exe = BIN_DIR / ("oar-ocr-cli.exe" if os.name == "nt" else "oar-ocr-cli")
if oar_exe.is_file():
    try:
        if os.name != "nt" and not os.access(oar_exe, os.X_OK):
            oar_exe.chmod(oar_exe.stat().st_mode | 0o755)
    except Exception:
        pass
    os.environ.setdefault("SOVEREIGN_OAR_OCR_BIN", str(oar_exe))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONUTF8", "1")

print(
    f"sovereign-ocr-mcpb launcher: python={sys.executable} ({sys.version.split()[0]}) "
    f"platform={sys.platform} bundle={BUNDLE_ROOT} "
    f"src={'ok' if (SRC_DIR / 'sovereign_ocr').is_dir() else 'MISSING'} "
    f"runtime-libs={'ok' if LIBS_DIR.is_dir() else 'absent'}",
    file=sys.stderr,
    flush=True,
)

from sovereign_ocr.mcp_server import run_stdio_server

if __name__ == "__main__":
    raise SystemExit(run_stdio_server())
