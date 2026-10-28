@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "SCRIPT_DIR=%~dp0"
set "BUNDLE_ROOT=%SCRIPT_DIR%.."

if exist "%BUNDLE_ROOT%\runtime-python\python.exe" (
    "%BUNDLE_ROOT%\runtime-python\python.exe" "%SCRIPT_DIR%run-sovereign-ocr.py" %*
    exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -3 "%SCRIPT_DIR%run-sovereign-ocr.py" %*
    exit /b %ERRORLEVEL%
)

python "%SCRIPT_DIR%run-sovereign-ocr.py" %*
exit /b %ERRORLEVEL%
