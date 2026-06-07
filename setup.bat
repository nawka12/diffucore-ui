@echo off
setlocal

echo === Diffucore UI Setup ===

pushd "%~dp0"

REM --- pick a Python interpreter ---
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo Be sure to check "Add Python to PATH" during installation, then re-run setup.bat.
    goto :error
)

REM --- submodule ---
if not exist "diffucore\src\diffucore\__init__.py" (
    echo [1/4] Initializing submodules...
    git submodule update --init --recursive
    if errorlevel 1 goto :error
) else (
    echo [1/4] Submodule already present.
)

REM --- venv ---
if not exist ".venv\" (
    echo [2/4] Creating virtualenv at .venv ...
    %PY% -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo [2/4] Virtualenv already exists.
)

set "VPY=.venv\Scripts\python.exe"

REM --- pip deps ---
echo [3/4] Installing Python dependencies...
"%VPY%" -m pip install --upgrade pip -q
if errorlevel 1 goto :error
REM Install the cu124 torch build first so requirements.txt / ultralytics don't
REM pull the default PyPI wheel (built for a newer CUDA than many drivers run).
"%VPY%" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :error
"%VPY%" -m pip install -q -r requirements.txt
if errorlevel 1 goto :error
"%VPY%" -m pip install -q -e diffucore
if errorlevel 1 goto :error

REM --- CUDA torch sanity check ---
"%VPY%" -c "import torch; assert torch.cuda.is_available()" 2>nul
if errorlevel 1 (
    echo [4/4] WARNING: torch present but CUDA unavailable ^(check NVIDIA driver vs cu124^).
) else (
    echo [4/4] CUDA torch OK.
)

echo.
echo === Setup complete ===
echo Run the UI:  launch.bat
popd
pause
exit /b 0

:error
echo.
echo === Setup failed ===
popd
pause
exit /b 1
