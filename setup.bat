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

REM --- Python version guard: the cu124 torch wheels exist for Python 3.10-3.13 only ---
%PY% -c "import sys; v=sys.version_info[:2]; ok=(3,10)<=v<=(3,13); print(f'Unsupported Python {v[0]}.{v[1]} - the CUDA 12.4 torch build needs Python 3.10-3.13.') if not ok else None; sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo Install a supported Python from https://www.python.org/downloads/ then re-run setup.bat.
    goto :error
)

REM --- submodule ---
if not exist "diffucore\src\diffucore\__init__.py" (
    if not exist ".git" (
        echo ERROR: the diffucore engine submodule is missing and this is not a git clone.
        echo GitHub's "Download ZIP" does not include submodules. Install Git from
        echo https://git-scm.com/ and clone instead:
        echo   git clone --recurse-submodules https://github.com/nawka12/diffucore-ui.git
        goto :error
    )
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
REM Install the CUDA torch build first so requirements.txt / ultralytics /
REM spandrel don't pull the default PyPI wheel (built for a newer CUDA than many
REM drivers run). torchvision is included because spandrel depends on it -
REM pulling it from PyPI would drag in a mismatched torch.
REM Pick the wheel by GPU arch: cu124 wheels stop at sm_90, so Blackwell
REM (RTX 50-series, compute cap 10.0+) needs the cu128 build instead.
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
where nvidia-smi >nul 2>nul && (
    nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader > "%TEMP%\dc_gpu.txt" 2>nul
    findstr /r "^1[0-9]\." "%TEMP%\dc_gpu.txt" >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    findstr /i /r "RTX 50[0-9][0-9]" "%TEMP%\dc_gpu.txt" >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    del "%TEMP%\dc_gpu.txt" >nul 2>nul
)
if /i not "%TORCH_INDEX%"=="https://download.pytorch.org/whl/cu124" echo Detected Blackwell-class GPU - using CUDA 12.8 torch wheels.
echo Downloading the CUDA torch build, ~2.5 GB - this is the slow part...
"%VPY%" -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 goto :error
"%VPY%" -m pip install -r requirements.txt
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
