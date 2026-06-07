@echo off
setlocal

echo === Diffucore UI Update ===

pushd "%~dp0"

set "VPY=.venv\Scripts\python.exe"

if not exist "%VPY%" (
    echo No virtualenv found at .venv
    echo Please run setup first:  setup.bat
    popd
    pause
    exit /b 1
)

REM --- pull latest UI code ---
echo [1/4] Pulling latest changes...
git pull --ff-only
if errorlevel 1 goto :error

REM --- sync submodule to the pinned revision ---
echo [2/4] Updating submodules...
git submodule update --init --recursive
if errorlevel 1 goto :error

REM --- refresh deps (requirements and the editable engine may have changed) ---
echo [3/4] Updating Python dependencies...
"%VPY%" -m pip install --upgrade pip -q
if errorlevel 1 goto :error
"%VPY%" -m pip install -q -r requirements.txt
if errorlevel 1 goto :error
"%VPY%" -m pip install -q -e diffucore
if errorlevel 1 goto :error

REM --- ensure CUDA torch is still present ---
REM On failure, uninstall first: a bare `pip install torch` is a no-op when a
REM mismatched wheel is already installed, so it could never repair a CPU build.
"%VPY%" -c "import torch; assert torch.cuda.is_available()" 2>nul
if errorlevel 1 (
    echo [4/4] Reinstalling CUDA torch...
    "%VPY%" -m pip uninstall -y -q torch
    "%VPY%" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 goto :error
) else (
    echo [4/4] CUDA torch OK.
)

echo.
echo === Update complete ===
echo Relaunch the UI:  launch.bat
popd
pause
exit /b 0

:error
echo.
echo === Update failed ===
popd
pause
exit /b 1
