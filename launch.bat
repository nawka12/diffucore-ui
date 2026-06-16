@echo off
setlocal

pushd "%~dp0"

set "VPY=.venv\Scripts\python.exe"

if not exist "%VPY%" (
    echo === Launch aborted ===
    echo No virtualenv found at .venv
    echo Please run setup first:  setup.bat
    popd
    pause
    exit /b 1
)

REM --- keep deps in sync with requirements.txt ---
REM Handles updating with `git pull` instead of update.bat: a newly added dep
REM (e.g. spandrel) would otherwise surface as a runtime import error. Hash-gated
REM via the venv Python, so it's a no-op when requirements.txt is unchanged.
set "STAMP=.venv\.requirements.sha256"
"%VPY%" -c "import hashlib,os,sys; h=hashlib.sha256(open('requirements.txt','rb').read()).hexdigest(); s=r'%STAMP%'; old=open(s).read().strip() if os.path.exists(s) else ''; sys.exit(1 if old!=h else 0)"
if errorlevel 1 (
    echo requirements.txt changed - syncing dependencies...
    "%VPY%" -m pip install -q -r requirements.txt
    "%VPY%" -c "import torch; assert torch.cuda.is_available()" 2>nul
    if errorlevel 1 (
        echo Repairing CUDA torch...
        call :repair_torch
    )
    "%VPY%" -c "import hashlib; open(r'%STAMP%','w').write(hashlib.sha256(open('requirements.txt','rb').read()).hexdigest())"
)

echo === Launching Diffucore UI ===
"%VPY%" backend\app.py --autolaunch %*

popd
exit /b

REM Reinstall CUDA torch from the right index. cu124 wheels stop at sm_90, so
REM Blackwell (RTX 50-series, compute cap 10.0+) needs the cu128 build instead.
:repair_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
where nvidia-smi >nul 2>nul && (
    nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader > "%TEMP%\dc_gpu.txt" 2>nul
    findstr /r "^1[0-9]\." "%TEMP%\dc_gpu.txt" >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    findstr /i /r "RTX 50[0-9][0-9]" "%TEMP%\dc_gpu.txt" >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    del "%TEMP%\dc_gpu.txt" >nul 2>nul
)
"%VPY%" -m pip uninstall -y -q torch torchvision
"%VPY%" -m pip install -q torch torchvision --index-url %TORCH_INDEX%
goto :eof
