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

echo === Launching Diffucore UI ===
"%VPY%" app.py %*

popd
