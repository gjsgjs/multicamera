@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1
set "demo_python=runtime\python.exe"
if not exist "%demo_python%" set "demo_python=.venv\Scripts\python.exe"
if not exist "%demo_python%" (
    echo Error: Python runtime not found. Extract the complete demo ZIP first.
    popd
    pause
    exit /b 1
)
echo Starting local stitching UI. Keep this window open.
echo URL: http://127.0.0.1:8765
"%demo_python%" -B "app\server.py" --open %*
set "exit_code=%errorlevel%"
popd
pause
exit /b %exit_code%
