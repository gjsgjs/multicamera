@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "panorama=output\dark-natural-ges.png"
if not "%~1"=="" set "panorama=%~1"
set "code=1"
if not exist ".venv\Scripts\python.exe" (
    echo Error: project Python not found in .venv.
    goto finish
)
rem Recompute natural/registration images; an existing panorama is never overwritten.
".venv\Scripts\python.exe" enhance_lowlight.py ^
    --input "datasets\dark" --output "datasets\dark_natural_v2" --overwrite ^
    --stitch "%panorama%"
set "code=%errorlevel%"
if not "%code%"=="0" echo Enhancement/stitching failed. See the error above.
:finish
popd
if /I not "%~2"=="--no-pause" pause
exit /b %code%
