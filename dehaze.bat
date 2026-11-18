@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "input=datasets\frog"
set "output=datasets\frog_dehazed"
if not "%~1"=="" set "input=%~1"
if not "%~2"=="" set "output=%~2"
if not exist ".venv\Scripts\python.exe" (
    echo Error: project Python not found in .venv.
    popd
    pause
    exit /b 1
)
".venv\Scripts\python.exe" dehaze.py --input "%input%" --output "%output%"
set "code=%errorlevel%"
echo.
if not "%code%"=="0" echo Failed. Check the error above; existing output is not overwritten.
popd
if /I not "%~3"=="--no-pause" pause
exit /b %code%
