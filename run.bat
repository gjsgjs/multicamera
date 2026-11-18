@echo off
setlocal EnableExtensions DisableDelayedExpansion

if /I "%~1"=="--help" goto help

pushd "%~dp0"
if errorlevel 1 exit /b 1

set "exit_code=1"
set "python_exe=.venv\Scripts\python.exe"
set "output=output\dark-ges.png"
if not "%~1"=="" set "output=%~1"

if not exist "%python_exe%" (
    echo Error: project Python not found: %python_exe%
    goto finish
)

for %%F in (00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15) do (
    if not exist "datasets\dark\%%F.png" (
        echo Error: missing input image: datasets\dark\%%F.png
        goto finish
    )
)

echo Stitching 16 overlapping images with SIFT, HED and GES-GSP...
echo Output: "%output%"

"%python_exe%" stitch.py stitch --images ^
    "datasets\dark\00.png" ^
    "datasets\dark\01.png" ^
    "datasets\dark\02.png" ^
    "datasets\dark\03.png" ^
    "datasets\dark\04.png" ^
    "datasets\dark\05.png" ^
    "datasets\dark\06.png" ^
    "datasets\dark\07.png" ^
    "datasets\dark\08.png" ^
    "datasets\dark\09.png" ^
    "datasets\dark\10.png" ^
    "datasets\dark\11.png" ^
    "datasets\dark\12.png" ^
    "datasets\dark\13.png" ^
    "datasets\dark\14.png" ^
    "datasets\dark\15.png" ^
    -o "%output%"
set "exit_code=%errorlevel%"

:finish
echo.
if "%exit_code%"=="0" (
    echo Finished successfully.
) else (
    echo Stitching failed. See the error above.
    echo If the output already exists, use a different output filename.
)
popd
if /I not "%~2"=="--no-pause" pause
exit /b %exit_code%

:help
echo Usage: run.bat [output.png] [--no-pause]
echo Default output: output\dark-ges.png
echo Input images: datasets\dark\00.png through 15.png
exit /b 0
