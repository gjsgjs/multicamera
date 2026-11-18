@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "output=output\frog-dehazed-ges.png"
if not "%~1"=="" set "output=%~1"
set "code=1"
if not exist ".venv\Scripts\python.exe" (
    echo Error: project Python not found in .venv.
    goto finish
)
for %%F in (00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15) do (
    if not exist "datasets\frog_dehazed\%%F.png" (
        echo Error: missing datasets\frog_dehazed\%%F.png. Run dehaze.bat first.
        goto finish
    )
)
".venv\Scripts\python.exe" stitch.py stitch --images ^
    "datasets\frog_dehazed\00.png" "datasets\frog_dehazed\01.png" ^
    "datasets\frog_dehazed\02.png" "datasets\frog_dehazed\03.png" ^
    "datasets\frog_dehazed\04.png" "datasets\frog_dehazed\05.png" ^
    "datasets\frog_dehazed\06.png" "datasets\frog_dehazed\07.png" ^
    "datasets\frog_dehazed\08.png" "datasets\frog_dehazed\09.png" ^
    "datasets\frog_dehazed\10.png" "datasets\frog_dehazed\11.png" ^
    "datasets\frog_dehazed\12.png" "datasets\frog_dehazed\13.png" ^
    "datasets\frog_dehazed\14.png" "datasets\frog_dehazed\15.png" ^
    -o "%output%"
set "code=%errorlevel%"
:finish
popd
if /I not "%~2"=="--no-pause" pause
exit /b %code%
