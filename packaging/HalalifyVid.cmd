@echo off
rem HalalifyVid launcher: drag one or more videos onto this file, or
rem double-click it and paste a file path or a link.
setlocal
set "APP=%~dp0halalify.exe"
cd /d "%~dp0"

if not exist "%~dp0data\bin\onnxruntime\onnxruntime.dll" (
    echo First run: installing ONNX Runtime and the models. This happens once.
    echo.
    "%APP%" setup
    if errorlevel 1 goto failed
    echo.
)

if "%~1"=="" goto ask

:each
if "%~1"=="" goto finished
"%APP%" "%~1"
if errorlevel 1 set "PROBLEM=1"
shift
goto each

:ask
set "INPUT="
set /p "INPUT=Drag a video here, or paste a file path or a link, then press Enter: "
if not defined INPUT goto :eof
set "INPUT=%INPUT:"=%"
"%APP%" "%INPUT%"
if errorlevel 1 set "PROBLEM=1"

:finished
echo.
if defined PROBLEM (
    echo Something went wrong; see the messages above.
) else (
    echo Done. Each cleaned video is next to its original, named NAME.halal.mp4
)
pause
exit /b

:failed
echo.
echo Setup failed; see the messages above.
pause
exit /b 1
