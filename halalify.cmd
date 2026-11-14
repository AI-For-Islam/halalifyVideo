@echo off
rem Runs the release build, keeping its runtime and models in .halalify beside
rem this script (unless HALALIFY_HOME already points somewhere else).
setlocal
if not defined HALALIFY_HOME set "HALALIFY_HOME=%~dp0.halalify"
if not exist "%~dp0target\release\halalify.exe" (
    echo halalify is not built yet. Run: cargo build --release
    exit /b 1
)
"%~dp0target\release\halalify.exe" %*
