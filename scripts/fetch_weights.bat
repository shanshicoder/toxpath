@echo off
REM ---------------------------------------------------------------------------
REM  HistoPLUS weights downloader (Windows).
REM  Wraps fetch_weights.ps1 with -ExecutionPolicy Bypass. All arguments are
REM  forwarded.
REM
REM  Usage:
REM    scripts\fetch_weights.bat
REM    scripts\fetch_weights.bat -Token "ghp_xxx"
REM    scripts\fetch_weights.bat -Force
REM ---------------------------------------------------------------------------
setlocal
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fetch_weights.ps1" %*
set RC=%ERRORLEVEL%
endlocal & exit /b %RC%
