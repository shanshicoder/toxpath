@echo off
REM ---------------------------------------------------------------------------
REM  ToxPath GitHub sync launcher (Windows).
REM  Wraps sync_github.ps1 with -ExecutionPolicy Bypass so it works even when
REM  the machine forbids running .ps1 scripts. All arguments are forwarded.
REM
REM  Usage:
REM    scripts\sync_github.bat
REM    scripts\sync_github.bat -Message "fix: sampling weights"
REM    scripts\sync_github.bat -DryRun
REM ---------------------------------------------------------------------------
setlocal
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_github.ps1" %*
set RC=%ERRORLEVEL%
endlocal & exit /b %RC%
