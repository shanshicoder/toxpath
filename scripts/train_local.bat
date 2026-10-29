@echo off
REM ---------------------------------------------------------------------------
REM  ToxPath training launcher (Windows).
REM  Wraps train_local.ps1 with -ExecutionPolicy Bypass so it works even when
REM  the machine forbids running .ps1 scripts. All arguments are forwarded.
REM
REM  Usage:
REM    scripts\train_local.bat -Mode smoke
REM    scripts\train_local.bat -Mode fold -Fold 1 -Epochs 20
REM    scripts\train_local.bat -Mode all -Epochs 20
REM    scripts\train_local.bat -Mode resume -Fold 1
REM ---------------------------------------------------------------------------
setlocal
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_local.ps1" %*
set RC=%ERRORLEVEL%
endlocal & exit /b %RC%
