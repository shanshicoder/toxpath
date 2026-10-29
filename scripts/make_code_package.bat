@echo off
REM ---------------------------------------------------------------------------
REM  Build a code-only zip for copying the project to the lab workstation.
REM  Wraps make_code_package.ps1 with -ExecutionPolicy Bypass.
REM
REM  Usage:
REM    scripts\make_code_package.bat
REM    scripts\make_code_package.bat -OutFile D:\transfer\toxpath_code.zip
REM ---------------------------------------------------------------------------
setlocal
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_code_package.ps1" %*
set RC=%ERRORLEVEL%
endlocal & exit /b %RC%
