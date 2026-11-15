@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup-minimax.ps1" %*
set "MVM_EXIT=%ERRORLEVEL%"
if not "%MVM_EXIT%"=="0" pause
exit /b %MVM_EXIT%
