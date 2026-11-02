@echo off
setlocal EnableExtensions
title Character Forge - Install and Run
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start.ps1" %*
set "FORGE_EXIT=%ERRORLEVEL%"
if not "%FORGE_EXIT%"=="0" (
  echo.
  echo Character Forge could not start. See the message above and data\logs\setup.log.
  pause
)
exit /b %FORGE_EXIT%
