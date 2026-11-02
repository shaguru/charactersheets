@echo off
setlocal EnableExtensions
title Character Forge - Setup
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup-direct.ps1" %*
set "FORGE_EXIT=%ERRORLEVEL%"
if not "%FORGE_EXIT%"=="0" pause
exit /b %FORGE_EXIT%
