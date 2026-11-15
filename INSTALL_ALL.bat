@echo off
setlocal
cd /d "%~dp0"
call SETUP.bat
if errorlevel 1 exit /b 1
call SETUP_MINIMAX.bat
if errorlevel 1 exit /b 1
call START.bat
exit /b %ERRORLEVEL%
