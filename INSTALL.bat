@echo off
rem First-time installation and subsequent launches use the same resumable setup.
call "%~dp0START.bat" %*
exit /b %ERRORLEVEL%
