@echo off
"%~dp0runtime\python.exe" "%~dp0bootstrap.py" %*
if errorlevel 1 (
    if "%~1"=="" pause
    exit /b 1
)
