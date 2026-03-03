@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_one_click.ps1"

if errorlevel 1 (
  echo.
  echo Launch failed. Please check the error output above.
  pause
)

endlocal