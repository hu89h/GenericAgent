@echo off
setlocal
cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..\..") do set "PROJECT_DIR=%%~fI"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_windows.ps1" -ProjectDir "%PROJECT_DIR%" -Mode PrepareOnly -SkipNpmInstall %*
echo.
echo If the installer failed, copy the error above and send it to the developer.
pause