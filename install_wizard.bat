@echo off
echo.
echo ================================================================================
echo                    CYBERNETIC PLANNING SYSTEM
echo                        Installation Wizard
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.9 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Detected Python version: %PYTHON_VERSION%

REM Run the installation wizard
echo.
echo Starting installation wizard...
echo.
python install_wizard.py

REM Check if installation was successful
if errorlevel 1 (
    echo.
    echo Installation failed. Check the error messages above.
    echo Installation log saved to: logs\installation.log
    echo.
    pause
    exit /b 1
)

echo.
echo Installation completed successfully!
echo.
echo You can now run the system using: run_gui.bat
echo.
pause
