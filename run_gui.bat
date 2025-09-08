@echo off
echo Starting Cybernetic Planning System GUI...
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo Error: Virtual environment not found
    echo Please run: python -m venv .venv
    echo Then run: .venv\Scripts\activate
    echo Then run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if Python is available in virtual environment
.venv\Scripts\python.exe --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not working in virtual environment
    echo Please reinstall the virtual environment
    pause
    exit /b 1
)

REM Run the GUI
.venv\Scripts\python.exe gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Check the error message above.
    pause
)
