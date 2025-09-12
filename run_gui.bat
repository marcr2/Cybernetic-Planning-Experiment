@echo off
echo Starting Cybernetic Planning System GUI...
echo.

REM Check if virtual environment exists
if not exist "C:\Users\marce\Desktop\Cybernetic-Planning-Experiment\.venv\Scripts\python.exe" (
    echo Error: Virtual environment not found
    echo Please run the installation wizard again
    pause
    exit / b 1
)

REM Run the GUI
"C:\Users\marce\Desktop\Cybernetic-Planning-Experiment\.venv\Scripts\python.exe" gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Check the error message above.
    pause
)
