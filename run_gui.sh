#!/bin / bash
echo "Starting Cybernetic Planning System GUI..."

# Check if virtual environment exists
if [ ! -f "C:\Users\marce\Desktop\Cybernetic-Planning-Experiment\.venv\Scripts\python.exe" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run the installation wizard again"
    exit 1
fi

# Run the GUI
"C:\Users\marce\Desktop\Cybernetic-Planning-Experiment\.venv\Scripts\python.exe" gui.py
