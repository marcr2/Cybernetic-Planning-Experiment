#!/bin/bash

echo ""
echo "================================================================================"
echo "                    CYBERNETIC PLANNING SYSTEM"
echo "                        Installation Wizard"
echo "================================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo ""
        echo "Please install Python 3.9 or higher:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "  CentOS/RHEL:   sudo yum install python3 python3-pip"
        echo "  macOS:         brew install python3"
        echo "  Or download from: https://www.python.org/downloads/"
        echo ""
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "Detected Python version: $PYTHON_VERSION"

# Check if version is 3.9+
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d'.' -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 9 ]); then
    echo "ERROR: Python 3.9 or higher is required. Found: $PYTHON_VERSION"
    echo ""
    echo "Please upgrade Python to version 3.9 or higher."
    exit 1
fi

# Run the installation wizard
echo ""
echo "Starting installation wizard..."
echo ""
$PYTHON_CMD install_wizard.py

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "Installation failed. Check the error messages above."
    echo "Installation log saved to: logs/installation.log"
    echo ""
    exit 1
fi

echo ""
echo "Installation completed successfully!"
echo ""
echo "You can now run the system using: ./run_gui.sh"
echo ""
