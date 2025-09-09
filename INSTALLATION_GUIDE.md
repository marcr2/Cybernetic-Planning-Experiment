# Cybernetic Planning System - Installation Guide

This comprehensive guide will walk you through installing the Cybernetic Planning System on your computer.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Manual Installation](#manual-installation)
4. [Platform-Specific Instructions](#platform-specific-instructions)
5. [Troubleshooting](#troubleshooting)
6. [Post-Installation](#post-installation)
7. [Uninstallation](#uninstallation)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.9 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free disk space
- **Internet**: Required for downloading dependencies

### Recommended Requirements
- **Memory**: 8GB+ RAM for large datasets
- **Storage**: 5GB+ free disk space
- **CPU**: Multi-core processor for optimization calculations
- **Graphics**: Dedicated graphics card for visualization (optional)

## Quick Installation

### 🚀 Automated Installation Wizard

The easiest way to install the system is using our automated installation wizard.

#### Windows Users

1. **Download the project** from the repository
2. **Extract the files** to your desired location (e.g., `C:\Users\YourName\Desktop\`)
3. **Navigate to the project folder** in File Explorer
4. **Double-click `install_wizard.bat`**
5. **Follow the on-screen instructions**

The wizard will:
- Check your Python installation
- Create a virtual environment
- Install all dependencies
- Set up configuration files
- Validate the installation
- Create launcher scripts

#### Linux/macOS Users

1. **Download the project** from the repository
2. **Extract the files** to your desired location
3. **Open a terminal** in the project directory
4. **Make the script executable**:
   ```bash
   chmod +x install_wizard.sh
   ```
5. **Run the installation wizard**:
   ```bash
   ./install_wizard.sh
   ```
6. **Follow the on-screen instructions**

### What Happens During Installation

The installation wizard performs these steps automatically:

1. **System Check**
   - Verifies Python version (3.9+ required)
   - Checks available disk space and memory
   - Validates system compatibility

2. **Environment Setup**
   - Creates a Python virtual environment (`.venv/`)
   - Upgrades pip to the latest version
   - Sets up project directory structure

3. **Dependency Installation**
   - Installs all required Python packages
   - Installs the project in development mode
   - Verifies all imports work correctly

4. **Configuration**
   - Creates system configuration files
   - Sets up API key templates
   - Generates security keys
   - Creates `.gitignore` file

5. **Validation**
   - Tests core functionality
   - Validates data processing capabilities
   - Checks GUI components
   - Runs performance tests

6. **Finalization**
   - Creates launcher scripts
   - Generates installation log
   - Displays success message

## Manual Installation

If the automated wizard doesn't work or you prefer manual control:

### Step 1: Install Python

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Verify installation: Open Command Prompt and run `python --version`

#### macOS
1. Install Homebrew if you don't have it:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```bash
   brew install python3
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### Linux (CentOS/RHEL)
```bash
sudo yum install python3 python3-pip
```

### Step 2: Download the Project

1. **Clone the repository** (if you have Git):
   ```bash
   git clone <repository-url>
   cd Cybernetic-Planning-Experiment
   ```

2. **Or download the ZIP file**:
   - Download from the repository
   - Extract to your desired location
   - Navigate to the extracted folder

### Step 3: Create Virtual Environment

```bash
# Windows
py -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### Step 5: Set Up Configuration

```bash
# Create necessary directories
mkdir -p data exports logs cache outputs config

# Run configuration setup
python -c "from src.cybernetic_planning.utils.config_setup import SystemInitializer; SystemInitializer(Path('.')).initialize_system()"
```

### Step 6: Verify Installation

```bash
# Test imports
python -c "from src.cybernetic_planning.planning_system import CyberneticPlanningSystem; print('Installation successful!')"

# Run validation
python -c "from src.cybernetic_planning.utils.system_validator import SystemValidator; validator = SystemValidator(Path('.'), 'python'); validator.run_comprehensive_validation()"
```

## Platform-Specific Instructions

### Windows

#### Prerequisites
- Windows 10 or later
- Python 3.9+ with pip
- Visual Studio Build Tools (for some packages)

#### Installation Steps
1. Run `install_wizard.bat` as administrator if needed
2. If you get permission errors, run PowerShell as administrator
3. Enable execution of scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### Common Issues
- **"Python not found"**: Add Python to your PATH environment variable
- **"Permission denied"**: Run Command Prompt as administrator
- **"Microsoft Visual C++ 14.0 is required"**: Install Visual Studio Build Tools

### macOS

#### Prerequisites
- macOS 10.14 or later
- Xcode Command Line Tools
- Homebrew (recommended)

#### Installation Steps
1. Install Xcode Command Line Tools: `xcode-select --install`
2. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
3. Install Python: `brew install python3`
4. Run the installation wizard: `./install_wizard.sh`

#### Common Issues
- **"Command not found"**: Add Homebrew to your PATH
- **"Permission denied"**: Use `chmod +x install_wizard.sh`
- **"SSL certificate error"**: Update certificates or use `--trusted-host` flag

### Linux

#### Prerequisites
- Ubuntu 18.04+ or equivalent
- Python 3.9+ with pip
- Build tools (gcc, make)

#### Installation Steps
1. Update package lists: `sudo apt update`
2. Install build tools: `sudo apt install build-essential python3-dev`
3. Install Python: `sudo apt install python3 python3-pip python3-venv`
4. Run the installation wizard: `./install_wizard.sh`

#### Common Issues
- **"Package not found"**: Update package lists with `sudo apt update`
- **"Permission denied"**: Use `sudo` for system-wide installation or fix permissions
- **"Import error"**: Install missing system libraries

## Troubleshooting

### Common Installation Issues

#### 1. Python Version Issues
**Problem**: "Python 3.9+ required"
**Solution**: 
- Check your Python version: `python --version`
- Install Python 3.9+ from [python.org](https://www.python.org/downloads/)
- Make sure Python is in your PATH

#### 2. Virtual Environment Issues
**Problem**: "Virtual environment not found"
**Solution**:
- Delete the `.venv` folder
- Recreate it: `python -m venv .venv`
- Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)

#### 3. Dependency Installation Issues
**Problem**: "Failed to install dependencies"
**Solution**:
- Update pip: `python -m pip install --upgrade pip`
- Clear pip cache: `pip cache purge`
- Install dependencies one by one to identify problematic packages
- Use `--no-cache-dir` flag: `pip install --no-cache-dir -r requirements.txt`

#### 4. Import Errors
**Problem**: "ModuleNotFoundError"
**Solution**:
- Ensure virtual environment is activated
- Reinstall the project: `pip install -e .`
- Check Python path: `python -c "import sys; print(sys.path)"`

#### 5. Permission Issues
**Problem**: "Permission denied" or "Access denied"
**Solution**:
- Run as administrator (Windows) or with sudo (Linux/macOS)
- Check file permissions
- Ensure you have write access to the project directory

### Getting Help

1. **Check the installation log**: Look in `logs/installation.log`
2. **Run validation**: Use the system validator to identify issues
3. **Check system requirements**: Ensure your system meets minimum requirements
4. **Search for error messages**: Many issues have known solutions online
5. **Contact support**: Create an issue in the repository with detailed error information

## Post-Installation

### First Run

1. **Launch the GUI**:
   - Windows: Double-click `run_gui.bat`
   - Linux/macOS: Run `./run_gui.sh`

2. **Test basic functionality**:
   - Load or generate sample data
   - Create a simple economic plan
   - Generate a report

3. **Explore the system**:
   - Read the user manual
   - Try different policy goals
   - Experiment with data visualization

### Configuration

1. **API Keys**: Edit `api_keys_config.py` to add your API keys
2. **System Settings**: Modify `config/system_config.json` for custom settings
3. **Data Sources**: Add your data files to the `data/` directory

### Updates

To update the system:
1. Pull the latest changes: `git pull origin main`
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Reinstall the project: `pip install -e .`

## Uninstallation

To completely remove the system:

1. **Deactivate virtual environment** (if active):
   ```bash
   deactivate
   ```

2. **Delete the project folder**:
   - Windows: Delete the folder in File Explorer
   - Linux/macOS: `rm -rf /path/to/project`

3. **Clean up Python packages** (optional):
   ```bash
   pip uninstall cybernetic-planning
   ```

4. **Remove virtual environment** (if you want to clean up):
   ```bash
   rm -rf .venv
   ```

## Support

If you encounter issues not covered in this guide:

1. **Check the FAQ** in the repository
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Your operating system and version
   - Python version
   - Complete error message
   - Installation log contents
   - Steps to reproduce the issue

---

*This installation guide is part of the Cybernetic Planning System documentation. For more information, see the main README.md file.*
