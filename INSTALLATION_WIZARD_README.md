# Installation Wizard - Quick Reference

This document provides a quick reference for the Cybernetic Planning System installation wizard.

## Files Created

### Main Installation Files
- `install_wizard.py` - Main Python installation wizard
- `install_wizard.bat` - Windows batch file launcher
- `install_wizard.sh` - Unix/Linux shell script launcher
- `test_installation.py` - Test script for the installation wizard

### Utility Modules
- `src/cybernetic_planning/utils/installer_utils.py` - Requirements checking and dependency installation
- `src/cybernetic_planning/utils/config_setup.py` - Configuration and security setup
- `src/cybernetic_planning/utils/system_validator.py` - System validation and testing

### Documentation
- `INSTALLATION_GUIDE.md` - Comprehensive installation guide
- `README.md` - Updated with installation wizard instructions

## Quick Start

### For Users
1. **Windows**: Double-click `install_wizard.bat`
2. **Linux/macOS**: Run `./install_wizard.sh`
3. Follow the on-screen instructions

### For Developers
1. Test the installation wizard: `python test_installation.py`
2. Run the main wizard: `python install_wizard.py`
3. Check installation logs in `logs/installation.log`

## Features

### Automated Installation
- ✅ Python version checking (3.9+ required)
- ✅ Virtual environment creation
- ✅ Dependency installation from requirements.txt
- ✅ Project installation in development mode
- ✅ Directory structure setup
- ✅ Configuration file creation
- ✅ Security key generation
- ✅ API key template creation
- ✅ System validation and testing
- ✅ Launcher script generation
- ✅ Comprehensive logging

### Cross-Platform Support
- ✅ Windows (batch files)
- ✅ Linux (shell scripts)
- ✅ macOS (shell scripts)
- ✅ Python 3.9+ compatibility

### Error Handling
- ✅ Detailed error messages
- ✅ Installation logging
- ✅ Rollback on failure
- ✅ Troubleshooting guidance

## Installation Process

1. **System Check**
   - Verify Python version
   - Check system resources
   - Validate project structure

2. **Environment Setup**
   - Create virtual environment
   - Upgrade pip
   - Set up directories

3. **Dependency Installation**
   - Install requirements.txt
   - Install project in dev mode
   - Verify imports

4. **Configuration**
   - Create config files
   - Set up security keys
   - Generate API templates

5. **Validation**
   - Test core functionality
   - Validate data processing
   - Check GUI components
   - Run performance tests

6. **Finalization**
   - Create launcher scripts
   - Generate installation log
   - Display success message

## Troubleshooting

### Common Issues
- **Python not found**: Install Python 3.9+ and add to PATH
- **Permission denied**: Run as administrator (Windows) or with sudo (Unix)
- **Virtual environment issues**: Delete `.venv` folder and try again
- **Dependency installation fails**: Update pip and clear cache

### Getting Help
1. Check `logs/installation.log` for detailed error messages
2. Run `python test_installation.py` to test components
3. Verify system requirements
4. Check the comprehensive `INSTALLATION_GUIDE.md`

## File Structure

```
Cybernetic-Planning-Experiment/
├── install_wizard.py              # Main installation wizard
├── install_wizard.bat             # Windows launcher
├── install_wizard.sh              # Unix launcher
├── test_installation.py           # Test script
├── INSTALLATION_GUIDE.md          # Comprehensive guide
├── INSTALLATION_WIZARD_README.md  # This file
├── src/cybernetic_planning/utils/
│   ├── installer_utils.py         # Requirements & dependencies
│   ├── config_setup.py            # Configuration & security
│   └── system_validator.py        # Validation & testing
└── logs/
    └── installation.log           # Installation log
```

## Usage Examples

### Basic Installation
```bash
# Windows
install_wizard.bat

# Linux/macOS
./install_wizard.sh
```

### Testing Components
```bash
python test_installation.py
```

### Manual Validation
```python
from src.cybernetic_planning.utils.system_validator import SystemValidator
from pathlib import Path

validator = SystemValidator(Path('.'), 'python')
results = validator.run_comprehensive_validation()
print(validator.generate_validation_report())
```

## Security Notes

- API keys are stored in `api_keys_config.py` (not committed to git)
- Encryption keys are generated automatically
- Sensitive files are excluded from version control
- All security files have restrictive permissions

## Maintenance

### Updating the Wizard
1. Modify the relevant utility modules
2. Test with `python test_installation.py`
3. Update documentation as needed

### Adding New Dependencies
1. Add to `requirements.txt`
2. Update the requirements checker if needed
3. Test the installation process

### Customizing Installation
1. Modify `install_wizard.py` for main process changes
2. Update utility modules for specific functionality
3. Add new validation tests as needed

---

*This installation wizard provides a robust, user-friendly way to install the Cybernetic Planning System across different platforms and environments.*
