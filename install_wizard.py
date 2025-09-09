#!/usr / bin / env python3
"""
Cybernetic Planning System - Installation Wizard

This wizard will guide you through the complete installation and setup
of the Cybernetic Planning System, including:
- Python environment setup - Dependency installation - Configuration setup - System validation - First - time usage guidance
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

class InstallationWizard:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / ".venv"
        self.python_executable = None
        self.pip_executable = None
        self.system_info = self._get_system_info()
        self.installation_log = []

    def _get_system_info(self) -> Dict:
        """Get system information for compatibility checking."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'python_version': sys.version,
            'python_executable': sys.executable
        }

    def log(self, message: str, level: str = "INFO"):
        """Log installation progress."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.installation_log.append(log_entry)

    def print_banner(self):
        """Print the installation wizard banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CYBERNETIC PLANNING SYSTEM                                ║
║                        Installation Wizard                                   ║
║                                                                              ║
║  AI - enhanced central planning software system using Input - Output analysis║
║  Implements Leontief models and labor - time accounting principles           ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        self.log("Checking Python version compatibility...")

        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 9):
            self.log(f"ERROR: Python {major}.{minor} detected. Python 3.9 + required.", "ERROR")
            return False

        self.log(f"✓ Python {major}.{minor} detected - Compatible")
        return True

    def check_system_requirements(self) -> bool:
        """Check system requirements and dependencies."""
        self.log("Checking system requirements...")

        # Check available disk space (simplified check)
        try:
            statvfs = os.statvfs(self.project_root) if hasattr(os, 'statvfs') else None
            if statvfs:
                free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
                if free_space_gb < 1.0:
                    self.log("WARNING: Less than 1GB free space available", "WARNING")
        except:
            pass

        # Check if we're in the right directory
        if not (self.project_root / "src" / "cybernetic_planning").exists():
            self.log("ERROR: Not in the correct project directory", "ERROR")
            return False

        self.log("✓ System requirements check passed")
        return True

    def setup_virtual_environment(self) -> bool:
        """Create and setup Python virtual environment."""
        self.log("Setting up Python virtual environment...")

        try:
            # Remove existing venv if it exists
            if self.venv_path.exists():
                self.log("Removing existing virtual environment...")
                shutil.rmtree(self.venv_path)

            # Create new virtual environment
            self.log("Creating virtual environment...")
            result = subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"ERROR: Failed to create virtual environment: {result.stderr}", "ERROR")
                return False

            # Determine Python and pip executables
            if platform.system() == "Windows":
                self.python_executable = self.venv_path / "Scripts" / "python.exe"
                self.pip_executable = self.venv_path / "Scripts" / "pip.exe"
            else:
                self.python_executable = self.venv_path / "bin" / "python"
                self.pip_executable = self.venv_path / "bin" / "pip"

            # Verify virtual environment
            if not self.python_executable.exists():
                self.log("ERROR: Virtual environment creation failed", "ERROR")
                return False

            self.log("✓ Virtual environment created successfully")
            return True

        except Exception as e:
            self.log(f"ERROR: Failed to setup virtual environment: {str(e)}", "ERROR")
            return False

    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version."""
        self.log("Upgrading pip to latest version...")

        try:
            result = subprocess.run([
                str(self.python_executable), "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"WARNING: Failed to upgrade pip: {result.stderr}", "WARNING")
                return False

            self.log("✓ Pip upgraded successfully")
            return True

        except Exception as e:
            self.log(f"WARNING: Failed to upgrade pip: {str(e)}", "WARNING")
            return False

    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        self.log("Installing project dependencies...")

        try:
            # Install from requirements.txt
            requirements_file = self.project_root / "requirements.txt"
            if not requirements_file.exists():
                self.log("ERROR: requirements.txt not found", "ERROR")
                return False

            self.log("Installing dependencies from requirements.txt...")
            result = subprocess.run([
                str(self.pip_executable), "install", "-r", str(requirements_file)
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"ERROR: Failed to install dependencies: {result.stderr}", "ERROR")
                return False

            # Install the project itself in development mode
            self.log("Installing project in development mode...")
            result = subprocess.run([
                str(self.pip_executable), "install", "-e", "."
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"WARNING: Failed to install project in development mode: {result.stderr}", "WARNING")

            self.log("✓ Dependencies installed successfully")
            return True

        except Exception as e:
            self.log(f"ERROR: Failed to install dependencies: {str(e)}", "ERROR")
            return False

    def setup_directories(self) -> bool:
        """Create necessary directories."""
        self.log("Setting up project directories...")

        directories = [
            "data",
            "exports",
            "logs",
            "cache",
            "outputs",
            "tests"
        ]

        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok = True)
                self.log(f"✓ Created directory: {directory}")

            return True

        except Exception as e:
            self.log(f"ERROR: Failed to create directories: {str(e)}", "ERROR")
            return False

    def create_configuration(self) -> bool:
        """Create initial configuration files."""
        self.log("Creating configuration files...")

        try:
            # Create basic configuration
            config = {
                "system": {
                    "name": "Cybernetic Planning System",
                    "version": "0.1.0",
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "installation_date": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "paths": {
                    "data_directory": "data",
                    "exports_directory": "exports",
                    "logs_directory": "logs",
                    "cache_directory": "cache"
                },
                "planning": {
                    "default_sectors": 8,
                    "default_years": 5,
                    "convergence_threshold": 0.005,
                    "max_iterations": 15
                }
            }

            config_file = self.project_root / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent = 2)

            self.log("✓ Configuration file created")
            return True

        except Exception as e:
            self.log(f"ERROR: Failed to create configuration: {str(e)}", "ERROR")
            return False

    def validate_installation(self) -> bool:
        """Validate the installation by running basic tests."""
        self.log("Validating installation...")

        try:
            # Test Python import
            self.log("Testing Python imports...")
            result = subprocess.run([
                str(self.python_executable), "-c",
                "import numpy, pandas, scipy, cvxpy, matplotlib; print('All core imports successful')"
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"ERROR: Import test failed: {result.stderr}", "ERROR")
                return False

            # Test project import
            self.log("Testing project imports...")
            result = subprocess.run([
                str(self.python_executable), "-c",
                "from src.cybernetic_planning.planning_system import CyberneticPlanningSystem; print('Project import successful')"
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"WARNING: Project import test failed: {result.stderr}", "WARNING")

            self.log("✓ Installation validation completed")
            return True

        except Exception as e:
            self.log(f"ERROR: Validation failed: {str(e)}", "ERROR")
            return False

    def create_launcher_scripts(self) -> bool:
        """Create launcher scripts for different platforms."""
        self.log("Creating launcher scripts...")

        try:
            # Windows batch file
            if platform.system() == "Windows":
                bat_content = f"""@echo off
echo Starting Cybernetic Planning System GUI...
echo.

REM Check if virtual environment exists
if not exist "{self.venv_path}\\Scripts\\python.exe" (
    echo Error: Virtual environment not found
    echo Please run the installation wizard again
    pause
    exit / b 1
)

REM Run the GUI
"{self.python_executable}" gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Check the error message above.
    pause
)
"""
                with open(self.project_root / "run_gui.bat", 'w') as f:
                    f.write(bat_content)

                self.log("✓ Windows launcher script created")

            # Unix shell script
            shell_content = f"""#!/bin / bash
echo "Starting Cybernetic Planning System GUI..."

# Check if virtual environment exists
if [ ! -f "{self.python_executable}" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run the installation wizard again"
    exit 1
fi

# Run the GUI
"{self.python_executable}" gui.py
"""
            with open(self.project_root / "run_gui.sh", 'w') as f:
                f.write(shell_content)

            # Make shell script executable
            os.chmod(self.project_root / "run_gui.sh", 0o755)
            self.log("✓ Unix launcher script created")

            return True

        except Exception as e:
            self.log(f"ERROR: Failed to create launcher scripts: {str(e)}", "ERROR")
            return False

    def save_installation_log(self):
        """Save installation log to file."""
        log_file = self.project_root / "logs" / "installation.log"
        log_file.parent.mkdir(exist_ok = True)

        with open(log_file, 'w') as f:
            f.write("Cybernetic Planning System - Installation Log\n")
            f.write("=" * 50 + "\n\n")
            for entry in self.installation_log:
                f.write(entry + "\n")

        self.log(f"Installation log saved to: {log_file}")

    def print_success_message(self):
        """Print installation success message."""
        success_message = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        INSTALLATION COMPLETE!                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎉 The Cybernetic Planning System has been successfully installed!

📋 What was installed:
   ✓ Python virtual environment (.venv/)
   ✓ All required dependencies
   ✓ Project configuration
   ✓ Launcher scripts
   ✓ Directory structure

🚀 How to run the system:
   Windows: Double - click run_gui.bat or run: .\\run_gui.bat
   Unix / Mac: Run: ./run_gui.sh

📚 Next steps:
   1. Run the GUI to start planning
   2. Load or generate economic data
   3. Create your first economic plan
   4. Explore the comprehensive reports

📖 Documentation: See README.md for detailed usage instructions

🔧 Troubleshooting: Check logs / installation.log if you encounter issues

Happy planning! 🏗️
        """
        print(success_message)

    def run_installation(self) -> bool:
        """Run the complete installation process."""
        self.print_banner()

        steps = [
            ("Checking Python version", self.check_python_version),
            ("Checking system requirements", self.check_system_requirements),
            ("Setting up virtual environment", self.setup_virtual_environment),
            ("Upgrading pip", self.upgrade_pip),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up directories", self.setup_directories),
            ("Creating configuration", self.create_configuration),
            ("Validating installation", self.validate_installation),
            ("Creating launcher scripts", self.create_launcher_scripts)
        ]

        for step_name, step_function in steps:
            self.log(f"Step: {step_name}")
            if not step_function():
                self.log(f"Installation failed at step: {step_name}", "ERROR")
                self.save_installation_log()
                return False
            self.log(f"✓ Completed: {step_name}")
            print()  # Add spacing between steps

        self.save_installation_log()
        self.print_success_message()
        return True

def main():
    """Main entry point for the installation wizard."""
    wizard = InstallationWizard()

    try:
        success = wizard.run_installation()
        if success:
            print("\nPress Enter to exit...")
            input()
        else:
            print("\nInstallation failed. Check the log for details.")
            print("Press Enter to exit...")
            input()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Press Enter to exit...")
        input()
        sys.exit(1)

if __name__ == "__main__":
    main()
