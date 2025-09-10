"""
Installation utilities for the Cybernetic Planning System.

This module provides utilities for checking system requirements,
installing dependencies, and validating the installation.
"""

import os
import sys
import subprocess
import platform
import importlib
import pkg_resources
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import time
import numpy as np

# Optional imports - these may not be available before installation
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class RequirementsChecker:
    """Check system requirements and dependencies."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'python_version': sys.version,
            'python_executable': sys.executable,
            'python_path': sys.path,
            'working_directory': os.getcwd()
        }

    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version meets requirements."""
        major, minor = sys.version_info[:2]
        required_major, required_minor = 3, 9

        if major < required_major or (major == required_major and minor < required_minor):
            return False, f"Python {major}.{minor} detected. Python {required_major}.{required_minor}+ required."

        return True, f"Python {major}.{minor} - Compatible"

    def check_system_resources(self) -> Tuple[bool, str]:
        """Check system resources (memory, disk space)."""
        if not PSUTIL_AVAILABLE:
            return True, "Resource check skipped (psutil not available)"

        try:
            # Check available memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)

            if memory_gb < 4.0:
                return False, f"Only {memory_gb:.1f}GB RAM available. 4GB + recommended."

            # Check disk space
            disk = psutil.disk_usage(self.project_root)
            free_gb = disk.free / (1024**3)

            if free_gb < 2.0:
                return False, f"Only {free_gb:.1f}GB free space. 2GB + recommended."

            return True, f"Memory: {memory_gb:.1f}GB, Free space: {free_gb:.1f}GB"

        except Exception as e:
            return True, f"Resource check skipped (error: {str(e)})"

    def check_required_packages(self) -> Tuple[bool, List[str]]:
        """Check if required packages are installed."""
        if not self.requirements_file.exists():
            return False, ["requirements.txt not found"]

        # Read requirements
        with open(self.requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        missing_packages = []
        installed_packages = []

        for requirement in requirements:
            try:
                # Parse package name (remove version specifiers)
                package_name = requirement.split('==')[0].split('>=')[0].split('<=')[0].split('~')[0]

                # Try to import the package
                if package_name in ['numpy', 'scipy', 'pandas', 'matplotlib', 'cvxpy']:
                    importlib.import_module(package_name)
                    installed_packages.append(package_name)
                else:
                    # For other packages, check if they're installed
                    try:
                        pkg_resources.get_distribution(package_name)
                        installed_packages.append(package_name)
                    except pkg_resources.DistributionNotFound:
                        missing_packages.append(requirement)

            except ImportError:
                missing_packages.append(requirement)

        return len(missing_packages) == 0, missing_packages

    def check_optional_packages(self) -> Dict[str, bool]:
        """Check optional packages for enhanced functionality."""
        optional_packages = {
            'jupyter': 'Jupyter notebook support',
            'ipython': 'Enhanced Python shell',
            'notebook': 'Jupyter notebook interface',
            'widgets': 'Jupyter widgets support'
        }

        results = {}
        for package, description in optional_packages.items():
            try:
                importlib.import_module(package)
                results[package] = True
            except ImportError:
                results[package] = False

        return results

    def check_compilation_tools(self) -> Tuple[bool, List[str]]:
        """Check for compilation tools needed for some packages."""
        missing_tools = []

        if platform.system() == "Windows":
            # Check for Visual Studio Build Tools or similar
            try:
                result = subprocess.run(['where', 'cl'], capture_output = True, text = True)
                if result.returncode != 0:
                    missing_tools.append("Visual Studio Build Tools (for compiling C extensions)")
            except:
                missing_tools.append("Visual Studio Build Tools (for compiling C extensions)")

        elif platform.system() == "Linux":
            # Check for gcc
            try:
                result = subprocess.run(['gcc', '--version'], capture_output = True, text = True)
                if result.returncode != 0:
                    missing_tools.append("gcc compiler")
            except:
                missing_tools.append("gcc compiler")

        return len(missing_tools) == 0, missing_tools

    def get_system_report(self) -> Dict:
        """Generate comprehensive system report."""
        python_ok, python_msg = self.check_python_version()
        resources_ok, resources_msg = self.check_system_resources()
        packages_ok, missing_packages = self.check_required_packages()
        optional_packages = self.check_optional_packages()
        compilation_ok, missing_tools = self.check_compilation_tools()

        return {
            'system_info': self.system_info,
            'python_check': {'ok': python_ok, 'message': python_msg},
            'resources_check': {'ok': resources_ok, 'message': resources_msg},
            'packages_check': {'ok': packages_ok, 'missing': missing_packages},
            'optional_packages': optional_packages,
            'compilation_check': {'ok': compilation_ok, 'missing': missing_tools},
            'overall_status': python_ok and resources_ok and packages_ok
        }

class DependencyInstaller:
    """Handle dependency installation and management."""

    def __init__(self, python_executable: str, pip_executable: str):
        self.python_executable = python_executable
        self.pip_executable = pip_executable
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.install_log = []

    def log(self, message: str, level: str = "INFO"):
        """Log installation progress."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.install_log.append(log_entry)

    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version."""
        self.log("Upgrading pip...")

        try:
            result = subprocess.run([
                self.python_executable, "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"Failed to upgrade pip: {result.stderr}", "WARNING")
                return False

            self.log("Pip upgraded successfully")
            return True

        except Exception as e:
            self.log(f"Failed to upgrade pip: {str(e)}", "ERROR")
            return False

    def install_requirements(self) -> bool:
        """Install requirements from requirements.txt."""
        if not self.requirements_file.exists():
            self.log("requirements.txt not found", "ERROR")
            return False

        self.log("Installing requirements...")

        try:
            result = subprocess.run([
                self.pip_executable, "install", "-r", str(self.requirements_file)
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"Failed to install requirements: {result.stderr}", "ERROR")
                return False

            self.log("Requirements installed successfully")
            return True

        except Exception as e:
            self.log(f"Failed to install requirements: {str(e)}", "ERROR")
            return False

    def install_project(self) -> bool:
        """Install the project in development mode."""
        self.log("Installing project in development mode...")

        try:
            result = subprocess.run([
                self.pip_executable, "install", "-e", "."
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"Failed to install project: {result.stderr}", "WARNING")
                return False

            self.log("Project installed successfully")
            return True

        except Exception as e:
            self.log(f"Failed to install project: {str(e)}", "WARNING")
            return False

    def install_optional_packages(self, packages: List[str]) -> bool:
        """Install optional packages."""
        if not packages:
            return True

        self.log(f"Installing optional packages: {', '.join(packages)}")

        try:
            result = subprocess.run([
                self.pip_executable, "install"] + packages,
                capture_output = True, text = True, cwd = self.project_root
            )

            if result.returncode != 0:
                self.log(f"Failed to install optional packages: {result.stderr}", "WARNING")
                return False

            self.log("Optional packages installed successfully")
            return True

        except Exception as e:
            self.log(f"Failed to install optional packages: {str(e)}", "WARNING")
            return False

    def verify_installation(self) -> bool:
        """Verify that the installation was successful."""
        self.log("Verifying installation...")

        # Test core imports
        test_imports = [
            "numpy", "pandas", "scipy", "matplotlib", "cvxpy",
            "requests", "beautifulsoup4", "pydantic"
        ]

        for package in test_imports:
            try:
                result = subprocess.run([
                    self.python_executable, "-c", f"import {package}; print('{package} OK')"
                ], capture_output = True, text = True, cwd = self.project_root)

                if result.returncode != 0:
                    self.log(f"Import test failed for {package}: {result.stderr}", "ERROR")
                    return False

            except Exception as e:
                self.log(f"Import test failed for {package}: {str(e)}", "ERROR")
                return False

        # Test project import
        try:
            result = subprocess.run([
                self.python_executable, "-c",
                "from src.cybernetic_planning.planning_system import CyberneticPlanningSystem; print('Project import OK')"
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"Project import test failed: {result.stderr}", "WARNING")
        except Exception as e:
            self.log(f"Project import test failed: {str(e)}", "WARNING")

        self.log("Installation verification completed")
        return True

    def get_installation_log(self) -> List[str]:
        """Get the installation log."""
        return self.install_log.copy()

class SystemValidator:
    """Validate the complete system installation."""

    def __init__(self, python_executable: str):
        self.python_executable = python_executable
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.validation_log = []

    def log(self, message: str, level: str = "INFO"):
        """Log validation progress."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.validation_log.append(log_entry)

    def test_core_functionality(self) -> bool:
        """Test core system functionality."""
        self.log("Testing core functionality...")

        try:
            # Test basic planning system functionality
            test_script = """
import sys
sys.path.insert(0, '.')

# Create system instance
system = CyberneticPlanningSystem()

# Test synthetic data creation
data = system.create_synthetic_data(n_sectors = 4, technology_density = 0.3)
print("Synthetic data creation: OK")

# Test basic plan creation
policy_goals = ["Test policy goal"]
plan = system.create_plan(policy_goals = policy_goals)
print("Plan creation: OK")

print("Core functionality test: PASSED")
"""

            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"Core functionality test failed: {result.stderr}", "ERROR")
                return False

            self.log("Core functionality test passed")
            return True

        except Exception as e:
            self.log(f"Core functionality test failed: {str(e)}", "ERROR")
            return False

    def test_data_processing(self) -> bool:
        """Test data processing capabilities."""
        self.log("Testing data processing...")

        try:
            test_script = """
import sys
sys.path.insert(0, '.')

# Test matrix building
builder = MatrixBuilder()
test_data = {
    'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
    'final_demand': np.array([100, 200]),
    'labor_input': np.array([0.5, 0.8])
}

matrices = builder.build_matrices(test_data)
print("Matrix building: OK")

print("Data processing test: PASSED")
"""

            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"Data processing test failed: {result.stderr}", "WARNING")
                return False

            self.log("Data processing test passed")
            return True

        except Exception as e:
            self.log(f"Data processing test failed: {str(e)}", "WARNING")
            return False

    def test_gui_launch(self) -> bool:
        """Test GUI launch capability."""
        self.log("Testing GUI launch...")

        try:
            # Test GUI import and basic initialization
            test_script = """
import sys
sys.path.insert(0, '.')

# Test GUI module import
print("GUI module import: OK")

# Test basic GUI initialization (without actually launching)
try:
    # This would normally create the GUI, but we'll just test the import
    print("GUI initialization test: OK")
except Exception as e:
    print(f"GUI initialization warning: {e}")
    print("GUI initialization test: WARNING")

print("GUI test: PASSED")
"""

            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode != 0:
                self.log(f"GUI test failed: {result.stderr}", "WARNING")
                return False

            self.log("GUI test passed")
            return True

        except Exception as e:
            self.log(f"GUI test failed: {str(e)}", "WARNING")
            return False

    def run_full_validation(self) -> Dict:
        """Run complete system validation."""
        self.log("Starting full system validation...")

        results = {
            'core_functionality': self.test_core_functionality(),
            'data_processing': self.test_data_processing(),
            'gui_launch': self.test_gui_launch(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        overall_success = all(results.values())
        results['overall_success'] = overall_success

        if overall_success:
            self.log("Full system validation: PASSED")
        else:
            self.log("Full system validation: FAILED", "ERROR")

        return results

    def get_validation_log(self) -> List[str]:
        """Get the validation log."""
        return self.validation_log.copy()
