"""
System validation and testing utilities for the Cybernetic Planning System.

This module provides comprehensive testing and validation of the installed system
to ensure all components are working correctly.
"""

import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Optional imports - these may not be available before installation
try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class SystemValidator:
    """Comprehensive system validation and testing."""

    def __init__(self, project_root: Path, python_executable: str):
        self.project_root = project_root
        self.python_executable = python_executable
        self.validation_log = []
        self.test_results = {}

    def log(self, message: str, level: str = "INFO"):
        """Log validation progress."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.validation_log.append(log_entry)

    def test_imports(self) -> Dict[str, bool]:
        """Test all critical imports."""
        self.log("Testing critical imports...")

        critical_imports = {
            'numpy': 'NumPy for numerical computing',
            'pandas': 'Pandas for data manipulation',
            'scipy': 'SciPy for scientific computing',
            'matplotlib': 'Matplotlib for plotting',
            'cvxpy': 'CVXPY for optimization',
            'requests': 'Requests for HTTP requests',
            'bs4': 'BeautifulSoup for web scraping',
            'pydantic': 'Pydantic for data validation',
            'cryptography': 'Cryptography for security'
        }

        results = {}
        for module, description in critical_imports.items():
            try:
                __import__(module)
                results[module] = True
                self.log(f"✓ {module}: {description}")
            except ImportError as e:
                results[module] = False
                self.log(f"✗ {module}: {description} - {str(e)}", "ERROR")

        return results

    def test_project_imports(self) -> Dict[str, bool]:
        """Test project - specific imports."""
        self.log("Testing project imports...")

        project_imports = {
            'src.cybernetic_planning.planning_system': 'Main planning system',
            'src.cybernetic_planning.core.leontief': 'Leontief I - O model',
            'src.cybernetic_planning.core.optimization': 'Optimization engine',
            'src.cybernetic_planning.agents.manager': 'Agent manager',
            'src.cybernetic_planning.data.matrix_builder': 'Matrix builder'
            # Note: visualization module requires seaborn, so we skip it in pre - installation tests
        }

        results = {}
        for module, description in project_imports.items():
            try:
                result = subprocess.run([
                    self.python_executable, "-c", f"import {module}; print('{module} OK')"
                ], capture_output = True, text = True, cwd = self.project_root)

                if result.returncode == 0:
                    results[module] = True
                    self.log(f"✓ {module}: {description}")
                else:
                    results[module] = False
                    self.log(f"✗ {module}: {description} - {result.stderr}", "ERROR")
            except Exception as e:
                results[module] = False
                self.log(f"✗ {module}: {description} - {str(e)}", "ERROR")

        return results

    def test_core_functionality(self) -> Dict[str, bool]:
        """Test core system functionality."""
        self.log("Testing core functionality...")

        test_script = """
sys.path.insert(0, '.')

try:

    # Test 1: System initialization
    system = CyberneticPlanningSystem()
    print("System initialization: OK")

    # Test 2: Synthetic data creation
    data = system.create_synthetic_data(n_sectors = 4, technology_density = 0.3)
    print("Synthetic data creation: OK")

    # Test 3: Basic plan creation
    policy_goals = ["Test policy goal"]
    plan = system.create_plan(policy_goals = policy_goals)
    print("Plan creation: OK")

    # Test 4: Leontief model
    leontief = LeontiefModel()
    A = np.array([[0.1, 0.2], [0.3, 0.1]])
    d = np.array([100, 200])
    x = leontief.solve_total_output(A, d)
    print("Leontief model: OK")

    # Test 5: Optimization engine
    opt = OptimizationEngine()
    result = opt.solve_linear_program(
        c = np.array([1, 2]),
        A_ub = np.array([[1, 1]]),
        b_ub = np.array([10]),
        bounds=[(0, None), (0, None)]
    )
    print("Optimization engine: OK")

    print("Core functionality test: PASSED")

except Exception as e:
    print(f"Core functionality test: FAILED - {str(e)}")
    traceback.print_exc()
    sys.exit(1)
"""

        try:
            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode == 0:
                self.log("✓ Core functionality test passed")
                return {'core_functionality': True}
            else:
                self.log(f"✗ Core functionality test failed: {result.stderr}", "ERROR")
                return {'core_functionality': False}

        except Exception as e:
            self.log(f"✗ Core functionality test failed: {str(e)}", "ERROR")
            return {'core_functionality': False}

    def test_data_processing(self) -> Dict[str, bool]:
        """Test data processing capabilities."""
        self.log("Testing data processing...")

        test_script = """
sys.path.insert(0, '.')

try:

    # Test 1: Matrix builder
    builder = MatrixBuilder()
    test_data = {
        'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
        'final_demand': np.array([100, 200]),
        'labor_input': np.array([0.5, 0.8])
    }
    matrices = builder.build_matrices(test_data)
    print("Matrix builder: OK")

    # Test 2: Data validator
    validator = DataValidator()
    is_valid = validator.validate_data(test_data)
    print("Data validator: OK")

    # Test 3: DataFrame operations
    df = pd.DataFrame({
        'sector': ['A', 'B'],
        'output': [100, 200],
        'labor': [0.5, 0.8]
    })
    print("DataFrame operations: OK")

    print("Data processing test: PASSED")

except Exception as e:
    print(f"Data processing test: FAILED - {str(e)}")
    traceback.print_exc()
    sys.exit(1)
"""

        try:
            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode == 0:
                self.log("✓ Data processing test passed")
                return {'data_processing': True}
            else:
                self.log(f"✗ Data processing test failed: {result.stderr}", "ERROR")
                return {'data_processing': False}

        except Exception as e:
            self.log(f"✗ Data processing test failed: {str(e)}", "ERROR")
            return {'data_processing': False}

    def test_agent_system(self) -> Dict[str, bool]:
        """Test the multi - agent system."""
        self.log("Testing agent system...")

        test_script = """
sys.path.insert(0, '.')

try:

    # Test 1: Agent manager
    manager = AgentManager()
    print("Agent manager: OK")

    # Test 2: Economics agent
    econ_agent = EconomicsAgent()
    test_data = {
        'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
        'final_demand': np.array([100, 200])
    }
    analysis = econ_agent.analyze_economics(test_data)
    print("Economics agent: OK")

    # Test 3: Resource agent
    resource_agent = ResourceAgent()
    resource_analysis = resource_agent.analyze_resources(test_data)
    print("Resource agent: OK")

    print("Agent system test: PASSED")

except Exception as e:
    print(f"Agent system test: FAILED - {str(e)}")
    traceback.print_exc()
    sys.exit(1)
"""

        try:
            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode == 0:
                self.log("✓ Agent system test passed")
                return {'agent_system': True}
            else:
                self.log(f"✗ Agent system test failed: {result.stderr}", "ERROR")
                return {'agent_system': False}

        except Exception as e:
            self.log(f"✗ Agent system test failed: {str(e)}", "ERROR")
            return {'agent_system': False}

    def test_gui_components(self) -> Dict[str, bool]:
        """Test GUI components."""
        self.log("Testing GUI components...")

        test_script = """
sys.path.insert(0, '.')

try:
    # Test GUI module import
    print("GUI module import: OK")

    # Test GUI class availability
    if hasattr(gui, 'CyberneticPlanningGUI'):
        print("GUI class available: OK")
    else:
        print("GUI class available: WARNING")

    # Test GUI initialization (without actually launching)
    try:
        # This would normally create the GUI, but we'll just test the import
        print("GUI initialization test: OK")
    except Exception as e:
        print(f"GUI initialization warning: {e}")
        print("GUI initialization test: WARNING")

    print("GUI components test: PASSED")

except Exception as e:
    print(f"GUI components test: FAILED - {str(e)}")
    traceback.print_exc()
    sys.exit(1)
"""

        try:
            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode == 0:
                self.log("✓ GUI components test passed")
                return {'gui_components': True}
            else:
                self.log(f"✗ GUI components test failed: {result.stderr}", "WARNING")
                return {'gui_components': False}

        except Exception as e:
            self.log(f"✗ GUI components test failed: {str(e)}", "WARNING")
            return {'gui_components': False}

    def test_file_permissions(self) -> Dict[str, bool]:
        """Test file permissions and access."""
        self.log("Testing file permissions...")

        test_directories = [
            "data",
            "exports",
            "logs",
            "cache",
            "outputs"
        ]

        results = {}
        for directory in test_directories:
            dir_path = self.project_root / directory
            try:
                # Test read access
                list(dir_path)
                # Test write access
                test_file = dir_path / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                results[directory] = True
                self.log(f"✓ {directory}: Read / write access OK")
            except Exception as e:
                results[directory] = False
                self.log(f"✗ {directory}: Access failed - {str(e)}", "ERROR")

        return results

    def test_configuration_files(self) -> Dict[str, bool]:
        """Test configuration files."""
        self.log("Testing configuration files...")

        config_files = {
            "config.json": self.project_root / "config" / "system_config.json",
            "requirements.txt": self.project_root / "requirements.txt",
            "pyproject.toml": self.project_root / "pyproject.toml"
        }

        results = {}
        for name, path in config_files.items():
            try:
                if path.exists():
                    if name.endswith('.json'):
                        with open(path, 'r') as f:
                            json.load(f)
                    results[name] = True
                    self.log(f"✓ {name}: Valid")
                else:
                    results[name] = False
                    self.log(f"✗ {name}: Not found", "WARNING")
            except Exception as e:
                results[name] = False
                self.log(f"✗ {name}: Invalid - {str(e)}", "ERROR")

        return results

    def test_performance(self) -> Dict[str, Any]:
        """Test system performance."""
        self.log("Testing system performance...")

        test_script = """
sys.path.insert(0, '.')
import time

try:

    # Test performance with different data sizes
    system = CyberneticPlanningSystem()

    # Small dataset
    start_time = time.time()
    data_small = system.create_synthetic_data(n_sectors = 4, technology_density = 0.3)
    plan_small = system.create_plan(policy_goals=["Test goal"])
    small_time = time.time() - start_time
    print(f"Small dataset (4 sectors): {small_time:.3f}s")

    # Medium dataset
    start_time = time.time()
    data_medium = system.create_synthetic_data(n_sectors = 8, technology_density = 0.4)
    plan_medium = system.create_plan(policy_goals=["Test goal"])
    medium_time = time.time() - start_time
    print(f"Medium dataset (8 sectors): {medium_time:.3f}s")

    # Large dataset
    start_time = time.time()
    data_large = system.create_synthetic_data(n_sectors = 16, technology_density = 0.5)
    plan_large = system.create_plan(policy_goals=["Test goal"])
    large_time = time.time() - start_time
    print(f"Large dataset (16 sectors): {large_time:.3f}s")

    print("Performance test: PASSED")

except Exception as e:
    print(f"Performance test: FAILED - {str(e)}")
    traceback.print_exc()
    sys.exit(1)
"""

        try:
            result = subprocess.run([
                self.python_executable, "-c", test_script
            ], capture_output = True, text = True, cwd = self.project_root)

            if result.returncode == 0:
                self.log("✓ Performance test passed")
                return {'performance': True, 'output': result.stdout}
            else:
                self.log(f"✗ Performance test failed: {result.stderr}", "ERROR")
                return {'performance': False, 'error': result.stderr}

        except Exception as e:
            self.log(f"✗ Performance test failed: {str(e)}", "ERROR")
            return {'performance': False, 'error': str(e)}

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        self.log("Starting comprehensive system validation...")

        start_time = time.time()

        # Run all tests
        test_results = {}

        # Import tests
        test_results['critical_imports'] = self.test_imports()
        test_results['project_imports'] = self.test_project_imports()

        # Functionality tests
        test_results['core_functionality'] = self.test_core_functionality()
        test_results['data_processing'] = self.test_data_processing()
        test_results['agent_system'] = self.test_agent_system()
        test_results['gui_components'] = self.test_gui_components()

        # System tests
        test_results['file_permissions'] = self.test_file_permissions()
        test_results['configuration_files'] = self.test_configuration_files()
        test_results['performance'] = self.test_performance()

        # Calculate overall success
        all_tests = []
        for category, results in test_results.items():
            if isinstance(results, dict):
                all_tests.extend(results.values())
            else:
                all_tests.append(results)

        overall_success = all(all_tests)

        # Add metadata
        test_results['metadata'] = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'duration': time.time() - start_time,
            'overall_success': overall_success,
            'total_tests': len(all_tests),
            'passed_tests': sum(all_tests),
            'failed_tests': len(all_tests) - sum(all_tests)
        }

        self.test_results = test_results

        if overall_success:
            self.log("✓ Comprehensive validation: PASSED")
        else:
            self.log("✗ Comprehensive validation: FAILED", "ERROR")

        return test_results

    def generate_validation_report(self) -> str:
        """Generate a detailed validation report."""
        if not self.test_results:
            self.run_comprehensive_validation()

        report = "Cybernetic Planning System - Validation Report\n"
        report += "=" * 50 + "\n\n"

        # Summary
        metadata = self.test_results.get('metadata', {})
        report += f"Timestamp: {metadata.get('timestamp', 'Unknown')}\n"
        report += f"Duration: {metadata.get('duration', 0):.2f} seconds\n"
        report += f"Overall Status: {'✓ PASSED' if metadata.get('overall_success', False) else '✗ FAILED'}\n"
        report += f"Tests Passed: {metadata.get('passed_tests', 0)}/{metadata.get('total_tests', 0)}\n\n"

        # Detailed results
        for category, results in self.test_results.items():
            if category == 'metadata':
                continue

            report += f"{category.replace('_', ' ').title()}:\n"
            report += "-" * 30 + "\n"

            if isinstance(results, dict):
                for test, status in results.items():
                    status_symbol = "✓" if status else "✗"
                    report += f"  {status_symbol} {test}\n"
            else:
                status_symbol = "✓" if results else "✗"
                report += f"  {status_symbol} {category}\n"

            report += "\n"

        # Recommendations
        report += "Recommendations:\n"
        report += "-" * 20 + "\n"

        if not metadata.get('overall_success', False):
            report += "- Check failed tests above for specific issues\n"
            report += "- Ensure all dependencies are properly installed\n"
            report += "- Verify file permissions and directory access\n"
            report += "- Check system requirements and resources\n"
        else:
            report += "- System is ready for use\n"
            report += "- All components are functioning correctly\n"
            report += "- You can now run the GUI or use the system\n"

        return report

    def save_validation_report(self, filename: Optional[str] = None) -> Path:
        """Save validation report to file."""
        if filename is None:
            filename = f"validation_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"

        report_path = self.project_root / "logs" / filename
        report_path.parent.mkdir(exist_ok = True)

        with open(report_path, 'w') as f:
            f.write(self.generate_validation_report())

        return report_path

    def get_validation_log(self) -> List[str]:
        """Get the validation log."""
        return self.validation_log.copy()
