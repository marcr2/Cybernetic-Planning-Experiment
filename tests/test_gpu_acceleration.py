"""
Tests for GPU acceleration functionality.

Tests GPU detection, solver selection, performance monitoring,
and integration with optimization and Leontief models.
"""

import unittest
import numpy as np
import tempfile
import json
import os
from pathlib import Path
import warnings

# Test imports
try:
    from src.cybernetic_planning.core.gpu_acceleration import (
        GPUDetector, GPUSolverSelector, GPUPerformanceMonitor, 
        GPUSettingsManager, create_gpu_optimized_arrays, convert_gpu_to_cpu
    )
    GPU_ACCELERATION_AVAILABLE = True
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False

try:
    from src.cybernetic_planning.core.optimization import ConstrainedOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

try:
    from src.cybernetic_planning.core.leontief import LeontiefModel
    LEONTIEF_AVAILABLE = True
except ImportError:
    LEONTIEF_AVAILABLE = False


class TestGPUDetector(unittest.TestCase):
    """Test GPU detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = GPUDetector()
    
    def test_gpu_detection(self):
        """Test GPU detection capabilities."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        gpu_info = self.detector.detect_gpu_capabilities()
        
        # Check that required keys are present
        required_keys = [
            'cupy_available', 'cuda_available', 'rocm_available', 
            'gpu_count', 'gpu_memory', 'compute_capability', 'gpu_names'
        ]
        for key in required_keys:
            self.assertIn(key, gpu_info)
        
        # Check data types
        self.assertIsInstance(gpu_info['cupy_available'], bool)
        self.assertIsInstance(gpu_info['cuda_available'], bool)
        self.assertIsInstance(gpu_info['rocm_available'], bool)
        self.assertIsInstance(gpu_info['gpu_count'], int)
        self.assertIsInstance(gpu_info['gpu_memory'], (int, float))
        self.assertIsInstance(gpu_info['gpu_names'], list)
    
    def test_gpu_availability(self):
        """Test GPU availability check."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        is_available = self.detector.is_gpu_available()
        self.assertIsInstance(is_available, bool)
    
    def test_memory_usage(self):
        """Test GPU memory usage reporting."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        memory_usage = self.detector.get_gpu_memory_usage()
        
        required_keys = ['used', 'free', 'total']
        for key in required_keys:
            self.assertIn(key, memory_usage)
            self.assertIsInstance(memory_usage[key], int)
            self.assertGreaterEqual(memory_usage[key], 0)


class TestGPUSolverSelector(unittest.TestCase):
    """Test GPU solver selection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.selector = GPUSolverSelector()
    
    def test_solver_detection(self):
        """Test solver detection."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Check that available solvers is a list
        self.assertIsInstance(self.selector.available_solvers, list)
    
    def test_solver_selection(self):
        """Test solver selection logic."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Test GPU solver selection
        gpu_solver = self.selector.select_gpu_solver(use_gpu=True, solver_preference="CuClarabel")
        if gpu_solver:
            self.assertIsInstance(gpu_solver, str)
        
        # Test CPU fallback
        cpu_solver = self.selector.select_gpu_solver(use_gpu=False)
        if cpu_solver:
            self.assertIsInstance(cpu_solver, str)


class TestGPUPerformanceMonitor(unittest.TestCase):
    """Test GPU performance monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = GPUPerformanceMonitor()
    
    def test_monitoring_control(self):
        """Test monitoring start/stop functionality."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Test initial state
        self.assertFalse(self.monitor.monitoring_enabled)
        
        # Test start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring_enabled)
        
        # Test stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_enabled)
    
    def test_benchmark_operation(self):
        """Test operation benchmarking."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Define test functions
        def gpu_func(x):
            return np.sum(x ** 2)
        
        def cpu_func(x):
            return np.sum(x ** 2)
        
        # Create test data
        test_data = np.random.rand(1000)
        
        # Run benchmark
        result = self.monitor.benchmark_operation(
            "test_operation", gpu_func, cpu_func, test_data
        )
        
        # Check result structure
        required_keys = [
            'operation', 'gpu_time', 'cpu_time', 'speedup', 
            'gpu_success', 'cpu_success'
        ]
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that at least one version succeeded
        self.assertTrue(result['gpu_success'] or result['cpu_success'])
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Add some test benchmarks
        self.monitor.benchmarks = {
            'test1': {
                'gpu_success': True, 'cpu_success': True,
                'gpu_time': 1.0, 'cpu_time': 2.0, 'speedup': 2.0
            },
            'test2': {
                'gpu_success': True, 'cpu_success': True,
                'gpu_time': 0.5, 'cpu_time': 1.5, 'speedup': 3.0
            }
        }
        
        summary = self.monitor.get_performance_summary()
        
        # Check summary structure
        if 'message' not in summary:
            required_keys = [
                'total_benchmarks', 'successful_benchmarks', 
                'average_speedup', 'max_speedup', 'min_speedup'
            ]
            for key in required_keys:
                self.assertIn(key, summary)
            
            self.assertEqual(summary['total_benchmarks'], 2)
            self.assertEqual(summary['successful_benchmarks'], 2)
            self.assertEqual(summary['average_speedup'], 2.5)
            self.assertEqual(summary['max_speedup'], 3.0)
            self.assertEqual(summary['min_speedup'], 2.0)


class TestGPUSettingsManager(unittest.TestCase):
    """Test GPU settings management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.settings_manager = GPUSettingsManager(self.config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.temp_dir)
    
    def test_default_settings(self):
        """Test default settings loading."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        settings = self.settings_manager.get_settings()
        
        # Check default values
        self.assertFalse(settings['enabled'])
        self.assertEqual(settings['solver'], 'CuClarabel')
        self.assertTrue(settings['monitoring']['show_utilization'])
        self.assertFalse(settings['monitoring']['benchmark_mode'])
        self.assertTrue(settings['fallback_to_cpu'])
    
    def test_settings_save_load(self):
        """Test settings save and load functionality."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Save settings
        success = self.settings_manager.save_settings(
            gpu_enabled=True,
            solver_type="SCS",
            monitoring_enabled=False,
            benchmark_mode=True
        )
        
        self.assertTrue(success)
        
        # Reload settings
        new_manager = GPUSettingsManager(self.config_path)
        settings = new_manager.get_settings()
        
        # Check that settings were saved correctly
        self.assertTrue(settings['enabled'])
        self.assertEqual(settings['solver'], 'SCS')
        self.assertFalse(settings['monitoring']['show_utilization'])
        self.assertTrue(settings['monitoring']['benchmark_mode'])
    
    def test_settings_methods(self):
        """Test individual settings methods."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Test initial state
        self.assertFalse(self.settings_manager.is_gpu_enabled())
        self.assertEqual(self.settings_manager.get_solver_preference(), 'CuClarabel')
        self.assertTrue(self.settings_manager.should_fallback_to_cpu())


class TestGPUArrayOperations(unittest.TestCase):
    """Test GPU array conversion operations."""
    
    def test_create_gpu_arrays(self):
        """Test GPU array creation."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Create test arrays
        arrays = [
            np.random.rand(100, 100),
            np.random.rand(50),
            np.random.rand(10, 10, 10)
        ]
        
        # Test GPU conversion
        gpu_arrays = create_gpu_optimized_arrays(arrays, use_gpu=True)
        
        # Check that arrays were converted
        self.assertEqual(len(gpu_arrays), len(arrays))
        
        # Test CPU fallback
        cpu_arrays = create_gpu_optimized_arrays(arrays, use_gpu=False)
        self.assertEqual(len(cpu_arrays), len(arrays))
    
    def test_convert_gpu_to_cpu(self):
        """Test GPU to CPU conversion."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Create test arrays
        arrays = [
            np.random.rand(100, 100),
            np.random.rand(50)
        ]
        
        # Convert to GPU and back
        gpu_arrays = create_gpu_optimized_arrays(arrays, use_gpu=True)
        cpu_arrays = convert_gpu_to_cpu(gpu_arrays)
        
        # Check that conversion worked
        self.assertEqual(len(cpu_arrays), len(arrays))
        
        # Check that arrays are numpy arrays
        for arr in cpu_arrays:
            self.assertIsInstance(arr, np.ndarray)


class TestGPUOptimizationIntegration(unittest.TestCase):
    """Test GPU integration with optimization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not OPTIMIZATION_AVAILABLE:
            self.skipTest("Optimization module not available")
        
        # Create test problem
        self.n = 20
        self.A = np.random.rand(self.n, self.n) * 0.1
        self.l = np.random.rand(self.n)
        self.d = np.random.rand(self.n) * 100
    
    def test_gpu_optimizer_creation(self):
        """Test GPU-enabled optimizer creation."""
        if not GPU_ACCELERATION_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            self.skipTest("Required modules not available")
        
        # Test GPU-enabled optimizer
        optimizer = ConstrainedOptimizer(
            self.A, self.l, self.d, use_gpu=True
        )
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.use_gpu)
    
    def test_gpu_optimizer_solve(self):
        """Test GPU-enabled optimizer solving."""
        if not GPU_ACCELERATION_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            self.skipTest("Required modules not available")
        
        optimizer = ConstrainedOptimizer(
            self.A, self.l, self.d, use_gpu=True
        )
        
        # Solve the problem
        result = optimizer.solve()
        
        # Check result structure
        required_keys = ['status', 'solution', 'feasible', 'gpu_accelerated']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check GPU acceleration status
        self.assertIsInstance(result['gpu_accelerated'], bool)
    
    def test_gpu_status(self):
        """Test GPU status reporting."""
        if not GPU_ACCELERATION_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            self.skipTest("Required modules not available")
        
        optimizer = ConstrainedOptimizer(
            self.A, self.l, self.d, use_gpu=True
        )
        
        status = optimizer.get_gpu_status()
        
        # Check status structure
        required_keys = ['gpu_available', 'gpu_enabled', 'gpu_arrays_created']
        for key in required_keys:
            self.assertIn(key, status)
    
    def test_benchmark_functionality(self):
        """Test GPU vs CPU benchmarking."""
        if not GPU_ACCELERATION_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            self.skipTest("Required modules not available")
        
        optimizer = ConstrainedOptimizer(
            self.A, self.l, self.d, use_gpu=True
        )
        
        # Run benchmark
        benchmark_result = optimizer.benchmark_gpu_vs_cpu()
        
        # Check result structure
        if 'error' not in benchmark_result:
            required_keys = [
                'operation', 'gpu_time', 'cpu_time', 'speedup',
                'gpu_success', 'cpu_success'
            ]
            for key in required_keys:
                self.assertIn(key, benchmark_result)


class TestGPULeontiefIntegration(unittest.TestCase):
    """Test GPU integration with Leontief model."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not LEONTIEF_AVAILABLE:
            self.skipTest("Leontief module not available")
        
        # Create test problem
        self.n = 10
        self.A = np.random.rand(self.n, self.n) * 0.1
        self.d = np.random.rand(self.n) * 100
    
    def test_gpu_leontief_creation(self):
        """Test GPU-enabled Leontief model creation."""
        if not GPU_ACCELERATION_AVAILABLE or not LEONTIEF_AVAILABLE:
            self.skipTest("Required modules not available")
        
        model = LeontiefModel(self.A, self.d, use_gpu=True)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.use_gpu)
    
    def test_gpu_leontief_computation(self):
        """Test GPU-enabled Leontief computations."""
        if not GPU_ACCELERATION_AVAILABLE or not LEONTIEF_AVAILABLE:
            self.skipTest("Required modules not available")
        
        model = LeontiefModel(self.A, self.d, use_gpu=True)
        
        # Test total output computation
        total_output = model.compute_total_output()
        
        self.assertIsInstance(total_output, np.ndarray)
        self.assertEqual(len(total_output), self.n)
    
    def test_gpu_leontief_status(self):
        """Test GPU status reporting for Leontief model."""
        if not GPU_ACCELERATION_AVAILABLE or not LEONTIEF_AVAILABLE:
            self.skipTest("Required modules not available")
        
        model = LeontiefModel(self.A, self.d, use_gpu=True)
        
        status = model.get_gpu_status()
        
        # Check status structure
        required_keys = ['gpu_available', 'gpu_enabled', 'gpu_arrays_created']
        for key in required_keys:
            self.assertIn(key, status)


class TestGPUErrorHandling(unittest.TestCase):
    """Test GPU error handling and fallbacks."""
    
    def test_gpu_unavailable_fallback(self):
        """Test fallback when GPU is unavailable."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # This test assumes GPU is not available
        # In a real test environment, you might need to mock this
        detector = GPUDetector()
        
        if not detector.is_gpu_available():
            # Test that CPU fallback works
            arrays = [np.random.rand(10, 10)]
            cpu_arrays = create_gpu_optimized_arrays(arrays, use_gpu=True)
            self.assertEqual(len(cpu_arrays), len(arrays))
    
    def test_invalid_gpu_settings(self):
        """Test handling of invalid GPU settings."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.skipTest("GPU acceleration module not available")
        
        # Test with invalid config file
        invalid_config_path = "/invalid/path/config.json"
        settings_manager = GPUSettingsManager(invalid_config_path)
        
        # Should not raise exception, should use defaults
        settings = settings_manager.get_settings()
        self.assertIsInstance(settings, dict)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
