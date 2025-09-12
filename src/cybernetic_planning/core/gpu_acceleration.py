"""
GPU Acceleration Module for Cybernetic Planning System

Provides GPU detection, solver selection, and performance monitoring
for accelerated economic computations.
"""

import os
import json
import warnings
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import cvxpy as cvx
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cvx = None

class GPUDetector:
    """Detects and manages GPU capabilities."""
    
    def __init__(self):
        self.gpu_info = {}
        self.cuda_available = False
        self.rocm_available = False
        self.gpu_memory = 0
        self.compute_capability = None
        
    def detect_gpu_capabilities(self) -> Dict[str, Any]:
        """
        Detect available GPU capabilities and return status.
        
        Returns:
            Dictionary with GPU information and availability status
        """
        gpu_info = {
            "cupy_available": CUPY_AVAILABLE,
            "cuda_available": False,
            "rocm_available": False,
            "gpu_count": 0,
            "gpu_memory": 0,
            "compute_capability": None,
            "gpu_names": [],
            "error": None
        }
        
        if not CUPY_AVAILABLE:
            gpu_info["error"] = "CuPy not available - install with: pip install cupy-cuda11x or cupy-cuda12x"
            return gpu_info
            
        try:
            # Get GPU count
            gpu_count = cp.cuda.runtime.getDeviceCount()
            gpu_info["gpu_count"] = gpu_count
            
            if gpu_count > 0:
                # Get GPU information
                gpu_names = []
                total_memory = 0
                
                for i in range(gpu_count):
                    with cp.cuda.Device(i):
                        # Get GPU name
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        gpu_name = props['name'].decode('utf-8')
                        gpu_names.append(gpu_name)
                        
                        # Get memory info
                        mem_info = cp.cuda.runtime.memGetInfo()
                        free_mem = mem_info[0]
                        total_mem = mem_info[1]
                        total_memory += total_mem
                        
                        # Get compute capability
                        major = props['major']
                        minor = props['minor']
                        compute_cap = f"{major}.{minor}"
                        
                        if i == 0:  # Use first GPU's compute capability
                            gpu_info["compute_capability"] = compute_cap
                
                gpu_info["gpu_names"] = gpu_names
                gpu_info["gpu_memory"] = total_memory
                gpu_info["cuda_available"] = True
                
                # Check for ROCm (AMD GPUs)
                try:
                    # This is a simple check - in practice, you'd check for ROCm-specific libraries
                    if any("AMD" in name or "Radeon" in name for name in gpu_names):
                        gpu_info["rocm_available"] = True
                except:
                    pass
                    
        except Exception as e:
            gpu_info["error"] = f"GPU detection failed: {str(e)}"
        
        # Set the overall GPU availability status
        gpu_info["gpu_available"] = self.is_gpu_available()
            
        self.gpu_info = gpu_info
        return gpu_info
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and functional."""
        if not CUPY_AVAILABLE:
            return False
            
        try:
            # Try to create a simple array on GPU
            test_array = cp.array([1, 2, 3])
            del test_array
            return True
        except:
            return False
    
    def get_gpu_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage."""
        if not self.is_gpu_available():
            return {"used": 0, "free": 0, "total": 0}
            
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mem = mem_info[0]
            total_mem = mem_info[1]
            used_mem = total_mem - free_mem
            
            return {
                "used": used_mem,
                "free": free_mem,
                "total": total_mem
            }
        except:
            return {"used": 0, "free": 0, "total": 0}


class GPUSolverSelector:
    """Selects appropriate GPU solvers for optimization problems."""
    
    def __init__(self):
        self.available_solvers = []
        self._detect_solvers()
    
    def _detect_solvers(self):
        """Detect available GPU solvers."""
        self.available_solvers = []
        
        if not CVXPY_AVAILABLE:
            return
            
        # Check for GPU-enabled solvers (removed CuClarabel as it requires juliacall)
        # Note: CuClarabel requires juliacall which we removed to fix PCRE2 errors
            
        try:
            # Check for other GPU solvers
            if hasattr(cvx, 'SCS'):
                self.available_solvers.append("SCS")
            if hasattr(cvx, 'ECOS'):
                self.available_solvers.append("ECOS")
            if hasattr(cvx, 'OSQP'):
                self.available_solvers.append("OSQP")
        except:
            pass
    
    def select_gpu_solver(self, use_gpu: bool = True, solver_preference: str = "SCS") -> Optional[str]:
        """
        Select appropriate solver based on GPU availability and user preference.
        
        Args:
            use_gpu: Whether to prefer GPU solvers
            solver_preference: Preferred solver name
            
        Returns:
            Selected solver name or None if no suitable solver found
        """
        if not use_gpu or not self.available_solvers:
            # Fall back to CPU solvers
            cpu_solvers = ["ECOS", "SCS", "OSQP", "CLARABEL"]
            for solver in cpu_solvers:
                if solver in self.available_solvers:
                    return solver
            return None
        
        # Prefer GPU solvers
        if solver_preference in self.available_solvers:
            return solver_preference
            
        # Return first available GPU solver
        return self.available_solvers[0] if self.available_solvers else None


class GPUPerformanceMonitor:
    """Monitors GPU performance and provides benchmarking."""
    
    def __init__(self):
        self.benchmarks = {}
        self.monitoring_enabled = False
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_enabled = True
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_enabled = False
        
    def benchmark_operation(self, operation_name: str, gpu_func, cpu_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance for an operation.
        
        Args:
            operation_name: Name of the operation being benchmarked
            gpu_func: GPU version of the function
            cpu_func: CPU version of the function
            *args, **kwargs: Arguments to pass to both functions
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        results = {
            "operation": operation_name,
            "gpu_time": None,
            "cpu_time": None,
            "speedup": None,
            "gpu_success": False,
            "cpu_success": False,
            "error": None
        }
        
        # Benchmark GPU version
        if CUPY_AVAILABLE:
            try:
                start_time = time.time()
                gpu_result = gpu_func(*args, **kwargs)
                gpu_time = time.time() - start_time
                results["gpu_time"] = gpu_time
                results["gpu_success"] = True
            except Exception as e:
                results["error"] = f"GPU operation failed: {str(e)}"
        
        # Benchmark CPU version
        try:
            start_time = time.time()
            cpu_result = cpu_func(*args, **kwargs)
            cpu_time = time.time() - start_time
            results["cpu_time"] = cpu_time
            results["cpu_success"] = True
        except Exception as e:
            if not results["error"]:
                results["error"] = f"CPU operation failed: {str(e)}"
        
        # Calculate speedup
        if results["gpu_success"] and results["cpu_success"]:
            results["speedup"] = results["cpu_time"] / results["gpu_time"]
        
        # Store benchmark
        self.benchmarks[operation_name] = results
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks."""
        if not self.benchmarks:
            return {"message": "No benchmarks available"}
        
        successful_benchmarks = {k: v for k, v in self.benchmarks.items() 
                               if v["gpu_success"] and v["cpu_success"]}
        
        if not successful_benchmarks:
            return {"message": "No successful benchmarks available"}
        
        speedups = [v["speedup"] for v in successful_benchmarks.values()]
        
        return {
            "total_benchmarks": len(self.benchmarks),
            "successful_benchmarks": len(successful_benchmarks),
            "average_speedup": np.mean(speedups),
            "max_speedup": np.max(speedups),
            "min_speedup": np.min(speedups),
            "benchmarks": successful_benchmarks
        }


class GPUSettingsManager:
    """Manages GPU settings and configuration persistence."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.settings = self._load_default_settings()
        self.load_settings()
    
    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default GPU settings."""
        return {
            "gpu_settings": {
                "enabled": False,
                "solver": "SCS",
                "monitoring": {
                    "show_utilization": True,
                    "benchmark_mode": False
                },
                "fallback_to_cpu": True
            }
        }
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from configuration file."""
        if not self.config_path.exists():
            return self.settings
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Merge with default settings
            if "gpu_settings" in config:
                self.settings["gpu_settings"].update(config["gpu_settings"])
            
            return self.settings
        except Exception as e:
            warnings.warn(f"Failed to load GPU settings: {e}")
            return self.settings
    
    def save_settings(self, gpu_enabled: bool, solver_type: str, 
                     monitoring_enabled: bool, benchmark_mode: bool) -> bool:
        """
        Save GPU settings to configuration file.
        
        Args:
            gpu_enabled: Whether GPU acceleration is enabled
            solver_type: Type of solver to use
            monitoring_enabled: Whether to show utilization
            benchmark_mode: Whether to run in benchmark mode
            
        Returns:
            True if settings saved successfully
        """
        try:
            # Update settings
            self.settings["gpu_settings"]["enabled"] = gpu_enabled
            self.settings["gpu_settings"]["solver"] = solver_type
            self.settings["gpu_settings"]["monitoring"]["show_utilization"] = monitoring_enabled
            self.settings["gpu_settings"]["monitoring"]["benchmark_mode"] = benchmark_mode
            
            # Load existing config
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Update with new GPU settings
            config["gpu_settings"] = self.settings["gpu_settings"]
            
            # Save updated config
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to save GPU settings: {e}")
            return False
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current GPU settings."""
        return self.settings["gpu_settings"].copy()
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return self.settings["gpu_settings"]["enabled"]
    
    def get_solver_preference(self) -> str:
        """Get preferred solver type."""
        return self.settings["gpu_settings"]["solver"]
    
    def should_fallback_to_cpu(self) -> bool:
        """Check if system should fallback to CPU when GPU fails."""
        return self.settings["gpu_settings"]["fallback_to_cpu"]


def create_gpu_optimized_arrays(arrays: List[np.ndarray], use_gpu: bool = True) -> List:
    """
    Convert numpy arrays to GPU arrays if GPU is available and enabled.
    
    Args:
        arrays: List of numpy arrays to convert
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        List of arrays (GPU or CPU)
    """
    if not use_gpu or not CUPY_AVAILABLE:
        return arrays
    
    try:
        gpu_arrays = []
        for arr in arrays:
            gpu_arrays.append(cp.asarray(arr))
        return gpu_arrays
    except Exception as e:
        warnings.warn(f"Failed to convert arrays to GPU: {e}")
        return arrays


def convert_gpu_to_cpu(arrays: List) -> List[np.ndarray]:
    """
    Convert GPU arrays back to CPU arrays.
    
    Args:
        arrays: List of arrays (GPU or CPU)
        
    Returns:
        List of numpy arrays
    """
    if not CUPY_AVAILABLE:
        return arrays
    
    cpu_arrays = []
    for arr in arrays:
        if hasattr(arr, 'get'):  # GPU array
            cpu_arrays.append(arr.get())
        else:  # Already CPU array
            cpu_arrays.append(arr)
    return cpu_arrays


# Global instances
gpu_detector = GPUDetector()
solver_selector = GPUSolverSelector()
performance_monitor = GPUPerformanceMonitor()
settings_manager = GPUSettingsManager()
