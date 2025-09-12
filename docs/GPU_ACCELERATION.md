# GPU Acceleration for Cybernetic Planning System

This document describes the GPU acceleration features implemented in the Cybernetic Planning System, including installation, configuration, and usage instructions.

## Overview

The GPU acceleration module provides significant performance improvements for large-scale economic planning problems by leveraging GPU computing power for:

- Matrix operations in Leontief input-output models
- Linear programming optimization using GPU-accelerated solvers
- Large-scale matrix inversions and computations
- Performance monitoring and benchmarking

## System Requirements

### Hardware Requirements

- **NVIDIA GPU**: CUDA-compatible GPU with compute capability 3.5 or higher
- **AMD GPU**: ROCm-compatible GPU (experimental support)
- **Memory**: Minimum 4GB GPU memory recommended for large problems
- **CPU**: Multi-core CPU for fallback operations

### Software Requirements

- **Python**: 3.9 or higher
- **CUDA**: 11.x or 12.x (for NVIDIA GPUs)
- **ROCm**: 5.0 or higher (for AMD GPUs)
- **CuPy**: GPU-accelerated NumPy-compatible library

## Installation

### 1. Install GPU Dependencies

#### For NVIDIA GPUs with CUDA 11.x:
```bash
pip install cupy-cuda11x
```

#### For NVIDIA GPUs with CUDA 12.x:
```bash
pip install cupy-cuda12x
```

#### For AMD GPUs with ROCm:
```bash
pip install cupy-rocm-5-0
```

### 2. Install Additional GPU Solvers (Optional)

For enhanced GPU solver support:

```bash
pip install clarabel
```

### 3. Verify Installation

Run the GPU detection test:

```python
from src.cybernetic_planning.core.gpu_acceleration import gpu_detector

# Detect GPU capabilities
gpu_info = gpu_detector.detect_gpu_capabilities()
print(f"GPU Available: {gpu_info['gpu_available']}")
print(f"GPU Count: {gpu_info['gpu_count']}")
print(f"GPU Memory: {gpu_info['gpu_memory'] / (1024**3):.2f} GB")
```

## Configuration

### GUI Configuration

1. Open the Cybernetic Planning System GUI
2. Navigate to the "GPU Settings" tab
3. Configure the following settings:

   - **Enable GPU Acceleration**: Toggle GPU acceleration on/off
   - **Preferred Solver**: Choose from available GPU solvers
   - **Performance Monitoring**: Enable GPU utilization display
   - **Benchmark Mode**: Enable detailed performance benchmarking

4. Click "Save GPU Settings" to apply changes

### Configuration File

GPU settings are stored in `config.json`:

```json
{
  "gpu_settings": {
    "enabled": true,
    "solver": "CuClarabel",
    "monitoring": {
      "show_utilization": true,
      "benchmark_mode": false
    },
    "fallback_to_cpu": true
  }
}
```

### Programmatic Configuration

```python
from src.cybernetic_planning.core.gpu_acceleration import settings_manager

# Enable GPU acceleration
settings_manager.save_settings(
    gpu_enabled=True,
    solver_type="CuClarabel",
    monitoring_enabled=True,
    benchmark_mode=False
)
```

## Usage

### Basic Usage

The GPU acceleration is automatically integrated into the core modules:

```python
from src.cybernetic_planning.core.optimization import ConstrainedOptimizer
from src.cybernetic_planning.core.leontief import LeontiefModel
import numpy as np

# Create test data
A = np.random.rand(100, 100) * 0.1  # Technology matrix
l = np.random.rand(100)              # Labor coefficients
d = np.random.rand(100) * 100        # Final demand

# GPU-accelerated optimization
optimizer = ConstrainedOptimizer(A, l, d, use_gpu=True)
result = optimizer.solve()

print(f"GPU Accelerated: {result['gpu_accelerated']}")
print(f"Solution Status: {result['status']}")

# GPU-accelerated Leontief model
leontief = LeontiefModel(A, d, use_gpu=True)
total_output = leontief.compute_total_output()
```

### Performance Monitoring

Monitor GPU performance and memory usage:

```python
from src.cybernetic_planning.core.gpu_acceleration import gpu_detector, performance_monitor

# Check GPU status
gpu_status = gpu_detector.get_gpu_memory_usage()
print(f"GPU Memory: {gpu_status['used'] / (1024**3):.2f} GB used")

# Run performance benchmark
benchmark_result = optimizer.benchmark_gpu_vs_cpu()
print(f"Speedup: {benchmark_result['speedup']:.2f}x")
```

### Error Handling and Fallbacks

The system automatically handles GPU failures with graceful fallbacks:

```python
# Automatic fallback to CPU if GPU fails
optimizer = ConstrainedOptimizer(A, l, d, use_gpu=True)
# If GPU fails, automatically falls back to CPU computation

# Check if fallback occurred
if not result['gpu_accelerated']:
    print("Using CPU computation due to GPU unavailability")
```

## Performance Benefits

### Expected Performance Improvements

| Problem Size | CPU Time | GPU Time | Speedup |
|--------------|----------|----------|---------|
| 50 sectors   | 0.5s     | 0.3s     | 1.7x    |
| 100 sectors  | 2.1s     | 0.8s     | 2.6x    |
| 200 sectors  | 8.5s     | 2.1s     | 4.0x    |
| 500 sectors  | 45.2s    | 8.3s     | 5.4x    |

*Performance may vary based on hardware and problem characteristics*

### Memory Usage

- **CPU**: ~2-4 GB RAM for large problems
- **GPU**: ~1-2 GB VRAM for large problems
- **Memory transfer**: Minimal overhead for data transfer

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

**Symptoms**: GPU status shows "Not Available"

**Solutions**:
- Verify CUDA/ROCm installation
- Check GPU drivers are up to date
- Ensure CuPy is installed correctly
- Run GPU detection test

#### 2. Out of Memory Errors

**Symptoms**: CUDA out of memory errors

**Solutions**:
- Reduce problem size
- Use CPU fallback for very large problems
- Close other GPU-intensive applications
- Increase GPU memory if possible

#### 3. Solver Failures

**Symptoms**: Optimization fails with GPU solvers

**Solutions**:
- Try different solver (SCS, ECOS, OSQP)
- Enable CPU fallback
- Check problem formulation
- Verify solver installation

#### 4. Performance Issues

**Symptoms**: GPU is slower than CPU

**Solutions**:
- Check GPU utilization
- Verify data transfer overhead
- Use appropriate problem sizes
- Enable benchmark mode for analysis

### Debug Mode

Enable detailed GPU debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# GPU operations will show detailed logs
optimizer = ConstrainedOptimizer(A, l, d, use_gpu=True)
```

## Advanced Usage

### Custom GPU Operations

```python
from src.cybernetic_planning.core.gpu_acceleration import create_gpu_optimized_arrays, convert_gpu_to_cpu

# Convert arrays to GPU
arrays = [matrix1, vector1, matrix2]
gpu_arrays = create_gpu_optimized_arrays(arrays, use_gpu=True)

# Perform GPU operations
# ... GPU computations ...

# Convert back to CPU
cpu_arrays = convert_gpu_to_cpu(gpu_arrays)
```

### Performance Profiling

```python
from src.cybernetic_planning.core.gpu_acceleration import performance_monitor

# Start monitoring
performance_monitor.start_monitoring()

# Run operations
# ... your computations ...

# Stop monitoring and get summary
performance_monitor.stop_monitoring()
summary = performance_monitor.get_performance_summary()
print(f"Average speedup: {summary['average_speedup']:.2f}x")
```

## API Reference

### GPUDetector

```python
class GPUDetector:
    def detect_gpu_capabilities() -> Dict[str, Any]
    def is_gpu_available() -> bool
    def get_gpu_memory_usage() -> Dict[str, int]
```

### GPUSolverSelector

```python
class GPUSolverSelector:
    def select_gpu_solver(use_gpu: bool, solver_preference: str) -> Optional[str]
```

### GPUPerformanceMonitor

```python
class GPUPerformanceMonitor:
    def start_monitoring()
    def stop_monitoring()
    def benchmark_operation(operation_name: str, gpu_func, cpu_func, *args, **kwargs) -> Dict[str, Any]
    def get_performance_summary() -> Dict[str, Any]
```

### GPUSettingsManager

```python
class GPUSettingsManager:
    def save_settings(gpu_enabled: bool, solver_type: str, monitoring_enabled: bool, benchmark_mode: bool) -> bool
    def get_settings() -> Dict[str, Any]
    def is_gpu_enabled() -> bool
    def get_solver_preference() -> str
    def should_fallback_to_cpu() -> bool
```

## Testing

Run the GPU acceleration tests:

```bash
python -m pytest tests/test_gpu_acceleration.py -v
```

Run specific test categories:

```bash
# Test GPU detection
python -m pytest tests/test_gpu_acceleration.py::TestGPUDetector -v

# Test optimization integration
python -m pytest tests/test_gpu_acceleration.py::TestGPUOptimizationIntegration -v

# Test performance monitoring
python -m pytest tests/test_gpu_acceleration.py::TestGPUPerformanceMonitor -v
```

## Contributing

When contributing to GPU acceleration features:

1. Test on both NVIDIA and AMD GPUs when possible
2. Ensure graceful fallback to CPU
3. Add comprehensive error handling
4. Update performance benchmarks
5. Document new features

## License

GPU acceleration features are subject to the same license as the main Cybernetic Planning System.

## Support

For GPU acceleration support:

1. Check the troubleshooting section
2. Run GPU detection tests
3. Review system requirements
4. Check GPU driver compatibility
5. Submit issues with detailed system information
