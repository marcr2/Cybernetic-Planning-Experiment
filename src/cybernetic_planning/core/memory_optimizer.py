"""
Memory Optimization Module

Implements memory optimization strategies for the cybernetic planning simulation.
This module provides memory-efficient data structures, circular buffers, and
memory management utilities to reduce memory usage and improve performance.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque
import gc
import psutil
import warnings
from dataclasses import dataclass
import weakref


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    available_memory_mb: float
    memory_usage_percent: float
    simulation_memory_mb: float
    cache_memory_mb: float
    matrix_memory_mb: float


class CircularBuffer:
    """
    Memory-efficient circular buffer for storing simulation history.
    
    Features:
    - Fixed memory footprint
    - Automatic overwriting of old data
    - Efficient indexing
    - Memory-mapped storage option
    """
    
    def __init__(self, max_size: int, dtype: np.dtype = np.float64):
        """
        Initialize circular buffer.
        
        Args:
            max_size: Maximum number of elements to store
            dtype: Data type for stored elements
        """
        self.max_size = max_size
        self.dtype = dtype
        self.buffer = np.zeros(max_size, dtype=dtype)
        self.head = 0
        self.size = 0
        self.total_added = 0
    
    def append(self, value: Union[float, np.ndarray]) -> None:
        """Add a value to the buffer."""
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = float(value.item())
            else:
                # For arrays, store only the first element or mean
                value = float(np.mean(value))
        
        self.buffer[self.head] = value
        self.head = (self.head + 1) % self.max_size
        
        if self.size < self.max_size:
            self.size += 1
        
        self.total_added += 1
    
    def get_recent(self, n: int) -> np.ndarray:
        """Get the most recent n values."""
        if n <= 0:
            return np.array([])
        
        n = min(n, self.size)
        if self.size < self.max_size:
            return self.buffer[:self.size][-n:]
        else:
            indices = [(self.head - n + i) % self.max_size for i in range(n)]
            return self.buffer[indices]
    
    def get_all(self) -> np.ndarray:
        """Get all stored values in chronological order."""
        if self.size < self.max_size:
            return self.buffer[:self.size].copy()
        else:
            indices = [(self.head + i) % self.max_size for i in range(self.max_size)]
            return self.buffer[indices].copy()
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.fill(0)
        self.head = 0
        self.size = 0
        self.total_added = 0


class MemoryEfficientMatrix:
    """
    Memory-efficient matrix wrapper with lazy evaluation and compression.
    """
    
    def __init__(self, matrix: np.ndarray, compress_threshold: float = 0.1):
        """
        Initialize memory-efficient matrix.
        
        Args:
            matrix: Input matrix
            compress_threshold: Sparsity threshold for compression
        """
        self.original_shape = matrix.shape
        self.compress_threshold = compress_threshold
        self._compressed = None
        self._dense = None
        
        # Determine if matrix should be compressed
        sparsity = np.count_nonzero(matrix == 0) / matrix.size
        if sparsity > compress_threshold:
            self._compressed = self._compress_matrix(matrix)
        else:
            self._dense = matrix.copy()
    
    def _compress_matrix(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Compress matrix using sparse representation."""
        from scipy.sparse import csr_matrix
        
        sparse_matrix = csr_matrix(matrix)
        return {
            'data': sparse_matrix.data,
            'indices': sparse_matrix.indices,
            'indptr': sparse_matrix.indptr,
            'shape': matrix.shape
        }
    
    def _decompress_matrix(self) -> np.ndarray:
        """Decompress matrix to dense format."""
        if self._compressed is not None:
            from scipy.sparse import csr_matrix
            sparse_matrix = csr_matrix(
                (self._compressed['data'], 
                 self._compressed['indices'], 
                 self._compressed['indptr']),
                shape=self._compressed['shape']
            )
            return sparse_matrix.toarray()
        else:
            return self._dense
    
    def get_matrix(self) -> np.ndarray:
        """Get the matrix (decompressed if necessary)."""
        if self._dense is not None:
            return self._dense
        else:
            return self._decompress_matrix()
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        if self._dense is not None:
            return self._dense.nbytes
        else:
            return (self._compressed['data'].nbytes + 
                   self._compressed['indices'].nbytes + 
                   self._compressed['indptr'].nbytes)


class MemoryOptimizer:
    """
    Main memory optimization manager for the simulation.
    """
    
    def __init__(self, max_memory_mb: int = 1000):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.monitored_objects = weakref.WeakSet()
        self.circular_buffers = {}
        self.memory_efficient_matrices = {}
        
    def create_circular_buffer(self, name: str, max_size: int, dtype: np.dtype = np.float64) -> CircularBuffer:
        """Create a circular buffer for storing time series data."""
        buffer = CircularBuffer(max_size, dtype)
        self.circular_buffers[name] = buffer
        return buffer
    
    def create_memory_efficient_matrix(self, name: str, matrix: np.ndarray, compress_threshold: float = 0.1) -> MemoryEfficientMatrix:
        """Create a memory-efficient matrix wrapper."""
        mem_matrix = MemoryEfficientMatrix(matrix, compress_threshold)
        self.memory_efficient_matrices[name] = mem_matrix
        return mem_matrix
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        
        # Calculate simulation-specific memory usage
        simulation_memory = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        # Estimate cache memory usage
        cache_memory = 0
        for buffer in self.circular_buffers.values():
            cache_memory += buffer.buffer.nbytes / (1024 * 1024)
        
        # Estimate matrix memory usage
        matrix_memory = 0
        for matrix in self.memory_efficient_matrices.values():
            matrix_memory += matrix.get_memory_usage() / (1024 * 1024)
        
        return MemoryStats(
            total_memory_mb=system_memory.total / (1024 * 1024),
            available_memory_mb=system_memory.available / (1024 * 1024),
            memory_usage_percent=system_memory.percent,
            simulation_memory_mb=simulation_memory,
            cache_memory_mb=cache_memory,
            matrix_memory_mb=matrix_memory
        )
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization operations."""
        stats_before = self.get_memory_stats()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear unused circular buffers
        buffers_to_remove = []
        for name, buffer in self.circular_buffers.items():
            if buffer.total_added == 0:  # Unused buffer
                buffers_to_remove.append(name)
        
        for name in buffers_to_remove:
            del self.circular_buffers[name]
        
        # Compress matrices if memory usage is high
        if stats_before.simulation_memory_mb > self.max_memory_mb * 0.8:
            for name, matrix in self.memory_efficient_matrices.items():
                if matrix._dense is not None:  # Not already compressed
                    # Recompress with higher threshold
                    matrix._compressed = matrix._compress_matrix(matrix._dense)
                    matrix._dense = None
        
        stats_after = self.get_memory_stats()
        
        return {
            'memory_freed_mb': stats_before.simulation_memory_mb - stats_after.simulation_memory_mb,
            'garbage_collected': collected,
            'buffers_removed': len(buffers_to_remove),
            'memory_usage_before_mb': stats_before.simulation_memory_mb,
            'memory_usage_after_mb': stats_after.simulation_memory_mb
        }
    
    def preallocate_arrays(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64, 
                          num_arrays: int = 1) -> List[np.ndarray]:
        """Pre-allocate arrays for better memory management."""
        arrays = []
        for _ in range(num_arrays):
            arrays.append(np.zeros(shape, dtype=dtype))
        return arrays
    
    def create_memory_pool(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64, 
                          pool_size: int = 10) -> List[np.ndarray]:
        """Create a pool of pre-allocated arrays for reuse."""
        return self.preallocate_arrays(shape, dtype, pool_size)
    
    def cleanup_unused_objects(self) -> int:
        """Clean up unused objects and return number of objects collected."""
        return gc.collect()
    
    def monitor_object(self, obj: Any) -> None:
        """Add an object to memory monitoring."""
        self.monitored_objects.add(obj)
    
    def get_monitored_objects_count(self) -> int:
        """Get the number of monitored objects."""
        return len(self.monitored_objects)


class SimulationMemoryManager:
    """
    High-level memory manager for simulation data.
    """
    
    def __init__(self, max_history_months: int = 120, max_memory_mb: int = 1000):
        """
        Initialize simulation memory manager.
        
        Args:
            max_history_months: Maximum number of months to keep in history
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_history_months = max_history_months
        self.memory_optimizer = MemoryOptimizer(max_memory_mb)
        
        # Create circular buffers for different metrics
        self.metrics_buffer = self.memory_optimizer.create_circular_buffer(
            'metrics', max_history_months
        )
        self.production_buffer = self.memory_optimizer.create_circular_buffer(
            'production', max_history_months
        )
        self.labor_buffer = self.memory_optimizer.create_circular_buffer(
            'labor', max_history_months
        )
        
        # Memory pools for temporary arrays
        self.temp_arrays_pool = self.memory_optimizer.create_memory_pool(
            (100,), np.float64, 20
        )
        self.temp_arrays_index = 0
    
    def get_temp_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """Get a temporary array from the pool."""
        if self.temp_arrays_index >= len(self.temp_arrays_pool):
            # Create new array if pool is exhausted
            return np.zeros(shape, dtype=dtype)
        
        array = self.temp_arrays_pool[self.temp_arrays_index]
        self.temp_arrays_index = (self.temp_arrays_index + 1) % len(self.temp_arrays_pool)
        
        # Resize if necessary
        if array.shape != shape:
            array = np.zeros(shape, dtype=dtype)
        
        return array
    
    def store_simulation_data(self, month: int, metrics: Dict[str, Any]) -> None:
        """Store simulation data in circular buffers."""
        # Store key metrics
        if 'total_economic_output' in metrics:
            self.metrics_buffer.append(metrics['total_economic_output'])
        
        if 'average_efficiency' in metrics:
            self.production_buffer.append(metrics['average_efficiency'])
        
        if 'labor_productivity' in metrics:
            self.labor_buffer.append(metrics['labor_productivity'])
    
    def get_recent_metrics(self, months: int = 12) -> Dict[str, np.ndarray]:
        """Get recent metrics data."""
        return {
            'economic_output': self.metrics_buffer.get_recent(months),
            'efficiency': self.production_buffer.get_recent(months),
            'labor_productivity': self.labor_buffer.get_recent(months)
        }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage and return statistics."""
        return self.memory_optimizer.optimize_memory()
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self.memory_optimizer.get_memory_stats()
    
    def cleanup(self) -> None:
        """Clean up all managed resources."""
        self.memory_optimizer.cleanup_unused_objects()
        self.temp_arrays_pool.clear()
        self.temp_arrays_index = 0
