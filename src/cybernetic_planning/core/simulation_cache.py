"""
Simulation Caching System

Implements intelligent caching for expensive computations in the cybernetic planning simulation.
This module provides caching for matrix operations, labor value calculations, and other
computationally expensive operations to significantly improve performance.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from scipy.sparse import issparse, csr_matrix, csc_matrix
import hashlib
import time
from dataclasses import dataclass
import warnings


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    computation_time: float = 0.0


class SimulationCache:
    """
    Intelligent caching system for simulation computations.
    
    Features:
    - Matrix operation caching
    - Labor value calculation caching
    - Technology efficiency caching
    - Automatic cache invalidation
    - Memory usage monitoring
    - Performance metrics
    """
    
    def __init__(self, max_cache_size: int = 1000, max_memory_mb: int = 500):
        """
        Initialize the simulation cache.
        
        Args:
            max_cache_size: Maximum number of cache entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_cache_size = max_cache_size
        self.max_memory_mb = max_memory_mb
        self.cache: Dict[str, CacheEntry] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.total_computation_time = 0.0
        
        # Cache keys for different operations
        self._matrix_cache_keys = set()
        self._labor_cache_keys = set()
        self._tech_efficiency_cache_keys = set()
        
    def _generate_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate a unique cache key for the given operation and arguments."""
        # Create a hash of the operation and arguments
        key_data = f"{operation}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_memory_usage(self, value: Any) -> int:
        """Estimate memory usage of a value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif issparse(value):
            return value.data.nbytes + value.indices.nbytes + value.indptr.nbytes
        elif isinstance(value, dict):
            return sum(self._estimate_memory_usage(v) for v in value.values())
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_memory_usage(v) for v in value)
        else:
            return 100  # Default estimate for small objects
    
    def _get_total_memory_usage(self) -> int:
        """Get total memory usage of cache in bytes."""
        total = 0
        for entry in self.cache.values():
            total += self._estimate_memory_usage(entry.value)
        return total
    
    def _cleanup_cache(self) -> None:
        """Clean up cache by removing least recently used entries."""
        if len(self.cache) <= self.max_cache_size:
            return
            
        # Sort by access count and timestamp (LRU)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )
        
        # Remove oldest 25% of entries
        entries_to_remove = len(sorted_entries) // 4
        for key, _ in sorted_entries[:entries_to_remove]:
            del self.cache[key]
    
    def _check_memory_limit(self) -> None:
        """Check if cache exceeds memory limit and clean up if necessary."""
        total_memory_bytes = self._get_total_memory_usage()
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        
        if total_memory_mb > self.max_memory_mb:
            # Remove least recently used entries until under limit
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1].access_count, x[1].timestamp)
            )
            
            for key, _ in sorted_entries:
                del self.cache[key]
                total_memory_bytes = self._get_total_memory_usage()
                total_memory_mb = total_memory_bytes / (1024 * 1024)
                if total_memory_mb <= self.max_memory_mb * 0.8:  # Leave some headroom
                    break
    
    def get(self, operation: str, *args, **kwargs) -> Optional[Any]:
        """
        Get a cached value for the given operation and arguments.
        
        Args:
            operation: Name of the operation
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cached value if found, None otherwise
        """
        key = self._generate_cache_key(operation, *args, **kwargs)
        
        if key in self.cache:
            self.cache[key].access_count += 1
            self.hit_count += 1
            return self.cache[key].value
        else:
            self.miss_count += 1
            return None
    
    def set(self, operation: str, value: Any, computation_time: float = 0.0, *args, **kwargs) -> None:
        """
        Set a cached value for the given operation and arguments.
        
        Args:
            operation: Name of the operation
            value: Value to cache
            computation_time: Time taken to compute the value
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        key = self._generate_cache_key(operation, *args, **kwargs)
        
        # Check memory limits before adding
        self._check_memory_limit()
        self._cleanup_cache()
        
        # Store the value
        self.cache[key] = CacheEntry(
            value=value,
            timestamp=time.time(),
            computation_time=computation_time
        )
        
        # Track operation-specific cache keys
        if operation.startswith('matrix_'):
            self._matrix_cache_keys.add(key)
        elif operation.startswith('labor_'):
            self._labor_cache_keys.add(key)
        elif operation.startswith('tech_efficiency_'):
            self._tech_efficiency_cache_keys.add(key)
        
        self.total_computation_time += computation_time
    
    def get_or_compute(self, operation: str, compute_func, *args, **kwargs) -> Any:
        """
        Get cached value or compute and cache it.
        
        Args:
            operation: Name of the operation
            compute_func: Function to compute the value if not cached
            *args: Positional arguments for compute_func
            **kwargs: Keyword arguments for compute_func
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        cached_value = self.get(operation, *args, **kwargs)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        start_time = time.time()
        value = compute_func(*args, **kwargs)
        computation_time = time.time() - start_time
        
        # Cache the result
        self.set(operation, value, computation_time, *args, **kwargs)
        
        return value
    
    def invalidate_operation(self, operation_pattern: str) -> None:
        """
        Invalidate cache entries matching the operation pattern.
        
        Args:
            operation_pattern: Pattern to match operation names
        """
        keys_to_remove = []
        for key in self.cache.keys():
            if operation_pattern in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def invalidate_matrix_cache(self) -> None:
        """Invalidate all matrix-related cache entries."""
        for key in list(self._matrix_cache_keys):
            if key in self.cache:
                del self.cache[key]
        self._matrix_cache_keys.clear()
    
    def invalidate_labor_cache(self) -> None:
        """Invalidate all labor-related cache entries."""
        for key in list(self._labor_cache_keys):
            if key in self.cache:
                del self.cache[key]
        self._labor_cache_keys.clear()
    
    def invalidate_tech_efficiency_cache(self) -> None:
        """Invalidate all technology efficiency cache entries."""
        for key in list(self._tech_efficiency_cache_keys):
            if key in self.cache:
                del self.cache[key]
        self._tech_efficiency_cache_keys.clear()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self._matrix_cache_keys.clear()
        self._labor_cache_keys.clear()
        self._tech_efficiency_cache_keys.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.total_computation_time = 0.0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        total_memory_bytes = self._get_total_memory_usage()
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_memory_mb': total_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'total_computation_time': self.total_computation_time,
            'matrix_cache_entries': len(self._matrix_cache_keys),
            'labor_cache_entries': len(self._labor_cache_keys),
            'tech_efficiency_cache_entries': len(self._tech_efficiency_cache_keys)
        }


class MatrixOperationCache:
    """
    Specialized cache for matrix operations with optimized storage.
    """
    
    def __init__(self, cache: SimulationCache):
        """Initialize with reference to main cache."""
        self.cache = cache
    
    def get_modified_matrices(
        self, 
        A: np.ndarray, 
        l: np.ndarray, 
        d: np.ndarray,
        tech_level: float,
        living_standards: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get or compute modified matrices with caching.
        
        Args:
            A: Technology matrix
            l: Labor vector
            d: Final demand vector
            tech_level: Current technology level
            living_standards: Current living standards
            
        Returns:
            Tuple of (A_modified, l_modified, d_modified)
        """
        def compute_modified_matrices(A, l, d, tech_level, living_standards):
            # Calculate technology-driven efficiency improvements
            n = A.shape[0]
            efficiency = np.ones(n)
            
            # Technology level affects all sectors differently
            tech_multiplier = 1.0 + tech_level * 0.5
            
            # Apply efficiency improvements
            A_modified = A.copy()
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    A_modified[i, j] = A[i, j] / efficiency[j]
            
            A_modified = np.clip(A_modified, 0, 0.99)
            
            # Update labor vector
            l_modified = l / efficiency
            living_standards_factor = 1.0 + living_standards * 0.3
            l_modified = l_modified / living_standards_factor
            
            # Update final demand
            d_modified = d.copy()
            living_standards_sectors = ['Healthcare', 'Education', 'Construction', 'Agriculture']
            # Note: This would need sector names to be passed in for full functionality
            
            return A_modified, l_modified, d_modified
        
        return self.cache.get_or_compute(
            'matrix_modified',
            compute_modified_matrices,
            A, l, d, tech_level, living_standards
        )
    
    def get_leontief_inverse(self, A: np.ndarray) -> np.ndarray:
        """
        Get or compute Leontief inverse with caching.
        
        Args:
            A: Technology matrix
            
        Returns:
            Leontief inverse matrix (I - A)^(-1)
        """
        def compute_leontief_inverse():
            n = A.shape[0]
            I = np.eye(n)
            
            if issparse(A):
                I_sparse = csr_matrix(I)
                return spsolve(I_sparse - A, I_sparse)
            else:
                return np.linalg.solve(I - A, I)
        
        return self.cache.get_or_compute(
            'matrix_leontief_inverse',
            compute_leontief_inverse,
            A
        )


class LaborValueCache:
    """
    Specialized cache for labor value calculations.
    """
    
    def __init__(self, cache: SimulationCache):
        """Initialize with reference to main cache."""
        self.cache = cache
    
    def get_labor_values(self, A: np.ndarray, l: np.ndarray) -> np.ndarray:
        """
        Get or compute labor values with caching.
        
        Args:
            A: Technology matrix
            l: Direct labor vector
            
        Returns:
            Labor values vector
        """
        def compute_labor_values():
            # This would integrate with the actual LaborValueCalculator
            # For now, return a simple computation
            leontief_inverse = MatrixOperationCache(self.cache).get_leontief_inverse(A)
            return leontief_inverse @ l
        
        return self.cache.get_or_compute(
            'labor_values',
            compute_labor_values,
            A, l
        )


class TechnologyEfficiencyCache:
    """
    Specialized cache for technology efficiency calculations.
    """
    
    def __init__(self, cache: SimulationCache):
        """Initialize with reference to main cache."""
        self.cache = cache
    
    def get_technology_efficiency(
        self, 
        tech_level: float, 
        sector_names: list,
        base_efficiency: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get or compute technology efficiency with caching.
        
        Args:
            tech_level: Current technology level
            sector_names: List of sector names
            base_efficiency: Base efficiency vector (optional)
            
        Returns:
            Technology efficiency vector
        """
        def compute_technology_efficiency():
            n = len(sector_names)
            efficiency = np.ones(n) if base_efficiency is None else base_efficiency.copy()
            
            tech_multiplier = 1.0 + tech_level * 0.5
            
            high_tech_sectors = ['Technology', 'Healthcare', 'Education', 'Research']
            tech_sectors = ['Technology', 'Manufacturing', 'Energy', 'Healthcare', 'Education']
            
            for i, sector in enumerate(sector_names):
                if sector in high_tech_sectors:
                    efficiency[i] = tech_multiplier * 1.2
                elif sector in tech_sectors:
                    efficiency[i] = tech_multiplier * 1.1
                else:
                    efficiency[i] = tech_multiplier * 1.05
            
            return efficiency
        
        return self.cache.get_or_compute(
            'tech_efficiency',
            compute_technology_efficiency,
            tech_level, sector_names, base_efficiency
        )
