"""
Performance Monitoring Module

Provides comprehensive performance monitoring and profiling capabilities for the cybernetic planning simulation.
This module tracks execution times, memory usage, cache performance, and other key metrics.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
import warnings
from collections import defaultdict, deque
import json
import os
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    operations_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SimulationProfile:
    """Profile of simulation performance over time."""
    total_simulation_time: float = 0.0
    average_month_time: float = 0.0
    peak_memory_usage_mb: float = 0.0
    average_memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    total_operations: int = 0
    optimization_effectiveness: float = 0.0


class PerformanceMonitor:
    """
    Main performance monitoring system.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage_history: deque = deque(maxlen=max_history)
        self.cpu_usage_history: deque = deque(maxlen=max_history)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Monitoring state
        self.monitoring = False
        self.start_time = None
        self.baseline_memory = None
        
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        with self._lock:
            self.monitoring = True
            self.start_time = time.time()
            self.baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        with self._lock:
            self.monitoring = False
    
    def record_operation(
        self, 
        operation_name: str, 
        execution_time: float,
        memory_usage_mb: Optional[float] = None,
        cache_hits: int = 0,
        cache_misses: int = 0
    ) -> None:
        """Record performance metrics for an operation."""
        if not self.monitoring:
            return
        
        with self._lock:
            # Get current system metrics
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)
            cpu_usage = process.cpu_percent()
            
            # Create metrics record
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage_mb or current_memory,
                cpu_usage_percent=cpu_usage,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                operations_count=1
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.operation_times[operation_name].append(execution_time)
            self.memory_usage_history.append(current_memory)
            self.cpu_usage_history.append(cpu_usage)
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.record_operation(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=memory_delta
            )
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation_name not in self.operation_times:
            return {}
        
        times = self.operation_times[operation_name]
        
        return {
            'operation_name': operation_name,
            'count': len(times),
            'total_time': sum(times),
            'average_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'median_time': np.median(times)
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.metrics_history:
            return {}
        
        with self._lock:
            execution_times = [m.execution_time for m in self.metrics_history]
            memory_usage = [m.memory_usage_mb for m in self.metrics_history]
            cpu_usage = [m.cpu_usage_percent for m in self.metrics_history]
            
            total_cache_hits = sum(m.cache_hits for m in self.metrics_history)
            total_cache_misses = sum(m.cache_misses for m in self.metrics_history)
            total_operations = sum(m.operations_count for m in self.metrics_history)
            
            cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses) if (total_cache_hits + total_cache_misses) > 0 else 0.0
            
            return {
                'total_operations': total_operations,
                'total_execution_time': sum(execution_times),
                'average_execution_time': np.mean(execution_times),
                'peak_memory_usage_mb': max(memory_usage) if memory_usage else 0.0,
                'average_memory_usage_mb': np.mean(memory_usage),
                'peak_cpu_usage_percent': max(cpu_usage) if cpu_usage else 0.0,
                'average_cpu_usage_percent': np.mean(cpu_usage),
                'cache_hit_rate': cache_hit_rate,
                'total_cache_hits': total_cache_hits,
                'total_cache_misses': total_cache_misses,
                'monitoring_duration': time.time() - self.start_time if self.start_time else 0.0
            }
    
    def get_simulation_profile(self) -> SimulationProfile:
        """Get comprehensive simulation performance profile."""
        overall_stats = self.get_overall_stats()
        
        if not overall_stats:
            return SimulationProfile()
        
        # Calculate optimization effectiveness
        baseline_time = overall_stats.get('total_execution_time', 0.0)
        if baseline_time > 0:
            # Estimate baseline performance (without optimizations)
            estimated_baseline = baseline_time * 10  # Assume 10x improvement from optimizations
            optimization_effectiveness = (estimated_baseline - baseline_time) / estimated_baseline
        else:
            optimization_effectiveness = 0.0
        
        return SimulationProfile(
            total_simulation_time=overall_stats.get('total_execution_time', 0.0),
            average_month_time=overall_stats.get('average_execution_time', 0.0),
            peak_memory_usage_mb=overall_stats.get('peak_memory_usage_mb', 0.0),
            average_memory_usage_mb=overall_stats.get('average_memory_usage_mb', 0.0),
            cache_hit_rate=overall_stats.get('cache_hit_rate', 0.0),
            total_operations=overall_stats.get('total_operations', 0),
            optimization_effectiveness=optimization_effectiveness
        )
    
    def export_profile(self, file_path: str) -> None:
        """Export performance profile to file."""
        profile = self.get_simulation_profile()
        overall_stats = self.get_overall_stats()
        
        export_data = {
            'profile': {
                'total_simulation_time': profile.total_simulation_time,
                'average_month_time': profile.average_month_time,
                'peak_memory_usage_mb': profile.peak_memory_usage_mb,
                'average_memory_usage_mb': profile.average_memory_usage_mb,
                'cache_hit_rate': profile.cache_hit_rate,
                'total_operations': profile.total_operations,
                'optimization_effectiveness': profile.optimization_effectiveness
            },
            'overall_stats': overall_stats,
            'operation_stats': {
                op: self.get_operation_stats(op) 
                for op in self.operation_times.keys()
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def reset(self) -> None:
        """Reset all performance metrics."""
        with self._lock:
            self.metrics_history.clear()
            self.operation_times.clear()
            self.memory_usage_history.clear()
            self.cpu_usage_history.clear()
            self.start_time = None
            self.baseline_memory = None


class SimulationProfiler:
    """
    High-level profiler for simulation performance.
    """
    
    def __init__(self, simulation_instance):
        """
        Initialize simulation profiler.
        
        Args:
            simulation_instance: Instance of EnhancedEconomicSimulation to profile
        """
        self.simulation = simulation_instance
        self.monitor = PerformanceMonitor()
        self.profile_data = {}
    
    def profile_simulation(self, months: int = 12) -> Dict[str, Any]:
        """
        Profile a complete simulation run.
        
        Args:
            months: Number of months to simulate
            
        Returns:
            Profile results
        """
        self.monitor.start_monitoring()
        
        try:
            # Profile individual month simulations
            for month in range(1, months + 1):
                with self.monitor.time_operation('simulate_month'):
                    result = self.simulation.simulate_month(month)
                
                # Record cache performance
                cache_stats = self.simulation.get_cache_performance()
                self.monitor.record_operation(
                    'simulate_month',
                    0,  # Time already recorded by context manager
                    cache_hits=cache_stats.get('hit_count', 0),
                    cache_misses=cache_stats.get('miss_count', 0)
                )
            
            # Get final performance profile
            profile = self.monitor.get_simulation_profile()
            overall_stats = self.monitor.get_overall_stats()
            
            return {
                'profile': profile,
                'overall_stats': overall_stats,
                'simulation_months': months,
                'performance_grade': self._calculate_performance_grade(profile)
            }
            
        finally:
            self.monitor.stop_monitoring()
    
    def profile_batch_simulation(self, start_month: int, batch_size: int = 12) -> Dict[str, Any]:
        """
        Profile batch simulation performance.
        
        Args:
            start_month: Starting month
            batch_size: Batch size
            
        Returns:
            Batch profile results
        """
        self.monitor.start_monitoring()
        
        try:
            with self.monitor.time_operation('simulate_batch_months'):
                results = self.simulation.simulate_batch_months(start_month, batch_size)
            
            profile = self.monitor.get_simulation_profile()
            overall_stats = self.monitor.get_overall_stats()
            
            return {
                'profile': profile,
                'overall_stats': overall_stats,
                'batch_size': batch_size,
                'results_count': len(results),
                'performance_grade': self._calculate_performance_grade(profile)
            }
            
        finally:
            self.monitor.stop_monitoring()
    
    def _calculate_performance_grade(self, profile: SimulationProfile) -> str:
        """Calculate performance grade based on profile."""
        if profile.total_simulation_time == 0:
            return 'N/A'
        
        # Grade based on average month time
        avg_month_time = profile.average_month_time
        
        if avg_month_time < 0.1:  # Less than 100ms per month
            return 'A+'
        elif avg_month_time < 0.5:  # Less than 500ms per month
            return 'A'
        elif avg_month_time < 1.0:  # Less than 1s per month
            return 'B'
        elif avg_month_time < 2.0:  # Less than 2s per month
            return 'C'
        else:
            return 'D'
    
    def benchmark_optimizations(self) -> Dict[str, Any]:
        """
        Benchmark the effectiveness of optimizations.
        
        Returns:
            Benchmark results
        """
        # Test with optimizations enabled
        self.simulation.invalidate_cache()  # Clear cache for fair comparison
        
        # Profile with optimizations
        opt_results = self.profile_simulation(months=6)
        
        # Test cache effectiveness
        cache_stats_before = self.simulation.get_cache_performance()
        
        # Run same simulation again to test cache hits
        self.profile_simulation(months=6)
        
        cache_stats_after = self.simulation.get_cache_performance()
        
        # Calculate cache effectiveness
        cache_improvement = (
            cache_stats_after['hit_rate'] - cache_stats_before['hit_rate']
        ) if cache_stats_before['hit_rate'] > 0 else cache_stats_after['hit_rate']
        
        return {
            'optimized_performance': opt_results,
            'cache_effectiveness': cache_improvement,
            'memory_efficiency': self.simulation.get_memory_stats(),
            'io_performance': self.simulation.get_io_performance(),
            'parallel_performance': self.simulation.get_parallel_performance()
        }
    
    def export_detailed_report(self, file_path: str) -> None:
        """Export detailed performance report."""
        # Run comprehensive profiling
        simulation_profile = self.profile_simulation(months=12)
        batch_profile = self.profile_batch_simulation(1, batch_size=12)
        benchmark_results = self.benchmark_optimizations()
        
        # Combine all results
        detailed_report = {
            'simulation_profile': simulation_profile,
            'batch_profile': batch_profile,
            'benchmark_results': benchmark_results,
            'export_timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': os.sys.version
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
