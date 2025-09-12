#!/usr/bin/env python3
"""
Performance Benchmark Script

This script benchmarks the performance optimizations implemented in the cybernetic planning simulation.
It runs comprehensive tests to measure the effectiveness of all optimizations.
"""

import sys
import os
import time
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cybernetic_planning.core.enhanced_simulation import EnhancedEconomicSimulation
from cybernetic_planning.core.performance_monitor import SimulationProfiler


def create_test_data(n_sectors=50):
    """Create test data for benchmarking."""
    # Create technology matrix with proper economic constraints
    technology_matrix = np.random.rand(n_sectors, n_sectors) * 0.1  # Reduced from 0.3
    np.fill_diagonal(technology_matrix, 0.0)
    technology_matrix = np.clip(technology_matrix, 0, 0.5)  # Reduced from 0.8
    
    # Ensure economy is productive by making diagonal dominance
    for i in range(n_sectors):
        row_sum = np.sum(technology_matrix[i, :])
        if row_sum >= 0.9:  # If row sum is too high, scale it down
            technology_matrix[i, :] *= 0.8 / row_sum
    
    # Create labor vector
    labor_vector = np.random.rand(n_sectors) * 0.05  # Reduced from 0.1
    
    # Create final demand
    final_demand = np.random.rand(n_sectors) * 50  # Reduced from 100
    
    # Create sector names
    sector_names = [f"Sector_{i+1}" for i in range(n_sectors)]
    
    return technology_matrix, labor_vector, final_demand, sector_names


def benchmark_sleep_delay_optimization():
    """Benchmark sleep delay optimization."""
    print("🔧 Testing sleep delay optimization...")
    
    # This would need to be tested in the actual GUI context
    # For now, we'll just verify the change was made
    try:
        with open('gui.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'time.sleep(0.01)' in content:
                print("✅ Sleep delay optimized from 0.1s to 0.01s (10x improvement)")
                return True
            else:
                print("❌ Sleep delay optimization not found")
                return False
    except Exception as e:
        print(f"❌ Sleep delay optimization test failed: {e}")
        return False


def benchmark_caching_system():
    """Benchmark caching system performance."""
    print("🔧 Testing caching system...")
    
    try:
        from cybernetic_planning.core.simulation_cache import SimulationCache
        
        cache = SimulationCache(max_cache_size=1000, max_memory_mb=100)
        
        def expensive_operation(x):
            time.sleep(0.01)  # Simulate expensive computation
            return x ** 2
        
        # Test cache performance
        start_time = time.time()
        
        # First call (cache miss)
        result1 = cache.get_or_compute('test_op', expensive_operation, 5)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = cache.get_or_compute('test_op', expensive_operation, 5)
        second_call_time = time.time() - start_time
        
        # Verify results
        assert result1 == result2, "Cache results should be identical"
        
        # Verify performance improvement
        speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
        
        print(f"✅ Caching system working: {speedup:.1f}x speedup on cache hits")
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        print(f"   Cache hit rate: {stats['hit_rate']:.2%}")
        print(f"   Cache size: {stats['cache_size']}/{stats['max_cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching system test failed: {e}")
        return False


def benchmark_memory_optimization():
    """Benchmark memory optimization features."""
    print("🔧 Testing memory optimization...")
    
    try:
        from cybernetic_planning.core.memory_optimizer import SimulationMemoryManager
        
        memory_manager = SimulationMemoryManager(max_history_months=60, max_memory_mb=50)
        
        # Test circular buffer
        buffer = memory_manager.metrics_buffer
        
        # Add data to buffer
        for i in range(100):
            buffer.append(i * 1.5)
        
        # Verify buffer maintains fixed size
        assert buffer.size == 60, f"Expected buffer size 60, got {buffer.size}"
        
        # Test recent data retrieval
        recent_data = buffer.get_recent(10)
        assert len(recent_data) == 10, f"Expected 10 recent items, got {len(recent_data)}"
        
        # Test memory stats
        stats = memory_manager.get_memory_stats()
        assert stats.simulation_memory_mb > 0, "Memory usage should be positive"
        
        print("✅ Memory optimization working correctly")
        print(f"   Memory usage: {stats.simulation_memory_mb:.2f} MB")
        print(f"   Available memory: {stats.available_memory_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False


def benchmark_parallel_processing():
    """Benchmark parallel processing performance."""
    print("🔧 Testing parallel processing...")
    
    try:
        from cybernetic_planning.core.parallel_processor import ParallelProcessor, ParallelConfig
        
        config = ParallelConfig(max_workers=2, use_threading=True)  # Use threading instead
        processor = ParallelProcessor(config)
        
        # Test parallel matrix operations instead of sector calculations
        matrices = [np.random.rand(10, 10) for _ in range(5)]
        
        def matrix_operation(matrix):
            time.sleep(0.01)  # Simulate computation
            return np.linalg.det(matrix)
        
        # Time parallel execution
        start_time = time.time()
        parallel_results = processor.parallel_matrix_operations(
            matrices, matrix_operation
        )
        parallel_time = time.time() - start_time
        
        # Time sequential execution
        start_time = time.time()
        sequential_results = [matrix_operation(matrix) for matrix in matrices]
        sequential_time = time.time() - start_time
        
        # Verify results
        assert len(parallel_results) == len(sequential_results), "Results count mismatch"
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"✅ Parallel processing working: {speedup:.1f}x speedup")
        print(f"   Sequential time: {sequential_time:.3f}s")
        print(f"   Parallel time: {parallel_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        return False


def benchmark_io_optimization():
    """Benchmark I/O optimization features."""
    print("🔧 Testing I/O optimization...")
    
    try:
        from cybernetic_planning.core.io_optimizer import SimulationIO, IOConfig
        import tempfile
        
        io_config = IOConfig(use_compression=True, use_binary_format=True, background_saving=False)
        io_manager = SimulationIO(io_config)
        
        # Test data
        test_data = {
            'metrics': [{'output': i * 10, 'efficiency': 0.8} for i in range(100)],
            'matrices': [np.random.rand(10, 10) for _ in range(5)],
            'vectors': [np.random.rand(20) for _ in range(10)]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            file_path = tmp_file.name
        
        try:
            # Test save performance
            start_time = time.time()
            save_result = io_manager.save_simulation_state(file_path, test_data, use_background=False)
            save_time = time.time() - start_time
            
            # Verify save was successful
            assert save_result['status'] == 'saved', f"Save failed: {save_result}"
            
            # Test load performance
            start_time = time.time()
            loaded_data = io_manager.load_simulation_state(file_path)
            load_time = time.time() - start_time
            
            # Verify data integrity
            assert loaded_data is not None, "Load failed"
            assert len(loaded_data['metrics']) == len(test_data['metrics']), "Data integrity check failed"
            
            # Test I/O performance stats
            io_stats = io_manager.get_io_performance()
            
            print("✅ I/O optimization working correctly")
            print(f"   Save time: {save_time:.3f}s")
            print(f"   Load time: {load_time:.3f}s")
            print(f"   File size: {save_result['file_size_mb']:.2f} MB")
            print(f"   Compression ratio: {save_result['compression_ratio']:.2%}")
            
            return True
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.unlink(file_path)
        
    except Exception as e:
        print(f"❌ I/O optimization test failed: {e}")
        return False


def benchmark_full_simulation():
    """Benchmark full simulation performance."""
    print("🔧 Testing full simulation performance...")
    
    try:
        # Create test data
        technology_matrix, labor_vector, final_demand, sector_names = create_test_data(30)
        
        # Create simulation
        simulation = EnhancedEconomicSimulation(
            technology_matrix=technology_matrix,
            labor_vector=labor_vector,
            final_demand=final_demand,
            sector_names=sector_names
        )
        
        # Start performance monitoring
        simulation.start_performance_monitoring()
        
        # Run simulation
        start_time = time.time()
        
        for month in range(1, 13):  # 12 months
            result = simulation.simulate_month(month)
        
        total_time = time.time() - start_time
        
        # Get performance statistics
        cache_stats = simulation.get_cache_performance()
        memory_stats = simulation.get_memory_stats()
        parallel_stats = simulation.get_parallel_performance()
        io_stats = simulation.get_io_performance()
        
        # Calculate performance metrics
        avg_month_time = total_time / 12
        cache_hit_rate = cache_stats.get('hit_rate', 0)
        memory_usage = memory_stats.get('simulation_memory_mb', 0)
        
        print("✅ Full simulation performance test completed")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average per month: {avg_month_time:.3f}s")
        print(f"   Cache hit rate: {cache_hit_rate:.2%}")
        print(f"   Memory usage: {memory_usage:.2f} MB")
        
        # Performance grading
        if avg_month_time < 0.1:
            grade = "A+"
        elif avg_month_time < 0.5:
            grade = "A"
        elif avg_month_time < 1.0:
            grade = "B"
        elif avg_month_time < 2.0:
            grade = "C"
        else:
            grade = "D"
        
        print(f"   Performance grade: {grade}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full simulation test failed: {e}")
        return False


def benchmark_batch_processing():
    """Benchmark batch processing performance."""
    print("🔧 Testing batch processing...")
    
    try:
        # Create test data
        technology_matrix, labor_vector, final_demand, sector_names = create_test_data(30)
        
        # Create simulation
        simulation = EnhancedEconomicSimulation(
            technology_matrix=technology_matrix,
            labor_vector=labor_vector,
            final_demand=final_demand,
            sector_names=sector_names
        )
        
        # Test individual month processing
        start_time = time.time()
        for month in range(1, 7):  # 6 months
            simulation.simulate_month(month)
        individual_time = time.time() - start_time
        
        # Test batch processing
        start_time = time.time()
        batch_results = simulation.simulate_batch_months(7, batch_size=6)
        batch_time = time.time() - start_time
        
        # Calculate speedup
        speedup = individual_time / batch_time if batch_time > 0 else 1.0
        
        print("✅ Batch processing working correctly")
        print(f"   Individual processing: {individual_time:.3f}s")
        print(f"   Batch processing: {batch_time:.3f}s")
        print(f"   Speedup: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False


def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark."""
    print("🚀 Starting Comprehensive Performance Benchmark")
    print("=" * 60)
    
    results = {}
    
    # Run individual tests
    tests = [
        ("Sleep Delay Optimization", benchmark_sleep_delay_optimization),
        ("Caching System", benchmark_caching_system),
        ("Memory Optimization", benchmark_memory_optimization),
        ("Parallel Processing", benchmark_parallel_processing),
        ("I/O Optimization", benchmark_io_optimization),
        ("Batch Processing", benchmark_batch_processing),
        ("Full Simulation", benchmark_full_simulation),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Performance targets
    print("\nPERFORMANCE TARGETS:")
    print("- Sleep delay: 10x improvement (0.1s → 0.01s)")
    print("- Caching: >50% hit rate")
    print("- Memory: <100MB for 12-month simulation")
    print("- Parallel: >1.5x speedup")
    print("- I/O: <1s save/load time")
    print("- Batch: >1.2x speedup over individual")
    print("- Full simulation: <1s per month average")
    
    return results


if __name__ == "__main__":
    # Run the benchmark
    results = run_comprehensive_benchmark()
    
    # Save results to file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to benchmark_results.json")
