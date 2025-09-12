"""
Parallel Processing Module

Implements parallel processing capabilities for the cybernetic planning simulation.
This module provides parallel execution for independent calculations, sector processing,
and batch operations to utilize multiple CPU cores effectively.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import time
import warnings
from dataclasses import dataclass
import queue
import psutil


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: int = 0  # 0 = auto-detect
    use_threading: bool = False  # True for I/O bound, False for CPU bound
    chunk_size: int = 1
    timeout: Optional[float] = None
    memory_limit_mb: int = 1000


class ParallelProcessor:
    """
    Main parallel processing manager for simulation computations.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel processor.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelConfig()
        
        # Auto-detect optimal number of workers
        if self.config.max_workers == 0:
            self.config.max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead
        
        # Performance monitoring
        self.execution_times = []
        self.parallel_efficiency = []
        
    def parallel_sector_calculations(
        self, 
        sectors_data: List[Dict[str, Any]], 
        calculation_func: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Process sector calculations in parallel.
        
        Args:
            sectors_data: List of sector data dictionaries
            calculation_func: Function to apply to each sector
            **kwargs: Additional arguments for calculation_func
            
        Returns:
            List of results for each sector
        """
        if len(sectors_data) == 0:
            return []
        
        # Use threading for I/O bound operations, processes for CPU bound
        executor_class = ThreadPoolExecutor if self.config.use_threading else ProcessPoolExecutor
        
        start_time = time.time()
        
        try:
            with executor_class(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_sector = {
                    executor.submit(calculation_func, sector_data, **kwargs): i
                    for i, sector_data in enumerate(sectors_data)
                }
                
                # Collect results in order
                results = [None] * len(sectors_data)
                
                for future in as_completed(future_to_sector, timeout=self.config.timeout):
                    sector_index = future_to_sector[future]
                    try:
                        results[sector_index] = future.result()
                    except Exception as e:
                        warnings.warn(f"Sector {sector_index} calculation failed: {e}")
                        results[sector_index] = None
                
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                
                # Calculate parallel efficiency
                sequential_time = execution_time * self.config.max_workers
                parallel_efficiency = (sequential_time / execution_time) / self.config.max_workers
                self.parallel_efficiency.append(parallel_efficiency)
                
                return results
                
        except Exception as e:
            warnings.warn(f"Parallel processing failed: {e}, falling back to sequential")
            return [calculation_func(sector_data, **kwargs) for sector_data in sectors_data]
    
    def parallel_matrix_operations(
        self, 
        matrices: List[np.ndarray], 
        operation_func: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Process matrix operations in parallel.
        
        Args:
            matrices: List of matrices to process
            operation_func: Function to apply to each matrix
            **kwargs: Additional arguments for operation_func
            
        Returns:
            List of results for each matrix
        """
        if len(matrices) == 0:
            return []
        
        # For CPU-intensive matrix operations, use processes
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_matrix = {
                    executor.submit(operation_func, matrix, **kwargs): i
                    for i, matrix in enumerate(matrices)
                }
                
                # Collect results in order
                results = [None] * len(matrices)
                
                for future in as_completed(future_to_matrix, timeout=self.config.timeout):
                    matrix_index = future_to_matrix[future]
                    try:
                        results[matrix_index] = future.result()
                    except Exception as e:
                        warnings.warn(f"Matrix {matrix_index} operation failed: {e}")
                        results[matrix_index] = None
                
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                
                return results
                
        except Exception as e:
            warnings.warn(f"Parallel matrix operations failed: {e}, falling back to sequential")
            return [operation_func(matrix, **kwargs) for matrix in matrices]
    
    def parallel_batch_simulation(
        self, 
        simulation_params: List[Dict[str, Any]], 
        simulation_func: Callable
    ) -> List[Any]:
        """
        Run multiple simulations in parallel.
        
        Args:
            simulation_params: List of parameter dictionaries for each simulation
            simulation_func: Function to run each simulation
            
        Returns:
            List of simulation results
        """
        if len(simulation_params) == 0:
            return []
        
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all simulations
                future_to_params = {
                    executor.submit(simulation_func, params): i
                    for i, params in enumerate(simulation_params)
                }
                
                # Collect results in order
                results = [None] * len(simulation_params)
                
                for future in as_completed(future_to_params, timeout=self.config.timeout):
                    param_index = future_to_params[future]
                    try:
                        results[param_index] = future.result()
                    except Exception as e:
                        warnings.warn(f"Simulation {param_index} failed: {e}")
                        results[param_index] = None
                
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                
                return results
                
        except Exception as e:
            warnings.warn(f"Parallel batch simulation failed: {e}, falling back to sequential")
            return [simulation_func(params) for params in simulation_params]
    
    def parallel_optimization(
        self, 
        optimization_problems: List[Dict[str, Any]], 
        solver_func: Callable
    ) -> List[Any]:
        """
        Solve multiple optimization problems in parallel.
        
        Args:
            optimization_problems: List of optimization problem dictionaries
            solver_func: Function to solve each optimization problem
            
        Returns:
            List of optimization results
        """
        if len(optimization_problems) == 0:
            return []
        
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all optimization problems
                future_to_problem = {
                    executor.submit(solver_func, problem): i
                    for i, problem in enumerate(optimization_problems)
                }
                
                # Collect results in order
                results = [None] * len(optimization_problems)
                
                for future in as_completed(future_to_problem, timeout=self.config.timeout):
                    problem_index = future_to_problem[future]
                    try:
                        results[problem_index] = future.result()
                    except Exception as e:
                        warnings.warn(f"Optimization {problem_index} failed: {e}")
                        results[problem_index] = None
                
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                
                return results
                
        except Exception as e:
            warnings.warn(f"Parallel optimization failed: {e}, falling back to sequential")
            return [solver_func(problem) for problem in optimization_problems]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get parallel processing performance statistics."""
        if not self.execution_times:
            return {
                'average_execution_time': 0.0,
                'total_executions': 0,
                'average_parallel_efficiency': 0.0,
                'max_workers': self.config.max_workers
            }
        
        return {
            'average_execution_time': np.mean(self.execution_times),
            'total_executions': len(self.execution_times),
            'average_parallel_efficiency': np.mean(self.parallel_efficiency) if self.parallel_efficiency else 0.0,
            'max_workers': self.config.max_workers,
            'cpu_count': mp.cpu_count(),
            'memory_usage_percent': psutil.virtual_memory().percent
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.execution_times.clear()
        self.parallel_efficiency.clear()


class SectorParallelProcessor:
    """
    Specialized parallel processor for sector-based calculations.
    """
    
    def __init__(self, parallel_processor: ParallelProcessor):
        """Initialize with reference to main parallel processor."""
        self.parallel_processor = parallel_processor
    
    def calculate_sector_efficiency(
        self, 
        sectors: List[Dict[str, Any]], 
        technology_level: float
    ) -> List[Dict[str, Any]]:
        """
        Calculate efficiency for multiple sectors in parallel.
        
        Args:
            sectors: List of sector data
            technology_level: Current technology level
            
        Returns:
            List of sector efficiency results
        """
        def calculate_single_sector_efficiency(sector_data):
            sector_name = sector_data.get('name', 'Unknown')
            base_efficiency = sector_data.get('base_efficiency', 1.0)
            
            # Technology level affects different sectors differently
            tech_multiplier = 1.0 + technology_level * 0.5
            
            high_tech_sectors = ['Technology', 'Healthcare', 'Education', 'Research']
            tech_sectors = ['Technology', 'Manufacturing', 'Energy', 'Healthcare', 'Education']
            
            if sector_name in high_tech_sectors:
                efficiency = tech_multiplier * 1.2
            elif sector_name in tech_sectors:
                efficiency = tech_multiplier * 1.1
            else:
                efficiency = tech_multiplier * 1.05
            
            return {
                'sector_name': sector_name,
                'efficiency': efficiency,
                'base_efficiency': base_efficiency,
                'technology_multiplier': tech_multiplier
            }
        
        return self.parallel_processor.parallel_sector_calculations(
            sectors, calculate_single_sector_efficiency
        )
    
    def calculate_sector_production(
        self, 
        sectors: List[Dict[str, Any]], 
        demand_vector: np.ndarray,
        technology_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Calculate production for multiple sectors in parallel.
        
        Args:
            sectors: List of sector data
            demand_vector: Final demand vector
            technology_matrix: Technology matrix
            
        Returns:
            List of sector production results
        """
        def calculate_single_sector_production(sector_data):
            sector_index = sector_data.get('index', 0)
            sector_name = sector_data.get('name', 'Unknown')
            
            # Calculate production using Leontief model
            # This is a simplified calculation for parallel processing
            demand = demand_vector[sector_index] if sector_index < len(demand_vector) else 0.0
            
            # Calculate intermediate demand
            intermediate_demand = np.sum(technology_matrix[:, sector_index] * demand_vector)
            
            total_production = demand + intermediate_demand
            
            return {
                'sector_name': sector_name,
                'sector_index': sector_index,
                'final_demand': demand,
                'intermediate_demand': intermediate_demand,
                'total_production': total_production,
                'efficiency': min(1.0, total_production / max(1e-10, demand))
            }
        
        return self.parallel_processor.parallel_sector_calculations(
            sectors, calculate_single_sector_production
        )


class MatrixParallelProcessor:
    """
    Specialized parallel processor for matrix operations.
    """
    
    def __init__(self, parallel_processor: ParallelProcessor):
        """Initialize with reference to main parallel processor."""
        self.parallel_processor = parallel_processor
    
    def parallel_matrix_inversion(
        self, 
        matrices: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Compute matrix inversions in parallel.
        
        Args:
            matrices: List of matrices to invert
            
        Returns:
            List of inverted matrices
        """
        def invert_matrix(matrix):
            try:
                return np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                # Return pseudo-inverse if matrix is singular
                return np.linalg.pinv(matrix)
        
        return self.parallel_processor.parallel_matrix_operations(
            matrices, invert_matrix
        )
    
    def parallel_matrix_multiplication(
        self, 
        matrix_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Compute matrix multiplications in parallel.
        
        Args:
            matrix_pairs: List of (matrix1, matrix2) tuples to multiply
            
        Returns:
            List of resulting matrices
        """
        def multiply_matrices(matrix_pair):
            matrix1, matrix2 = matrix_pair
            return matrix1 @ matrix2
        
        # Convert pairs to single matrices for parallel processing
        matrices = [matrix_pairs]
        
        return self.parallel_processor.parallel_matrix_operations(
            matrices, multiply_matrices
        )
    
    def parallel_eigenvalue_calculation(
        self, 
        matrices: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Calculate eigenvalues and eigenvectors in parallel.
        
        Args:
            matrices: List of matrices for eigenvalue calculation
            
        Returns:
            List of eigenvalue/eigenvector results
        """
        def calculate_eigenvalues(matrix):
            try:
                eigenvals, eigenvecs = np.linalg.eig(matrix)
                return {
                    'eigenvalues': eigenvals,
                    'eigenvectors': eigenvecs,
                    'spectral_radius': np.max(np.abs(eigenvals))
                }
            except np.linalg.LinAlgError as e:
                return {
                    'eigenvalues': None,
                    'eigenvectors': None,
                    'spectral_radius': None,
                    'error': str(e)
                }
        
        return self.parallel_processor.parallel_matrix_operations(
            matrices, calculate_eigenvalues
        )
