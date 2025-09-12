"""
Enhanced Economic Simulation System

This module provides an improved simulation system that uses proper economic optimization
instead of random factors, with focus on living standards and labor-time optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import warnings

from .leontief import LeontiefModel
from .optimization import ConstrainedOptimizer
from .labor_values import LaborValueCalculator
from .simulation_cache import SimulationCache, MatrixOperationCache, LaborValueCache, TechnologyEfficiencyCache
from .memory_optimizer import SimulationMemoryManager
from .parallel_processor import ParallelProcessor, SectorParallelProcessor, MatrixParallelProcessor, ParallelConfig
from .io_optimizer import SimulationIO, IOConfig
from .performance_monitor import PerformanceMonitor, SimulationProfiler
from ..utils.population_health_tracker import PopulationHealthTracker


@dataclass
class SimulationMetrics:
    """Metrics for simulation performance."""
    total_economic_output: float = 0.0
    average_efficiency: float = 0.0
    living_standards_index: float = 0.0
    technology_level: float = 0.0
    labor_productivity: float = 0.0
    resource_utilization: float = 0.0
    consumer_demand_fulfillment: float = 0.0


class EnhancedEconomicSimulation:
    """
    Enhanced economic simulation that uses proper optimization algorithms.
    
    Features:
    - Labor-time optimization using linear programming
    - Living standards maximization
    - Technology-driven productivity improvements
    - Resource constraint handling
    - Dynamic sector efficiency calculation
    """
    
    def __init__(
        self,
        technology_matrix: np.ndarray,
        labor_vector: np.ndarray,
        final_demand: np.ndarray,
        resource_matrix: Optional[np.ndarray] = None,
        max_resources: Optional[np.ndarray] = None,
        sector_names: Optional[List[str]] = None
    ):
        """
        Initialize the enhanced simulation.
        
        Args:
            technology_matrix: Input-output technology matrix A
            labor_vector: Direct labor input vector l
            final_demand: Final demand vector d
            resource_matrix: Resource constraint matrix R (optional)
            max_resources: Maximum resource availability R_max (optional)
            sector_names: Names of economic sectors
        """
        print(f"DEBUG: EnhancedEconomicSimulation.__init__ called")
        print(f"DEBUG: technology_matrix type: {type(technology_matrix)}, shape: {getattr(technology_matrix, 'shape', 'No shape')}")
        print(f"DEBUG: labor_vector type: {type(labor_vector)}, shape: {getattr(labor_vector, 'shape', 'No shape')}")
        print(f"DEBUG: final_demand type: {type(final_demand)}, shape: {getattr(final_demand, 'shape', 'No shape')}")
        print(f"DEBUG: sector_names type: {type(sector_names)}, value: {sector_names}")
        
        self.A = np.asarray(technology_matrix)
        print(f"DEBUG: self.A created, shape: {self.A.shape}")
        
        self.l = np.asarray(labor_vector).flatten()
        print(f"DEBUG: self.l created, shape: {self.l.shape}")
        
        self.d = np.asarray(final_demand).flatten()
        print(f"DEBUG: self.d created, shape: {self.d.shape}")
        
        self.R = resource_matrix
        self.R_max = max_resources
        
        print(f"DEBUG: About to set sector_names, self.l length: {len(self.l)}")
        if sector_names is not None:
            self.sector_names = sector_names
        else:
            self.sector_names = [f"Sector_{i+1}" for i in range(len(self.l))]
        print(f"DEBUG: sector_names set: {self.sector_names[:5]}...")
        
        # Validate inputs
        print(f"DEBUG: About to validate inputs")
        self._validate_inputs()
        print(f"DEBUG: Input validation completed")
        
        # Initialize components
        self.leontief_model = LeontiefModel(self.A, self.d)
        self.labor_calculator = LaborValueCalculator(self.A, self.l)
        
        # Initialize caching system
        self.cache = SimulationCache(max_cache_size=2000, max_memory_mb=1000)
        self.matrix_cache = MatrixOperationCache(self.cache)
        self.labor_cache = LaborValueCache(self.cache)
        self.tech_efficiency_cache = TechnologyEfficiencyCache(self.cache)
        
        # Initialize memory management
        self.memory_manager = SimulationMemoryManager(max_history_months=120, max_memory_mb=1000)
        
        # Initialize parallel processing
        parallel_config = ParallelConfig(max_workers=0, use_threading=False)
        self.parallel_processor = ParallelProcessor(parallel_config)
        self.sector_processor = SectorParallelProcessor(self.parallel_processor)
        self.matrix_processor = MatrixParallelProcessor(self.parallel_processor)
        
        # Initialize I/O optimization
        io_config = IOConfig(use_compression=True, use_binary_format=True, background_saving=True)
        self.io_manager = SimulationIO(io_config)
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.profiler = SimulationProfiler(self)
        
        # Simulation state
        self.current_technology_level = 0.0
        self.current_living_standards = 0.5
        self.simulation_history: List[SimulationMetrics] = []
        
        # Optimization parameters
        self.optimization_weights = {
            'labor_efficiency': 0.4,
            'living_standards': 0.3,
            'resource_utilization': 0.2,
            'technology_growth': 0.1
        }
    
    def _validate_inputs(self) -> None:
        """Validate input matrices and vectors."""
        n = self.A.shape[0]
        
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Technology matrix must be square")
        
        if self.l.shape[0] != n or self.d.shape[0] != n:
            raise ValueError("All vectors must have same dimension as technology matrix")
        
        if self.R is not None and self.R_max is not None:
            if self.R.shape[1] != n:
                raise ValueError("Resource matrix must have same number of columns as technology matrix")
            if self.R.shape[0] != self.R_max.shape[0]:
                raise ValueError("Resource matrix and max resources must have compatible dimensions")
    
    def simulate_month(
        self,
        month: int,
        population_health_tracker: Optional[PopulationHealthTracker] = None,
        use_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Simulate one month of economic activity.
        
        Args:
            month: Current month number
            population_health_tracker: Population health tracker for living standards
            use_optimization: Whether to use optimization (True) or simple Leontief (False)
        
        Returns:
            Dictionary with simulation results
        """
        # Get technology and living standards from tracker if available
        if population_health_tracker:
            tech_level = population_health_tracker.current_technology_level
            living_standards = population_health_tracker.current_living_standards
        else:
            tech_level = self.current_technology_level
            living_standards = self.current_living_standards
        
        # Use cached matrix operations
        A_modified, l_modified, d_modified = self.matrix_cache.get_modified_matrices(
            self.A, self.l, self.d, tech_level, living_standards
        )
        
        if use_optimization:
            # Use constrained optimization for labor-time minimization
            results = self._run_optimization(A_modified, l_modified, d_modified)
        else:
            # Use simple Leontief model
            results = self._run_leontief_simulation(A_modified, l_modified, d_modified)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(results, population_health_tracker)
        
        # Store results
        self.simulation_history.append(metrics)
        
        # Store in memory-efficient buffers
        self.memory_manager.store_simulation_data(month, {
            'total_economic_output': metrics.total_economic_output,
            'average_efficiency': metrics.average_efficiency,
            'labor_productivity': metrics.labor_productivity
        })
        
        return {
            'month': month,
            'production': results['production'],
            'labor_allocation': results['labor_allocation'],
            'resource_usage': results.get('resource_usage', {}),
            'metrics': metrics,
            'technology_level': self.current_technology_level,
            'living_standards': self.current_living_standards
        }
    
    def _calculate_technology_efficiency(self) -> np.ndarray:
        """Calculate technology-driven efficiency improvements for each sector."""
        return self.tech_efficiency_cache.get_technology_efficiency(
            self.current_technology_level, 
            self.sector_names
        )
    
    def _apply_technology_improvements(self, A: np.ndarray, efficiency: np.ndarray) -> np.ndarray:
        """Apply technology improvements to the technology matrix."""
        # Technology improvements reduce input requirements (more efficient production)
        A_improved = A.copy()
        
        # Apply efficiency improvements to reduce input coefficients
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                # More efficient production means less input needed per unit output
                A_improved[i, j] = A[i, j] / efficiency[j]
        
        # Ensure matrix remains economically valid
        A_improved = np.clip(A_improved, 0, 0.99)  # Prevent values >= 1
        
        return A_improved
    
    def _apply_labor_improvements(self, l: np.ndarray, efficiency: np.ndarray) -> np.ndarray:
        """Apply technology and living standards improvements to labor requirements."""
        # Technology reduces labor requirements per unit output
        l_improved = l / efficiency
        
        # Living standards improvements also reduce labor requirements
        # (better working conditions, more motivated workers)
        living_standards_factor = 1.0 + self.current_living_standards * 0.3
        l_improved = l_improved / living_standards_factor
        
        return l_improved
    
    def _optimize_final_demand_for_living_standards(self, d: np.ndarray) -> np.ndarray:
        """Optimize final demand to maximize living standards."""
        d_optimized = d.copy()
        
        # Increase demand for living standards-related sectors
        living_standards_sectors = ['Healthcare', 'Education', 'Construction', 'Agriculture']
        
        for i, sector in enumerate(self.sector_names):
            if sector in living_standards_sectors:
                # Increase demand for these sectors based on living standards goals
                multiplier = 1.0 + self.current_living_standards * 0.5
                d_optimized[i] *= multiplier
        
        return d_optimized
    
    def _run_optimization(
        self, 
        A: np.ndarray, 
        l: np.ndarray, 
        d: np.ndarray
    ) -> Dict[str, Any]:
        """Run constrained optimization for labor-time minimization."""
        try:
            optimizer = ConstrainedOptimizer(
                technology_matrix=A,
                direct_labor=l,
                final_demand=d,
                resource_matrix=self.R,
                max_resources=self.R_max
            )
            
            result = optimizer.solve()
            
            if result['feasible']:
                total_output = result['solution']
                total_labor_cost = result['total_labor_cost']
                
                # Calculate sector-specific production
                production = {}
                labor_allocation = {}
                
                for i, sector in enumerate(self.sector_names):
                    production[sector] = {
                        'target': d[i],
                        'actual': total_output[i],
                        'efficiency': min(1.0, total_output[i] / max(1e-10, d[i]))
                    }
                    labor_allocation[sector] = l[i] * total_output[i]
                
                return {
                    'production': production,
                    'labor_allocation': labor_allocation,
                    'total_labor_cost': total_labor_cost,
                    'total_output': total_output,
                    'optimization_success': True
                }
            else:
                # Fallback to Leontief if optimization fails
                return self._run_leontief_simulation(A, l, d)
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}, falling back to Leontief model")
            return self._run_leontief_simulation(A, l, d)
    
    def _run_leontief_simulation(
        self, 
        A: np.ndarray, 
        l: np.ndarray, 
        d: np.ndarray
    ) -> Dict[str, Any]:
        """Run simple Leontief simulation."""
        leontief = LeontiefModel(A, d)
        total_output = leontief.compute_total_output()
        total_labor_cost = np.dot(l, total_output)
        
        # Calculate sector-specific production
        production = {}
        labor_allocation = {}
        
        for i, sector in enumerate(self.sector_names):
            production[sector] = {
                'target': d[i],
                'actual': total_output[i],
                'efficiency': min(1.0, total_output[i] / max(1e-10, d[i]))
            }
            labor_allocation[sector] = l[i] * total_output[i]
        
        return {
            'production': production,
            'labor_allocation': labor_allocation,
            'total_labor_cost': total_labor_cost,
            'total_output': total_output,
            'optimization_success': False
        }
    
    def _calculate_comprehensive_metrics(
        self, 
        results: Dict[str, Any], 
        population_health_tracker: Optional[PopulationHealthTracker] = None
    ) -> SimulationMetrics:
        """Calculate comprehensive simulation metrics."""
        production = results['production']
        total_labor_cost = results['total_labor_cost']
        total_output = results['total_output']
        
        # Calculate total economic output
        total_economic_output = np.sum(total_output)
        
        # Calculate average efficiency
        efficiencies = [sector_data['efficiency'] for sector_data in production.values()]
        average_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        
        # Calculate living standards
        if population_health_tracker:
            living_standards = population_health_tracker.current_living_standards
            technology_level = population_health_tracker.current_technology_level
        else:
            living_standards = self.current_living_standards
            technology_level = self.current_technology_level
        
        # Calculate labor productivity (output per unit labor)
        labor_productivity = total_economic_output / max(1e-10, total_labor_cost)
        
        # Calculate resource utilization
        resource_utilization = 1.0
        if self.R is not None and self.R_max is not None:
            resource_usage = self.R @ total_output
            resource_utilization = np.mean(resource_usage / (self.R_max + 1e-10))
        
        # Calculate consumer demand fulfillment
        consumer_demand_fulfillment = average_efficiency  # Simplified for now
        
        return SimulationMetrics(
            total_economic_output=total_economic_output,
            average_efficiency=average_efficiency,
            living_standards_index=living_standards,
            technology_level=technology_level,
            labor_productivity=labor_productivity,
            resource_utilization=resource_utilization,
            consumer_demand_fulfillment=consumer_demand_fulfillment
        )
    
    def update_technology_level(self, new_level: float) -> None:
        """Update the current technology level."""
        self.current_technology_level = min(1.0, max(0.0, new_level))
    
    def update_living_standards(self, new_standards: float) -> None:
        """Update the current living standards level."""
        self.current_living_standards = min(1.0, max(0.0, new_standards))
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get a summary of the simulation performance."""
        if not self.simulation_history:
            return {}
        
        latest_metrics = self.simulation_history[-1]
        cache_stats = self.cache.get_cache_stats()
        
        return {
            'total_economic_output': latest_metrics.total_economic_output,
            'average_efficiency': latest_metrics.average_efficiency,
            'living_standards_index': latest_metrics.living_standards_index,
            'technology_level': latest_metrics.technology_level,
            'labor_productivity': latest_metrics.labor_productivity,
            'resource_utilization': latest_metrics.resource_utilization,
            'consumer_demand_fulfillment': latest_metrics.consumer_demand_fulfillment,
            'simulation_months': len(self.simulation_history),
            'cache_performance': cache_stats
        }
    
    def invalidate_cache(self) -> None:
        """Invalidate all cached computations."""
        self.cache.clear()
    
    def invalidate_technology_cache(self) -> None:
        """Invalidate technology-related cache entries."""
        self.cache.invalidate_tech_efficiency_cache()
        self.cache.invalidate_matrix_cache()
    
    def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return self.cache.get_cache_stats()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage and return statistics."""
        return self.memory_manager.optimize_memory_usage()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = self.memory_manager.get_memory_stats()
        return {
            'total_memory_mb': stats.total_memory_mb,
            'available_memory_mb': stats.available_memory_mb,
            'memory_usage_percent': stats.memory_usage_percent,
            'simulation_memory_mb': stats.simulation_memory_mb,
            'cache_memory_mb': stats.cache_memory_mb,
            'matrix_memory_mb': stats.matrix_memory_mb
        }
    
    def get_recent_metrics(self, months: int = 12) -> Dict[str, np.ndarray]:
        """Get recent simulation metrics from memory-efficient storage."""
        return self.memory_manager.get_recent_metrics(months)
    
    def get_parallel_performance(self) -> Dict[str, Any]:
        """Get parallel processing performance statistics."""
        return self.parallel_processor.get_performance_stats()
    
    def parallel_sector_analysis(self, sectors_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform parallel sector efficiency analysis."""
        return self.sector_processor.calculate_sector_efficiency(
            sectors_data, self.current_technology_level
        )
    
    def parallel_production_calculation(self, sectors_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate sector production in parallel."""
        return self.sector_processor.calculate_sector_production(
            sectors_data, self.d, self.A
        )
    
    def save_simulation_state(self, file_path: str, use_background: bool = True) -> Dict[str, Any]:
        """Save complete simulation state with I/O optimization."""
        simulation_data = {
            'technology_level': self.current_technology_level,
            'living_standards': self.current_living_standards,
            'simulation_history': [asdict(metric) for metric in self.simulation_history],
            'cache_stats': self.cache.get_cache_stats(),
            'memory_stats': self.get_memory_stats(),
            'parallel_stats': self.get_parallel_performance()
        }
        return self.io_manager.save_simulation_state(file_path, simulation_data, use_background)
    
    def load_simulation_state(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load complete simulation state with I/O optimization."""
        return self.io_manager.load_simulation_state(file_path)
    
    def save_simulation_results(self, file_path: str, use_background: bool = True) -> Dict[str, Any]:
        """Save simulation results with I/O optimization."""
        results = []
        for i, metrics in enumerate(self.simulation_history):
            results.append({
                'month': i + 1,
                'metrics': asdict(metrics),
                'technology_level': self.current_technology_level,
                'living_standards': self.current_living_standards
            })
        return self.io_manager.save_simulation_results(file_path, results, use_background)
    
    def get_io_performance(self) -> Dict[str, Any]:
        """Get I/O performance statistics."""
        return self.io_manager.get_io_performance()
    
    def start_performance_monitoring(self) -> None:
        """Start performance monitoring."""
        self.performance_monitor.start_monitoring()
    
    def stop_performance_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.performance_monitor.stop_monitoring()
    
    def get_performance_profile(self) -> Dict[str, Any]:
        """Get comprehensive performance profile."""
        return self.profiler.profile_simulation(months=12)
    
    def benchmark_optimizations(self) -> Dict[str, Any]:
        """Benchmark optimization effectiveness."""
        return self.profiler.benchmark_optimizations()
    
    def export_performance_report(self, file_path: str) -> None:
        """Export detailed performance report."""
        self.profiler.export_detailed_report(file_path)
    
    def simulate_batch_months(
        self, 
        start_month: int, 
        batch_size: int = 12,
        population_health_tracker: Optional[PopulationHealthTracker] = None,
        use_optimization: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple months in a single batch for better performance.
        
        Args:
            start_month: Starting month number
            batch_size: Number of months to process in batch
            population_health_tracker: Population health tracker for living standards
            use_optimization: Whether to use optimization (True) or simple Leontief (False)
        
        Returns:
            List of simulation results for each month in the batch
        """
        results = []
        
        # Pre-allocate arrays for better performance
        tech_levels = np.zeros(batch_size)
        living_standards_levels = np.zeros(batch_size)
        
        # Get initial values
        if population_health_tracker:
            initial_tech = population_health_tracker.current_technology_level
            initial_living = population_health_tracker.current_living_standards
        else:
            initial_tech = self.current_technology_level
            initial_living = self.current_living_standards
        
        # Process batch without individual UI updates
        for i in range(batch_size):
            month = start_month + i
            
            # Update technology and living standards (simplified for batch processing)
            if population_health_tracker:
                tech_levels[i] = population_health_tracker.current_technology_level
                living_standards_levels[i] = population_health_tracker.current_living_standards
            else:
                tech_levels[i] = self.current_technology_level
                living_standards_levels[i] = self.current_living_standards
            
            # Use cached matrix operations for the batch
            A_modified, l_modified, d_modified = self.matrix_cache.get_modified_matrices(
                self.A, self.l, self.d, tech_levels[i], living_standards_levels[i]
            )
            
            if use_optimization:
                # Use constrained optimization for labor-time minimization
                month_results = self._run_optimization(A_modified, l_modified, d_modified)
            else:
                # Use simple Leontief model
                month_results = self._run_leontief_simulation(A_modified, l_modified, d_modified)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(month_results, population_health_tracker)
            
            # Store results
            self.simulation_history.append(metrics)
            
            results.append({
                'month': month,
                'production': month_results['production'],
                'labor_allocation': month_results['labor_allocation'],
                'resource_usage': month_results.get('resource_usage', {}),
                'metrics': metrics,
                'technology_level': tech_levels[i],
                'living_standards': living_standards_levels[i]
            })
        
        return results
