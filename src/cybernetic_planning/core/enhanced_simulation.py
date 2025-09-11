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
        self.A = np.asarray(technology_matrix)
        self.l = np.asarray(labor_vector).flatten()
        self.d = np.asarray(final_demand).flatten()
        self.R = resource_matrix
        self.R_max = max_resources
        self.sector_names = sector_names or [f"Sector_{i+1}" for i in range(len(self.l))]
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize components
        self.leontief_model = LeontiefModel(self.A, self.d)
        self.labor_calculator = LaborValueCalculator(self.A, self.l)
        
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
        # Calculate technology-driven efficiency improvements
        tech_efficiency = self._calculate_technology_efficiency()
        
        # Update technology matrix based on current technology level
        A_modified = self._apply_technology_improvements(self.A, tech_efficiency)
        
        # Update labor vector based on technology and living standards
        l_modified = self._apply_labor_improvements(self.l, tech_efficiency)
        
        # Calculate living standards-optimized final demand
        d_modified = self._optimize_final_demand_for_living_standards(self.d)
        
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
        n = len(self.sector_names)
        efficiency = np.ones(n)
        
        # Technology level affects all sectors differently
        tech_multiplier = 1.0 + self.current_technology_level * 0.5  # Up to 50% improvement
        
        # Different sectors benefit differently from technology
        tech_sectors = ['Technology', 'Manufacturing', 'Energy', 'Healthcare', 'Education']
        high_tech_sectors = ['Technology', 'Healthcare', 'Education', 'Research']
        
        for i, sector in enumerate(self.sector_names):
            if sector in high_tech_sectors:
                efficiency[i] = tech_multiplier * 1.2  # High-tech sectors benefit more
            elif sector in tech_sectors:
                efficiency[i] = tech_multiplier * 1.1  # Medium-tech sectors benefit moderately
            else:
                efficiency[i] = tech_multiplier * 1.05  # Low-tech sectors benefit less
        
        return efficiency
    
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
        
        return {
            'total_economic_output': latest_metrics.total_economic_output,
            'average_efficiency': latest_metrics.average_efficiency,
            'living_standards_index': latest_metrics.living_standards_index,
            'technology_level': latest_metrics.technology_level,
            'labor_productivity': latest_metrics.labor_productivity,
            'resource_utilization': latest_metrics.resource_utilization,
            'consumer_demand_fulfillment': latest_metrics.consumer_demand_fulfillment,
            'simulation_months': len(self.simulation_history)
        }
