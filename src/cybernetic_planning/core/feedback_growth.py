"""
Feedback - Driven Growth System

Implements dynamic growth rates based on industry performance, demand fulfillment,
and sectoral balance. Growth rates are determined by actual economic performance
rather than fixed parameters.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class IndustryPerformance:
    """Performance metrics for a specific industry sector."""
    sector_id: int
    output: float
    demand: float
    demand_fulfillment_rate: float
    labor_efficiency: float
    technology_level: float
    growth_potential: float
    bottleneck_severity: float

@dataclass
class GrowthRates:
    """Dynamic growth rates based on economic performance."""
    population_growth: float
    living_standards_growth: float
    technology_improvement: float
    capital_accumulation: float
    sectoral_growth_rates: np.ndarray

class FeedbackGrowthSystem:
    """
    Manages feedback - driven growth based on industry performance.

    This system analyzes:
    - Demand fulfillment rates by sector - Labor efficiency improvements - Technology advancement potential - Capital accumulation needs - Sectoral bottlenecks and growth opportunities
    """

    def __init__(
        self,
        n_sectors: int,
        dept_I_indices: List[int],
        dept_II_indices: List[int],
        dept_III_indices: List[int],
        base_population_growth: float = 0.01,
        base_living_standards_growth: float = 0.02,
        base_technology_rate: float = 0.02,
        base_capital_rate: float = 0.15
    ):
        """
        Initialize the feedback growth system.

        Args:
            n_sectors: Number of economic sectors
            dept_I_indices: Indices for Department I (means of production)
            dept_II_indices: Indices for Department II (consumer goods)
            dept_III_indices: Indices for Department III (services)
            base_population_growth: Base population growth rate
            base_living_standards_growth: Base living standards growth rate
            base_technology_rate: Base technology improvement rate
            base_capital_rate: Base capital accumulation rate
        """
        self.n_sectors = n_sectors
        self.dept_I_indices = dept_I_indices
        self.dept_II_indices = dept_II_indices
        self.dept_III_indices = dept_III_indices

        # Base growth rates (used as fallbacks)
        self.base_population_growth = base_population_growth
        self.base_living_standards_growth = base_living_standards_growth
        self.base_technology_rate = base_technology_rate
        self.base_capital_rate = base_capital_rate

        # Performance history for trend analysis
        self.performance_history: List[Dict[str, Any]] = []

    def analyze_industry_performance(
        self,
        current_plan: Dict[str, Any],
        previous_plan: Optional[Dict[str, Any]] = None
    ) -> List[IndustryPerformance]:
        """
        Analyze performance of each industry sector.

        Args:
            current_plan: Current year's economic plan
            previous_plan: Previous year's plan (for trend analysis)

        Returns:
            List of industry performance metrics
        """
        total_output = current_plan.get("total_output", np.zeros(self.n_sectors))
        final_demand = current_plan.get("final_demand", np.zeros(self.n_sectors))
        labor_vector = current_plan.get("labor_vector", np.zeros(self.n_sectors))
        technology_matrix = current_plan.get("technology_matrix", np.eye(self.n_sectors))

        performance_metrics = []

        for i in range(self.n_sectors):
            # Calculate demand fulfillment rate
            if final_demand[i] > 0:
                # Net output = total output - intermediate demand
                intermediate_demand = np.sum(technology_matrix[:, i] * total_output)
                net_output = total_output[i] - intermediate_demand
                demand_fulfillment_rate = min(net_output / final_demand[i], 2.0)  # Cap at 200%
            else:
                demand_fulfillment_rate = 1.0

            # Calculate labor efficiency (output per unit labor)
            if labor_vector[i] > 0:
                labor_efficiency = total_output[i] / labor_vector[i]
            else:
                labor_efficiency = 0.0

            # Calculate technology level (inverse of input requirements)
            technology_level = 1.0 / (np.sum(technology_matrix[:, i]) + 1e-10)

            # Calculate growth potential based on demand fulfillment
            if demand_fulfillment_rate < 0.8:  # Under - supplied
                growth_potential = 1.5  # High growth potential
            elif demand_fulfillment_rate < 1.2:  # Well - balanced
                growth_potential = 1.0  # Normal growth
            else:  # Over - supplied
                growth_potential = 0.5  # Low growth potential

            # Calculate bottleneck severity
            bottleneck_severity = max(0, 1.0 - demand_fulfillment_rate)

            # Determine department
            if i in self.dept_I_indices:
                department = "I"
            elif i in self.dept_II_indices:
                department = "II"
            else:
                department = "III"

            performance = IndustryPerformance(
                sector_id = i,
                output = total_output[i],
                demand = final_demand[i],
                demand_fulfillment_rate = demand_fulfillment_rate,
                labor_efficiency = labor_efficiency,
                technology_level = technology_level,
                growth_potential = growth_potential,
                bottleneck_severity = bottleneck_severity
            )

            performance_metrics.append(performance)

        return performance_metrics

    def calculate_dynamic_growth_rates(
        self,
        performance_metrics: List[IndustryPerformance],
        year: int
    ) -> GrowthRates:
        """
        Calculate dynamic growth rates based on industry performance.

        Args:
            performance_metrics: Industry performance analysis
            year: Current year

        Returns:
            Dynamic growth rates
        """
        # Analyze department - level performance
        dept_I_performance = [p for p in performance_metrics if p.sector_id in self.dept_I_indices]
        dept_II_performance = [p for p in performance_metrics if p.sector_id in self.dept_II_indices]
        dept_III_performance = [p for p in performance_metrics if p.sector_id in self.dept_III_indices]

        # Calculate department averages
        dept_I_fulfillment = np.mean([p.demand_fulfillment_rate for p in dept_I_performance])
        dept_II_fulfillment = np.mean([p.demand_fulfillment_rate for p in dept_II_performance])
        dept_III_fulfillment = np.mean([p.demand_fulfillment_rate for p in dept_III_performance])

        dept_I_efficiency = np.mean([p.labor_efficiency for p in dept_I_performance])
        dept_II_efficiency = np.mean([p.labor_efficiency for p in dept_II_performance])
        dept_III_efficiency = np.mean([p.labor_efficiency for p in dept_III_performance])

        # Calculate overall economic health
        overall_fulfillment = np.mean([p.demand_fulfillment_rate for p in performance_metrics])
        overall_efficiency = np.mean([p.labor_efficiency for p in performance_metrics])

        # Population growth based on consumer goods availability
        # If consumer goods are well - supplied, population can grow
        population_growth = self.base_population_growth * (0.5 + dept_II_fulfillment * 0.5)
        population_growth = np.clip(population_growth, 0.01, 0.04)  # 1% to 4%

        # Living standards growth based on service sector performance
        # Better services enable higher living standards
        living_standards_growth = self.base_living_standards_growth * (0.5 + dept_III_fulfillment * 0.5)
        living_standards_growth = np.clip(living_standards_growth, 0.02, 0.08)  # 2% to 8%

        # Technology improvement based on Department I performance
        # Better means of production enable faster technological progress
        # BUT ensure it's always less than demand growth to prevent decreasing output
        technology_improvement = self.base_technology_rate * (0.5 + dept_I_fulfillment * 0.5)
        max_tech_improvement = (population_growth + living_standards_growth) * 0.8  # Max 80% of demand growth
        technology_improvement = np.clip(technology_improvement, 0.005, max_tech_improvement)

        # Capital accumulation based on overall economic health
        # Strong performance enables more investment
        capital_accumulation = self.base_capital_rate * (0.5 + overall_fulfillment * 0.5)
        capital_accumulation = np.clip(capital_accumulation, 0.05, 0.25)  # 5% to 25%

        # Calculate sectoral growth rates
        sectoral_growth_rates = np.zeros(self.n_sectors)

        for i, performance in enumerate(performance_metrics):
            # Base growth from population and living standards
            if i in self.dept_II_indices:  # Consumer goods
                base_growth = population_growth + living_standards_growth
            elif i in self.dept_III_indices:  # Services
                base_growth = living_standards_growth
            else:  # Department I (means of production)
                base_growth = capital_accumulation

            # Adjust based on performance
            if performance.demand_fulfillment_rate < 0.8:  # Under - supplied
                growth_multiplier = 1.5  # Accelerate growth
            elif performance.demand_fulfillment_rate > 1.5:  # Over - supplied
                growth_multiplier = 0.7  # Slow growth
            else:
                growth_multiplier = 1.0  # Normal growth

            # Apply efficiency bonus / penalty
            efficiency_bonus = 0.0
            efficiency_penalty = 0.0

            if performance.labor_efficiency > overall_efficiency * 1.1:
                efficiency_bonus = 0.1  # 10% bonus for high efficiency
            elif performance.labor_efficiency < overall_efficiency * 0.9:
                efficiency_penalty = -0.1  # 10% penalty for low efficiency

            sectoral_growth_rates[i] = base_growth * growth_multiplier + efficiency_bonus + efficiency_penalty
            sectoral_growth_rates[i] = np.clip(sectoral_growth_rates[i], 0.0, 0.15)  # 0% to 15%

        return GrowthRates(
            population_growth = population_growth,
            living_standards_growth = living_standards_growth,
            technology_improvement = technology_improvement,
            capital_accumulation = capital_accumulation,
            sectoral_growth_rates = sectoral_growth_rates
        )

    def generate_adaptive_demands(
        self,
        base_final_demand: np.ndarray,
        growth_rates: GrowthRates,
        year: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adaptive consumption and investment demands based on growth rates.

        Args:
            base_final_demand: Base final demand vector
            growth_rates: Calculated growth rates
            year: Current year

        Returns:
            Tuple of (consumption_demand, investment_demand)
        """
        # Calculate total growth rate
        total_growth_rate = growth_rates.population_growth + growth_rates.living_standards_growth

        # Apply sectoral growth rates to consumption demand
        consumption_demand = base_final_demand.copy()
        for i in range(self.n_sectors):
            if i in self.dept_II_indices:  # Consumer goods grow with population + living standards
                consumption_demand[i] *= (1 + total_growth_rate) ** (year - 1)
            elif i in self.dept_III_indices:  # Services grow with living standards
                consumption_demand[i] *= (1 + growth_rates.living_standards_growth) ** (year - 1)
            else:  # Department I grows with capital accumulation
                consumption_demand[i] *= (1 + growth_rates.capital_accumulation) ** (year - 1)

        # Generate investment demand based on sectoral performance
        investment_demand = np.zeros_like(base_final_demand)

        for i in range(self.n_sectors):
            if i in self.dept_I_indices:  # Invest in means of production
                # Investment grows with capital accumulation rate
                investment_demand[i] = base_final_demand[i] * growth_rates.capital_accumulation * (1 + growth_rates.capital_accumulation) ** (year - 1)
            elif i in self.dept_II_indices:  # Some investment in consumer goods production
                investment_demand[i] = base_final_demand[i] * growth_rates.capital_accumulation * 0.3 * (1 + growth_rates.capital_accumulation) ** (year - 1)
            # Department III (services) typically requires less investment

        return consumption_demand, investment_demand

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance analysis."""
        if not self.performance_history:
            return {"message": "No performance data available"}

        latest = self.performance_history[-1]

        return {
            "overall_fulfillment": latest.get("overall_fulfillment", 0),
            "overall_efficiency": latest.get("overall_efficiency", 0),
            "dept_I_performance": latest.get("dept_I_fulfillment", 0),
            "dept_II_performance": latest.get("dept_II_fulfillment", 0),
            "dept_III_performance": latest.get("dept_III_fulfillment", 0),
            "growth_rates": latest.get("growth_rates", {}),
            "bottlenecks": latest.get("bottlenecks", [])
        }
