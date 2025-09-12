"""
Population Health Tracking System

Tracks population health metrics over time including living standards growth,
consumer demand fulfillment, population per month, and technology level per month.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
import json

@dataclass
class MonthlyHealthMetrics:
    """Monthly population health metrics."""
    month: int
    year: int
    month_in_year: int
    population: float
    technology_level: float
    living_standards_index: float
    consumer_demand_fulfillment: float
    r_and_d_output: float
    healthcare_access: float
    education_access: float
    housing_quality: float
    nutrition_index: float
    employment_rate: float
    income_per_capita: float
    life_expectancy: float
    birth_rate: float
    death_rate: float
    net_migration: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PopulationHealthSummary:
    """Summary of population health over time."""
    total_months: int
    start_date: datetime
    end_date: datetime
    initial_population: float
    final_population: float
    population_growth_rate: float
    average_technology_level: float
    technology_growth_rate: float
    average_living_standards: float
    living_standards_growth_rate: float
    average_consumer_demand_fulfillment: float
    demand_fulfillment_trend: str
    health_indicators: Dict[str, float]
    monthly_data: List[MonthlyHealthMetrics]

class PopulationHealthTracker:
    """
    Tracks population health metrics over time for the simulation.
    
    Features:
    - Monthly population tracking
    - Technology level growth based on R&D output
    - Living standards calculation
    - Consumer demand fulfillment tracking
    - Health indicators monitoring
    - Trend analysis and reporting
    """
    
    def __init__(self, initial_population: float = 1000000, initial_technology_level: float = 0.0):
        """
        Initialize the population health tracker.
        
        Args:
            initial_population: Starting population
            initial_technology_level: Starting technology level (0.0 to 1.0)
        """
        self.initial_population = initial_population
        self.initial_technology_level = initial_technology_level
        self.current_population = initial_population
        self.current_technology_level = initial_technology_level
        
        # Monthly tracking data
        self.monthly_metrics: List[MonthlyHealthMetrics] = []
        self.current_month = 0
        
        # R&D sector tracking
        self.rd_sector_indices: List[int] = []
        self.rd_output_history: List[float] = []
        
        # Health indicators
        self.health_indicators = {
            'life_expectancy': 75.0,  # years
            'birth_rate': 0.012,      # per year
            'death_rate': 0.008,      # per year
            'net_migration': 0.002,   # per year
            'employment_rate': 0.95,  # 95%
            'income_per_capita': 50000,  # dollars
        }
        
        # Living standards components
        self.living_standards_weights = {
            'healthcare_access': 0.25,
            'education_access': 0.20,
            'housing_quality': 0.15,
            'nutrition_index': 0.15,
            'employment_rate': 0.10,
            'income_per_capita': 0.10,
            'technology_level': 0.05
        }
    
    def set_rd_sectors(self, rd_sector_indices: List[int]):
        """Set the indices of R&D related sectors."""
        self.rd_sector_indices = rd_sector_indices
    
    def update_monthly_metrics(
        self,
        production_data: Dict[str, Any],
        resource_data: Dict[str, Any],
        labor_data: Dict[str, Any],
        sector_mapping: Optional[Dict[str, int]] = None
    ) -> MonthlyHealthMetrics:
        """
        Update monthly population health metrics.
        
        Args:
            production_data: Production results for the month
            resource_data: Resource allocation data
            labor_data: Labor allocation data
            sector_mapping: Mapping of sector names to indices
            
        Returns:
            Monthly health metrics for the current month
        """
        self.current_month += 1
        year = self.current_month // 12
        month_in_year = self.current_month % 12
        
        # Calculate R&D output
        rd_output = self._calculate_rd_output(production_data, sector_mapping)
        self.rd_output_history.append(rd_output)
        
        # Update technology level based on R&D output
        self._update_technology_level(rd_output)
        
        # Calculate living standards
        living_standards = self._calculate_living_standards(
            production_data, resource_data, labor_data, sector_mapping
        )
        
        # Calculate consumer demand fulfillment
        consumer_demand_fulfillment = self._calculate_consumer_demand_fulfillment(
            production_data, sector_mapping
        )
        
        # Update population based on health indicators
        self._update_population()
        
        # Calculate health indicators
        health_indicators = self._calculate_health_indicators(
            production_data, resource_data, labor_data, sector_mapping
        )
        
        # Create monthly metrics
        monthly_metrics = MonthlyHealthMetrics(
            month=self.current_month,
            year=year,
            month_in_year=month_in_year,
            population=self.current_population,
            technology_level=self.current_technology_level,
            living_standards_index=living_standards,
            consumer_demand_fulfillment=consumer_demand_fulfillment,
            r_and_d_output=rd_output,
            healthcare_access=health_indicators['healthcare_access'],
            education_access=health_indicators['education_access'],
            housing_quality=health_indicators['housing_quality'],
            nutrition_index=health_indicators['nutrition_index'],
            employment_rate=health_indicators['employment_rate'],
            income_per_capita=health_indicators['income_per_capita'],
            life_expectancy=health_indicators['life_expectancy'],
            birth_rate=health_indicators['birth_rate'],
            death_rate=health_indicators['death_rate'],
            net_migration=health_indicators['net_migration']
        )
        
        self.monthly_metrics.append(monthly_metrics)
        return monthly_metrics
    
    def _calculate_rd_output(
        self, 
        production_data: Dict[str, Any], 
        sector_mapping: Optional[Dict[str, int]] = None
    ) -> float:
        """Calculate R&D sector output."""
        total_rd_output = 0.0
        
        # Try multiple approaches to find R&D sectors
        rd_sector_names = [
            'Technology', 'Technology and Software', 'Healthcare', 'Healthcare and Biotechnology',
            'Education', 'Education and Research', 'Professional Services', 'Research and Development',
            'Science', 'Innovation', 'R&D', 'Research'
        ]
        
        # First try: Look for R&D sectors by name in production_data
        for sector_name in rd_sector_names:
            if sector_name in production_data:
                sector_data = production_data[sector_name]
                if isinstance(sector_data, dict):
                    total_rd_output += sector_data.get('actual', 0)
                else:
                    total_rd_output += float(sector_data)
        
        # Second try: Look for technology-related keywords in sector names
        if total_rd_output == 0:
            for sector_name, sector_data in production_data.items():
                if isinstance(sector_data, dict):
                    sector_output = sector_data.get('actual', 0)
                else:
                    sector_output = float(sector_data)
                
                # Check if sector name contains R&D keywords
                sector_lower = sector_name.lower()
                if any(keyword in sector_lower for keyword in ['tech', 'research', 'development', 'innovation', 'science', 'rd']):
                    total_rd_output += sector_output
        
        # Third try: Use sector indices if available
        if total_rd_output == 0 and self.rd_sector_indices:
            for sector_idx in self.rd_sector_indices:
                if sector_idx < len(production_data.get('sectors', [])):
                    sector_data = production_data['sectors'][sector_idx]
                    total_rd_output += sector_data.get('actual', 0)
        
        # Fourth try: Use a percentage of total production as R&D proxy
        if total_rd_output == 0:
            total_production = 0
            for sector_name, sector_data in production_data.items():
                if isinstance(sector_data, dict):
                    total_production += sector_data.get('actual', 0)
                else:
                    total_production += float(sector_data)
            
            # Assume 5% of total production is R&D
            total_rd_output = total_production * 0.05
        
        return total_rd_output
    
    def _update_technology_level(self, rd_output: float):
        """Update technology level based on R&D output."""
        # Technology growth proportional to R&D output
        # More realistic scaling: 0.01 per 100,000 units of R&D output
        base_tech_growth = (rd_output / 100000.0) * 0.01
        
        # Apply diminishing returns - technology gets harder to improve as it advances
        diminishing_factor = 1.0 - (self.current_technology_level * 0.7)
        tech_growth = base_tech_growth * max(0.1, diminishing_factor)
        
        # Add a small base growth rate to ensure continuous progress
        base_progress = 0.001  # 0.1% per month base growth
        
        # Technology level also depends on population size (more people = more innovation)
        population_factor = min(2.0, self.current_population / 1000000.0)  # Cap at 2x for large populations
        
        total_tech_growth = (tech_growth + base_progress) * population_factor
        
        # Update technology level
        self.current_technology_level = min(1.0, self.current_technology_level + total_tech_growth)
    
    def _calculate_living_standards(
        self,
        production_data: Dict[str, Any],
        resource_data: Dict[str, Any],
        labor_data: Dict[str, Any],
        sector_mapping: Optional[Dict[str, int]] = None
    ) -> float:
        """Calculate living standards index."""
        living_standards = 0.0
        
        # Calculate total economic output for normalization
        total_output = 0
        for sector_name, sector_data in production_data.items():
            if isinstance(sector_data, dict):
                total_output += sector_data.get('actual', 0)
            else:
                total_output += float(sector_data)
        
        # Normalize by population
        per_capita_output = total_output / max(1, self.current_population)
        
        # Healthcare access (based on healthcare sector output)
        healthcare_output = self._get_sector_output(production_data, 'Healthcare', sector_mapping)
        healthcare_access = min(1.0, healthcare_output / max(1, per_capita_output * 1000))  # More realistic normalization
        living_standards += healthcare_access * self.living_standards_weights['healthcare_access']
        
        # Education access (based on education sector output)
        education_output = self._get_sector_output(production_data, 'Education', sector_mapping)
        education_access = min(1.0, education_output / max(1, per_capita_output * 1000))
        living_standards += education_access * self.living_standards_weights['education_access']
        
        # Housing quality (based on construction sector output)
        housing_output = self._get_sector_output(production_data, 'Construction', sector_mapping)
        housing_quality = min(1.0, housing_output / max(1, per_capita_output * 1000))
        living_standards += housing_quality * self.living_standards_weights['housing_quality']
        
        # Nutrition index (based on agriculture sector output)
        nutrition_output = self._get_sector_output(production_data, 'Agriculture', sector_mapping)
        nutrition_index = min(1.0, nutrition_output / max(1, per_capita_output * 1000))
        living_standards += nutrition_index * self.living_standards_weights['nutrition_index']
        
        # Employment rate (from labor data)
        employment_rate = labor_data.get('employment_rate', 0.95)
        living_standards += employment_rate * self.living_standards_weights['employment_rate']
        
        # Income per capita (calculated from total output)
        income_per_capita = per_capita_output / 1000.0  # Convert to thousands
        income_index = min(1.0, income_per_capita / 50.0)  # Normalize to $50k per capita
        living_standards += income_index * self.living_standards_weights['income_per_capita']
        
        # Technology level contribution
        tech_contribution = min(1.0, self.current_technology_level)
        living_standards += tech_contribution * self.living_standards_weights['technology_level']
        
        # Add resource availability factor
        resource_availability = 1.0
        if resource_data:
            total_resources = 0
            if isinstance(resource_data, dict):
                for resource_value in resource_data.values():
                    if isinstance(resource_value, dict):
                        actual_value = resource_value.get('actual', 0)
                        # Handle case where actual_value is a list
                        if isinstance(actual_value, list):
                            # Sum all numeric values in the list
                            for item in actual_value:
                                if isinstance(item, (int, float)):
                                    total_resources += item
                        else:
                            total_resources += actual_value
                    else:
                        total_resources += float(resource_value) if isinstance(resource_value, (int, float)) else 0
            resource_availability = min(1.0, total_resources / max(1, total_output))
        
        # Apply resource availability as a multiplier
        living_standards *= resource_availability
        
        return min(1.0, max(0.0, living_standards))
    
    def _calculate_consumer_demand_fulfillment(
        self,
        production_data: Dict[str, Any],
        sector_mapping: Optional[Dict[str, int]] = None
    ) -> float:
        """Calculate consumer demand fulfillment rate."""
        consumer_sectors = ['Food and Agriculture', 'Healthcare', 'Education', 'Transportation']
        total_fulfillment = 0.0
        sector_count = 0
        
        for sector in consumer_sectors:
            sector_output = self._get_sector_output(production_data, sector, sector_mapping)
            sector_target = self._get_sector_target(production_data, sector, sector_mapping)
            
            if sector_target > 0:
                fulfillment = min(2.0, sector_output / sector_target)  # Cap at 200%
                total_fulfillment += fulfillment
                sector_count += 1
        
        return total_fulfillment / sector_count if sector_count > 0 else 1.0
    
    def _calculate_health_indicators(
        self,
        production_data: Dict[str, Any],
        resource_data: Dict[str, Any],
        labor_data: Dict[str, Any],
        sector_mapping: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """Calculate health indicators."""
        # Base indicators
        indicators = self.health_indicators.copy()
        
        # Adjust based on living standards
        living_standards = self._calculate_living_standards(
            production_data, resource_data, labor_data, sector_mapping
        )
        
        # Life expectancy increases with living standards
        indicators['life_expectancy'] = 65.0 + (living_standards * 20.0)
        
        # Birth rate decreases with living standards (demographic transition)
        indicators['birth_rate'] = 0.020 - (living_standards * 0.010)
        
        # Death rate decreases with living standards
        indicators['death_rate'] = 0.015 - (living_standards * 0.010)
        
        # Employment rate increases with economic output
        total_output = 0
        if isinstance(production_data.get('sectors'), dict):
            for sector_data in production_data['sectors'].values():
                if isinstance(sector_data, dict):
                    total_output += sector_data.get('actual', 0)
                else:
                    total_output += float(sector_data) if isinstance(sector_data, (int, float)) else 0
        else:
            # Calculate total output from production_data directly
            for sector_name, sector_data in production_data.items():
                if isinstance(sector_data, dict):
                    total_output += sector_data.get('actual', 0)
                else:
                    total_output += float(sector_data) if isinstance(sector_data, (int, float)) else 0
        
        indicators['employment_rate'] = min(0.98, 0.80 + (total_output / 10000000.0) * 0.18)
        
        # Income per capita based on total output
        indicators['income_per_capita'] = max(20000, total_output / self.current_population)
        
        # Healthcare access based on healthcare sector
        healthcare_output = self._get_sector_output(production_data, 'Healthcare', sector_mapping)
        indicators['healthcare_access'] = min(1.0, healthcare_output / 500000.0)
        
        # Education access based on education sector
        education_output = self._get_sector_output(production_data, 'Education', sector_mapping)
        indicators['education_access'] = min(1.0, education_output / 500000.0)
        
        # Housing quality based on construction sector
        housing_output = self._get_sector_output(production_data, 'Construction', sector_mapping)
        indicators['housing_quality'] = min(1.0, housing_output / 500000.0)
        
        # Nutrition index based on agriculture sector
        nutrition_output = self._get_sector_output(production_data, 'Agriculture', sector_mapping)
        indicators['nutrition_index'] = min(1.0, nutrition_output / 500000.0)
        
        return indicators
    
    def _update_population(self):
        """Update population based on health indicators."""
        if not self.monthly_metrics:
            return
        
        # Get latest health indicators
        latest_metrics = self.monthly_metrics[-1]
        
        # Calculate population change
        birth_rate = latest_metrics.birth_rate / 12.0  # Monthly rate
        death_rate = latest_metrics.death_rate / 12.0  # Monthly rate
        migration_rate = latest_metrics.net_migration / 12.0  # Monthly rate
        
        # Apply population change
        population_change = self.current_population * (birth_rate - death_rate + migration_rate)
        self.current_population = max(1000, self.current_population + population_change)
    
    def _get_sector_output(
        self, 
        production_data: Dict[str, Any], 
        sector_name: str, 
        sector_mapping: Optional[Dict[str, int]] = None
    ) -> float:
        """Get output for a specific sector."""
        if sector_mapping and sector_name in sector_mapping:
            sector_idx = sector_mapping[sector_name]
            if 'sectors' in production_data and sector_idx < len(production_data['sectors']):
                sector_data = production_data['sectors'][sector_idx]
                if isinstance(sector_data, dict):
                    return sector_data.get('actual', 0)
                else:
                    return float(sector_data) if isinstance(sector_data, (int, float)) else 0.0
        
        # Try direct sector name lookup
        if sector_name in production_data:
            sector_data = production_data[sector_name]
            if isinstance(sector_data, dict):
                return sector_data.get('actual', 0)
            else:
                return float(sector_data) if isinstance(sector_data, (int, float)) else 0.0
        
        return 0.0
    
    def _get_sector_target(
        self, 
        production_data: Dict[str, Any], 
        sector_name: str, 
        sector_mapping: Optional[Dict[str, int]] = None
    ) -> float:
        """Get target for a specific sector."""
        if sector_mapping and sector_name in sector_mapping:
            sector_idx = sector_mapping[sector_name]
            if 'sectors' in production_data and sector_idx < len(production_data['sectors']):
                sector_data = production_data['sectors'][sector_idx]
                if isinstance(sector_data, dict):
                    return sector_data.get('target', 0)
                else:
                    return float(sector_data) if isinstance(sector_data, (int, float)) else 0.0
        
        # Try direct sector name lookup
        if sector_name in production_data:
            sector_data = production_data[sector_name]
            if isinstance(sector_data, dict):
                return sector_data.get('target', 0)
            else:
                return float(sector_data) if isinstance(sector_data, (int, float)) else 0.0
        
        return 0.0
    
    def get_population_health_summary(self) -> PopulationHealthSummary:
        """Get comprehensive population health summary."""
        if not self.monthly_metrics:
            return PopulationHealthSummary(
                total_months=0,
                start_date=datetime.now(),
                end_date=datetime.now(),
                initial_population=self.initial_population,
                final_population=self.current_population,
                population_growth_rate=0.0,
                average_technology_level=self.current_technology_level,
                technology_growth_rate=0.0,
                average_living_standards=0.0,
                living_standards_growth_rate=0.0,
                average_consumer_demand_fulfillment=0.0,
                demand_fulfillment_trend="stable",
                health_indicators=self.health_indicators,
                monthly_data=[]
            )
        
        # Calculate summary statistics
        total_months = len(self.monthly_metrics)
        start_date = self.monthly_metrics[0].timestamp
        end_date = self.monthly_metrics[-1].timestamp
        
        initial_population = self.monthly_metrics[0].population
        final_population = self.monthly_metrics[-1].population
        population_growth_rate = (final_population - initial_population) / initial_population if initial_population > 0 else 0.0
        
        # Technology level statistics
        tech_levels = [m.technology_level for m in self.monthly_metrics]
        average_technology_level = np.mean(tech_levels)
        technology_growth_rate = (tech_levels[-1] - tech_levels[0]) / tech_levels[0] if tech_levels[0] > 0 else 0.0
        
        # Living standards statistics
        living_standards_list = [m.living_standards_index for m in self.monthly_metrics]
        average_living_standards = np.mean(living_standards_list)
        living_standards_growth_rate = (living_standards_list[-1] - living_standards_list[0]) / living_standards_list[0] if living_standards_list[0] > 0 else 0.0
        
        # Consumer demand fulfillment statistics
        demand_fulfillment = [m.consumer_demand_fulfillment for m in self.monthly_metrics]
        average_consumer_demand_fulfillment = np.mean(demand_fulfillment)
        
        # Determine demand fulfillment trend
        if len(demand_fulfillment) >= 3:
            recent_trend = np.mean(demand_fulfillment[-3:]) - np.mean(demand_fulfillment[:3])
            if recent_trend > 0.05:
                demand_fulfillment_trend = "improving"
            elif recent_trend < -0.05:
                demand_fulfillment_trend = "declining"
            else:
                demand_fulfillment_trend = "stable"
        else:
            demand_fulfillment_trend = "stable"
        
        # Calculate average health indicators
        avg_health_indicators = {}
        for key in self.health_indicators.keys():
            values = [getattr(m, key, 0) for m in self.monthly_metrics if hasattr(m, key)]
            avg_health_indicators[key] = np.mean(values) if values else 0.0
        
        return PopulationHealthSummary(
            total_months=total_months,
            start_date=start_date,
            end_date=end_date,
            initial_population=initial_population,
            final_population=final_population,
            population_growth_rate=population_growth_rate,
            average_technology_level=average_technology_level,
            technology_growth_rate=technology_growth_rate,
            average_living_standards=average_living_standards,
            living_standards_growth_rate=living_standards_growth_rate,
            average_consumer_demand_fulfillment=average_consumer_demand_fulfillment,
            demand_fulfillment_trend=demand_fulfillment_trend,
            health_indicators=avg_health_indicators,
            monthly_data=self.monthly_metrics
        )
    
    def export_monthly_data(self, filepath: str):
        """Export monthly data to JSON file."""
        data = {
            'metadata': {
                'initial_population': self.initial_population,
                'initial_technology_level': self.initial_technology_level,
                'total_months': len(self.monthly_metrics),
                'export_date': datetime.now().isoformat()
            },
            'monthly_metrics': [
                {
                    'month': m.month,
                    'year': m.year,
                    'month_in_year': m.month_in_year,
                    'population': m.population,
                    'technology_level': m.technology_level,
                    'living_standards_index': m.living_standards_index,
                    'consumer_demand_fulfillment': m.consumer_demand_fulfillment,
                    'r_and_d_output': m.r_and_d_output,
                    'healthcare_access': m.healthcare_access,
                    'education_access': m.education_access,
                    'housing_quality': m.housing_quality,
                    'nutrition_index': m.nutrition_index,
                    'employment_rate': m.employment_rate,
                    'income_per_capita': m.income_per_capita,
                    'life_expectancy': m.life_expectancy,
                    'birth_rate': m.birth_rate,
                    'death_rate': m.death_rate,
                    'net_migration': m.net_migration,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.monthly_metrics
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_technology_growth_rate(self) -> float:
        """Get current technology growth rate."""
        if len(self.rd_output_history) < 2:
            return 0.0
        
        # Calculate growth rate based on recent R&D output trend
        recent_output = np.mean(self.rd_output_history[-3:]) if len(self.rd_output_history) >= 3 else self.rd_output_history[-1]
        earlier_output = np.mean(self.rd_output_history[:3]) if len(self.rd_output_history) >= 3 else self.rd_output_history[0]
        
        if earlier_output > 0:
            return (recent_output - earlier_output) / earlier_output
        return 0.0
