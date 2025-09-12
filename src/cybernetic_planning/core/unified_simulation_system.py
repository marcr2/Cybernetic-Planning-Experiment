"""
Unified Cybernetic-Spatial Simulation System

Merges map-based spatial simulation with dynamic economic simulation
into a single, cohesive system that accounts for both geographical
constraints and economic processes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Import existing components
from .map_based_simulator import MapBasedSimulator, MapTile, Settlement, InfrastructureSegment, DisasterEvent
from .dynamic_planning import DynamicPlanner
from .enhanced_simulation import EnhancedEconomicSimulation
from ..utils.simulation_integration import EconomicSimulation, SimulationConfig
from ..planning_system import CyberneticPlanningSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedSimulationState:
    """Unified state tracking both spatial and economic systems."""
    
    # Time management
    current_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    current_day: int = 0
    current_month: int = 1
    current_year: int = 2024
    
    # Spatial state
    map_generated: bool = False
    settlements_active: int = 0
    infrastructure_segments: int = 0
    active_disasters: int = 0
    
    # Economic state
    economic_plan_active: bool = False
    sectors_active: int = 0
    total_economic_output: float = 0.0
    capital_stock_total: float = 0.0
    
    # Integration metrics
    spatial_economic_efficiency: float = 0.0
    logistics_friction_total: float = 0.0
    disaster_impact_on_economy: float = 0.0
    
    # Performance tracking
    simulation_start_time: datetime = field(default_factory=datetime.now)
    step_times: List[float] = field(default_factory=list)
    memory_usage_mb: float = 0.0

@dataclass
class SpatialEconomicMapping:
    """Maps economic sectors to spatial locations with constraints."""
    
    sector_id: str
    settlement_id: str
    production_capacity: float
    terrain_suitability: float
    infrastructure_access: float
    resource_availability: float
    logistics_cost_multiplier: float = 1.0
    disaster_vulnerability: float = 0.1

@dataclass
class UnifiedSimulationConfig:
    """Configuration for unified simulation system."""
    
    # Spatial parameters
    map_width: int = 200
    map_height: int = 200
    terrain_distribution: Dict[str, float] = field(default_factory=lambda: {
        "flatland": 0.4,
        "forest": 0.3,
        "mountain": 0.2,
        "water": 0.1
    })
    num_cities: int = 5
    num_towns: int = 15
    total_population: int = 1000000
    rural_population_percent: float = 0.3
    urban_concentration: str = "medium"
    
    # Economic parameters
    n_sectors: int = 15
    technology_density: float = 0.4
    resource_count: int = 8
    policy_goals: List[str] = field(default_factory=lambda: [
        "Increase industrial production",
        "Improve living standards",
        "Develop infrastructure"
    ])
    
    # Simulation parameters
    simulation_duration_months: int = 60  # 5-year plan
    spatial_update_frequency: str = "daily"  # daily, weekly, monthly
    economic_update_frequency: str = "monthly"  # monthly, quarterly, annual
    disaster_probability: float = 0.05
    
    # Integration parameters
    enable_bidirectional_feedback: bool = True
    enable_spatial_constraints: bool = True
    enable_disaster_economic_impact: bool = True
    enable_infrastructure_economic_feedback: bool = True

class SpatialEconomicIntegration:
    """Enhanced integration between spatial and economic systems."""
    
    def __init__(self, unified_system: 'UnifiedSimulationSystem'):
        self.unified_system = unified_system
        self.sector_settlement_mappings: Dict[str, SpatialEconomicMapping] = {}
        self.feedback_coefficients = {
            'economic_to_infrastructure': 0.1,
            'infrastructure_to_economic': 0.15,
            'disaster_to_economic': 0.2,
            'logistics_to_economic': 0.05
        }
    
    def map_economic_activity_to_spatial(self, economic_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Map economic sectors to spatial locations with realistic constraints."""
        if not self.unified_system.map_simulator or not economic_plan:
            return {"success": False, "error": "Missing map simulator or economic plan"}
        
        try:
            sectors = economic_plan.get('sectors', [])
            settlements = list(self.unified_system.map_simulator.settlements.values())
            
            if not settlements:
                return {"success": False, "error": "No settlements available for mapping"}
            
            # Clear existing mappings
            self.sector_settlement_mappings.clear()
            
            # Create enhanced mapping based on settlement characteristics
            for i, sector in enumerate(sectors):
                # Select settlement based on economic importance and suitability
                settlement = self._select_optimal_settlement(sector, settlements)
                
                # Calculate spatial constraints
                production_capacity = self._calculate_production_capacity(sector, settlement)
                terrain_suitability = self._calculate_terrain_suitability(sector, settlement)
                infrastructure_access = self._calculate_infrastructure_access(settlement)
                resource_availability = self._calculate_resource_availability(sector, settlement)
                
                # Create mapping
                mapping = SpatialEconomicMapping(
                    sector_id=sector.get('id', f'sector_{i}'),
                    settlement_id=settlement.id,
                    production_capacity=production_capacity,
                    terrain_suitability=terrain_suitability,
                    infrastructure_access=infrastructure_access,
                    resource_availability=resource_availability,
                    logistics_cost_multiplier=self._calculate_logistics_multiplier(settlement),
                    disaster_vulnerability=self._calculate_disaster_vulnerability(settlement)
                )
                
                self.sector_settlement_mappings[mapping.sector_id] = mapping
                
                # Update settlement with sector assignment
                if not hasattr(settlement, 'sectors'):
                    settlement.sectors = []
                settlement.sectors.append(sector)
            
            logger.info(f"Mapped {len(sectors)} sectors to {len(settlements)} settlements")
            
            return {
                "success": True,
                "mappings_created": len(self.sector_settlement_mappings),
                "average_production_capacity": np.mean([m.production_capacity for m in self.sector_settlement_mappings.values()]),
                "average_terrain_suitability": np.mean([m.terrain_suitability for m in self.sector_settlement_mappings.values()])
            }
            
        except Exception as e:
            logger.error(f"Failed to map economic activity to spatial: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_spatial_economic_costs(self, production_targets: Dict[str, float]) -> Dict[str, Any]:
        """Calculate how spatial factors affect economic efficiency."""
        if not self.sector_settlement_mappings:
            return {"success": False, "error": "No sector-settlement mappings available"}
        
        try:
            total_spatial_cost = 0.0
            spatial_constraints = {}
            
            for sector_id, mapping in self.sector_settlement_mappings.items():
                target_output = production_targets.get(sector_id, 0.0)
                
                # Calculate spatial efficiency factors
                terrain_efficiency = mapping.terrain_suitability
                infrastructure_efficiency = mapping.infrastructure_access
                resource_efficiency = mapping.resource_availability
                logistics_efficiency = 1.0 / mapping.logistics_cost_multiplier
                
                # Combined spatial efficiency
                spatial_efficiency = (terrain_efficiency * 0.3 + 
                                    infrastructure_efficiency * 0.3 + 
                                    resource_efficiency * 0.2 + 
                                    logistics_efficiency * 0.2)
                
                # Calculate spatial cost multiplier
                spatial_cost_multiplier = 1.0 / max(spatial_efficiency, 0.1)
                spatial_cost = target_output * spatial_cost_multiplier
                
                spatial_constraints[sector_id] = {
                    "spatial_efficiency": spatial_efficiency,
                    "spatial_cost_multiplier": spatial_cost_multiplier,
                    "spatial_cost": spatial_cost,
                    "terrain_suitability": terrain_efficiency,
                    "infrastructure_access": infrastructure_efficiency,
                    "resource_availability": resource_efficiency,
                    "logistics_efficiency": logistics_efficiency
                }
                
                total_spatial_cost += spatial_cost
            
            return {
                "success": True,
                "total_spatial_cost": total_spatial_cost,
                "spatial_constraints": spatial_constraints,
                "average_spatial_efficiency": np.mean([c["spatial_efficiency"] for c in spatial_constraints.values()])
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate spatial economic costs: {e}")
            return {"success": False, "error": str(e)}
    
    def simulate_disaster_impact(self, disaster_event: DisasterEvent) -> Dict[str, Any]:
        """Simulate how disasters affect both spatial and economic systems."""
        try:
            # Find affected settlements
            affected_settlements = []
            for settlement in self.unified_system.map_simulator.settlements.values():
                distance = np.sqrt((settlement.x - disaster_event.x)**2 + (settlement.y - disaster_event.y)**2)
                if distance <= disaster_event.radius:
                    affected_settlements.append(settlement)
            
            # Calculate economic impact
            economic_impact = 0.0
            affected_sectors = []
            
            for settlement in affected_settlements:
                # Calculate settlement damage
                distance = np.sqrt((settlement.x - disaster_event.x)**2 + (settlement.y - disaster_event.y)**2)
                damage_factor = max(0, 1.0 - distance / disaster_event.radius) * disaster_event.intensity
                
                # Find sectors in this settlement
                for sector_id, mapping in self.sector_settlement_mappings.items():
                    if mapping.settlement_id == settlement.id:
                        sector_impact = damage_factor * mapping.disaster_vulnerability
                        economic_impact += sector_impact
                        affected_sectors.append({
                            "sector_id": sector_id,
                            "settlement_id": settlement.id,
                            "damage_factor": damage_factor,
                            "economic_impact": sector_impact
                        })
            
            return {
                "success": True,
                "disaster_type": disaster_event.disaster_type.value,
                "affected_settlements": len(affected_settlements),
                "affected_sectors": len(affected_sectors),
                "total_economic_impact": economic_impact,
                "sector_impacts": affected_sectors
            }
            
        except Exception as e:
            logger.error(f"Failed to simulate disaster impact: {e}")
            return {"success": False, "error": str(e)}
    
    def _select_optimal_settlement(self, sector: Dict[str, Any], settlements: List[Settlement]) -> Settlement:
        """Select the most suitable settlement for a sector."""
        # Simple heuristic: prefer larger settlements for industrial sectors
        sector_type = sector.get('name', '').lower()
        
        if any(word in sector_type for word in ['industrial', 'manufacturing', 'heavy']):
            # Prefer cities for industrial sectors
            cities = [s for s in settlements if s.settlement_type.value == 'city']
            if cities:
                return max(cities, key=lambda s: s.population)
        
        # Default: select settlement with highest economic importance
        return max(settlements, key=lambda s: s.economic_importance)
    
    def _calculate_production_capacity(self, sector: Dict[str, Any], settlement: Settlement) -> float:
        """Calculate production capacity based on settlement characteristics."""
        base_capacity = settlement.population / 1000.0  # Base capacity per 1000 people
        
        # Adjust based on settlement type
        if settlement.settlement_type.value == 'city':
            base_capacity *= 2.0
        elif settlement.settlement_type.value == 'town':
            base_capacity *= 1.5
        
        # Adjust based on economic importance
        base_capacity *= (1.0 + settlement.economic_importance)
        
        return base_capacity
    
    def _calculate_terrain_suitability(self, sector: Dict[str, Any], settlement: Settlement) -> float:
        """Calculate terrain suitability for a sector."""
        # Get terrain type at settlement location
        tile = self.unified_system.map_simulator.map_tiles.get((settlement.x, settlement.y))
        if not tile:
            return 0.5  # Default suitability
        
        sector_type = sector.get('name', '').lower()
        terrain_type = tile.terrain_type.value
        
        # Terrain suitability matrix
        suitability_matrix = {
            'flatland': {'agricultural': 1.0, 'industrial': 0.9, 'service': 0.8},
            'forest': {'agricultural': 0.6, 'industrial': 0.7, 'service': 0.5},
            'mountain': {'agricultural': 0.3, 'industrial': 0.8, 'service': 0.4},
            'water': {'agricultural': 0.2, 'industrial': 0.4, 'service': 0.9},
            'coastal': {'agricultural': 0.7, 'industrial': 0.9, 'service': 1.0}
        }
        
        # Determine sector category
        if any(word in sector_type for word in ['agricultural', 'farming', 'food']):
            category = 'agricultural'
        elif any(word in sector_type for word in ['industrial', 'manufacturing', 'mining']):
            category = 'industrial'
        else:
            category = 'service'
        
        return suitability_matrix.get(terrain_type, {}).get(category, 0.5)
    
    def _calculate_infrastructure_access(self, settlement: Settlement) -> float:
        """Calculate infrastructure access for a settlement."""
        # Count infrastructure connections
        connections = len(settlement.infrastructure_connections)
        
        # Base access from connections
        access = min(1.0, connections / 5.0)  # Normalize to 0-1
        
        # Boost for cities (better infrastructure)
        if settlement.settlement_type.value == 'city':
            access = min(1.0, access + 0.3)
        
        return access
    
    def _calculate_resource_availability(self, sector: Dict[str, Any], settlement: Settlement) -> float:
        """Calculate resource availability for a sector."""
        # Get tile resource level
        tile = self.unified_system.map_simulator.map_tiles.get((settlement.x, settlement.y))
        if not tile:
            return 0.5
        
        # Base resource availability
        resource_availability = tile.resource_level
        
        # Adjust based on settlement size (larger settlements have better resource access)
        resource_availability *= (1.0 + settlement.population / 100000.0)
        
        return min(1.0, resource_availability)
    
    def _calculate_logistics_multiplier(self, settlement: Settlement) -> float:
        """Calculate logistics cost multiplier for a settlement."""
        # More connections = lower logistics costs
        connections = len(settlement.infrastructure_connections)
        multiplier = max(0.5, 1.0 - connections * 0.1)
        
        # Cities have better logistics
        if settlement.settlement_type.value == 'city':
            multiplier *= 0.8
        
        return multiplier
    
    def _calculate_disaster_vulnerability(self, settlement: Settlement) -> float:
        """Calculate disaster vulnerability for a settlement."""
        # Larger settlements are more vulnerable
        vulnerability = min(1.0, settlement.population / 500000.0)
        
        # Cities are more vulnerable
        if settlement.settlement_type.value == 'city':
            vulnerability *= 1.2
        
        return min(1.0, vulnerability)

class UnifiedSimulationSystem:
    """
    Main unified simulation system that merges spatial and economic simulation.
    
    This system coordinates:
    - Map-based spatial simulation (terrain, settlements, infrastructure)
    - Dynamic economic simulation (sectors, capital accumulation, planning)
    - Bidirectional feedback between spatial and economic systems
    - Unified time management with different update frequencies
    - Comprehensive reporting and analysis
    """
    
    def __init__(self, config: Optional[UnifiedSimulationConfig] = None):
        """Initialize the unified simulation system."""
        self.config = config or UnifiedSimulationConfig()
        self.state = UnifiedSimulationState()
        
        # Initialize subsystems
        self.map_simulator: Optional[MapBasedSimulator] = None
        self.economic_simulator: Optional[EnhancedEconomicSimulation] = None
        self.dynamic_planner: Optional[DynamicPlanner] = None
        self.planning_system: Optional[CyberneticPlanningSystem] = None
        
        # Integration components
        self.spatial_economic_integration = SpatialEconomicIntegration(self)
        
        # Simulation state
        self.is_initialized = False
        self.is_running = False
        self.simulation_results = []
        
        # Performance tracking
        self.performance_metrics = {
            "initialization_time": 0.0,
            "total_simulation_time": 0.0,
            "average_step_time": 0.0,
            "memory_usage_mb": 0.0
        }
        
        logger.info("Unified simulation system initialized")
    
    def create_unified_simulation(self, 
                                map_params: Optional[Dict[str, Any]] = None,
                                economic_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a unified simulation with both spatial and economic components.
        
        Args:
            map_params: Parameters for map generation
            economic_params: Parameters for economic simulation
        
        Returns:
            Dictionary with creation results
        """
        try:
            start_time = datetime.now()
            
            # Update config with provided parameters
            if map_params:
                self._update_map_config(map_params)
            if economic_params:
                self._update_economic_config(economic_params)
            
            # Create planning system
            logger.info("Creating cybernetic planning system...")
            self.planning_system = CyberneticPlanningSystem()
            
            # Create synthetic economic data
            logger.info("Creating synthetic economic data...")
            data_result = self.planning_system.create_synthetic_data(
                n_sectors=self.config.n_sectors,
                technology_density=self.config.technology_density,
                resource_count=self.config.resource_count
            )
            
            if not data_result.get("success", False):
                return {"success": False, "error": f"Failed to create synthetic data: {data_result.get('error', 'Unknown error')}"}
            
            # Extract the actual data from the result
            synthetic_data = data_result.get("data", {})
            
            # Create economic plan
            logger.info("Creating economic plan...")
            plan_result = self.planning_system.create_plan(
                policy_goals=self.config.policy_goals,
                use_optimization=True
            )
            
            if not plan_result.get("success", False):
                return {"success": False, "error": f"Failed to create economic plan: {plan_result.get('error', 'Unknown error')}"}
            
            # Extract the actual plan from the result
            economic_plan = plan_result.get("plan", {})
            
            self.state.economic_plan_active = True
            self.state.sectors_active = len(economic_plan.get('sectors', []))
            self.state.total_economic_output = economic_plan.get('total_economic_output', 0.0)
            
            # Create map-based simulation
            logger.info("Creating map-based simulation...")
            map_result = self.planning_system.create_map_based_simulation(
                map_width=self.config.map_width,
                map_height=self.config.map_height,
                terrain_distribution=self.config.terrain_distribution,
                num_cities=self.config.num_cities,
                num_towns=self.config.num_towns,
                total_population=self.config.total_population,
                rural_population_percent=self.config.rural_population_percent,
                urban_concentration=self.config.urban_concentration
            )
            
            if not map_result.get("success", False):
                return {"success": False, "error": f"Failed to create map simulation: {map_result.get('error', 'Unknown error')}"}
            
            self.map_simulator = self.planning_system.map_simulator
            self.state.map_generated = True
            self.state.settlements_active = len(self.map_simulator.settlements)
            self.state.infrastructure_segments = len(self.map_simulator.infrastructure_segments)
            
            
            # Set disaster probability
            self.map_simulator.disaster_probability = self.config.disaster_probability
            
            # Integrate spatial and economic systems
            logger.info("Integrating spatial and economic systems...")
            integration_result = self.spatial_economic_integration.map_economic_activity_to_spatial(economic_plan)
            
            if not integration_result.get("success", False):
                return {"success": False, "error": f"Failed to integrate systems: {integration_result.get('error', 'Unknown error')}"}
            
            # Initialize economic simulation components
            self._initialize_economic_simulation(economic_plan, synthetic_data)
            
            # Mark as initialized
            self.is_initialized = True
            
            initialization_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["initialization_time"] = initialization_time
            
            logger.info(f"Unified simulation created successfully in {initialization_time:.2f} seconds")
            
            return {
                "success": True,
                "message": "Unified simulation created successfully",
                "initialization_time": initialization_time,
                "spatial_summary": {
                    "map_dimensions": (self.config.map_width, self.config.map_height),
                    "settlements": self.state.settlements_active,
                    "infrastructure_segments": self.state.infrastructure_segments,
                    "terrain_distribution": self.config.terrain_distribution
                },
                "economic_summary": {
                    "sectors": self.state.sectors_active,
                    "total_output": self.state.total_economic_output,
                    "policy_goals": self.config.policy_goals
                },
                "integration_summary": {
                    "sector_settlement_mappings": len(self.spatial_economic_integration.sector_settlement_mappings),
                    "average_production_capacity": integration_result.get("average_production_capacity", 0.0),
                    "average_terrain_suitability": integration_result.get("average_terrain_suitability", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to create unified simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def run_unified_simulation(self, 
                             duration_months: Optional[int] = None,
                             spatial_update_frequency: Optional[str] = None,
                             economic_update_frequency: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the unified simulation for specified duration.
        
        Args:
            duration_months: Duration in months (uses config if None)
            spatial_update_frequency: Spatial update frequency (uses config if None)
            economic_update_frequency: Economic update frequency (uses config if None)
        
        Returns:
            Dictionary with simulation results
        """
        if not self.is_initialized:
            return {"success": False, "error": "Simulation not initialized. Please create unified simulation first."}
        
        try:
            # Update configuration
            if duration_months:
                self.config.simulation_duration_months = duration_months
            if spatial_update_frequency:
                self.config.spatial_update_frequency = spatial_update_frequency
            if economic_update_frequency:
                self.config.economic_update_frequency = economic_update_frequency
            
            logger.info(f"Starting unified simulation for {self.config.simulation_duration_months} months")
            logger.info(f"Spatial updates: {self.config.spatial_update_frequency}")
            logger.info(f"Economic updates: {self.config.economic_update_frequency}")
            
            start_time = datetime.now()
            self.is_running = True
            self.simulation_results = []
            
            # Calculate total days
            total_days = self.config.simulation_duration_months * 30
            
            # Run simulation loop
            for day in range(total_days):
                step_start_time = datetime.now()
                
                # Update simulation state
                self.state.current_day = day
                self.state.current_date = self.state.simulation_start_time + timedelta(days=day)
                self.state.current_month = (day // 30) + 1
                self.state.current_year = self.state.simulation_start_time.year + (day // 365)
                
                # Determine what to update this step
                should_update_spatial = self._should_update_spatial(day)
                should_update_economic = self._should_update_economic(day)
                
                # Run unified simulation step
                step_result = self._unified_simulation_step(
                    day=day,
                    update_spatial=should_update_spatial,
                    update_economic=should_update_economic
                )
                
                # Store results
                self.simulation_results.append(step_result)
                
                # Update performance metrics
                step_time = (datetime.now() - step_start_time).total_seconds()
                self.state.step_times.append(step_time)
                
                # Log progress
                if day % 30 == 0:  # Every month
                    month = day // 30 + 1
                    logger.info(f"Completed month {month}/{self.config.simulation_duration_months}")
            
            # Calculate final metrics
            total_simulation_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["total_simulation_time"] = total_simulation_time
            self.performance_metrics["average_step_time"] = np.mean(self.state.step_times) if self.state.step_times else 0.0
            
            self.is_running = False
            
            logger.info(f"Unified simulation completed in {total_simulation_time:.2f} seconds")
            
            return {
                "success": True,
                "simulation_duration_days": total_days,
                "simulation_duration_months": self.config.simulation_duration_months,
                "total_simulation_time": total_simulation_time,
                "average_step_time": self.performance_metrics["average_step_time"],
                "final_state": self._get_final_state_summary(),
                "performance_metrics": self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            self.is_running = False
            return {"success": False, "error": str(e)}
    
    def generate_unified_report(self) -> Dict[str, Any]:
        """Generate comprehensive report combining spatial and economic metrics."""
        if not self.simulation_results:
            return {"success": False, "error": "No simulation results available"}
        
        try:
            # Extract metrics from simulation results
            spatial_metrics = self._extract_spatial_metrics()
            economic_metrics = self._extract_economic_metrics()
            integration_metrics = self._extract_integration_metrics()
            
            # Calculate efficiency analysis
            efficiency_analysis = self._analyze_spatial_economic_efficiency()
            
            # Generate comprehensive report
            report = {
                "success": True,
                "report_generated_at": datetime.now().isoformat(),
                "simulation_summary": {
                    "duration_months": self.config.simulation_duration_months,
                    "total_days": len(self.simulation_results),
                    "spatial_updates": self.config.spatial_update_frequency,
                    "economic_updates": self.config.economic_update_frequency
                },
                "spatial_metrics": spatial_metrics,
                "economic_metrics": economic_metrics,
                "integration_metrics": integration_metrics,
                "efficiency_analysis": efficiency_analysis,
                "performance_metrics": self.performance_metrics,
                "recommendations": self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate unified report: {e}")
            return {"success": False, "error": str(e)}
    
    def _unified_simulation_step(self, day: int, update_spatial: bool, update_economic: bool) -> Dict[str, Any]:
        """Execute one unified simulation step."""
        step_result = {
            "day": day,
            "date": self.state.current_date.isoformat(),
            "update_spatial": update_spatial,
            "update_economic": update_economic,
            "spatial_metrics": {},
            "economic_metrics": {},
            "integration_metrics": {},
            "disaster_events": {"new_disasters": [], "ongoing_disasters": []},
            "feedback_effects": {}
        }
        
        # Update spatial systems
        if update_spatial and self.map_simulator:
            spatial_result = self.map_simulator.simulate_time_step()
            step_result["spatial_metrics"] = {
                "logistics_friction": spatial_result.get("total_logistics_friction", 0.0),
                "active_disasters": spatial_result.get("active_disasters", 0),
                "settlements_count": spatial_result.get("settlements_count", 0),
                "infrastructure_segments": spatial_result.get("infrastructure_segments", 0)
            }
            
            # Update state
            self.state.logistics_friction_total = spatial_result.get("total_logistics_friction", 0.0)
            self.state.active_disasters = spatial_result.get("active_disasters", 0)
            
            # Handle disaster events
            disaster_events = spatial_result.get("disaster_events", {})
            step_result["disaster_events"] = disaster_events
            
            # Calculate disaster economic impact
            if disaster_events.get("new_disasters"):
                for disaster in disaster_events["new_disasters"]:
                    impact_result = self.spatial_economic_integration.simulate_disaster_impact(disaster)
                    if impact_result.get("success"):
                        step_result["feedback_effects"]["disaster_economic_impact"] = impact_result.get("total_economic_impact", 0.0)
                        self.state.disaster_impact_on_economy += impact_result.get("total_economic_impact", 0.0)
        
        # Update economic systems
        if update_economic and self.economic_simulator:
            # Calculate spatial constraints for economic planning
            if self.spatial_economic_integration.sector_settlement_mappings:
                production_targets = {mapping.sector_id: 1000.0 for mapping in self.spatial_economic_integration.sector_settlement_mappings.values()}
                spatial_costs = self.spatial_economic_integration.calculate_spatial_economic_costs(production_targets)
                
                if spatial_costs.get("success"):
                    step_result["integration_metrics"]["spatial_economic_costs"] = spatial_costs.get("total_spatial_cost", 0.0)
                    step_result["integration_metrics"]["average_spatial_efficiency"] = spatial_costs.get("average_spatial_efficiency", 0.0)
            
            # Run economic simulation step
            economic_result = self._run_economic_step()
            step_result["economic_metrics"] = economic_result
            
            # Update state
            self.state.total_economic_output = economic_result.get("total_economic_output", self.state.total_economic_output)
            self.state.capital_stock_total = economic_result.get("total_capital_stock", self.state.capital_stock_total)
        
        # Calculate integration metrics
        step_result["integration_metrics"]["spatial_economic_efficiency"] = self._calculate_integration_efficiency()
        
        return step_result
    
    def _should_update_spatial(self, day: int) -> bool:
        """Determine if spatial systems should be updated this day."""
        if self.config.spatial_update_frequency == "daily":
            return True
        elif self.config.spatial_update_frequency == "weekly":
            return day % 7 == 0
        elif self.config.spatial_update_frequency == "monthly":
            return day % 30 == 0
        return False
    
    def _should_update_economic(self, day: int) -> bool:
        """Determine if economic systems should be updated this day."""
        if self.config.economic_update_frequency == "daily":
            return True
        elif self.config.economic_update_frequency == "weekly":
            return day % 7 == 0
        elif self.config.economic_update_frequency == "monthly":
            return day % 30 == 0
        elif self.config.economic_update_frequency == "quarterly":
            return day % 90 == 0
        elif self.config.economic_update_frequency == "annual":
            return day % 365 == 0
        return False
    
    def _run_economic_step(self) -> Dict[str, Any]:
        """Run one economic simulation step."""
        if not self.economic_simulator:
            # Fallback if no economic simulator is available
            return {
                "total_economic_output": self.state.total_economic_output * (1.0 + np.random.normal(0, 0.01)),
                "total_capital_stock": self.state.capital_stock_total * (1.0 + np.random.normal(0, 0.005)),
                "sectors_active": self.state.sectors_active,
                "plan_fulfillment_rate": 0.85 + np.random.normal(0, 0.05)
            }
        
        try:
            # Run actual economic simulation for current month
            month = self.state.current_month
            economic_result = self.economic_simulator.simulate_month(
                month=month,
                population_health_tracker=None,
                use_optimization=True
            )
            
            # Extract key metrics from the economic result
            metrics = economic_result.get('metrics', {})
            if hasattr(metrics, 'total_economic_output'):
                total_economic_output = metrics.total_economic_output
            else:
                total_economic_output = economic_result.get('total_economic_output', 0.0)
            
            # Calculate sectors active from production data
            production = economic_result.get('production', {})
            sectors_active = len(production) if production else self.state.sectors_active
            
            # Use capital stock from state or calculate from output
            total_capital_stock = self.state.capital_stock_total
            if total_economic_output > 0:
                total_capital_stock = total_economic_output * 0.1  # 10% of output as capital stock
            
            # Calculate plan fulfillment rate from production efficiency
            if hasattr(metrics, 'average_efficiency'):
                plan_fulfillment_rate = min(metrics.average_efficiency, 1.0)
            else:
                plan_fulfillment_rate = 0.85
            
            return {
                "total_economic_output": total_economic_output,
                "total_capital_stock": total_capital_stock,
                "sectors_active": sectors_active,
                "plan_fulfillment_rate": plan_fulfillment_rate,
                "economic_metrics": economic_result
            }
            
        except Exception as e:
            logger.error(f"Economic simulation step failed: {e}")
            # Fallback to simple growth model
            return {
                "total_economic_output": self.state.total_economic_output * (1.0 + np.random.normal(0, 0.01)),
                "total_capital_stock": self.state.capital_stock_total * (1.0 + np.random.normal(0, 0.005)),
                "sectors_active": self.state.sectors_active,
                "plan_fulfillment_rate": 0.85 + np.random.normal(0, 0.05)
            }
    
    def _calculate_integration_efficiency(self) -> float:
        """Calculate overall spatial-economic integration efficiency."""
        if not self.spatial_economic_integration.sector_settlement_mappings:
            return 0.0
        
        # Calculate average efficiency from mappings
        efficiencies = []
        for mapping in self.spatial_economic_integration.sector_settlement_mappings.values():
            efficiency = (mapping.terrain_suitability * 0.3 + 
                         mapping.infrastructure_access * 0.3 + 
                         mapping.resource_availability * 0.2 + 
                         (1.0 / mapping.logistics_cost_multiplier) * 0.2)
            efficiencies.append(efficiency)
        
        return np.mean(efficiencies) if efficiencies else 0.0
    
    def _update_map_config(self, map_params: Dict[str, Any]):
        """Update map configuration with provided parameters."""
        for key, value in map_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _update_economic_config(self, economic_params: Dict[str, Any]):
        """Update economic configuration with provided parameters."""
        for key, value in economic_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _initialize_economic_simulation(self, plan_result: Dict[str, Any], synthetic_data: Dict[str, Any]):
        """Initialize economic simulation components."""
        try:
            logger.info("Starting economic simulation initialization...")
            
            # Use synthetic data directly since plan_result doesn't have economic_data
            economic_data = synthetic_data
            try:
                data_keys = list(economic_data.keys()) if economic_data is not None else 'Empty'
            except:
                data_keys = 'Error'
            logger.info(f"Using synthetic data: {data_keys}")
            
            # Get matrices from economic data
            technology_matrix = economic_data.get('technology_matrix')
            labor_vector = economic_data.get('labor_input')
            if labor_vector is None:
                labor_vector = economic_data.get('labor_vector')
            final_demand = economic_data.get('final_demand')
            resource_matrix = economic_data.get('resource_matrix')
            max_resources = economic_data.get('max_resources')
            sector_names = economic_data.get('sectors')
            
            # Debug logging
            try:
                tech_shape = getattr(technology_matrix, 'shape', 'N/A') if technology_matrix is not None else 'N/A'
            except:
                tech_shape = 'Error'
            logger.info(f"Economic simulation init - technology_matrix: {type(technology_matrix)}, shape: {tech_shape}")
            
            try:
                labor_shape = getattr(labor_vector, 'shape', 'N/A') if labor_vector is not None else 'N/A'
            except:
                labor_shape = 'Error'
            logger.info(f"Economic simulation init - labor_vector: {type(labor_vector)}, shape: {labor_shape}")
            
            try:
                demand_shape = getattr(final_demand, 'shape', 'N/A') if final_demand is not None else 'N/A'
            except:
                demand_shape = 'Error'
            logger.info(f"Economic simulation init - final_demand: {type(final_demand)}, shape: {demand_shape}")
            
            try:
                sector_length = len(sector_names) if sector_names is not None else 'N/A'
            except:
                sector_length = 'Error'
            logger.info(f"Economic simulation init - sector_names: {type(sector_names)}, length: {sector_length}")
            
            # Simple validation - just check if the required data exists
            logger.info("Checking data validity...")
            
            # Check each variable individually to avoid numpy array boolean evaluation issues
            tech_matrix_valid = technology_matrix is not None
            labor_vector_valid = labor_vector is not None
            final_demand_valid = final_demand is not None
            sector_names_valid = sector_names is not None
            
            logger.info(f"technology_matrix is not None: {tech_matrix_valid}")
            logger.info(f"labor_vector is not None: {labor_vector_valid}")
            logger.info(f"final_demand is not None: {final_demand_valid}")
            logger.info(f"sector_names is not None: {sector_names_valid}")
            
            if (tech_matrix_valid and 
                labor_vector_valid and 
                final_demand_valid and 
                sector_names_valid):
                
                logger.info("All required data present, creating economic simulation...")
                
                try:
                    # Create the enhanced economic simulation
                    logger.info("About to create EnhancedEconomicSimulation...")
                    self.economic_simulator = EnhancedEconomicSimulation(
                        technology_matrix=technology_matrix,
                        labor_vector=labor_vector,
                        final_demand=final_demand,
                        resource_matrix=resource_matrix,
                        max_resources=max_resources,
                        sector_names=sector_names
                    )
                    logger.info("EnhancedEconomicSimulation created successfully")
                except Exception as e:
                    logger.error(f"Error creating EnhancedEconomicSimulation: {e}")
                    raise
                
                # Set initial economic output from the plan
                economic_plan = plan_result.get('plan', {})
                if 'total_economic_output' in economic_plan:
                    self.current_economic_output = economic_plan['total_economic_output']
                else:
                    self.current_economic_output = 0.0
                
                logger.info("Economic simulation initialized successfully")
                return True
            else:
                logger.error("Missing required data for economic simulation")
                logger.error(f"tech_matrix: {technology_matrix is not None}, labor: {labor_vector is not None}, demand: {final_demand is not None}, sectors: {sector_names is not None}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize economic simulation: {e}")
            return False
    
    def _get_final_state_summary(self) -> Dict[str, Any]:
        """Get summary of final simulation state."""
        return {
            "spatial_state": {
                "map_generated": self.state.map_generated,
                "settlements_active": self.state.settlements_active,
                "infrastructure_segments": self.state.infrastructure_segments,
                "active_disasters": self.state.active_disasters,
                "logistics_friction_total": self.state.logistics_friction_total
            },
            "economic_state": {
                "economic_plan_active": self.state.economic_plan_active,
                "sectors_active": self.state.sectors_active,
                "total_economic_output": self.state.total_economic_output,
                "capital_stock_total": self.state.capital_stock_total
            },
            "integration_state": {
                "spatial_economic_efficiency": self.state.spatial_economic_efficiency,
                "disaster_impact_on_economy": self.state.disaster_impact_on_economy,
                "sector_settlement_mappings": len(self.spatial_economic_integration.sector_settlement_mappings)
            }
        }
    
    def _extract_spatial_metrics(self) -> Dict[str, Any]:
        """Extract spatial metrics from simulation results."""
        if not self.simulation_results:
            return {}
        
        logistics_frictions = [r.get("spatial_metrics", {}).get("logistics_friction", 0.0) for r in self.simulation_results]
        disaster_counts = [r.get("spatial_metrics", {}).get("active_disasters", 0) for r in self.simulation_results]
        
        return {
            "average_logistics_friction": np.mean(logistics_frictions) if logistics_frictions else 0.0,
            "max_logistics_friction": np.max(logistics_frictions) if logistics_frictions else 0.0,
            "total_disasters": sum(len(r.get("disaster_events", {}).get("new_disasters", [])) for r in self.simulation_results),
            "average_active_disasters": np.mean(disaster_counts) if disaster_counts else 0.0,
            "infrastructure_stability": 1.0 - (np.std(logistics_frictions) / np.mean(logistics_frictions)) if logistics_frictions and np.mean(logistics_frictions) > 0 else 1.0
        }
    
    def _extract_economic_metrics(self) -> Dict[str, Any]:
        """Extract economic metrics from simulation results."""
        if not self.simulation_results:
            return {}
        
        economic_outputs = [r.get("economic_metrics", {}).get("total_economic_output", 0.0) for r in self.simulation_results]
        capital_stocks = [r.get("economic_metrics", {}).get("total_capital_stock", 0.0) for r in self.simulation_results]
        
        return {
            "average_economic_output": np.mean(economic_outputs) if economic_outputs else 0.0,
            "final_economic_output": economic_outputs[-1] if economic_outputs else 0.0,
            "economic_growth_rate": ((economic_outputs[-1] - economic_outputs[0]) / economic_outputs[0]) if economic_outputs and economic_outputs[0] > 0 else 0.0,
            "average_capital_stock": np.mean(capital_stocks) if capital_stocks else 0.0,
            "capital_accumulation_rate": ((capital_stocks[-1] - capital_stocks[0]) / capital_stocks[0]) if capital_stocks and capital_stocks[0] > 0 else 0.0
        }
    
    def _extract_integration_metrics(self) -> Dict[str, Any]:
        """Extract integration metrics from simulation results."""
        if not self.simulation_results:
            return {}
        
        spatial_efficiencies = [r.get("integration_metrics", {}).get("average_spatial_efficiency", 0.0) for r in self.simulation_results]
        spatial_costs = [r.get("integration_metrics", {}).get("spatial_economic_costs", 0.0) for r in self.simulation_results]
        
        return {
            "average_spatial_efficiency": np.mean(spatial_efficiencies) if spatial_efficiencies else 0.0,
            "average_spatial_economic_costs": np.mean(spatial_costs) if spatial_costs else 0.0,
            "integration_stability": 1.0 - (np.std(spatial_efficiencies) / np.mean(spatial_efficiencies)) if spatial_efficiencies and np.mean(spatial_efficiencies) > 0 else 1.0,
            "total_disaster_economic_impact": sum(r.get("feedback_effects", {}).get("disaster_economic_impact", 0.0) for r in self.simulation_results)
        }
    
    def _analyze_spatial_economic_efficiency(self) -> Dict[str, Any]:
        """Analyze spatial-economic efficiency."""
        if not self.spatial_economic_integration.sector_settlement_mappings:
            return {"analysis": "No sector-settlement mappings available"}
        
        mappings = list(self.spatial_economic_integration.sector_settlement_mappings.values())
        
        # Analyze efficiency distribution
        efficiencies = []
        for mapping in mappings:
            efficiency = (mapping.terrain_suitability * 0.3 + 
                         mapping.infrastructure_access * 0.3 + 
                         mapping.resource_availability * 0.2 + 
                         (1.0 / mapping.logistics_cost_multiplier) * 0.2)
            efficiencies.append(efficiency)
        
        return {
            "average_efficiency": np.mean(efficiencies),
            "efficiency_std": np.std(efficiencies),
            "min_efficiency": np.min(efficiencies),
            "max_efficiency": np.max(efficiencies),
            "efficiency_distribution": {
                "high_efficiency": len([e for e in efficiencies if e > 0.8]),
                "medium_efficiency": len([e for e in efficiencies if 0.5 <= e <= 0.8]),
                "low_efficiency": len([e for e in efficiencies if e < 0.5])
            },
            "bottlenecks": self._identify_efficiency_bottlenecks(mappings)
        }
    
    def _identify_efficiency_bottlenecks(self, mappings: List[SpatialEconomicMapping]) -> List[Dict[str, Any]]:
        """Identify efficiency bottlenecks in sector-settlement mappings."""
        bottlenecks = []
        
        for mapping in mappings:
            if mapping.terrain_suitability < 0.5:
                bottlenecks.append({
                    "sector_id": mapping.sector_id,
                    "settlement_id": mapping.settlement_id,
                    "bottleneck_type": "terrain_suitability",
                    "severity": 0.5 - mapping.terrain_suitability,
                    "recommendation": "Consider relocating to more suitable terrain"
                })
            
            if mapping.infrastructure_access < 0.5:
                bottlenecks.append({
                    "sector_id": mapping.sector_id,
                    "settlement_id": mapping.settlement_id,
                    "bottleneck_type": "infrastructure_access",
                    "severity": 0.5 - mapping.infrastructure_access,
                    "recommendation": "Invest in infrastructure development"
                })
            
            if mapping.logistics_cost_multiplier > 1.5:
                bottlenecks.append({
                    "sector_id": mapping.sector_id,
                    "settlement_id": mapping.settlement_id,
                    "bottleneck_type": "logistics_costs",
                    "severity": mapping.logistics_cost_multiplier - 1.0,
                    "recommendation": "Improve transportation connections"
                })
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        # Analyze spatial metrics
        spatial_metrics = self._extract_spatial_metrics()
        if spatial_metrics.get("average_logistics_friction", 0) > 1000:
            recommendations.append("High logistics friction detected. Consider infrastructure investment.")
        
        if spatial_metrics.get("total_disasters", 0) > 10:
            recommendations.append("Frequent disasters affecting economy. Implement disaster resilience measures.")
        
        # Analyze economic metrics
        economic_metrics = self._extract_economic_metrics()
        if economic_metrics.get("economic_growth_rate", 0) < 0.02:
            recommendations.append("Low economic growth rate. Review sector allocation and investment priorities.")
        
        # Analyze integration metrics
        integration_metrics = self._extract_integration_metrics()
        if integration_metrics.get("average_spatial_efficiency", 0) < 0.6:
            recommendations.append("Low spatial-economic efficiency. Optimize sector-settlement mappings.")
        
        return recommendations
