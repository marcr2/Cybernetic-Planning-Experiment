"""
Transportation and Distribution Integration

This module integrates transportation costs, stockpile management, and supply chain
optimization into the main economic planning system.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import transportation and distribution systems
from ..data.resource_flow_model import (
    ResourceFlowModel, TransportationCostCalculator, 
    ResourceDefinition, ResourceType
)
from ..utils.transportation_system import (
    TransportationSystem, Location, TransportMode, VehicleType
)
from ..utils.regional_stockpiles import (
    StockpileManager, StockpileFacility, StockpileType, StorageZone
)
from ..utils.cargo_distribution import (
    SupplyChainOptimizer, SupplyNode, SupplyDemand, CargoCategory
)
from ..utils.simulation_integration import EconomicSimulation, SimulationConfig


@dataclass
class TransportationIntegrationConfig:
    """Configuration for transportation and distribution integration."""
    enable_transportation_costs: bool = True
    enable_stockpile_management: bool = True
    enable_supply_chain_optimization: bool = True
    enable_real_time_updates: bool = True
    transportation_cost_weight: float = 0.3  # Weight in economic optimization
    stockpile_cost_weight: float = 0.1
    supply_chain_efficiency_weight: float = 0.2


class TransportationIntegration:
    """
    Main integration class for transportation, stockpiles, and supply chain.
    
    This class coordinates all transportation and distribution systems with
    the economic planning system to provide comprehensive resource flow analysis.
    """
    
    def __init__(self, config: TransportationIntegrationConfig = None):
        """Initialize the transportation integration system."""
        self.config = config or TransportationIntegrationConfig()
        
        # Initialize subsystems
        self.resource_flow_model = None
        self.transportation_system = None
        self.stockpile_manager = None
        self.supply_chain_optimizer = None
        self.economic_simulation = None
        
        # Integration state
        self.is_initialized = False
        self.current_locations = {}
        self.current_resource_flows = {}
        self.transportation_costs = {}
        self.stockpile_status = {}
        
        # Performance tracking
        self.integration_metrics = {
            "total_transportation_cost": 0.0,
            "total_stockpile_cost": 0.0,
            "supply_chain_efficiency": 0.0,
            "last_update": None
        }
    
    def initialize(self, 
                   sector_data: Dict[str, Any],
                   resource_data: Dict[str, Any],
                   location_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize the transportation integration system.
        
        Args:
            sector_data: Sector definitions and characteristics
            resource_data: Resource definitions and consumption patterns
            location_data: Geographic locations and infrastructure data
            
        Returns:
            Initialization results and status
        """
        try:
            # Initialize resource flow model
            self.resource_flow_model = ResourceFlowModel()
            self.resource_flow_model.load_from_data(sector_data, resource_data)
            
            # Initialize transportation system
            self.transportation_system = TransportationSystem()
            self._initialize_locations(location_data)
            
            # Initialize stockpile management
            if self.config.enable_stockpile_management:
                self.stockpile_manager = StockpileManager()
                self._initialize_stockpiles()
            
            # Initialize supply chain optimization
            if self.config.enable_supply_chain_optimization:
                self.supply_chain_optimizer = SupplyChainOptimizer()
                self._initialize_supply_chain()
            
            # Initialize economic simulation
            sim_config = SimulationConfig(
                starting_year=2024,
                num_regions=len(self.current_locations),
                debug_mode=True
            )
            self.economic_simulation = EconomicSimulation(sim_config)
            self.economic_simulation.initialize()
            
            self.is_initialized = True
            self.integration_metrics["last_update"] = datetime.now()
            
            return {
                "status": "success",
                "initialized_systems": [
                    "resource_flow_model",
                    "transportation_system",
                    "stockpile_manager" if self.config.enable_stockpile_management else None,
                    "supply_chain_optimizer" if self.config.enable_supply_chain_optimization else None,
                    "economic_simulation"
                ],
                "locations_created": len(self.current_locations),
                "resources_loaded": len(self.resource_flow_model.resources),
                "sectors_loaded": len(self.resource_flow_model.sector_names)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "initialized_systems": []
            }
    
    def _initialize_locations(self, location_data: Optional[Dict[str, Any]] = None):
        """Initialize geographic locations for transportation."""
        if location_data:
            # Use provided location data
            for loc_id, loc_info in location_data.items():
                location = Location(
                    id=loc_id,
                    name=loc_info.get("name", f"Location_{loc_id}"),
                    lat=loc_info.get("lat", 0.0),
                    lon=loc_info.get("lon", 0.0),
                    properties=loc_info.get("properties", {})
                )
                self.transportation_system.add_location(location)
                self.current_locations[loc_id] = location
        else:
            # Create default locations
            default_locations = [
                {"id": "NYC", "name": "New York City", "lat": 40.7128, "lon": -74.0060, "population": 8000000},
                {"id": "LA", "name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "population": 4000000},
                {"id": "CHI", "name": "Chicago", "lat": 41.8781, "lon": -87.6298, "population": 2700000},
                {"id": "HOU", "name": "Houston", "lat": 29.7604, "lon": -95.3698, "population": 2300000},
                {"id": "PHX", "name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "population": 1600000}
            ]
            
            for loc_info in default_locations:
                location = Location(
                    id=loc_info["id"],
                    name=loc_info["name"],
                    lat=loc_info["lat"],
                    lon=loc_info["lon"],
                    properties={"population": loc_info["population"]}
                )
                self.transportation_system.add_location(location)
                self.current_locations[loc_info["id"]] = location
    
    def _initialize_stockpiles(self):
        """Initialize stockpile facilities."""
        if not self.stockpile_manager:
            return
            
        # Create stockpile facilities in major locations
        major_locations = sorted(
            self.current_locations.values(),
            key=lambda x: x.properties.get('population', 0),
            reverse=True
        )[:5]  # Top 5 locations
        
        for i, location in enumerate(major_locations):
            # Create storage zones
            zones = [
                StorageZone(f"ZONE_{i}_01", "ambient", 5000),
                StorageZone(f"ZONE_{i}_02", "refrigerated", 2000),
                StorageZone(f"ZONE_{i}_03", "controlled_atmosphere", 1000)
            ]
            
            facility = StockpileFacility(
                facility_id=f"FACILITY_{i:03d}",
                name=f"{location.name} Regional Stockpile",
                location=(location.lat, location.lon),
                facility_type=StockpileType.DISTRIBUTION_CENTER,
                storage_zones=zones,
                total_capacity=8000,
                current_inventory={}
            )
            
            self.stockpile_manager.add_facility(facility)
    
    def _initialize_supply_chain(self):
        """Initialize supply chain optimization."""
        if not self.supply_chain_optimizer:
            return
            
        # Create supply nodes from locations
        for location in self.current_locations.values():
            population = location.properties.get('population', 50000)
            
            if population > 100000:
                node_type = "distribution_center"
                capacity = 50000
            elif population > 50000:
                node_type = "warehouse"
                capacity = 20000
            else:
                node_type = "consumer"
                capacity = 5000
            
            supply_node = SupplyNode(
                location=location,
                node_type=node_type,
                capacity=capacity,
                current_inventory={},
                storage_cost=0.1,
                handling_capacity={
                    CargoCategory.CONSUMER: capacity * 0.8,
                    CargoCategory.ESSENTIAL: capacity * 0.2
                }
            )
            
            # Add to supply chain optimizer (would need to modify SupplyChainOptimizer)
            # This is a placeholder for the actual integration
    
    def calculate_comprehensive_costs(self, 
                                    sector_outputs: np.ndarray,
                                    resource_flows: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive costs including transportation, stockpiles, and supply chain.
        
        Args:
            sector_outputs: Sector output levels
            resource_flows: Resource flow requirements (if not provided, will be calculated)
            
        Returns:
            Comprehensive cost analysis
        """
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        results = {
            "transportation_costs": {},
            "stockpile_costs": {},
            "supply_chain_costs": {},
            "total_costs": {},
            "optimization_recommendations": []
        }
        
        # Calculate resource flows if not provided
        if resource_flows is None:
            resource_flows = self._calculate_resource_flows(sector_outputs)
        
        # Calculate transportation costs
        if self.config.enable_transportation_costs:
            results["transportation_costs"] = self._calculate_transportation_costs(resource_flows)
        
        # Calculate stockpile costs
        if self.config.enable_stockpile_management:
            results["stockpile_costs"] = self._calculate_stockpile_costs(resource_flows)
        
        # Calculate supply chain costs
        if self.config.enable_supply_chain_optimization:
            results["supply_chain_costs"] = self._calculate_supply_chain_costs(resource_flows)
        
        # Calculate total costs
        results["total_costs"] = self._calculate_total_costs(results)
        
        # Generate optimization recommendations
        results["optimization_recommendations"] = self._generate_optimization_recommendations(results)
        
        # Update integration metrics
        self._update_integration_metrics(results)
        
        return results
    
    def _calculate_resource_flows(self, sector_outputs: np.ndarray) -> Dict[Tuple[str, str], float]:
        """Calculate resource flows based on sector outputs."""
        # This would integrate with the resource flow model
        # For now, return a simplified version
        flows = {}
        
        for sector_idx, output in enumerate(sector_outputs):
            if output > 0:
                # Calculate resource requirements for this sector
                resource_requirements = self.resource_flow_model.calculate_total_resource_demand(
                    np.array([output])
                )
                
                # Create flows to major locations
                for resource_id, demand in resource_requirements.items():
                    for location_id in self.current_locations.keys():
                        flows[(resource_id, location_id)] = demand * 0.2  # Simplified distribution
        
        return flows
    
    def _calculate_transportation_costs(self, resource_flows: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
        """Calculate transportation costs for resource flows."""
        if not self.transportation_system:
            return {}
        
        costs = {}
        total_cost = 0.0
        
        for (resource_id, destination_id), quantity in resource_flows.items():
            if resource_id not in self.resource_flow_model.resources:
                continue
            if destination_id not in self.current_locations:
                continue
            
            resource = self.resource_flow_model.resources[resource_id]
            destination = self.current_locations[destination_id]
            
            # Find origin (producing sector location)
            origin_id = resource.producing_sector_id
            if origin_id not in self.current_locations:
                continue
            origin = self.current_locations[origin_id]
            
            # Calculate costs for different transport modes
            transport_costs = {}
            for transport_mode in TransportMode:
                cost_details = self.resource_flow_model.transportation_calculator.calculate_transportation_cost(
                    resource=resource,
                    quantity=quantity,
                    origin=origin,
                    destination=destination,
                    transport_mode=transport_mode
                )
                transport_costs[transport_mode.value] = cost_details
                total_cost += cost_details["total_cost"]
            
            costs[(resource_id, destination_id)] = transport_costs
        
        return {
            "individual_costs": costs,
            "total_transportation_cost": total_cost,
            "cost_breakdown": self._analyze_transportation_costs(costs)
        }
    
    def _calculate_stockpile_costs(self, resource_flows: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
        """Calculate stockpile management costs."""
        if not self.stockpile_manager:
            return {}
        
        # This would integrate with the stockpile manager
        # For now, return a simplified calculation
        total_cost = 0.0
        facility_costs = {}
        
        for facility_id, facility in self.stockpile_manager.facilities.items():
            # Calculate storage costs based on inventory
            storage_cost = facility.total_capacity * 0.1  # $0.10 per unit capacity
            total_cost += storage_cost
            facility_costs[facility_id] = {
                "storage_cost": storage_cost,
                "capacity_utilization": len(facility.current_inventory) / facility.total_capacity,
                "facility_type": facility.facility_type.value
            }
        
        return {
            "total_stockpile_cost": total_cost,
            "facility_costs": facility_costs,
            "cost_breakdown": self._analyze_stockpile_costs(facility_costs)
        }
    
    def _calculate_supply_chain_costs(self, resource_flows: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
        """Calculate supply chain optimization costs."""
        if not self.supply_chain_optimizer:
            return {}
        
        # This would integrate with the supply chain optimizer
        # For now, return a simplified calculation
        total_cost = 0.0
        
        # Calculate distribution costs
        for (resource_id, destination_id), quantity in resource_flows.items():
            # Simplified distribution cost calculation
            distribution_cost = quantity * 0.05  # $0.05 per unit
            total_cost += distribution_cost
        
        return {
            "total_supply_chain_cost": total_cost,
            "distribution_efficiency": 0.85,  # Placeholder
            "cost_breakdown": {
                "distribution": total_cost * 0.7,
                "inventory_management": total_cost * 0.2,
                "optimization": total_cost * 0.1
            }
        }
    
    def _calculate_total_costs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total costs across all systems."""
        total_cost = 0.0
        cost_breakdown = {}
        
        if "transportation_costs" in results and results["transportation_costs"]:
            transport_cost = results["transportation_costs"].get("total_transportation_cost", 0)
            total_cost += transport_cost
            cost_breakdown["transportation"] = transport_cost
        
        if "stockpile_costs" in results and results["stockpile_costs"]:
            stockpile_cost = results["stockpile_costs"].get("total_stockpile_cost", 0)
            total_cost += stockpile_cost
            cost_breakdown["stockpiles"] = stockpile_cost
        
        if "supply_chain_costs" in results and results["supply_chain_costs"]:
            supply_chain_cost = results["supply_chain_costs"].get("total_supply_chain_cost", 0)
            total_cost += supply_chain_cost
            cost_breakdown["supply_chain"] = supply_chain_cost
        
        return {
            "total_cost": total_cost,
            "cost_breakdown": cost_breakdown,
            "cost_percentage": {
                category: (cost / total_cost * 100) if total_cost > 0 else 0
                for category, cost in cost_breakdown.items()
            }
        }
    
    def _analyze_transportation_costs(self, costs: Dict) -> Dict[str, Any]:
        """Analyze transportation cost patterns."""
        if not costs:
            return {}
        
        # Calculate cost statistics
        all_costs = []
        mode_costs = {}
        
        for flow_costs in costs.values():
            for mode, cost_details in flow_costs.items():
                cost = cost_details["total_cost"]
                all_costs.append(cost)
                if mode not in mode_costs:
                    mode_costs[mode] = []
                mode_costs[mode].append(cost)
        
        return {
            "average_cost": np.mean(all_costs) if all_costs else 0,
            "total_cost": sum(all_costs),
            "mode_efficiency": {
                mode: np.mean(costs) if costs else 0
                for mode, costs in mode_costs.items()
            },
            "cost_distribution": {
                "min": min(all_costs) if all_costs else 0,
                "max": max(all_costs) if all_costs else 0,
                "median": np.median(all_costs) if all_costs else 0
            }
        }
    
    def _analyze_stockpile_costs(self, facility_costs: Dict) -> Dict[str, Any]:
        """Analyze stockpile cost patterns."""
        if not facility_costs:
            return {}
        
        total_capacity = sum(facility["storage_cost"] for facility in facility_costs.values())
        avg_utilization = np.mean([
            facility["capacity_utilization"] for facility in facility_costs.values()
        ])
        
        return {
            "total_capacity_cost": total_capacity,
            "average_utilization": avg_utilization,
            "facility_count": len(facility_costs),
            "utilization_distribution": {
                "underutilized": sum(1 for f in facility_costs.values() if f["capacity_utilization"] < 0.5),
                "optimal": sum(1 for f in facility_costs.values() if 0.5 <= f["capacity_utilization"] <= 0.8),
                "overutilized": sum(1 for f in facility_costs.values() if f["capacity_utilization"] > 0.8)
            }
        }
    
    def _generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on cost analysis."""
        recommendations = []
        
        # Transportation optimization recommendations
        if "transportation_costs" in results and results["transportation_costs"]:
            transport_analysis = results["transportation_costs"].get("cost_breakdown", {})
            if transport_analysis.get("average_cost", 0) > 1000:
                recommendations.append({
                    "category": "transportation",
                    "priority": "high",
                    "recommendation": "Consider consolidating shipments to reduce transportation costs",
                    "potential_savings": "15-25%"
                })
        
        # Stockpile optimization recommendations
        if "stockpile_costs" in results and results["stockpile_costs"]:
            stockpile_analysis = results["stockpile_costs"].get("cost_breakdown", {})
            if stockpile_analysis.get("average_utilization", 0) < 0.6:
                recommendations.append({
                    "category": "stockpiles",
                    "priority": "medium",
                    "recommendation": "Optimize stockpile utilization to reduce storage costs",
                    "potential_savings": "10-20%"
                })
        
        # Supply chain optimization recommendations
        if "supply_chain_costs" in results and results["supply_chain_costs"]:
            supply_chain_analysis = results["supply_chain_costs"]
            if supply_chain_analysis.get("distribution_efficiency", 0) < 0.8:
                recommendations.append({
                    "category": "supply_chain",
                    "priority": "medium",
                    "recommendation": "Improve distribution efficiency through route optimization",
                    "potential_savings": "5-15%"
                })
        
        return recommendations
    
    def _update_integration_metrics(self, results: Dict[str, Any]):
        """Update integration performance metrics."""
        self.integration_metrics["total_transportation_cost"] = results.get("total_costs", {}).get("cost_breakdown", {}).get("transportation", 0)
        self.integration_metrics["total_stockpile_cost"] = results.get("total_costs", {}).get("cost_breakdown", {}).get("stockpiles", 0)
        self.integration_metrics["supply_chain_efficiency"] = results.get("supply_chain_costs", {}).get("distribution_efficiency", 0)
        self.integration_metrics["last_update"] = datetime.now()
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics."""
        return {
            "is_initialized": self.is_initialized,
            "enabled_systems": {
                "transportation_costs": self.config.enable_transportation_costs,
                "stockpile_management": self.config.enable_stockpile_management,
                "supply_chain_optimization": self.config.enable_supply_chain_optimization
            },
            "metrics": self.integration_metrics,
            "locations": len(self.current_locations),
            "resources": len(self.resource_flow_model.resources) if self.resource_flow_model else 0
        }
