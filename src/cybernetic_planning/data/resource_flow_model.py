"""
Resource Flow and Transportation Cost Model

This module implements a comprehensive resource flow and transportation cost model
that integrates with the existing cybernetic planning system. It defines resources,
maps them to production sectors, models consumption across the economy, and calculates
transportation costs for resource distribution.

Key Components:
1. Resource Data Structure - Defines all available resources with metadata
2. Resource Input Matrix (R) - Maps resource consumption to sector outputs
3. Transportation Cost Framework - Calculates costs for resource distribution
4. Integration Layer - Connects with existing economic planning system
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
from datetime import datetime
from pathlib import Path

from .sector_mapper import SectorMapper
from ..utils.transportation_system import (
    Location, TransportMode, VehicleType, VehicleSpec, 
    calculate_distance, TransportationSystem
)


class ResourceType(Enum):
    """Categories of resources in the economic system."""
    RAW_MATERIAL = "raw_material"           # Iron ore, crude oil, timber
    PROCESSED_MATERIAL = "processed_material"  # Steel, gasoline, lumber
    MANUFACTURED_GOOD = "manufactured_good"    # Electronics, machinery, vehicles
    ENERGY = "energy"                       # Electricity, fuel, natural gas
    LABOR = "labor"                         # Skilled, unskilled, technical
    SERVICES = "services"                   # Healthcare, education, transportation
    AGRICULTURAL = "agricultural"           # Crops, livestock, food products
    CONSTRUCTION = "construction"           # Cement, concrete, building materials
    TECHNOLOGY = "technology"               # Software, hardware, data
    FINANCIAL = "financial"                 # Capital, credit, insurance


@dataclass
class ResourceDefinition:
    """
    Primary data structure defining a resource in the economic system.
    
    This is the core data structure that defines all available resources
    with their essential attributes for economic planning and transportation.
    """
    resource_id: str                        # Unique identifier (e.g., "R001", "R002")
    resource_name: str                      # Human-readable name (e.g., "Iron Ore", "Crude Oil")
    resource_type: ResourceType             # Category of resource
    producing_sector_id: str                # ID of primary producing sector
    base_unit: str                          # Standard unit of measurement
    density: float = 1.0                    # kg/m³ for transportation calculations
    value_per_unit: float = 0.0             # Base economic value per unit
    perishability: float = 0.0              # 0.0 = non-perishable, 1.0 = highly perishable
    hazard_class: int = 0                   # 0 = safe, 1-9 = hazardous materials
    storage_requirements: List[str] = field(default_factory=list)  # Special storage needs
    transport_restrictions: List[str] = field(default_factory=list)  # Transport limitations
    substitutability: float = 0.5           # 0.0 = unique, 1.0 = highly substitutable
    criticality: float = 0.5                # 0.0 = optional, 1.0 = critical for economy
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional properties


@dataclass
class ResourceConsumption:
    """
    Defines resource consumption for a specific sector-resource pair.
    """
    resource_id: str
    sector_id: str
    consumption_rate: float                 # Units of resource per unit of sector output
    consumption_type: str = "direct"        # direct, indirect, capital
    efficiency_factor: float = 1.0          # Technology/efficiency multiplier
    substitutability: float = 0.5           # How easily this resource can be substituted


class ResourceInputMatrix:
    """
    Resource Input Matrix (R) - Maps resource consumption to sector outputs.
    
    Dimensions: (Number of Resources x Number of Sectors)
    R[i][j] = units of resource i required to produce one unit of output for sector j
    """
    
    def __init__(self, resources: List[ResourceDefinition], sectors: List[str]):
        """
        Initialize the Resource Input Matrix.
        
        Args:
            resources: List of resource definitions
            sectors: List of sector IDs
        """
        self.resources = {r.resource_id: r for r in resources}
        self.sectors = sectors
        self.n_resources = len(resources)
        self.n_sectors = len(sectors)
        
        # Create resource and sector index mappings
        self.resource_index = {r.resource_id: i for i, r in enumerate(resources)}
        self.sector_index = {s: i for i, s in enumerate(sectors)}
        
        # Initialize the matrix
        self.matrix = np.zeros((self.n_resources, self.n_sectors))
        self.consumption_data = {}  # Detailed consumption information
        
    def set_consumption(self, resource_id: str, sector_id: str, 
                       consumption_rate: float, consumption_type: str = "direct",
                       efficiency_factor: float = 1.0) -> None:
        """
        Set resource consumption rate for a sector.
        
        Args:
            resource_id: ID of the resource
            sector_id: ID of the consuming sector
            consumption_rate: Units of resource per unit of sector output
            consumption_type: Type of consumption (direct, indirect, capital)
            efficiency_factor: Technology/efficiency multiplier
        """
        if resource_id not in self.resource_index:
            raise ValueError(f"Unknown resource ID: {resource_id}")
        if sector_id not in self.sector_index:
            raise ValueError(f"Unknown sector ID: {sector_id}")
            
        resource_idx = self.resource_index[resource_id]
        sector_idx = self.sector_index[sector_id]
        
        # Set matrix value
        self.matrix[resource_idx, sector_idx] = consumption_rate
        
        # Store detailed consumption data
        key = (resource_id, sector_id)
        self.consumption_data[key] = ResourceConsumption(
            resource_id=resource_id,
            sector_id=sector_id,
            consumption_rate=consumption_rate,
            consumption_type=consumption_type,
            efficiency_factor=efficiency_factor
        )
    
    def get_consumption(self, resource_id: str, sector_id: str) -> float:
        """Get consumption rate for a resource-sector pair."""
        if resource_id not in self.resource_index or sector_id not in self.sector_index:
            return 0.0
        resource_idx = self.resource_index[resource_id]
        sector_idx = self.sector_index[sector_id]
        return self.matrix[resource_idx, sector_idx]
    
    def get_total_resource_demand(self, resource_id: str, sector_outputs: np.ndarray) -> float:
        """
        Calculate total demand for a resource given sector outputs.
        
        Args:
            resource_id: ID of the resource
            sector_outputs: Array of sector outputs (n_sectors,)
            
        Returns:
            Total demand for the resource
        """
        if resource_id not in self.resource_index:
            return 0.0
        resource_idx = self.resource_index[resource_id]
        return np.dot(self.matrix[resource_idx, :], sector_outputs)
    
    def get_sector_resource_requirements(self, sector_id: str) -> Dict[str, float]:
        """
        Get all resource requirements for a sector.
        
        Args:
            sector_id: ID of the sector
            
        Returns:
            Dictionary mapping resource_id to consumption_rate
        """
        if sector_id not in self.sector_index:
            return {}
        sector_idx = self.sector_index[sector_id]
        requirements = {}
        for resource_id, resource_idx in self.resource_index.items():
            rate = self.matrix[resource_idx, sector_idx]
            if rate > 0:
                requirements[resource_id] = rate
        return requirements
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert matrix to dictionary format for serialization."""
        return {
            "matrix": self.matrix.tolist(),
            "resources": list(self.resources.keys()),
            "sectors": self.sectors,
            "consumption_data": {
                f"{r}_{s}": {
                    "consumption_rate": data.consumption_rate,
                    "consumption_type": data.consumption_type,
                    "efficiency_factor": data.efficiency_factor
                }
                for (r, s), data in self.consumption_data.items()
            }
        }


class TransportationCostCalculator:
    """
    Transportation cost calculation framework.
    
    Calculates costs for transporting resources between locations using
    different transportation modes and methods.
    """
    
    def __init__(self, transportation_system: Optional[TransportationSystem] = None):
        """
        Initialize the transportation cost calculator.
        
        Args:
            transportation_system: Existing transportation system instance
        """
        self.transportation_system = transportation_system or TransportationSystem()
        self.cost_multipliers = self._initialize_cost_multipliers()
        
    def _initialize_cost_multipliers(self) -> Dict[str, Dict[str, float]]:
        """Initialize cost multipliers for different resource types and transport modes."""
        return {
            "raw_material": {
                "truck": 1.0,
                "rail": 0.6,
                "ship": 0.3,
                "pipeline": 0.2,
                "aircraft": 3.0
            },
            "processed_material": {
                "truck": 1.0,
                "rail": 0.7,
                "ship": 0.4,
                "pipeline": 0.3,
                "aircraft": 2.5
            },
            "manufactured_good": {
                "truck": 1.0,
                "rail": 0.8,
                "ship": 0.5,
                "pipeline": 0.0,  # Not applicable
                "aircraft": 2.0
            },
            "energy": {
                "truck": 1.0,
                "rail": 0.5,
                "ship": 0.3,
                "pipeline": 0.1,
                "aircraft": 4.0
            },
            "agricultural": {
                "truck": 1.0,
                "rail": 0.6,
                "ship": 0.4,
                "pipeline": 0.0,  # Not applicable
                "aircraft": 3.5
            }
        }
    
    def calculate_transportation_cost(
        self,
        resource: ResourceDefinition,
        quantity: float,
        origin: Location,
        destination: Location,
        transport_mode: TransportMode,
        vehicle_type: Optional[VehicleType] = None
    ) -> Dict[str, Any]:
        """
        Calculate transportation cost for moving a resource.
        
        Args:
            resource: Resource being transported
            quantity: Amount of resource to transport
            origin: Origin location
            destination: Destination location
            transport_mode: Transportation mode to use
            vehicle_type: Specific vehicle type (optional)
            
        Returns:
            Dictionary containing cost breakdown and transportation details
        """
        # Calculate distance
        distance = calculate_distance(origin, destination)
        
        # Get base cost per unit distance for transport mode
        base_cost_per_km = self._get_base_cost_per_km(transport_mode, vehicle_type)
        
        # Apply resource-specific cost multiplier
        resource_type_key = resource.resource_type.value
        if resource_type_key in self.cost_multipliers:
            transport_mode_key = transport_mode.value
            if transport_mode_key in self.cost_multipliers[resource_type_key]:
                cost_multiplier = self.cost_multipliers[resource_type_key][transport_mode_key]
            else:
                cost_multiplier = 1.0  # Default multiplier
        else:
            cost_multiplier = 1.0
        
        # Calculate base transportation cost
        base_cost = distance * base_cost_per_km * cost_multiplier
        
        # Apply quantity-based scaling
        quantity_factor = self._calculate_quantity_factor(quantity, resource, transport_mode)
        
        # Apply resource-specific factors
        resource_factor = self._calculate_resource_factor(resource)
        
        # Calculate final cost
        total_cost = base_cost * quantity_factor * resource_factor
        
        # Calculate additional costs
        fuel_cost = self._calculate_fuel_cost(distance, transport_mode, quantity, resource)
        labor_cost = self._calculate_labor_cost(distance, transport_mode, quantity)
        maintenance_cost = self._calculate_maintenance_cost(distance, transport_mode)
        insurance_cost = self._calculate_insurance_cost(total_cost, resource)
        
        # Calculate environmental impact
        emissions = self._calculate_emissions(distance, transport_mode, quantity, resource)
        
        return {
            "total_cost": total_cost,
            "base_cost": base_cost,
            "fuel_cost": fuel_cost,
            "labor_cost": labor_cost,
            "maintenance_cost": maintenance_cost,
            "insurance_cost": insurance_cost,
            "distance_km": distance,
            "quantity": quantity,
            "cost_per_unit": total_cost / max(quantity, 1),
            "cost_per_km": total_cost / max(distance, 1),
            "emissions_kg_co2": emissions,
            "transport_mode": transport_mode.value,
            "vehicle_type": vehicle_type.value if vehicle_type else None,
            "resource_type": resource.resource_type.value,
            "origin": origin.id,
            "destination": destination.id
        }
    
    def _get_base_cost_per_km(self, transport_mode: TransportMode, 
                             vehicle_type: Optional[VehicleType] = None) -> float:
        """Get base cost per kilometer for transport mode."""
        if vehicle_type and vehicle_type in self.transportation_system.vehicle_specs:
            return self.transportation_system.vehicle_specs[vehicle_type].operating_cost
        else:
            # Default costs per km for different modes
            default_costs = {
                TransportMode.TRUCK: 1.0,
                TransportMode.RAIL: 0.3,
                TransportMode.SHIP: 0.1,
                TransportMode.PIPELINE: 0.05,
                TransportMode.AIRCRAFT: 5.0
            }
            return default_costs.get(transport_mode, 1.0)
    
    def _calculate_quantity_factor(self, quantity: float, resource: ResourceDefinition, 
                                 transport_mode: TransportMode) -> float:
        """Calculate quantity-based cost scaling factor."""
        # Base quantity factor (economies of scale)
        if quantity <= 1000:  # Small quantities
            return 1.5
        elif quantity <= 10000:  # Medium quantities
            return 1.0
        else:  # Large quantities
            return 0.8
    
    def _calculate_resource_factor(self, resource: ResourceDefinition) -> float:
        """Calculate resource-specific cost factors."""
        factor = 1.0
        
        # Hazardous materials cost more to transport
        if resource.hazard_class > 0:
            factor *= (1.0 + resource.hazard_class * 0.3)
        
        # Perishable goods may require special handling
        if resource.perishability > 0.5:
            factor *= 1.2
        
        # High-value goods may require additional security
        if resource.value_per_unit > 1000:
            factor *= 1.1
        
        return factor
    
    def _calculate_fuel_cost(self, distance: float, transport_mode: TransportMode, 
                           quantity: float, resource: ResourceDefinition) -> float:
        """Calculate fuel cost for transportation."""
        # Base fuel consumption rates (L/km)
        fuel_rates = {
            TransportMode.TRUCK: 0.3,
            TransportMode.RAIL: 0.1,
            TransportMode.SHIP: 0.05,
            TransportMode.PIPELINE: 0.01,
            TransportMode.AIRCRAFT: 2.0
        }
        
        fuel_rate = fuel_rates.get(transport_mode, 0.3)
        fuel_price = 1.5  # $/L (simplified)
        
        # Adjust for cargo weight
        weight_factor = 1.0 + (quantity * resource.density / 10000) * 0.1
        
        return distance * fuel_rate * fuel_price * weight_factor
    
    def _calculate_labor_cost(self, distance: float, transport_mode: TransportMode, 
                            quantity: float) -> float:
        """Calculate labor cost for transportation."""
        # Labor rates per hour
        labor_rates = {
            TransportMode.TRUCK: 25.0,
            TransportMode.RAIL: 30.0,
            TransportMode.SHIP: 35.0,
            TransportMode.PIPELINE: 20.0,
            TransportMode.AIRCRAFT: 50.0
        }
        
        labor_rate = labor_rates.get(transport_mode, 25.0)
        travel_time = distance / 80.0  # Assume 80 km/h average speed
        
        return travel_time * labor_rate
    
    def _calculate_maintenance_cost(self, distance: float, transport_mode: TransportMode) -> float:
        """Calculate maintenance cost for transportation."""
        # Maintenance rates per km
        maintenance_rates = {
            TransportMode.TRUCK: 0.1,
            TransportMode.RAIL: 0.05,
            TransportMode.SHIP: 0.02,
            TransportMode.PIPELINE: 0.01,
            TransportMode.AIRCRAFT: 0.5
        }
        
        maintenance_rate = maintenance_rates.get(transport_mode, 0.1)
        return distance * maintenance_rate
    
    def _calculate_insurance_cost(self, base_cost: float, resource: ResourceDefinition) -> float:
        """Calculate insurance cost based on resource value and risk."""
        insurance_rate = 0.02  # 2% base rate
        
        # Adjust for resource value
        if resource.value_per_unit > 1000:
            insurance_rate *= 1.5
        
        # Adjust for hazard class
        if resource.hazard_class > 0:
            insurance_rate *= (1.0 + resource.hazard_class * 0.2)
        
        return base_cost * insurance_rate
    
    def _calculate_emissions(self, distance: float, transport_mode: TransportMode, 
                           quantity: float, resource: ResourceDefinition) -> float:
        """Calculate CO2 emissions for transportation."""
        # Emissions factors (kg CO2 per km per kg of cargo)
        emission_factors = {
            TransportMode.TRUCK: 0.1,
            TransportMode.RAIL: 0.03,
            TransportMode.SHIP: 0.01,
            TransportMode.PIPELINE: 0.005,
            TransportMode.AIRCRAFT: 0.5
        }
        
        emission_factor = emission_factors.get(transport_mode, 0.1)
        cargo_weight = quantity * resource.density
        
        return distance * emission_factor * cargo_weight


class ResourceFlowModel:
    """
    Main resource flow model integrating all components.
    
    This class coordinates resource definitions, consumption matrices,
    and transportation cost calculations to provide a comprehensive
    resource flow and cost model for the economic planning system.
    """
    
    def __init__(self, sector_mapper: Optional[SectorMapper] = None):
        """
        Initialize the resource flow model.
        
        Args:
            sector_mapper: Sector mapping instance for integration
        """
        self.sector_mapper = sector_mapper or SectorMapper()
        self.resources = {}  # resource_id -> ResourceDefinition
        self.resource_matrix = None  # ResourceInputMatrix
        self.transportation_calculator = TransportationCostCalculator()
        self.sectors = []
        
    def add_resource(self, resource: ResourceDefinition) -> None:
        """Add a resource definition to the model."""
        self.resources[resource.resource_id] = resource
    
    def add_resources(self, resources: List[ResourceDefinition]) -> None:
        """Add multiple resource definitions to the model."""
        for resource in resources:
            self.add_resource(resource)
    
    def initialize_resource_matrix(self, sectors: List[str]) -> None:
        """
        Initialize the resource input matrix.
        
        Args:
            sectors: List of sector IDs
        """
        self.sectors = sectors
        resource_list = list(self.resources.values())
        self.resource_matrix = ResourceInputMatrix(resource_list, sectors)
    
    def set_resource_consumption(self, resource_id: str, sector_id: str, 
                               consumption_rate: float, **kwargs) -> None:
        """Set resource consumption rate for a sector."""
        if self.resource_matrix is None:
            raise ValueError("Resource matrix not initialized. Call initialize_resource_matrix first.")
        self.resource_matrix.set_consumption(resource_id, sector_id, consumption_rate, **kwargs)
    
    def calculate_total_resource_demand(self, sector_outputs: np.ndarray) -> Dict[str, float]:
        """
        Calculate total demand for all resources given sector outputs.
        
        Args:
            sector_outputs: Array of sector outputs (n_sectors,)
            
        Returns:
            Dictionary mapping resource_id to total demand
        """
        if self.resource_matrix is None:
            return {}
        
        total_demand = {}
        for resource_id in self.resources.keys():
            demand = self.resource_matrix.get_total_resource_demand(resource_id, sector_outputs)
            total_demand[resource_id] = demand
        
        return total_demand
    
    def calculate_transportation_costs(self, resource_flows: Dict[Tuple[str, str], float],
                                     locations: Dict[str, Location]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Calculate transportation costs for resource flows.
        
        Args:
            resource_flows: Dictionary mapping (resource_id, destination_id) to quantity
            locations: Dictionary mapping location_id to Location objects
            
        Returns:
            Dictionary mapping (resource_id, destination_id) to cost details
        """
        costs = {}
        
        for (resource_id, destination_id), quantity in resource_flows.items():
            if resource_id not in self.resources:
                continue
            if destination_id not in locations:
                continue
                
            resource = self.resources[resource_id]
            destination = locations[destination_id]
            
            # Find origin (producing sector location)
            origin_id = resource.producing_sector_id
            if origin_id not in locations:
                continue
            origin = locations[origin_id]
            
            # Calculate costs for different transport modes
            transport_costs = {}
            for transport_mode in TransportMode:
                cost_details = self.transportation_calculator.calculate_transportation_cost(
                    resource=resource,
                    quantity=quantity,
                    origin=origin,
                    destination=destination,
                    transport_mode=transport_mode
                )
                transport_costs[transport_mode.value] = cost_details
            
            costs[(resource_id, destination_id)] = transport_costs
        
        return costs
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of all resources in the model."""
        return {
            "total_resources": len(self.resources),
            "resource_types": list(set(r.resource_type.value for r in self.resources.values())),
            "producing_sectors": list(set(r.producing_sector_id for r in self.resources.values())),
            "resources": {
                resource_id: {
                    "name": resource.resource_name,
                    "type": resource.resource_type.value,
                    "producing_sector": resource.producing_sector_id,
                    "base_unit": resource.base_unit,
                    "density": resource.density,
                    "value_per_unit": resource.value_per_unit
                }
                for resource_id, resource in self.resources.items()
            }
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the complete resource flow model to a file."""
        model_data = {
            "resources": {
                resource_id: {
                    "resource_id": resource.resource_id,
                    "resource_name": resource.resource_name,
                    "resource_type": resource.resource_type.value,
                    "producing_sector_id": resource.producing_sector_id,
                    "base_unit": resource.base_unit,
                    "density": resource.density,
                    "value_per_unit": resource.value_per_unit,
                    "perishability": resource.perishability,
                    "hazard_class": resource.hazard_class,
                    "storage_requirements": resource.storage_requirements,
                    "transport_restrictions": resource.transport_restrictions,
                    "substitutability": resource.substitutability,
                    "criticality": resource.criticality,
                    "metadata": resource.metadata
                }
                for resource_id, resource in self.resources.items()
            },
            "resource_matrix": self.resource_matrix.to_dict() if self.resource_matrix else None,
            "sectors": self.sectors,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load a resource flow model from a file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Load resources
        self.resources = {}
        for resource_id, resource_data in model_data["resources"].items():
            resource = ResourceDefinition(
                resource_id=resource_data["resource_id"],
                resource_name=resource_data["resource_name"],
                resource_type=ResourceType(resource_data["resource_type"]),
                producing_sector_id=resource_data["producing_sector_id"],
                base_unit=resource_data["base_unit"],
                density=resource_data["density"],
                value_per_unit=resource_data["value_per_unit"],
                perishability=resource_data["perishability"],
                hazard_class=resource_data["hazard_class"],
                storage_requirements=resource_data["storage_requirements"],
                transport_restrictions=resource_data["transport_restrictions"],
                substitutability=resource_data["substitutability"],
                criticality=resource_data["criticality"],
                metadata=resource_data["metadata"]
            )
            self.resources[resource_id] = resource
        
        # Load resource matrix
        if model_data["resource_matrix"]:
            self.sectors = model_data["resource_matrix"]["sectors"]
            resource_list = list(self.resources.values())
            self.resource_matrix = ResourceInputMatrix(resource_list, self.sectors)
            
            # Restore matrix data
            matrix_data = model_data["resource_matrix"]["matrix"]
            self.resource_matrix.matrix = np.array(matrix_data)
            
            # Restore consumption data
            for key, data in model_data["resource_matrix"]["consumption_data"].items():
                resource_id, sector_id = key.split("_", 1)
                self.resource_matrix.consumption_data[(resource_id, sector_id)] = ResourceConsumption(
                    resource_id=resource_id,
                    sector_id=sector_id,
                    consumption_rate=data["consumption_rate"],
                    consumption_type=data["consumption_type"],
                    efficiency_factor=data["efficiency_factor"]
                )


def create_example_resource_model() -> ResourceFlowModel:
    """
    Create an example resource flow model with sample data.
    
    This function demonstrates how to set up a resource flow model
    with example resources and consumption patterns.
    """
    # Create example resources
    example_resources = [
        ResourceDefinition(
            resource_id="R001",
            resource_name="Iron Ore",
            resource_type=ResourceType.RAW_MATERIAL,
            producing_sector_id="S051",  # Oil and Gas Extraction (using as mining)
            base_unit="ton",
            density=5000.0,  # kg/m³
            value_per_unit=50.0,  # $/ton
            criticality=0.9
        ),
        ResourceDefinition(
            resource_id="R002",
            resource_name="Crude Oil",
            resource_type=ResourceType.RAW_MATERIAL,
            producing_sector_id="S051",  # Oil and Gas Extraction
            base_unit="barrel",
            density=850.0,  # kg/m³
            value_per_unit=60.0,  # $/barrel
            hazard_class=3,
            criticality=0.95
        ),
        ResourceDefinition(
            resource_id="R003",
            resource_name="Steel",
            resource_type=ResourceType.PROCESSED_MATERIAL,
            producing_sector_id="S160",  # Metal Products Manufacturing
            base_unit="ton",
            density=7800.0,  # kg/m³
            value_per_unit=500.0,  # $/ton
            criticality=0.8
        ),
        ResourceDefinition(
            resource_id="R004",
            resource_name="Electricity",
            resource_type=ResourceType.ENERGY,
            producing_sector_id="S053",  # Electric Power Generation
            base_unit="MWh",
            density=0.0,  # Not applicable for energy
            value_per_unit=50.0,  # $/MWh
            criticality=1.0
        ),
        ResourceDefinition(
            resource_name="Medicines",
            resource_id="R005",
            resource_type=ResourceType.MANUFACTURED_GOOD,
            producing_sector_id="S007",  # Pharmaceuticals
            base_unit="unit",
            density=1000.0,  # kg/m³
            value_per_unit=100.0,  # $/unit
            perishability=0.3,
            criticality=0.9
        ),
        ResourceDefinition(
            resource_id="R006",
            resource_name="Farm Equipment",
            resource_type=ResourceType.MANUFACTURED_GOOD,
            producing_sector_id="S039",  # Farm Equipment Manufacturing
            base_unit="unit",
            density=2000.0,  # kg/m³
            value_per_unit=50000.0,  # $/unit
            criticality=0.7
        )
    ]
    
    # Create model
    model = ResourceFlowModel()
    model.add_resources(example_resources)
    
    # Initialize with example sectors
    example_sectors = ["S001", "S002", "S003", "S007", "S039", "S051", "S053", "S160"]
    model.initialize_resource_matrix(example_sectors)
    
    # Set example consumption patterns
    consumption_patterns = [
        # (resource_id, sector_id, consumption_rate, consumption_type)
        ("R001", "S160", 1.2, "direct"),  # Steel manufacturing needs iron ore
        ("R002", "S053", 0.1, "direct"),  # Power generation needs oil
        ("R003", "S039", 0.5, "direct"),  # Farm equipment needs steel
        ("R004", "S007", 0.01, "direct"),  # Pharmaceuticals need electricity
        ("R004", "S160", 0.05, "direct"),  # Steel manufacturing needs electricity
        ("R005", "S001", 0.001, "direct"),  # Healthcare needs medicines
        ("R006", "S002", 0.0001, "direct"),  # Agriculture needs farm equipment
    ]
    
    for resource_id, sector_id, rate, cons_type in consumption_patterns:
        model.set_resource_consumption(resource_id, sector_id, rate, consumption_type=cons_type)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating example resource flow model...")
    model = create_example_resource_model()
    
    print("\nResource Summary:")
    summary = model.get_resource_summary()
    print(f"Total resources: {summary['total_resources']}")
    print(f"Resource types: {summary['resource_types']}")
    
    print("\nExample transportation cost calculation:")
    # Create example locations
    origin = Location("MINE001", "Iron Mine", 40.0, -100.0, location_type="mine")
    destination = Location("FACTORY001", "Steel Factory", 41.0, -99.0, location_type="factory")
    
    # Calculate cost for transporting iron ore
    iron_ore = model.resources["R001"]
    cost_details = model.transportation_calculator.calculate_transportation_cost(
        resource=iron_ore,
        quantity=1000.0,  # 1000 tons
        origin=origin,
        destination=destination,
        transport_mode=TransportMode.TRUCK
    )
    
    print(f"Transporting {cost_details['quantity']} tons of {iron_ore.resource_name}")
    print(f"Distance: {cost_details['distance_km']:.1f} km")
    print(f"Total cost: ${cost_details['total_cost']:.2f}")
    print(f"Cost per ton: ${cost_details['cost_per_unit']:.2f}")
    print(f"Emissions: {cost_details['emissions_kg_co2']:.1f} kg CO2")
    
    # Save model
    model.save_model("example_resource_model.json")
    print("\nModel saved to example_resource_model.json")
