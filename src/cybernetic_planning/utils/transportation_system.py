#!/usr / bin / env python3
"""
Multi - Modal Transportation System for Cybernetic Planning

Implements aircraft, rail, and truck transportation with optimization algorithms
for fuel - efficient routing, capacity management, and multi - modal coordination.
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class TransportMode(Enum):
    """Transportation modes available in the system."""
    TRUCK = "truck"
    RAIL = "rail"
    AIRCRAFT = "aircraft"
    PIPELINE = "pipeline"
    SHIP = "ship"

class VehicleType(Enum):
    """Specific vehicle types within transport modes."""
    # Truck types
    LIGHT_TRUCK = "light_truck"
    HEAVY_TRUCK = "heavy_truck"
    SEMI_TRAILER = "semi_trailer"
    TANKER_TRUCK = "tanker_truck"

    # Rail types
    FREIGHT_TRAIN = "freight_train"
    PASSENGER_TRAIN = "passenger_train"
    HIGH_SPEED_RAIL = "high_speed_rail"

    # Aircraft types
    CARGO_PLANE = "cargo_plane"
    PASSENGER_PLANE = "passenger_plane"
    DRONE = "drone"
    HELICOPTER = "helicopter"

@dataclass
class Location:
    """Geographic location with coordinates and properties."""
    id: str
    name: str
    lat: float
    lon: float
    elevation: float = 0.0
    location_type: str = "city"  # city, warehouse, port, airport, etc.
    properties: Dict[str, Any] = field(default_factory = dict)

@dataclass
class VehicleSpec:
    """Vehicle specifications for transport calculations."""
    vehicle_type: VehicleType
    max_capacity: float  # kg
    max_volume: float    # cubic meters
    fuel_consumption: float  # L / km for trucks, L / h for aircraft
    max_speed: float     # km / h
    max_range: float     # km
    operating_cost: float  # cost per km
    fuel_type: str       # diesel, jet_fuel, electric
    emissions_factor: float  # kg CO2 / L fuel
    crew_requirements: int
    maintenance_cost: float  # per hour

@dataclass
class TransportRoute:
    """Route between two locations for a specific transport mode."""
    origin: Location
    destination: Location
    transport_mode: TransportMode
    distance: float  # km
    travel_time: float  # hours
    terrain_difficulty: float = 1.0  # multiplier for fuel consumption
    infrastructure_quality: float = 1.0  # condition of roads / rails / airways
    capacity_constraints: Dict[str, float] = field(default_factory = dict)
    tolls_fees: float = 0.0
    properties: Dict[str, Any] = field(default_factory = dict)

@dataclass
class CargoItem:
    """Individual cargo item for transportation."""
    id: str
    cargo_type: str
    weight: float  # kg
    volume: float  # cubic meters
    value: float   # monetary value
    priority: int = 1  # 1 = low, 5 = urgent
    special_requirements: List[str] = field(default_factory = list)  # refrigerated, hazardous, etc.
    origin: Location = None
    destination: Location = None
    deadline: Optional[float] = None  # hours from now

@dataclass
class TransportPlan:
    """Complete transportation plan for cargo movement."""
    cargo: List[CargoItem]
    route_segments: List[Tuple[TransportRoute, VehicleType]]
    total_distance: float
    total_time: float
    total_cost: float
    total_emissions: float
    fuel_required: Dict[str, float]  # fuel_type -> amount
    vehicles_required: Dict[VehicleType, int]

class RouteOptimizer:
    """Base class for route optimization algorithms."""

    @abstractmethod
    def optimize_route(self, cargo: List[CargoItem], available_routes: List[TransportRoute],
                      vehicle_specs: Dict[VehicleType, VehicleSpec]) -> TransportPlan:
        """Optimize transportation route for given cargo and constraints."""
        pass

class FuelEfficientOptimizer(RouteOptimizer):
    """Route optimizer focused on fuel efficiency and environmental impact."""

    def __init__(self, fuel_weight: float = 0.4, time_weight: float = 0.3, cost_weight: float = 0.3):
        self.fuel_weight = fuel_weight
        self.time_weight = time_weight
        self.cost_weight = cost_weight

    def optimize_route(self, cargo: List[CargoItem], available_routes: List[TransportRoute],
                      vehicle_specs: Dict[VehicleType, VehicleSpec]) -> TransportPlan:
        """Optimize routes using multi - criteria optimization focusing on fuel efficiency."""
        if not cargo:
            return TransportPlan([], [], 0, 0, 0, 0, {}, {})

        # Group cargo by origin - destination pairs
        cargo_groups = self._group_cargo_by_od_pairs(cargo)

        all_route_segments = []
        total_distance = 0
        total_time = 0
        total_cost = 0
        total_emissions = 0
        fuel_required = {}
        vehicles_required = {}

        for (origin, destination), cargo_items in cargo_groups.items():
            # Find best route for this OD pair
            best_plan = self._find_best_route_for_cargo_group(
                cargo_items, available_routes, vehicle_specs, origin, destination
            )

            all_route_segments.extend(best_plan.route_segments)
            total_distance += best_plan.total_distance
            total_time += best_plan.total_time
            total_cost += best_plan.total_cost
            total_emissions += best_plan.total_emissions

            # Aggregate fuel requirements
            for fuel_type, amount in best_plan.fuel_required.items():
                fuel_required[fuel_type] = fuel_required.get(fuel_type, 0) + amount

            # Aggregate vehicle requirements
            for vehicle_type, count in best_plan.vehicles_required.items():
                vehicles_required[vehicle_type] = vehicles_required.get(vehicle_type, 0) + count

        return TransportPlan(
            cargo = cargo,
            route_segments = all_route_segments,
            total_distance = total_distance,
            total_time = total_time,
            total_cost = total_cost,
            total_emissions = total_emissions,
            fuel_required = fuel_required,
            vehicles_required = vehicles_required
        )

    def _group_cargo_by_od_pairs(self, cargo: List[CargoItem]) -> Dict[Tuple[str, str], List[CargoItem]]:
        """Group cargo items by origin - destination pairs."""
        groups = {}
        for item in cargo:
            key = (item.origin.id, item.destination.id)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        return groups

    def _find_best_route_for_cargo_group(self, cargo_items: List[CargoItem],
                                       available_routes: List[TransportRoute],
                                       vehicle_specs: Dict[VehicleType, VehicleSpec],
                                       origin: Location, destination: Location) -> TransportPlan:
        """Find the best route for a group of cargo items with same origin / destination."""

        # Calculate total cargo requirements
        total_weight = sum(item.weight for item in cargo_items)
        total_volume = sum(item.volume for item in cargo_items)

        # Find applicable routes
        applicable_routes = [
            route for route in available_routes
            if route.origin.id == origin.id and route.destination.id == destination.id
        ]

        if not applicable_routes:
            # No direct routes - would need to implement multi - hop routing
            return TransportPlan(cargo_items, [], 0, 0, float('inf'), 0, {}, {})

        best_plan = None
        best_score = float('inf')

        # Evaluate each route with different vehicle types
        for route in applicable_routes:
            suitable_vehicles = self._get_suitable_vehicles(route.transport_mode, vehicle_specs)

            for vehicle_type in suitable_vehicles:
                vehicle_spec = vehicle_specs[vehicle_type]

                # Check if vehicle can handle the cargo
                if total_weight > vehicle_spec.max_capacity or total_volume > vehicle_spec.max_volume:
                    continue  # Vehicle too small

                # Calculate transportation metrics
                fuel_consumption = self._calculate_fuel_consumption(route, vehicle_spec, total_weight)
                travel_time = route.distance / vehicle_spec.max_speed
                operating_cost = route.distance * vehicle_spec.operating_cost + route.tolls_fees
                emissions = fuel_consumption * vehicle_spec.emissions_factor

                # Calculate multi - criteria score
                score = (self.fuel_weight * fuel_consumption + self.time_weight * travel_time + self.cost_weight * operating_cost)

                if score < best_score:
                    best_score = score
                    best_plan = TransportPlan(
                        cargo = cargo_items,
                        route_segments=[(route, vehicle_type)],
                        total_distance = route.distance,
                        total_time = travel_time,
                        total_cost = operating_cost,
                        total_emissions = emissions,
                        fuel_required={vehicle_spec.fuel_type: fuel_consumption},
                        vehicles_required={vehicle_type: 1}
                    )

        return best_plan or TransportPlan(cargo_items, [], 0, 0, float('inf'), 0, {}, {})

    def _get_suitable_vehicles(self, transport_mode: TransportMode,
                              vehicle_specs: Dict[VehicleType, VehicleSpec]) -> List[VehicleType]:
        """Get vehicle types suitable for a transport mode."""
        suitable = []
        for vehicle_type, spec in vehicle_specs.items():
            if transport_mode == TransportMode.TRUCK and vehicle_type.value.endswith('_truck'):
                suitable.append(vehicle_type)
            elif transport_mode == TransportMode.RAIL and vehicle_type.value.endswith('_train'):
                suitable.append(vehicle_type)
            elif transport_mode == TransportMode.AIRCRAFT and vehicle_type.value in ['cargo_plane', 'drone']:
                suitable.append(vehicle_type)
        return suitable

    def _calculate_fuel_consumption(self, route: TransportRoute, vehicle_spec: VehicleSpec,
                                  cargo_weight: float) -> float:
        """Calculate fuel consumption for a route considering cargo weight and terrain."""
        base_consumption = route.distance * vehicle_spec.fuel_consumption

        # Adjust for cargo weight (heavier cargo increases consumption)
        weight_factor = 1.0 + (cargo_weight / vehicle_spec.max_capacity) * 0.3

        # Adjust for terrain difficulty
        terrain_factor = route.terrain_difficulty

        # Adjust for infrastructure quality (poor roads increase consumption)
        infrastructure_factor = 2.0 - route.infrastructure_quality

        return base_consumption * weight_factor * terrain_factor * infrastructure_factor

class CapacityOptimizer(RouteOptimizer):
    """Route optimizer focused on maximizing vehicle capacity utilization."""

    def optimize_route(self, cargo: List[CargoItem], available_routes: List[TransportRoute],
                      vehicle_specs: Dict[VehicleType, VehicleSpec]) -> TransportPlan:
        """Optimize routes to maximize capacity utilization and minimize vehicle count."""
        # Implementation for capacity - focused optimization
        # This would implement bin - packing style algorithms for cargo consolidation
        pass

class MultiModalCoordinator:
    """Coordinates transfers between different transportation modes."""

    def __init__(self, transfer_hubs: List[Location]):
        self.transfer_hubs = transfer_hubs  # airports, rail stations, etc.
        self.transfer_costs = {}  # hub_id -> {from_mode: {to_mode: cost}}
        self.transfer_times = {}  # hub_id -> {from_mode: {to_mode: time}}

    def optimize_multimodal_route(self, cargo: List[CargoItem],
                                available_routes: List[TransportRoute],
                                vehicle_specs: Dict[VehicleType, VehicleSpec]) -> TransportPlan:
        """Optimize routes that may use multiple transportation modes."""
        # This would implement complex multi - modal routing algorithms
        # considering transfer costs, times, and capacity constraints
        pass

    def add_transfer_hub(self, hub: Location, transfer_matrix: Dict[str, Dict[str, Dict[str, float]]]):
        """Add a transfer hub with associated transfer costs and times."""
        self.transfer_costs[hub.id] = transfer_matrix.get('costs', {})
        self.transfer_times[hub.id] = transfer_matrix.get('times', {})

class FleetManager:
    """Manages vehicle fleet and scheduling."""

    def __init__(self):
        self.available_vehicles = {}  # vehicle_type -> count
        self.vehicle_schedules = {}   # vehicle_id -> list of scheduled routes
        self.maintenance_schedules = {}

    def schedule_transport(self, transport_plan: TransportPlan) -> Dict[str, Any]:
        """Schedule vehicles for a transport plan."""
        schedule_result = {
            'success': True,
            'vehicle_assignments': {},
            'delays': {},
            'conflicts': []
        }

        for vehicle_type, required_count in transport_plan.vehicles_required.items():
            available = self.available_vehicles.get(vehicle_type, 0)
            if available < required_count:
                schedule_result['success'] = False
                schedule_result['conflicts'].append(
                    f"Insufficient {vehicle_type.value}: need {required_count}, have {available}"
                )

        return schedule_result

    def update_vehicle_availability(self, vehicle_type: VehicleType, count: int):
        """Update available vehicle count."""
        self.available_vehicles[vehicle_type] = count

class TransportationSystem:
    """Main transportation system coordinating all transportation activities."""

    def __init__(self):
        self.locations = {}  # location_id -> Location
        self.routes = []     # List of available routes
        self.vehicle_specs = self._initialize_default_vehicle_specs()
        self.optimizers = {
            'fuel_efficient': FuelEfficientOptimizer(),
            'capacity_focused': CapacityOptimizer()
        }
        self.multimodal_coordinator = MultiModalCoordinator([])
        self.fleet_manager = FleetManager()

        # Performance metrics
        self.total_distance_traveled = 0.0
        self.total_fuel_consumed = {}  # fuel_type -> amount
        self.total_emissions = 0.0
        self.transport_history = []

    def add_location(self, location: Location):
        """Add a new location to the transportation network."""
        self.locations[location.id] = location

    def add_route(self, route: TransportRoute):
        """Add a new transportation route."""
        self.routes.append(route)

    def create_transport_plan(self, cargo: List[CargoItem],
                            optimization_strategy: str = 'fuel_efficient') -> TransportPlan:
        """Create optimized transportation plan for given cargo."""
        optimizer = self.optimizers.get(optimization_strategy)
        if not optimizer:
            raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")

        return optimizer.optimize_route(cargo, self.routes, self.vehicle_specs)

    def execute_transport_plan(self, plan: TransportPlan) -> Dict[str, Any]:
        """Execute a transportation plan and update system state."""
        # Check vehicle availability
        schedule_result = self.fleet_manager.schedule_transport(plan)
        if not schedule_result['success']:
            return {'success': False, 'errors': schedule_result['conflicts']}

        # Update performance metrics
        self.total_distance_traveled += plan.total_distance
        self.total_emissions += plan.total_emissions

        for fuel_type, amount in plan.fuel_required.items():
            self.total_fuel_consumed[fuel_type] = self.total_fuel_consumed.get(fuel_type, 0) + amount

        # Record in history
        self.transport_history.append(plan)

        return {'success': True, 'plan_executed': plan}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get transportation system performance metrics."""
        return {
            'total_distance_traveled': self.total_distance_traveled,
            'total_fuel_consumed': dict(self.total_fuel_consumed),
            'total_emissions': self.total_emissions,
            'completed_transports': len(self.transport_history),
            'average_distance_per_transport': (
                self.total_distance_traveled / max(1, len(self.transport_history))
            ),
            'fuel_efficiency': self._calculate_fuel_efficiency()
        }

    def _calculate_fuel_efficiency(self) -> Dict[str, float]:
        """Calculate fuel efficiency metrics."""
        if not self.total_fuel_consumed or self.total_distance_traveled == 0:
            return {}

        efficiency = {}
        for fuel_type, consumed in self.total_fuel_consumed.items():
            efficiency[f"{fuel_type}_per_km"] = consumed / self.total_distance_traveled

        return efficiency

    def _initialize_default_vehicle_specs(self) -> Dict[VehicleType, VehicleSpec]:
        """Initialize default vehicle specifications."""
        return {
            # Truck specifications
            VehicleType.LIGHT_TRUCK: VehicleSpec(
                vehicle_type = VehicleType.LIGHT_TRUCK,
                max_capacity = 3500,  # kg
                max_volume = 20,      # m³
                fuel_consumption = 0.12,  # L / km
                max_speed = 90,       # km / h
                max_range = 500,      # km
                operating_cost = 0.8,  # $/km
                fuel_type="diesel",
                emissions_factor = 2.68,  # kg CO2 / L
                crew_requirements = 1,
                maintenance_cost = 15  # $/hour
            ),

            VehicleType.HEAVY_TRUCK: VehicleSpec(
                vehicle_type = VehicleType.HEAVY_TRUCK,
                max_capacity = 26000,  # kg
                max_volume = 80,       # m³
                fuel_consumption = 0.35,  # L / km
                max_speed = 80,        # km / h
                max_range = 800,       # km
                operating_cost = 1.5,  # $/km
                fuel_type="diesel",
                emissions_factor = 2.68,  # kg CO2 / L
                crew_requirements = 1,
                maintenance_cost = 25  # $/hour
            ),

            # Rail specifications
            VehicleType.FREIGHT_TRAIN: VehicleSpec(
                vehicle_type = VehicleType.FREIGHT_TRAIN,
                max_capacity = 4000000,  # kg (4000 tons)
                max_volume = 10000,      # m³
                fuel_consumption = 15,    # L / km
                max_speed = 120,         # km / h
                max_range = 2000,        # km
                operating_cost = 3.0,    # $/km
                fuel_type="diesel",
                emissions_factor = 2.68,  # kg CO2 / L
                crew_requirements = 2,
                maintenance_cost = 100   # $/hour
            ),

            # Aircraft specifications
            VehicleType.CARGO_PLANE: VehicleSpec(
                vehicle_type = VehicleType.CARGO_PLANE,
                max_capacity = 100000,   # kg
                max_volume = 500,        # m³
                fuel_consumption = 3000,  # L / hour
                max_speed = 850,         # km / h
                max_range = 5000,        # km
                operating_cost = 15.0,   # $/km
                fuel_type="jet_fuel",
                emissions_factor = 3.16,  # kg CO2 / L
                crew_requirements = 2,
                maintenance_cost = 500   # $/hour
            ),

            VehicleType.DRONE: VehicleSpec(
                vehicle_type = VehicleType.DRONE,
                max_capacity = 25,       # kg
                max_volume = 0.1,        # m³
                fuel_consumption = 0,    # electric
                max_speed = 60,          # km / h
                max_range = 100,         # km
                operating_cost = 0.3,    # $/km
                fuel_type="electric",
                emissions_factor = 0,    # assuming clean electricity
                crew_requirements = 0,   # autonomous
                maintenance_cost = 5     # $/hour
            )
        }

# Utility functions for route calculations

def calculate_distance(loc1: Location, loc2: Location) -> float:
    """Calculate great circle distance between two locations."""
    R = 6371  # Earth's radius in km

    lat1, lon1 = math.radians(loc1.lat), math.radians(loc1.lon)
    lat2, lon2 = math.radians(loc2.lat), math.radians(loc2.lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def generate_route_network(locations: List[Location],
                         transport_modes: List[TransportMode]) -> List[TransportRoute]:
    """Generate a complete route network between all location pairs."""
    routes = []

    for i, origin in enumerate(locations):
        for j, destination in enumerate(locations):
            if i != j:  # Don't create self - routes
                distance = calculate_distance(origin, destination)

                for mode in transport_modes:
                    # Adjust travel characteristics based on transport mode
                    if mode == TransportMode.TRUCK:
                        # Roads might not be direct - add 20% to distance
                        actual_distance = distance * 1.2
                        terrain_difficulty = 1.1  # roads affected by terrain
                    elif mode == TransportMode.RAIL:
                        # Rails even less direct - add 40%
                        actual_distance = distance * 1.4
                        terrain_difficulty = 1.2
                    elif mode == TransportMode.AIRCRAFT:
                        # Aircraft can fly direct
                        actual_distance = distance
                        terrain_difficulty = 1.0  # unaffected by terrain
                    else:
                        actual_distance = distance
                        terrain_difficulty = 1.0

                    route = TransportRoute(
                        origin = origin,
                        destination = destination,
                        transport_mode = mode,
                        distance = actual_distance,
                        travel_time = 0,  # Will be calculated based on vehicle speed
                        terrain_difficulty = terrain_difficulty,
                        infrastructure_quality = 0.8  # Assume decent infrastructure
                    )
                    routes.append(route)

    return routes

if __name__ == "__main__":
    # Example usage
    system = TransportationSystem()

    # Add some test locations
    locations = [
        Location("NYC", "New York City", 40.7128, -74.0060),
        Location("CHI", "Chicago", 41.8781, -87.6298),
        Location("LAX", "Los Angeles", 34.0522, -118.2437)
    ]

    for loc in locations:
        system.add_location(loc)

    # Generate route network
    routes = generate_route_network(locations, [TransportMode.TRUCK, TransportMode.AIRCRAFT])
    for route in routes:
        system.add_route(route)

    # Create some test cargo
    cargo = [
        CargoItem("C001", "electronics", 1500, 10, 50000, 3, [], locations[0], locations[1]),
        CargoItem("C002", "food", 2000, 15, 10000, 2, ["refrigerated"], locations[0], locations[2])
    ]

    # Create transport plan
    plan = system.create_transport_plan(cargo, 'fuel_efficient')
    print(f"Transport plan: {plan.total_distance:.1f} km, {plan.total_time:.1f} hours, ${plan.total_cost:.2f}")

    # Execute plan
    result = system.execute_transport_plan(plan)
    print(f"Execution result: {result['success']}")

    # Get performance metrics
    metrics = system.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
