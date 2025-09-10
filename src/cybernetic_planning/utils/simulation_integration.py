#!/usr / bin / env python3
"""
Simulation Integration System

Integrates all simulation components including transportation, cargo distribution,
stockpiles, infrastructure, and visualization into a cohesive simulation environment.
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

# Import all simulation components
try:
    # Try relative imports first
    from .transportation_system import TransportationSystem, Location, CargoItem, VehicleType
    from .cargo_distribution import SupplyChainOptimizer, SupplyNode, SupplyDemand, CargoCategory
    from .regional_stockpiles import StockpileManager, StockpileFacility, StockpileType, StorageZone
    from .infrastructure_network import InfrastructureBuilder, TerrainAnalyzer, NetworkNode, InfrastructureType
    from .map_visualization import MapGenerator, InteractiveMap
except ImportError as e:
    print(f"Warning: Could not use relative imports: {e}")
    # Try absolute imports
    try:
        from src.cybernetic_planning.utils.transportation_system import TransportationSystem, Location, CargoItem, VehicleType
        from src.cybernetic_planning.utils.cargo_distribution import SupplyChainOptimizer, SupplyNode, SupplyDemand, CargoCategory
        from src.cybernetic_planning.utils.regional_stockpiles import StockpileManager, StockpileFacility, StockpileType, StorageZone
        from src.cybernetic_planning.utils.infrastructure_network import InfrastructureBuilder, TerrainAnalyzer, NetworkNode, InfrastructureType
        from src.cybernetic_planning.utils.map_visualization import MapGenerator, InteractiveMap
    except ImportError as e2:
        print(f"Warning: Could not import simulation components: {e2}")
        # Create placeholder classes to prevent crashes
        class TransportationSystem: pass
        class Location: pass
        class CargoItem: pass
        class VehicleType: pass
        class SupplyChainOptimizer: pass
        class SupplyNode: pass
        class SupplyDemand: pass
        class CargoCategory: pass
        class StockpileManager: pass
        class StockpileFacility: pass
        class StockpileType: pass
        class StorageZone: pass
        class InfrastructureBuilder: pass
        class TerrainAnalyzer: pass
        class NetworkNode: pass
        class InfrastructureType: pass
        class MapGenerator: pass
        class InteractiveMap: pass

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    simulation_name: str = "Economic Planning Simulation"
    duration_days: int = 365
    time_step_hours: int = 24
    map_size_km: int = 500
    grid_resolution: int = 200
    num_cities: int = 10
    num_regions: int = 5
    starting_year: int = 2025
    enable_transportation: bool = True
    enable_stockpiles: bool = True
    enable_infrastructure: bool = True
    enable_visualization: bool = True
    auto_save_interval: int = 30  # days
    debug_mode: bool = False

@dataclass
class SimulationState:
    """Current state of the simulation."""
    current_day: int = 0
    current_date: datetime = field(default_factory = datetime.now)
    total_economic_output: float = 0.0
    total_transportation_cost: float = 0.0
    total_infrastructure_cost: float = 0.0
    system_efficiency: float = 1.0
    population_satisfaction: float = 0.8
    events_log: List[Dict[str, Any]] = field(default_factory = list)
    metrics_history: List[Dict[str, Any]] = field(default_factory = list)

class EconomicSimulation:
    """Main simulation controller integrating all systems."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = SimulationState()
        self.state.current_date = datetime(config.starting_year, 1, 1)

        # Initialize all subsystems
        self.transportation_system = None
        self.supply_chain_optimizer = None
        self.stockpile_manager = None
        self.infrastructure_builder = None
        self.terrain_analyzer = None
        self.map_generator = None

        # Simulation data
        self.locations = {}
        self.supply_nodes = {}
        self.network_nodes = {}

        self.is_initialized = False
        self.is_running = False

        # Performance tracking
        self.initialization_time = 0.0
        self.step_times = []

        # Output directory
        self.output_dir = Path("outputs") / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents = True, exist_ok = True)

        self._log_event("simulation_created", {"config": asdict(config)})

    def initialize(self) -> Dict[str, Any]:
        """Initialize all simulation systems."""
        start_time = datetime.now()
        self._log_event("initialization_started")

        try:
            # Initialize terrain and map
            if self.config.enable_infrastructure or self.config.enable_visualization:
                self._initialize_terrain()

            # Initialize transportation system
            if self.config.enable_transportation:
                self._initialize_transportation()

            # Initialize stockpile management
            if self.config.enable_stockpiles:
                self._initialize_stockpiles()

            # Initialize infrastructure
            if self.config.enable_infrastructure:
                self._initialize_infrastructure()

            # Initialize supply chain
            self._initialize_supply_chain()

            # Initialize visualization
            if self.config.enable_visualization:
                self._initialize_visualization()

            self.initialization_time = (datetime.now() - start_time).total_seconds()
            self.is_initialized = True

            result = {
                "success": True,
                "initialization_time": self.initialization_time,
                "systems_initialized": self._get_initialized_systems(),
                "locations_created": len(self.locations),
                "message": "Simulation initialized successfully"
            }

            self._log_event("initialization_completed", result)
            return result

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "initialization_time": (datetime.now() - start_time).total_seconds()
            }
            self._log_event("initialization_failed", error_result)
            return error_result

    def _initialize_terrain(self):
        """Initialize terrain analysis system."""
        if self.config.debug_mode:
            print("Initializing terrain system...")

        self.terrain_analyzer = TerrainAnalyzer(
            grid_size=(self.config.grid_resolution, self.config.grid_resolution),
            cell_size_km = self.config.map_size_km / self.config.grid_resolution
        )

        # Generate cities based on terrain
        self._generate_cities_from_terrain()

    def _initialize_transportation(self):
        """Initialize transportation system."""
        if self.config.debug_mode:
            print("Initializing transportation system...")

        self.transportation_system = TransportationSystem()

        # Add all locations to transportation system
        for location in self.locations.values():
            self.transportation_system.add_location(location)

        # Generate transportation routes
        from .transportation_system import generate_route_network, TransportMode
        routes = generate_route_network(
            list(self.locations.values()),
            [TransportMode.TRUCK, TransportMode.RAIL, TransportMode.AIRCRAFT]
        )

        for route in routes:
            self.transportation_system.add_route(route)

    def _initialize_stockpiles(self):
        """Initialize stockpile management system."""
        if self.config.debug_mode:
            print("Initializing stockpile system...")

        self.stockpile_manager = StockpileManager()

        # Create stockpile facilities in major cities
        major_cities = sorted(self.locations.values(),
                            key = lambda x: x.properties.get('population', 0),
                            reverse = True)[:self.config.num_regions]

        for i, city in enumerate(major_cities):
            # Create storage zones
            zones = [
                StorageZone(f"ZONE_{i}_01", "ambient", 5000),
                StorageZone(f"ZONE_{i}_02", "refrigerated", 2000),
                StorageZone(f"ZONE_{i}_03", "controlled_atmosphere", 1000)
            ]

            facility = StockpileFacility(
                facility_id = f"FACILITY_{i:03d}",
                name = f"{city.name} Regional Stockpile",
                location=(city.lat, city.lon),
                facility_type = StockpileType.DISTRIBUTION_CENTER,
                storage_zones = zones,
                total_capacity = 8000,
                current_inventory={}  # Initialize empty inventory
            )

            self.stockpile_manager.add_facility(facility)

    def _initialize_infrastructure(self):
        """Initialize infrastructure system."""
        if self.config.debug_mode:
            print("Initializing infrastructure system...")

        if not self.terrain_analyzer:
            self._initialize_terrain()

        self.infrastructure_builder = InfrastructureBuilder(self.terrain_analyzer)

        # Create network nodes from cities
        for location in self.locations.values():
            population = location.properties.get('population', 50000)

            node_type = "city" if population > 100000 else "town"
            if population > 500000:
                node_type = "major_city"

            # Convert lat / lon to grid coordinates (simplified)
            grid_x = int((location.lon + 180) / 360 * self.config.grid_resolution)
            grid_y = int((location.lat + 90) / 180 * self.config.grid_resolution)

            network_node = NetworkNode(
                node_id = location.id,
                name = location.name,
                location=(grid_x, grid_y),
                node_type = node_type,
                population = population,
                economic_importance = location.properties.get('economic_importance', 0.5),
                connectivity_priority = 1 if population > 500000 else 2 if population > 100000 else 3
            )

            self.network_nodes[location.id] = network_node

    def _initialize_supply_chain(self):
        """Initialize supply chain optimization."""
        if self.config.debug_mode:
            print("Initializing supply chain system...")

        self.supply_chain_optimizer = SupplyChainOptimizer()

        # Create supply nodes from locations
        for location in self.locations.values():
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
                location = location,
                node_type = node_type,
                capacity = capacity,
                current_inventory={},
                storage_cost = 0.1,
                handling_capacity={CargoCategory.CONSUMER: capacity * 0.8, CargoCategory.ESSENTIAL: capacity * 0.2}
            )

            self.supply_nodes[location.id] = supply_node

    def _initialize_visualization(self):
        """Initialize visualization system."""
        if self.config.debug_mode:
            print("Initializing visualization system...")

        try:
            # Calculate map bounds based on locations
            if self.locations:
                lats = [loc.lat for loc in self.locations.values()]
                lons = [loc.lon for loc in self.locations.values()]

                map_bounds = (
                    (min(lats) - 1, min(lons) - 1),
                    (max(lats) + 1, max(lons) + 1)
                )
            else:
                # Default bounds
                map_bounds = ((40.0, -80.0), (50.0, -70.0))

            self.map_generator = MapGenerator(map_bounds)

        except ImportError:
            print("Warning: Visualization system not available")
            self.config.enable_visualization = False

    def _generate_cities_from_terrain(self):
        """Generate cities based on terrain analysis."""
        if not self.terrain_analyzer:
            return

        # Find suitable locations for cities based on terrain
        suitable_locations = []

        for x in range(0, self.config.grid_resolution, 10):  # Sample every 10 cells
            for y in range(0, self.config.grid_resolution, 10):
                cell = self.terrain_analyzer.get_terrain_cell(x, y)
                if cell and cell.construction_difficulty < 2.0 and cell.population_density > 50:
                    # Convert grid coordinates to lat / lon (simplified - covering a reasonable region)
                    # Map to region around 40 - 50 lat, -80 to - 70 lon (Northeast US)
                    lat = 40 + (y / self.config.grid_resolution) * 10  # 40 to 50 degrees
                    lon = -80 + (x / self.config.grid_resolution) * 10  # -80 to - 70 degrees

                    suitable_locations.append({
                        'x': x, 'y': y, 'lat': lat, 'lon': lon,
                        'suitability': cell.economic_value * cell.population_density / cell.construction_difficulty
                    })

        # Sort by suitability and select top locations
        suitable_locations.sort(key = lambda x: x['suitability'], reverse = True)
        selected_locations = suitable_locations[:self.config.num_cities]

        # If not enough suitable locations found, generate some basic ones
        if len(selected_locations) < self.config.num_cities:
            print(f"Only found {len(selected_locations)} suitable locations, generating {self.config.num_cities - len(selected_locations)} additional locations")
            for i in range(len(selected_locations), self.config.num_cities):
                x = (i * 20) % self.config.grid_resolution
                y = (i * 15) % self.config.grid_resolution
                lat = 40 + (y / self.config.grid_resolution) * 10
                lon = -80 + (x / self.config.grid_resolution) * 10

                selected_locations.append({
                    'x': x, 'y': y, 'lat': lat, 'lon': lon,
                    'suitability': 50.0  # Default suitability
                })

        # Create location objects
        city_names = [
            "Central City", "Northport", "Eastdale", "Westbridge", "Southfield",
            "Riverside", "Hillcrest", "Valleyview", "Meadowbrook", "Stonegate",
            "Lakewood", "Greenfield", "Fairview", "Millville", "Oakwood"
        ]

        for i, loc_data in enumerate(selected_locations):
            city_name = city_names[i % len(city_names)] + f" {i//len(city_names) + 1}" if i >= len(city_names) else city_names[i]

            # Estimate population based on suitability
            base_population = max(25000, min(1000000, int(loc_data['suitability'] * 10000)))

            location = Location(
                id = f"CITY_{i:03d}",
                name = city_name,
                lat = loc_data['lat'],
                lon = loc_data['lon'],
                location_type="city" if base_population > 100000 else "town"
            )

            location.properties = {
                'population': base_population,
                'economic_importance': min(1.0, loc_data['suitability'] / 100),
                'grid_coordinates': (loc_data['x'], loc_data['y'])
            }

            self.locations[location.id] = location

    def run_simulation(self, duration_days: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete simulation."""
        if not self.is_initialized:
            init_result = self.initialize()
            if not init_result["success"]:
                return init_result

        duration = duration_days or self.config.duration_days
        self.is_running = True

        self._log_event("simulation_started", {"duration_days": duration})

        try:
            for day in range(duration):
                step_result = self.step_simulation()

                if not step_result["success"]:
                    self._log_event("simulation_failed", {"day": day, "error": step_result.get("error")})
                    break

                # Auto - save periodically
                if day % self.config.auto_save_interval == 0:
                    self.save_state()

                # Log progress
                if day % 30 == 0 or self.config.debug_mode:
                    print(f"Simulation day {day}: {step_result['summary']}")

            self.is_running = False

            final_result = {
                "success": True,
                "total_days": duration,
                "final_state": asdict(self.state),
                "performance_metrics": self._calculate_final_metrics()
            }

            self._log_event("simulation_completed", final_result)
            return final_result

        except Exception as e:
            self.is_running = False
            error_result = {
                "success": False,
                "error": str(e),
                "days_completed": self.state.current_day
            }
            self._log_event("simulation_error", error_result)
            return error_result

    def step_simulation(self) -> Dict[str, Any]:
        """Execute one simulation step (day)."""
        if not self.is_initialized:
            return {"success": False, "error": "Simulation not initialized"}

        step_start_time = datetime.now()

        try:
            # Update simulation state
            self.state.current_day += 1
            self.state.current_date += timedelta(days = 1)

            step_metrics = {
                "day": self.state.current_day,
                "date": self.state.current_date.isoformat(),
                "transportation_activity": 0,
                "supply_chain_activity": 0,
                "stockpile_activity": 0,
                "infrastructure_activity": 0
            }

            # Run transportation system
            if self.transportation_system:
                transport_metrics = self._run_transportation_step()
                step_metrics.update(transport_metrics)

            # Run supply chain optimization
            if self.supply_chain_optimizer and self.supply_nodes:
                supply_metrics = self._run_supply_chain_step()
                step_metrics.update(supply_metrics)

            # Run stockpile management
            if self.stockpile_manager:
                stockpile_metrics = self._run_stockpile_step()
                step_metrics.update(stockpile_metrics)

            # Run infrastructure management
            if self.infrastructure_builder:
                infrastructure_metrics = self._run_infrastructure_step()
                step_metrics.update(infrastructure_metrics)

            # Update system - wide metrics
            self._update_system_metrics(step_metrics)

            step_time = (datetime.now() - step_start_time).total_seconds()
            self.step_times.append(step_time)

            return {
                "success": True,
                "day": self.state.current_day,
                "step_time": step_time,
                "metrics": step_metrics,
                "summary": f"Economic output: ${self.state.total_economic_output:,.0f}, Efficiency: {self.state.system_efficiency:.2%}"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_transportation_step(self) -> Dict[str, Any]:
        """Run transportation system for one step."""
        # Generate some cargo for transportation
        cargo_items = []

        if len(self.locations) >= 2:
            locations_list = list(self.locations.values())
            for i in range(3):  # Generate 3 cargo items per day
                origin = np.random.choice(locations_list)
                destination = np.random.choice(locations_list)

                if origin != destination:
                    cargo = CargoItem(
                        id = f"CARGO_{self.state.current_day}_{i:03d}",
                        cargo_type = np.random.choice(["food", "electronics", "machinery"]),
                        weight = np.random.uniform(1000, 5000),
                        volume = np.random.uniform(5, 25),
                        value = np.random.uniform(10000, 100000),
                        priority = np.random.randint(1, 6),
                        origin = origin,
                        destination = destination
                    )
                    cargo_items.append(cargo)

        if cargo_items:
            # Create transport plan
            plan = self.transportation_system.create_transport_plan(cargo_items)

            # Execute transport plan
            execution_result = self.transportation_system.execute_transport_plan(plan)

            if execution_result["success"]:
                self.state.total_transportation_cost += plan.total_cost
                return {
                    "transportation_activity": len(cargo_items),
                    "transportation_cost": plan.total_cost,
                    "transportation_distance": plan.total_distance
                }

        return {"transportation_activity": 0}

    def _run_supply_chain_step(self) -> Dict[str, Any]:
        """Run supply chain optimization for one step."""
        # Create sample supply demands
        supply_demands = []
        supply_nodes_list = list(self.supply_nodes.values())

        if len(supply_nodes_list) >= 2:
            # Create a supply demand pair
            suppliers = [n for n in supply_nodes_list if n.node_type in ["warehouse", "distribution_center"]]
            consumers = [n for n in supply_nodes_list if n.node_type == "consumer"]

            # Fallback if no specific types found
            if not suppliers:
                suppliers = supply_nodes_list[:len(supply_nodes_list)//2]
            if not consumers:
                consumers = supply_nodes_list[len(supply_nodes_list)//2:]

            if suppliers and consumers:
                supplier = np.random.choice(suppliers)
                consumer = np.random.choice(consumers)

            demand = SupplyDemand(
                item_type="electronics",
                supplier_nodes=[supplier],
                consumer_nodes=[consumer],
                supply_quantities={supplier.location.id: 1000},
                demand_quantities={consumer.location.id: 800}
            )
            supply_demands.append(demand)

        if supply_demands and self.transportation_system:
            # Optimize supply chain
            result = self.supply_chain_optimizer.optimize_supply_chain(
                supply_nodes_list, supply_demands, self.transportation_system, time_horizon = 1
            )

            return {
                "supply_chain_activity": len(result["distribution_schedules"]),
                "supply_chain_cost": result["total_cost"],
                "service_level": result["service_level"]
            }

        return {"supply_chain_activity": 0}

    def _run_stockpile_step(self) -> Dict[str, Any]:
        """Run stockpile management for one step."""
        # Check system status
        status = self.stockpile_manager.check_system_status()

        return {
            "stockpile_activity": status["total_facilities"],
            "stockpile_alerts": status["total_alerts"],
            "stockpile_utilization": status.get("performance_metrics", {}).get("system_utilization", 0)
        }

    def _run_infrastructure_step(self) -> Dict[str, Any]:
        """Run infrastructure management for one step."""
        # Get network status
        status = self.infrastructure_builder.get_network_status()

        return {
            "infrastructure_activity": status["total_segments"],
            "infrastructure_connectivity": status.get("average_connectivity", 0)
        }

    def _update_system_metrics(self, step_metrics: Dict[str, Any]):
        """Update system - wide metrics."""
        # Update economic output based on activities
        transport_contribution = step_metrics.get("transportation_activity", 0) * 10000
        supply_contribution = step_metrics.get("supply_chain_activity", 0) * 50000

        self.state.total_economic_output += transport_contribution + supply_contribution

        # Update system efficiency based on service levels and utilization
        service_level = step_metrics.get("service_level", 0.8)
        utilization = step_metrics.get("stockpile_utilization", 0.5)

        # Ensure valid values
        if np.isnan(service_level) or service_level is None:
            service_level = 0.8
        if np.isnan(utilization) or utilization is None:
            utilization = 0.5

        self.state.system_efficiency = (service_level + utilization) / 2

        # Update population satisfaction (simplified)
        alert_penalty = min(0.1, step_metrics.get("stockpile_alerts", 0) * 0.01)
        self.state.population_satisfaction = max(0.5, 0.9 - alert_penalty)

        # Store metrics history
        self.state.metrics_history.append({
            "day": self.state.current_day,
            **step_metrics,
            "economic_output": self.state.total_economic_output,
            "system_efficiency": self.state.system_efficiency,
            "population_satisfaction": self.state.population_satisfaction
        })

    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final performance metrics."""
        if not self.state.metrics_history:
            return {}

        return {
            "average_step_time": np.mean(self.step_times) if self.step_times else 0,
            "total_economic_output": self.state.total_economic_output,
            "average_system_efficiency": np.mean([m.get("system_efficiency", 0) for m in self.state.metrics_history]),
            "average_population_satisfaction": np.mean([m.get("population_satisfaction", 0) for m in self.state.metrics_history]),
            "total_transportation_cost": self.state.total_transportation_cost,
            "total_infrastructure_cost": self.state.total_infrastructure_cost
        }

    def _get_initialized_systems(self) -> List[str]:
        """Get list of successfully initialized systems."""
        systems = []
        if self.transportation_system:
            systems.append("transportation")
        if self.supply_chain_optimizer:
            systems.append("supply_chain")
        if self.stockpile_manager:
            systems.append("stockpiles")
        if self.infrastructure_builder:
            systems.append("infrastructure")
        if self.terrain_analyzer:
            systems.append("terrain")
        if self.map_generator:
            systems.append("visualization")
        return systems

    def _log_event(self, event_type: str, data: Dict[str, Any] = None):
        """Log a simulation event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "simulation_day": self.state.current_day,
            "event_type": event_type,
            "data": data or {}
        }

        self.state.events_log.append(event)

        if self.config.debug_mode:
            print(f"Event: {event_type} - {data}")

    def save_state(self, filename: Optional[str] = None) -> str:
        """Save simulation state to file."""
        if not filename:
            filename = f"simulation_state_day_{self.state.current_day:04d}.json"

        filepath = self.output_dir / filename

        # Prepare serializable state
        save_data = {
            "config": asdict(self.config),
            "state": asdict(self.state),
            "locations": {lid: {
                "id": loc.id,
                "name": loc.name,
                "lat": loc.lat,
                "lon": loc.lon,
                "location_type": loc.location_type,
                "properties": loc.properties
            } for lid, loc in self.locations.items()},
            "systems_initialized": self._get_initialized_systems(),
            "performance_metrics": self._calculate_final_metrics()
        }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent = 2, default = str)

        return str(filepath)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive simulation status."""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "current_day": self.state.current_day,
            "current_date": self.state.current_date.isoformat(),
            "systems_initialized": self._get_initialized_systems(),
            "locations_count": len(self.locations),
            "total_economic_output": self.state.total_economic_output,
            "system_efficiency": self.state.system_efficiency,
            "population_satisfaction": self.state.population_satisfaction,
            "recent_events": self.state.events_log[-5:] if self.state.events_log else []
        }

def create_test_simulation() -> EconomicSimulation:
    """Create a test simulation with default configuration."""
    config = SimulationConfig(
        simulation_name="Test Economic Simulation",
        duration_days = 30,
        map_size_km = 200,
        grid_resolution = 100,
        num_cities = 5,
        num_regions = 3,
        debug_mode = True
    )

    return EconomicSimulation(config)

if __name__ == "__main__":
    # Test simulation creation and initialization
    print("Creating test simulation...")
    sim = create_test_simulation()

    print("Initializing simulation...")
    init_result = sim.initialize()

    if init_result["success"]:
        print(f"Simulation initialized successfully in {init_result['initialization_time']:.2f} seconds")
        print(f"Systems initialized: {', '.join(init_result['systems_initialized'])}")
        print(f"Locations created: {init_result['locations_created']}")

        # Run a few simulation steps
        print("\nRunning simulation steps...")
        for i in range(3):
            step_result = sim.step_simulation()
            if step_result["success"]:
                print(f"Day {step_result['day']}: {step_result['summary']}")
            else:
                print(f"Step failed: {step_result['error']}")
                break

        # Get final status
        status = sim.get_status()
        print(f"\nFinal status: {status['current_day']} days completed")
        print(f"Economic output: ${status['total_economic_output']:,.0f}")
        print(f"System efficiency: {status['system_efficiency']:.2%}")

        # Save state
        save_path = sim.save_state()
        print(f"State saved to: {save_path}")

    else:
        print(f"Simulation initialization failed: {init_result.get('error', 'Unknown error')}")
