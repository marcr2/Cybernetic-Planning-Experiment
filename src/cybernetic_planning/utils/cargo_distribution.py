#!/usr / bin / env python3
"""
Cargo Distribution Optimization System

Implements advanced cargo distribution algorithms including supply chain optimization,
cargo prioritization, scheduling, and cost - benefit analysis for multi - modal transport.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
import numpy as np

try:
        CargoItem, Location, TransportPlan, TransportationSystem,
        TransportMode, VehicleType, calculate_distance
    )
except ImportError:
    try:
            CargoItem, Location, TransportPlan, TransportationSystem,
            TransportMode, VehicleType, calculate_distance
        )
    except ImportError:
        # Create placeholder classes
        class CargoItem: pass
        class Location: pass
        class TransportPlan: pass
        class TransportationSystem: pass
        class TransportMode: pass
        class VehicleType: pass
        def calculate_distance(a, b): return 100.0

class CargoCategory(Enum):
    """Categories of cargo with different handling requirements."""
    ESSENTIAL = "essential"           # Food, medicine, fuel
    INDUSTRIAL = "industrial"         # Raw materials, machinery
    CONSUMER = "consumer"            # Manufactured goods
    LUXURY = "luxury"                # Non - essential items
    HAZARDOUS = "hazardous"          # Dangerous materials
    PERISHABLE = "perishable"        # Time - sensitive items

class DeliveryPriority(Enum):
    """Priority levels for cargo delivery."""
    EMERGENCY = 5    # Medical supplies, disaster relief
    URGENT = 4       # Essential services, high - value items
    NORMAL = 3       # Standard delivery
    LOW = 2          # Non - critical items
    BULK = 1         # Large quantity, flexible timing

@dataclass
class SupplyNode:
    """Node in the supply chain network."""
    location: Location
    node_type: str  # "supplier", "warehouse", "distribution_center", "consumer"
    capacity: float  # Maximum throughput per time period
    current_inventory: Dict[str, float]  # item_type -> quantity
    storage_cost: float  # Cost per unit per time period
    handling_capacity: Dict[CargoCategory, float]  # Specialized handling capabilities
    operating_hours: Tuple[int, int] = (0, 24)  # (start_hour, end_hour)
    properties: Dict[str, Any] = field(default_factory = dict)

@dataclass
class SupplyDemand:
    """Supply and demand information for cargo items."""
    item_type: str
    supplier_nodes: List[SupplyNode]
    consumer_nodes: List[SupplyNode]
    supply_quantities: Dict[str, float]  # node_id -> available quantity
    demand_quantities: Dict[str, float]  # node_id -> required quantity
    seasonal_factors: Dict[int, float] = field(default_factory = lambda: {i: 1.0 for i in range(12)})
    price_elasticity: float = 1.0

@dataclass
class DistributionSchedule:
    """Schedule for cargo distribution."""
    cargo_batches: List[List[CargoItem]]
    transport_plans: List[TransportPlan]
    schedule_timeline: List[Tuple[float, str, Dict[str, Any]]]  # (time, event_type, details)
    total_cost: float
    total_time: float
    resource_utilization: Dict[str, float]
    performance_metrics: Dict[str, Any]

class InventoryManager:
    """Manages inventory levels at supply chain nodes."""

    def __init__(self):
        self.inventory_levels = {}  # node_id -> {item_type -> quantity}
        self.safety_stock_levels = {}  # node_id -> {item_type -> min_quantity}
        self.reorder_points = {}  # node_id -> {item_type -> reorder_level}
        self.inventory_costs = {}  # node_id -> {item_type -> cost_per_unit_per_day}

    def update_inventory(self, node_id: str, item_type: str, quantity_change: float):
        """Update inventory level at a node."""
        if node_id not in self.inventory_levels:
            self.inventory_levels[node_id] = {}

        current = self.inventory_levels[node_id].get(item_type, 0)
        self.inventory_levels[node_id][item_type] = max(0, current + quantity_change)

    def check_reorder_needed(self, node_id: str, item_type: str) -> bool:
        """Check if reordering is needed for an item at a node."""
        current_level = self.inventory_levels.get(node_id, {}).get(item_type, 0)
        reorder_point = self.reorder_points.get(node_id, {}).get(item_type, 0)
        return current_level <= reorder_point

    def calculate_optimal_order_quantity(self, node_id: str, item_type: str,
                                       demand_rate: float, lead_time: float) -> float:
        """Calculate optimal order quantity using EOQ model."""
        # Economic Order Quantity formula
        holding_cost = self.inventory_costs.get(node_id, {}).get(item_type, 1.0)
        ordering_cost = 100  # Fixed cost per order (should be configurable)

        if holding_cost > 0 and demand_rate > 0:
            eoq = math.sqrt(2 * demand_rate * ordering_cost / holding_cost)
            return eoq
        return demand_rate * lead_time  # Fallback to lead - time demand

class CargoConsolidationOptimizer:
    """Optimizes cargo consolidation to maximize vehicle utilization."""

    def __init__(self, consolidation_threshold: float = 0.8):
        self.consolidation_threshold = consolidation_threshold  # Min utilization before consolidating

    def consolidate_cargo(self, cargo_items: List[CargoItem],
                         vehicle_specs: Dict[VehicleType, Any]) -> List[List[CargoItem]]:
        """Group cargo items into consolidated shipments."""
        # Group by origin - destination pairs
        od_groups = defaultdict(list)
        for item in cargo_items:
            key = (item.origin.id, item.destination.id)
            od_groups[key].append(item)

        consolidated_batches = []

        for (origin_id, dest_id), items in od_groups.items():
            # Sort items by priority (highest first)
            items.sort(key = lambda x: x.priority, reverse = True)

            # Use bin - packing algorithm to create batches
            batches = self._bin_pack_cargo(items, vehicle_specs)
            consolidated_batches.extend(batches)

        return consolidated_batches

    def _bin_pack_cargo(self, items: List[CargoItem],
                       vehicle_specs: Dict[VehicleType, Any]) -> List[List[CargoItem]]:
        """Use bin - packing algorithm to optimize cargo consolidation."""
        batches = []

        # Choose the most appropriate vehicle type for this route
        # For simplicity, use heavy truck as default
        vehicle_capacity = 26000  # kg
        vehicle_volume = 80       # m³

        current_batch = []
        current_weight = 0
        current_volume = 0

        for item in items:
            # Check if item fits in current batch
            if (current_weight + item.weight <= vehicle_capacity and
                current_volume + item.volume <= vehicle_volume):
                current_batch.append(item)
                current_weight += item.weight
                current_volume += item.volume
            else:
                # Start new batch if current one has items
                if current_batch:
                    batches.append(current_batch)

                current_batch = [item]
                current_weight = item.weight
                current_volume = item.volume

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches

class RoutingOptimizer:
    """Advanced routing optimization with multiple objectives."""

    def __init__(self):
        self.cost_weight = 0.4
        self.time_weight = 0.3
        self.reliability_weight = 0.2
        self.environmental_weight = 0.1

    def optimize_distribution_routes(self, supply_demands: List[SupplyDemand],
                                   supply_nodes: List[SupplyNode],
                                   transportation_system: TransportationSystem) -> DistributionSchedule:
        """Optimize routes for multiple supply - demand pairs."""

        # Generate cargo items from supply - demand pairs
        all_cargo = self._generate_cargo_from_demands(supply_demands, supply_nodes)

        # Consolidate cargo for efficiency
        consolidator = CargoConsolidationOptimizer()
        consolidated_batches = consolidator.consolidate_cargo(all_cargo, transportation_system.vehicle_specs)

        # Create transport plans for each batch
        transport_plans = []
        total_cost = 0
        total_time = 0

        for batch in consolidated_batches:
            if batch:  # Only process non - empty batches
                plan = transportation_system.create_transport_plan(batch, 'fuel_efficient')
                transport_plans.append(plan)
                total_cost += plan.total_cost
                total_time = max(total_time, plan.total_time)  # Parallel execution assumption

        # Generate schedule timeline
        timeline = self._generate_schedule_timeline(consolidated_batches, transport_plans)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(transport_plans, all_cargo)

        # Calculate resource utilization
        resource_utilization = self._calculate_resource_utilization(transport_plans, transportation_system)

        return DistributionSchedule(
            cargo_batches = consolidated_batches,
            transport_plans = transport_plans,
            schedule_timeline = timeline,
            total_cost = total_cost,
            total_time = total_time,
            resource_utilization = resource_utilization,
            performance_metrics = performance_metrics
        )

    def _generate_cargo_from_demands(self, supply_demands: List[SupplyDemand],
                                   supply_nodes: List[SupplyNode]) -> List[CargoItem]:
        """Convert supply - demand pairs into cargo items."""
        cargo_items = []
        cargo_id_counter = 1

        for demand in supply_demands:
            for consumer_node_id, demand_qty in demand.demand_quantities.items():
                # Find consumer node
                consumer_node = next((n for n in supply_nodes if n.location.id == consumer_node_id), None)
                if not consumer_node:
                    continue

                # Allocate from suppliers based on availability and cost
                remaining_demand = demand_qty

                for supplier_node_id, supply_qty in demand.supply_quantities.items():
                    if remaining_demand <= 0:
                        break

                    supplier_node = next((n for n in supply_nodes if n.location.id == supplier_node_id), None)
                    if not supplier_node:
                        continue

                    # Determine quantity to ship from this supplier
                    ship_qty = min(remaining_demand, supply_qty)

                    if ship_qty > 0:
                        # Estimate cargo properties based on item type
                        weight = ship_qty * self._estimate_item_weight(demand.item_type)
                        volume = ship_qty * self._estimate_item_volume(demand.item_type)
                        value = ship_qty * self._estimate_item_value(demand.item_type)
                        priority = self._determine_priority(demand.item_type)

                        cargo_item = CargoItem(
                            id = f"CARGO_{cargo_id_counter:05d}",
                            cargo_type = demand.item_type,
                            weight = weight,
                            volume = volume,
                            value = value,
                            priority = priority,
                            origin = supplier_node.location,
                            destination = consumer_node.location
                        )

                        cargo_items.append(cargo_item)
                        cargo_id_counter += 1
                        remaining_demand -= ship_qty

        return cargo_items

    def _estimate_item_weight(self, item_type: str) -> float:
        """Estimate weight per unit for different item types."""
        weight_map = {
            "food": 1.0,        # kg per unit
            "electronics": 0.5,
            "machinery": 50.0,
            "textiles": 0.3,
            "chemicals": 2.0,
            "pharmaceuticals": 0.1,
            "fuel": 0.8
        }
        return weight_map.get(item_type, 1.0)

    def _estimate_item_volume(self, item_type: str) -> float:
        """Estimate volume per unit for different item types."""
        volume_map = {
            "food": 0.001,      # m³ per unit
            "electronics": 0.002,
            "machinery": 0.05,
            "textiles": 0.003,
            "chemicals": 0.002,
            "pharmaceuticals": 0.0005,
            "fuel": 0.001
        }
        return volume_map.get(item_type, 0.001)

    def _estimate_item_value(self, item_type: str) -> float:
        """Estimate value per unit for different item types."""
        value_map = {
            "food": 5.0,        # $ per unit
            "electronics": 100.0,
            "machinery": 1000.0,
            "textiles": 20.0,
            "chemicals": 50.0,
            "pharmaceuticals": 200.0,
            "fuel": 2.0
        }
        return value_map.get(item_type, 10.0)

    def _determine_priority(self, item_type: str) -> int:
        """Determine priority level based on item type."""
        priority_map = {
            "pharmaceuticals": 5,  # Emergency
            "food": 4,            # Urgent
            "fuel": 4,            # Urgent
            "machinery": 3,       # Normal
            "electronics": 3,     # Normal
            "chemicals": 2,       # Low
            "textiles": 2         # Low
        }
        return priority_map.get(item_type, 3)

    def _generate_schedule_timeline(self, cargo_batches: List[List[CargoItem]],
                                  transport_plans: List[TransportPlan]) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Generate a timeline of scheduled events."""
        timeline = []
        current_time = 0.0

        for i, (batch, plan) in enumerate(zip(cargo_batches, transport_plans)):
            if not batch:
                continue

            # Departure event
            timeline.append((current_time, "departure", {
                "batch_id": i,
                "cargo_count": len(batch),
                "origin": batch[0].origin.name if batch else "Unknown",
                "transport_plan": plan
            }))

            # Arrival event
            arrival_time = current_time + plan.total_time
            timeline.append((arrival_time, "arrival", {
                "batch_id": i,
                "cargo_count": len(batch),
                "destination": batch[0].destination.name if batch else "Unknown",
                "transport_plan": plan
            }))

        return sorted(timeline, key = lambda x: x[0])

    def _calculate_performance_metrics(self, transport_plans: List[TransportPlan],
                                     all_cargo: List[CargoItem]) -> Dict[str, Any]:
        """Calculate various performance metrics."""
        if not transport_plans:
            return {}

        total_distance = sum(plan.total_distance for plan in transport_plans)
        total_cost = sum(plan.total_cost for plan in transport_plans)
        total_emissions = sum(plan.total_emissions for plan in transport_plans)
        total_cargo_value = sum(item.value for item in all_cargo)

        return {
            "total_shipments": len(transport_plans),
            "total_cargo_items": len(all_cargo),
            "average_distance_per_shipment": total_distance / len(transport_plans),
            "cost_per_km": total_cost / max(total_distance, 1),
            "cost_per_cargo_value": total_cost / max(total_cargo_value, 1),
            "emissions_per_km": total_emissions / max(total_distance, 1),
            "cargo_consolidation_ratio": len(all_cargo) / max(len(transport_plans), 1)
        }

    def _calculate_resource_utilization(self, transport_plans: List[TransportPlan],
                                      transportation_system: TransportationSystem) -> Dict[str, float]:
        """Calculate resource utilization percentages."""
        vehicle_usage = defaultdict(int)
        fuel_usage = defaultdict(float)

        for plan in transport_plans:
            for vehicle_type, count in plan.vehicles_required.items():
                vehicle_usage[vehicle_type.value] += count

            for fuel_type, amount in plan.fuel_required.items():
                fuel_usage[fuel_type] += amount

        # Calculate utilization percentages (would need fleet size data)
        utilization = {}
        for vehicle_type, used_count in vehicle_usage.items():
            available = transportation_system.fleet_manager.available_vehicles.get(
                VehicleType(vehicle_type), 100  # Default fleet size
            )
            utilization[f"{vehicle_type}_utilization"] = min(used_count / max(available, 1), 1.0)

        return utilization

class SupplyChainOptimizer:
    """Optimizes the entire supply chain network."""

    def __init__(self):
        self.inventory_manager = InventoryManager()
        self.routing_optimizer = RoutingOptimizer()

    def optimize_supply_chain(self, supply_nodes: List[SupplyNode],
                            supply_demands: List[SupplyDemand],
                            transportation_system: TransportationSystem,
                            time_horizon: int = 30) -> Dict[str, Any]:
        """Optimize the entire supply chain over a time horizon."""

        optimization_result = {
            "distribution_schedules": [],
            "inventory_recommendations": {},
            "total_cost": 0,
            "service_level": 0,
            "recommendations": []
        }

        # Simulate over time horizon
        for day in range(time_horizon):
            # Check inventory levels and generate reorder recommendations
            reorder_recommendations = self._check_inventory_levels(supply_nodes, supply_demands, day)

            # Optimize distribution for current demands
            schedule = self.routing_optimizer.optimize_distribution_routes(
                supply_demands, supply_nodes, transportation_system
            )

            optimization_result["distribution_schedules"].append(schedule)
            optimization_result["total_cost"] += schedule.total_cost

            # Update inventory levels based on deliveries
            self._update_inventory_from_schedule(schedule, supply_nodes)

        # Calculate average service level
        optimization_result["service_level"] = self._calculate_service_level(
            optimization_result["distribution_schedules"]
        )

        # Generate recommendations
        optimization_result["recommendations"] = self._generate_recommendations(
            optimization_result
        )

        return optimization_result

    def _check_inventory_levels(self, supply_nodes: List[SupplyNode],
                              supply_demands: List[SupplyDemand],
                              current_day: int) -> List[Dict[str, Any]]:
        """Check inventory levels and generate reorder recommendations."""
        recommendations = []

        for node in supply_nodes:
            if node.node_type in ["warehouse", "distribution_center"]:
                for demand in supply_demands:
                    for item_type in demand.demand_quantities.keys():
                        if self.inventory_manager.check_reorder_needed(node.location.id, item_type):
                            # Calculate demand rate (simplified)
                            daily_demand = sum(demand.demand_quantities.values()) / 30  # Monthly to daily

                            # Calculate optimal order quantity
                            optimal_qty = self.inventory_manager.calculate_optimal_order_quantity(
                                node.location.id, item_type, daily_demand, 7  # 7 - day lead time
                            )

                            recommendations.append({
                                "node_id": node.location.id,
                                "item_type": item_type,
                                "recommended_order_quantity": optimal_qty,
                                "urgency": "high" if daily_demand > 100 else "normal",
                                "day": current_day
                            })

        return recommendations

    def _update_inventory_from_schedule(self, schedule: DistributionSchedule,
                                      supply_nodes: List[SupplyNode]):
        """Update inventory levels based on distribution schedule."""
        for plan in schedule.transport_plans:
            for cargo_item in plan.cargo:
                # Decrease inventory at origin
                self.inventory_manager.update_inventory(
                    cargo_item.origin.id, cargo_item.cargo_type, -cargo_item.weight
                )

                # Increase inventory at destination
                self.inventory_manager.update_inventory(
                    cargo_item.destination.id, cargo_item.cargo_type, cargo_item.weight
                )

    def _calculate_service_level(self, schedules: List[DistributionSchedule]) -> float:
        """Calculate average service level across all schedules."""
        if not schedules:
            return 0.0

        total_on_time = 0
        total_deliveries = 0

        for schedule in schedules:
            for plan in schedule.transport_plans:
                total_deliveries += len(plan.cargo)
                # Assume 95% on - time delivery rate (would need actual tracking)
                total_on_time += len(plan.cargo) * 0.95

        return total_on_time / max(total_deliveries, 1)

    def _generate_recommendations(self, optimization_result: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        avg_cost = optimization_result["total_cost"] / max(len(optimization_result["distribution_schedules"]), 1)
        service_level = optimization_result["service_level"]

        if avg_cost > 10000:  # Threshold for high cost
            recommendations.append("Consider consolidating more shipments to reduce transportation costs")

        if service_level < 0.9:  # Below 90% service level
            recommendations.append("Increase safety stock levels to improve service level")

        # Analyze vehicle utilization
        schedules = optimization_result["distribution_schedules"]
        if schedules:
            avg_utilization = np.mean([
                list(schedule.resource_utilization.values())
                for schedule in schedules
                if schedule.resource_utilization
            ])

            if avg_utilization < 0.7:  # Low utilization
                recommendations.append("Improve vehicle utilization through better consolidation")

        return recommendations

if __name__ == "__main__":
    # Example usage
    from .transportation_system import TransportationSystem, Location

    # Create transportation system
    transport_system = TransportationSystem()

    # Create test locations
    locations = [
        Location("WAREHOUSE_1", "Central Warehouse", 40.7128, -74.0060),
        Location("STORE_1", "Retail Store 1", 41.8781, -87.6298),
        Location("STORE_2", "Retail Store 2", 34.0522, -118.2437)
    ]

    for loc in locations:
        transport_system.add_location(loc)

    # Create supply nodes
    supply_nodes = [
        SupplyNode(
            location = locations[0],
            node_type="warehouse",
            capacity = 10000,
            current_inventory={"electronics": 5000, "food": 3000},
            storage_cost = 0.1,
            handling_capacity={CargoCategory.CONSUMER: 1000, CargoCategory.ESSENTIAL: 500}
        ),
        SupplyNode(
            location = locations[1],
            node_type="consumer",
            capacity = 1000,
            current_inventory={},
            storage_cost = 0.2,
            handling_capacity={CargoCategory.CONSUMER: 200}
        ),
        SupplyNode(
            location = locations[2],
            node_type="consumer",
            capacity = 1000,
            current_inventory={},
            storage_cost = 0.2,
            handling_capacity={CargoCategory.CONSUMER: 200}
        )
    ]

    # Create supply demands
    supply_demands = [
        SupplyDemand(
            item_type="electronics",
            supplier_nodes=[supply_nodes[0]],
            consumer_nodes=[supply_nodes[1], supply_nodes[2]],
            supply_quantities={"WAREHOUSE_1": 5000},
            demand_quantities={"STORE_1": 1500, "STORE_2": 2000}
        )
    ]

    # Optimize supply chain
    optimizer = SupplyChainOptimizer()
    result = optimizer.optimize_supply_chain(supply_nodes, supply_demands, transport_system)

    print(f"Supply chain optimization result:")
    print(f"Total cost: ${result['total_cost']:.2f}")
    print(f"Service level: {result['service_level']:.1%}")
    print(f"Number of schedules: {len(result['distribution_schedules'])}")
    for rec in result['recommendations']:
        print(f"Recommendation: {rec}")
