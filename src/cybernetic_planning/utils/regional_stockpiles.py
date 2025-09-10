#!/usr / bin / env python3
"""
Regional Stockpile Management System

Implements comprehensive regional inventory management including capacity constraints,
storage costs, automatic reorder points, safety stock calculations, turnover tracking,
and expiration / degradation management.
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta

class StockpileType(Enum):
    """Types of stockpile facilities."""
    STRATEGIC_RESERVE = "strategic_reserve"      # National / regional emergency reserves
    DISTRIBUTION_CENTER = "distribution_center" # Regional distribution hubs
    WAREHOUSE = "warehouse"                      # General storage
    COLD_STORAGE = "cold_storage"               # Temperature - controlled storage
    HAZMAT_FACILITY = "hazmat_facility"         # Hazardous materials storage
    FUEL_DEPOT = "fuel_depot"                   # Fuel and energy storage
    GRAIN_ELEVATOR = "grain_elevator"           # Agricultural products
    RAW_MATERIALS = "raw_materials"             # Industrial inputs

class ItemCondition(Enum):
    """Condition states of stored items."""
    EXCELLENT = "excellent"     # 100% value
    GOOD = "good"              # 90 - 99% value
    FAIR = "fair"              # 70 - 89% value
    POOR = "poor"              # 50 - 69% value
    EXPIRED = "expired"        # 0% value, disposal needed

class StorageCategory(Enum):
    """Categories of items with different storage requirements."""
    FOOD_PERISHABLE = "food_perishable"        # Days to weeks shelf life
    FOOD_NONPERISHABLE = "food_nonperishable"  # Years shelf life
    PHARMACEUTICALS = "pharmaceuticals"         # Controlled temperature / humidity
    CHEMICALS = "chemicals"                     # Special handling required
    ELECTRONICS = "electronics"                # Moisture sensitive
    TEXTILES = "textiles"                      # Pest control needed
    MACHINERY = "machinery"                    # Corrosion prevention
    FUEL = "fuel"                              # Flammable, volatile
    RAW_MATERIALS = "raw_materials"            # Various requirements

@dataclass
class StoredItem:
    """Individual item or batch stored in a stockpile."""
    id: str
    item_type: str
    category: StorageCategory
    quantity: float
    unit: str  # kg, liters, pieces, etc.
    condition: ItemCondition
    storage_date: datetime
    expiration_date: Optional[datetime]
    value_per_unit: float
    storage_requirements: List[str] = field(default_factory = list)
    batch_number: str = ""
    supplier: str = ""
    properties: Dict[str, Any] = field(default_factory = dict)

@dataclass
class StorageZone:
    """Specific storage zone within a stockpile facility."""
    zone_id: str
    zone_type: str  # "ambient", "refrigerated", "frozen", "controlled_atmosphere"
    capacity: float  # Maximum storage capacity
    current_utilization: float = 0.0
    temperature_range: Tuple[float, float] = (15, 25)  # Celsius
    humidity_range: Tuple[float, float] = (40, 60)     # Percentage
    special_features: List[str] = field(default_factory = list)
    operating_cost: float = 0.0  # Cost per unit per day
    energy_consumption: float = 0.0  # kWh per day

@dataclass
class StockpileFacility:
    """Regional stockpile facility with multiple storage zones."""
    facility_id: str
    name: str
    location: Tuple[float, float]  # lat, lon
    facility_type: StockpileType
    storage_zones: List[StorageZone]
    total_capacity: float
    current_inventory: Dict[str, List[StoredItem]]  # item_type -> items
    operating_hours: Tuple[int, int] = (0, 24)
    staff_count: int = 10
    security_level: str = "standard"  # standard, high, maximum
    environmental_controls: Dict[str, Any] = field(default_factory = dict)
    properties: Dict[str, Any] = field(default_factory = dict)

@dataclass
class ReorderRule:
    """Automatic reordering rule for stockpile items."""
    item_type: str
    facility_id: str
    reorder_point: float      # Quantity at which to reorder
    safety_stock: float       # Minimum quantity to maintain
    order_quantity: float     # Standard order quantity (EOQ)
    max_stock_level: float    # Maximum inventory level
    lead_time_days: int       # Expected delivery time
    supplier_priority: List[str] = field(default_factory = list)
    seasonal_adjustment: Dict[int, float] = field(default_factory = dict)  # month -> multiplier
    cost_per_unit: float = 0.0
    holding_cost_rate: float = 0.2  # Annual holding cost as % of value

@dataclass
class StockpileAlert:
    """Alert for stockpile management issues."""
    alert_id: str
    facility_id: str
    alert_type: str  # "low_stock", "expired", "capacity", "condition_degraded"
    severity: str    # "low", "medium", "high", "critical"
    item_type: str
    current_level: float
    threshold_level: float
    message: str
    timestamp: datetime
    resolved: bool = False

class DegradationModel:
    """Models item degradation over time."""

    def __init__(self):
        # Degradation rates per day for different categories
        self.degradation_rates = {
            StorageCategory.FOOD_PERISHABLE: 0.05,      # 5% per day
            StorageCategory.FOOD_NONPERISHABLE: 0.0001, # 0.01% per day
            StorageCategory.PHARMACEUTICALS: 0.002,      # 0.2% per day
            StorageCategory.CHEMICALS: 0.001,            # 0.1% per day
            StorageCategory.ELECTRONICS: 0.0005,         # 0.05% per day
            StorageCategory.TEXTILES: 0.0003,            # 0.03% per day
            StorageCategory.MACHINERY: 0.0002,           # 0.02% per day
            StorageCategory.FUEL: 0.003,                 # 0.3% per day
            StorageCategory.RAW_MATERIALS: 0.0001        # 0.01% per day
        }

        # Environmental factor multipliers
        self.temperature_factors = {
            "too_hot": 2.0,      # Doubles degradation rate
            "optimal": 1.0,      # Normal rate
            "too_cold": 1.2      # 20% increase
        }

        self.humidity_factors = {
            "too_humid": 1.5,    # 50% increase
            "optimal": 1.0,      # Normal rate
            "too_dry": 1.3       # 30% increase
        }

    def calculate_degradation(self, item: StoredItem, storage_zone: StorageZone,
                            days_elapsed: int) -> Tuple[ItemCondition, float]:
        """Calculate item degradation after specified days."""
        base_rate = self.degradation_rates.get(item.category, 0.001)

        # Environmental adjustments
        temp_factor = self._get_temperature_factor(storage_zone)
        humidity_factor = self._get_humidity_factor(storage_zone)

        adjusted_rate = base_rate * temp_factor * humidity_factor
        total_degradation = min(1.0, adjusted_rate * days_elapsed)

        # Determine new condition
        remaining_quality = 1.0 - total_degradation

        if remaining_quality > 0.95:
            new_condition = ItemCondition.EXCELLENT
        elif remaining_quality > 0.85:
            new_condition = ItemCondition.GOOD
        elif remaining_quality > 0.65:
            new_condition = ItemCondition.FAIR
        elif remaining_quality > 0.45:
            new_condition = ItemCondition.POOR
        else:
            new_condition = ItemCondition.EXPIRED

        return new_condition, remaining_quality

    def _get_temperature_factor(self, storage_zone: StorageZone) -> float:
        """Get temperature degradation factor."""
        # Simplified - would need actual temperature monitoring
        return self.temperature_factors["optimal"]

    def _get_humidity_factor(self, storage_zone: StorageZone) -> float:
        """Get humidity degradation factor."""
        # Simplified - would need actual humidity monitoring
        return self.humidity_factors["optimal"]

class InventoryOptimizer:
    """Optimizes inventory levels and reorder policies."""

    def calculate_eoq(self, annual_demand: float, ordering_cost: float,
                     holding_cost: float) -> float:
        """Calculate Economic Order Quantity."""
        if holding_cost <= 0 or annual_demand <= 0:
            return annual_demand / 12  # Monthly demand as fallback

        return math.sqrt(2 * annual_demand * ordering_cost / holding_cost)

    def calculate_reorder_point(self, daily_demand: float, lead_time_days: int,
                              service_level: float = 0.95) -> float:
        """Calculate reorder point with safety stock."""
        # Basic reorder point
        expected_demand = daily_demand * lead_time_days

        # Safety stock calculation (simplified)
        demand_std = daily_demand * 0.2  # Assume 20% coefficient of variation
        lead_time_std = lead_time_days * 0.1  # Assume 10% lead time variation

        # Z - score for service level
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z_score = z_scores.get(service_level, 1.65)

        safety_stock = z_score * math.sqrt(
            (lead_time_days * demand_std**2) + (daily_demand * lead_time_std)**2
        )

        return expected_demand + safety_stock

    def calculate_safety_stock(self, daily_demand: float, lead_time_days: int,
                             service_level: float = 0.95) -> float:
        """Calculate safety stock level."""
        reorder_point = self.calculate_reorder_point(daily_demand, lead_time_days, service_level)
        expected_demand = daily_demand * lead_time_days
        return reorder_point - expected_demand

    def optimize_abc_classification(self, items: List[StoredItem]) -> Dict[str, str]:
        """Classify items using ABC analysis."""
        # Calculate annual value for each item type
        item_values = defaultdict(float)
        for item in items:
            annual_value = item.value_per_unit * item.quantity * 12  # Assume monthly turnover
            item_values[item.item_type] += annual_value

        # Sort by value (descending)
        sorted_items = sorted(item_values.items(), key = lambda x: x[1], reverse = True)

        total_value = sum(item_values.values())
        cumulative_value = 0
        classification = {}

        for item_type, value in sorted_items:
            cumulative_percentage = (cumulative_value + value) / total_value

            if cumulative_percentage <= 0.8:
                classification[item_type] = 'A'  # High value items (80% of value)
            elif cumulative_percentage <= 0.95:
                classification[item_type] = 'B'  # Medium value items (15% of value)
            else:
                classification[item_type] = 'C'  # Low value items (5% of value)

            cumulative_value += value

        return classification

class StockpileManager:
    """Main manager for regional stockpile operations."""

    def __init__(self):
        self.facilities = {}  # facility_id -> StockpileFacility
        self.reorder_rules = {}  # facility_id -> {item_type -> ReorderRule}
        self.alerts = deque(maxlen = 1000)  # Recent alerts
        self.degradation_model = DegradationModel()
        self.inventory_optimizer = InventoryOptimizer()
        self.performance_metrics = {}

        # System parameters
        self.alert_check_interval = 24  # hours
        self.degradation_check_interval = 24  # hours
        self.last_check_time = datetime.now()

    def add_facility(self, facility: StockpileFacility):
        """Add a new stockpile facility."""
        self.facilities[facility.facility_id] = facility
        self.reorder_rules[facility.facility_id] = {}

    def add_reorder_rule(self, rule: ReorderRule):
        """Add an automatic reorder rule."""
        if rule.facility_id not in self.reorder_rules:
            self.reorder_rules[rule.facility_id] = {}
        self.reorder_rules[rule.facility_id][rule.item_type] = rule

    def receive_items(self, facility_id: str, items: List[StoredItem]) -> Dict[str, Any]:
        """Process incoming items to a facility."""
        if facility_id not in self.facilities:
            return {"success": False, "error": "Facility not found"}

        facility = self.facilities[facility_id]
        result = {"success": True, "items_stored": [], "items_rejected": []}

        for item in items:
            # Check capacity
            if self._check_capacity_available(facility, item):
                # Find appropriate storage zone
                zone = self._find_best_storage_zone(facility, item)
                if zone:
                    self._store_item(facility, item, zone)
                    result["items_stored"].append(item.id)
                else:
                    result["items_rejected"].append({
                        "item_id": item.id,
                        "reason": "No suitable storage zone available"
                    })
            else:
                result["items_rejected"].append({
                    "item_id": item.id,
                    "reason": "Insufficient capacity"
                })

        # Update metrics
        self._update_facility_metrics(facility)

        return result

    def dispatch_items(self, facility_id: str, item_type: str,
                      quantity: float) -> Dict[str, Any]:
        """Dispatch items from a facility."""
        if facility_id not in self.facilities:
            return {"success": False, "error": "Facility not found"}

        facility = self.facilities[facility_id]
        available_items = facility.current_inventory.get(item_type, [])

        if not available_items:
            return {"success": False, "error": "Item type not found"}

        # Sort items by FIFO (First In, First Out) and condition
        available_items.sort(key = lambda x: (x.storage_date, x.condition.value))

        dispatched_items = []
        remaining_quantity = quantity

        for item in available_items.copy():
            if remaining_quantity <= 0:
                break

            if item.quantity <= remaining_quantity:
                # Take entire item
                dispatched_items.append(item)
                available_items.remove(item)
                remaining_quantity -= item.quantity
            else:
                # Partial item dispatch
                dispatched_item = StoredItem(
                    id = f"{item.id}_partial",
                    item_type = item.item_type,
                    category = item.category,
                    quantity = remaining_quantity,
                    unit = item.unit,
                    condition = item.condition,
                    storage_date = item.storage_date,
                    expiration_date = item.expiration_date,
                    value_per_unit = item.value_per_unit,
                    batch_number = item.batch_number,
                    supplier = item.supplier
                )
                dispatched_items.append(dispatched_item)
                item.quantity -= remaining_quantity
                remaining_quantity = 0

        # Check if reorder is needed
        self._check_reorder_needed(facility_id, item_type)

        return {
            "success": True,
            "dispatched_items": dispatched_items,
            "total_dispatched": quantity - remaining_quantity,
            "remaining_requested": remaining_quantity
        }

    def check_system_status(self) -> Dict[str, Any]:
        """Comprehensive system status check."""
        current_time = datetime.now()

        # Update degradation and check alerts periodically
        hours_since_check = (current_time - self.last_check_time).total_seconds() / 3600

        if hours_since_check >= self.alert_check_interval:
            self._update_item_conditions()
            self._check_all_alerts()
            self.last_check_time = current_time

        # Compile system status
        status = {
            "total_facilities": len(self.facilities),
            "total_alerts": len([a for a in self.alerts if not a.resolved]),
            "critical_alerts": len([a for a in self.alerts if a.severity == "critical" and not a.resolved]),
            "system_health": "healthy",  # Will be determined by alerts
            "facility_status": {},
            "inventory_summary": self._get_inventory_summary(),
            "performance_metrics": self._calculate_system_performance()
        }

        # Determine system health
        if status["critical_alerts"] > 0:
            status["system_health"] = "critical"
        elif status["total_alerts"] > 10:
            status["system_health"] = "warning"

        # Get facility - specific status
        for facility_id, facility in self.facilities.items():
            status["facility_status"][facility_id] = self._get_facility_status(facility)

        return status

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []

        for facility_id, facility in self.facilities.items():
            # Check capacity utilization
            utilization = self._calculate_capacity_utilization(facility)

            if utilization > 0.9:
                recommendations.append({
                    "type": "capacity_expansion",
                    "facility_id": facility_id,
                    "priority": "high",
                    "message": f"Facility {facility.name} is {utilization:.1%} full - consider expansion",
                    "estimated_cost": self._estimate_expansion_cost(facility),
                    "benefit": "Prevent stockouts and improve service level"
                })

            if utilization < 0.3:
                recommendations.append({
                    "type": "consolidation",
                    "facility_id": facility_id,
                    "priority": "medium",
                    "message": f"Facility {facility.name} is underutilized at {utilization:.1%}",
                    "estimated_savings": facility.properties.get("operating_cost", 1000) * 0.3,
                    "benefit": "Reduce operating costs through consolidation"
                })

            # Check for expired items
            expired_count = self._count_expired_items(facility)
            if expired_count > 0:
                recommendations.append({
                    "type": "inventory_rotation",
                    "facility_id": facility_id,
                    "priority": "high",
                    "message": f"{expired_count} items expired at {facility.name}",
                    "action": "Improve FIFO rotation and reduce order quantities"
                })

            # ABC analysis recommendations
            abc_classification = self._get_abc_analysis(facility)
            if abc_classification:
                recommendations.append({
                    "type": "abc_optimization",
                    "facility_id": facility_id,
                    "priority": "medium",
                    "message": "Optimize inventory policies based on ABC classification",
                    "details": abc_classification
                })

        return sorted(recommendations, key = lambda x:
                     {"critical": 4, "high": 3, "medium": 2, "low": 1}[x["priority"]],
                     reverse = True)

    def _check_capacity_available(self, facility: StockpileFacility,
                                item: StoredItem) -> bool:
        """Check if facility has capacity for new item."""
        current_total = sum(
            sum(item.quantity for item in items)
            for items in facility.current_inventory.values()
        )
        return current_total + item.quantity <= facility.total_capacity

    def _find_best_storage_zone(self, facility: StockpileFacility,
                              item: StoredItem) -> Optional[StorageZone]:
        """Find the best storage zone for an item."""
        suitable_zones = []

        for zone in facility.storage_zones:
            # Check basic compatibility
            if self._is_zone_suitable(zone, item):
                utilization = zone.current_utilization / zone.capacity
                suitable_zones.append((zone, utilization))

        if not suitable_zones:
            return None

        # Sort by utilization (prefer less utilized zones)
        suitable_zones.sort(key = lambda x: x[1])
        return suitable_zones[0][0]

    def _is_zone_suitable(self, zone: StorageZone, item: StoredItem) -> bool:
        """Check if a storage zone is suitable for an item."""
        # Basic capacity check
        if zone.current_utilization + item.quantity > zone.capacity:
            return False

        # Temperature requirements
        if item.category == StorageCategory.FOOD_PERISHABLE:
            return zone.zone_type in ["refrigerated", "frozen"]
        elif item.category == StorageCategory.PHARMACEUTICALS:
            return zone.zone_type in ["controlled_atmosphere", "refrigerated"]
        elif item.category == StorageCategory.CHEMICALS:
            return "chemical_safe" in zone.special_features

        return True  # Default to suitable for ambient storage

    def _store_item(self, facility: StockpileFacility, item: StoredItem,
                   zone: StorageZone):
        """Store an item in a specific zone."""
        if item.item_type not in facility.current_inventory:
            facility.current_inventory[item.item_type] = []

        facility.current_inventory[item.item_type].append(item)
        zone.current_utilization += item.quantity

    def _update_item_conditions(self):
        """Update condition of all stored items based on degradation."""
        for facility in self.facilities.values():
            for item_type, items in facility.current_inventory.items():
                for item in items:
                    # Find the storage zone (simplified - assumes item knows its zone)
                    zone = facility.storage_zones[0]  # Simplified

                    days_stored = (datetime.now() - item.storage_date).days
                    new_condition, _ = self.degradation_model.calculate_degradation(
                        item, zone, days_stored
                    )

                    if new_condition != item.condition:
                        item.condition = new_condition

                        # Generate alert if item has degraded significantly
                        if new_condition == ItemCondition.EXPIRED:
                            self._create_alert(
                                facility.facility_id, "expired", "high",
                                item_type, 0, 1,
                                f"Item {item.id} has expired and needs disposal"
                            )

    def _check_all_alerts(self):
        """Check for various alert conditions across all facilities."""
        for facility_id, facility in self.facilities.items():
            for item_type, items in facility.current_inventory.items():
                current_stock = sum(item.quantity for item in items)

                # Check reorder rules
                if facility_id in self.reorder_rules and item_type in self.reorder_rules[facility_id]:
                    rule = self.reorder_rules[facility_id][item_type]

                    if current_stock <= rule.reorder_point:
                        self._create_alert(
                            facility_id, "low_stock", "medium", item_type,
                            current_stock, rule.reorder_point,
                            f"Stock level {current_stock:.1f} below reorder point {rule.reorder_point:.1f}"
                        )

                    if current_stock <= rule.safety_stock:
                        self._create_alert(
                            facility_id, "critical_stock", "critical", item_type,
                            current_stock, rule.safety_stock,
                            f"Stock level {current_stock:.1f} below safety stock {rule.safety_stock:.1f}"
                        )

            # Check capacity alerts
            utilization = self._calculate_capacity_utilization(facility)
            if utilization > 0.95:
                self._create_alert(
                    facility_id, "capacity", "high", "facility",
                    utilization, 0.95,
                    f"Facility capacity at {utilization:.1%}"
                )

    def _create_alert(self, facility_id: str, alert_type: str, severity: str,
                     item_type: str, current_level: float, threshold_level: float,
                     message: str):
        """Create a new alert."""
        alert = StockpileAlert(
            alert_id = f"ALERT_{len(self.alerts)+1:05d}",
            facility_id = facility_id,
            alert_type = alert_type,
            severity = severity,
            item_type = item_type,
            current_level = current_level,
            threshold_level = threshold_level,
            message = message,
            timestamp = datetime.now()
        )
        self.alerts.append(alert)

    def _check_reorder_needed(self, facility_id: str, item_type: str):
        """Check if reordering is needed after dispatch."""
        if facility_id not in self.reorder_rules:
            return

        rule = self.reorder_rules[facility_id].get(item_type)
        if not rule:
            return

        facility = self.facilities[facility_id]
        current_stock = sum(
            item.quantity for item in facility.current_inventory.get(item_type, [])
        )

        if current_stock <= rule.reorder_point:
            # Generate reorder recommendation
            order_qty = rule.order_quantity

            # Adjust for seasonal factors
            current_month = datetime.now().month
            seasonal_factor = rule.seasonal_adjustment.get(current_month, 1.0)
            adjusted_qty = order_qty * seasonal_factor

            self._create_alert(
                facility_id, "reorder_needed", "medium", item_type,
                current_stock, rule.reorder_point,
                f"Reorder {adjusted_qty:.1f} units of {item_type}"
            )

    def _get_inventory_summary(self) -> Dict[str, Any]:
        """Get system - wide inventory summary."""
        total_items = 0
        total_value = 0
        item_types = set()

        for facility in self.facilities.values():
            for item_type, items in facility.current_inventory.items():
                item_types.add(item_type)
                for item in items:
                    total_items += 1
                    total_value += item.quantity * item.value_per_unit

        return {
            "total_items": total_items,
            "total_value": total_value,
            "unique_item_types": len(item_types),
            "item_types": list(item_types)
        }

    def _calculate_system_performance(self) -> Dict[str, Any]:
        """Calculate system - wide performance metrics."""
        total_capacity = sum(f.total_capacity for f in self.facilities.values())
        total_utilized = sum(
            sum(sum(item.quantity for item in items)
                for items in f.current_inventory.values())
            for f in self.facilities.values()
        )

        return {
            "system_utilization": total_utilized / max(total_capacity, 1),
            "average_facility_utilization": np.mean([
                self._calculate_capacity_utilization(f) for f in self.facilities.values()
            ]),
            "alert_rate": len([a for a in self.alerts if not a.resolved]) / len(self.facilities),
            "inventory_turnover": self._calculate_inventory_turnover()
        }

    def _calculate_capacity_utilization(self, facility: StockpileFacility) -> float:
        """Calculate capacity utilization for a facility."""
        total_stored = sum(
            sum(item.quantity for item in items)
            for items in facility.current_inventory.values()
        )
        return total_stored / max(facility.total_capacity, 1)

    def _get_facility_status(self, facility: StockpileFacility) -> Dict[str, Any]:
        """Get comprehensive status for a facility."""
        utilization = self._calculate_capacity_utilization(facility)
        facility_alerts = [a for a in self.alerts if a.facility_id == facility.facility_id and not a.resolved]

        return {
            "name": facility.name,
            "type": facility.facility_type.value,
            "capacity_utilization": utilization,
            "total_inventory_value": sum(
                sum(item.quantity * item.value_per_unit for item in items)
                for items in facility.current_inventory.values()
            ),
            "active_alerts": len(facility_alerts),
            "critical_alerts": len([a for a in facility_alerts if a.severity == "critical"]),
            "item_types_stored": len(facility.current_inventory),
            "health_status": "healthy" if len(facility_alerts) == 0 else "warning"
        }

    def _calculate_inventory_turnover(self) -> float:
        """Calculate average inventory turnover rate."""
        # Simplified calculation - would need historical data for accurate turnover
        return 12.0  # Assume monthly turnover

    def _count_expired_items(self, facility: StockpileFacility) -> int:
        """Count expired items in a facility."""
        count = 0
        for items in facility.current_inventory.values():
            count += len([item for item in items if item.condition == ItemCondition.EXPIRED])
        return count

    def _get_abc_analysis(self, facility: StockpileFacility) -> Dict[str, str]:
        """Get ABC classification for facility items."""
        all_items = []
        for items in facility.current_inventory.values():
            all_items.extend(items)

        return self.inventory_optimizer.optimize_abc_classification(all_items)

    def _estimate_expansion_cost(self, facility: StockpileFacility) -> float:
        """Estimate cost of expanding facility capacity."""
        # Simplified cost estimation
        return facility.total_capacity * 100  # $100 per unit of capacity

    def _update_facility_metrics(self, facility: StockpileFacility):
        """Update performance metrics for a facility."""
        # This would update various KPIs and metrics
        pass

if __name__ == "__main__":
    # Example usage
    manager = StockpileManager()

    # Create storage zones
    zones = [
        StorageZone("ZONE_01", "ambient", 10000, temperature_range=(15, 25)),
        StorageZone("ZONE_02", "refrigerated", 5000, temperature_range=(2, 8)),
        StorageZone("ZONE_03", "frozen", 3000, temperature_range=(-18, -15))
    ]

    # Create facility
    facility = StockpileFacility(
        facility_id="FAC_001",
        name="Regional Distribution Center",
        location=(40.7128, -74.0060),
        facility_type = StockpileType.DISTRIBUTION_CENTER,
        storage_zones = zones,
        total_capacity = 18000
    )

    manager.add_facility(facility)

    # Add reorder rules
    rule = ReorderRule(
        item_type="canned_food",
        facility_id="FAC_001",
        reorder_point = 1000,
        safety_stock = 500,
        order_quantity = 2000,
        max_stock_level = 5000,
        lead_time_days = 7
    )

    manager.add_reorder_rule(rule)

    # Create test items
    test_items = [
        StoredItem(
            id="ITEM_001",
            item_type="canned_food",
            category = StorageCategory.FOOD_NONPERISHABLE,
            quantity = 2000,
            unit="kg",
            condition = ItemCondition.EXCELLENT,
            storage_date = datetime.now() - timedelta(days = 10),
            expiration_date = datetime.now() + timedelta(days = 365),
            value_per_unit = 3.50
        )
    ]

    # Receive items
    result = manager.receive_items("FAC_001", test_items)
    print(f"Receive result: {result}")

    # Check system status
    status = manager.check_system_status()
    print(f"System status: {status['system_health']}")
    print(f"Total facilities: {status['total_facilities']}")
    print(f"Active alerts: {status['total_alerts']}")

    # Get recommendations
    recommendations = manager.get_optimization_recommendations()
    for rec in recommendations[:3]:  # Show top 3
        print(f"Recommendation: {rec['message']}")
