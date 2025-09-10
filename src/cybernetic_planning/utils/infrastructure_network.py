#!/usr / bin / env python3
"""
Infrastructure Network Generation System

Implements terrain analysis, route planning algorithms, cost modeling,
and network generation for both road and rail networks based on
topographical constraints, population density, and construction feasibility.
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict
import random

class TerrainType(Enum):
    """Types of terrain affecting construction."""
    PLAINS = "plains"           # Easy construction
    HILLS = "hills"             # Moderate construction
    MOUNTAINS = "mountains"     # Difficult construction
    VALLEYS = "valleys"         # Moderate construction
    FORESTS = "forests"         # Moderate construction
    WETLANDS = "wetlands"       # Difficult construction
    DESERT = "desert"           # Moderate construction
    COASTLINE = "coastline"     # Moderate construction
    RIVERS = "rivers"           # Difficult construction (requires bridges)
    URBAN = "urban"             # Expensive construction

class InfrastructureType(Enum):
    """Types of infrastructure to construct."""
    ROAD = "road"
    HIGHWAY = "highway"
    RAILWAY = "railway"
    HIGH_SPEED_RAIL = "high_speed_rail"
    BRIDGE = "bridge"
    TUNNEL = "tunnel"
    POWER_LINE = "power_line"
    PIPELINE = "pipeline"
    FIBER_OPTIC = "fiber_optic"

@dataclass
class TerrainCell:
    """Individual cell in the terrain grid."""
    x: int
    y: int
    elevation: float  # meters above sea level
    terrain_type: TerrainType
    slope: float  # degrees
    population_density: float  # people per km²
    economic_value: float  # relative importance for connectivity
    construction_difficulty: float  # 1.0 = easy, 5.0 = extremely difficult
    environmental_sensitivity: float  # 0.0 = no restrictions, 1.0 = protected
    existing_infrastructure: Set[InfrastructureType] = field(default_factory = set)

@dataclass
class ConstructionSpec:
    """Specifications for constructing different infrastructure types."""
    infrastructure_type: InfrastructureType
    base_cost_per_km: float  # Base cost in currency units
    terrain_multipliers: Dict[TerrainType, float]  # Cost multipliers by terrain
    slope_limits: Tuple[float, float]  # (max_slope_degrees, extreme_slope_multiplier)
    max_elevation_change: float  # Maximum elevation change per km
    environmental_impact: float  # Environmental impact factor
    construction_speed: float  # km per day
    maintenance_cost_annual: float  # Annual maintenance per km

@dataclass
class NetworkNode:
    """Node in the infrastructure network."""
    node_id: str
    name: str
    location: Tuple[int, int]  # Grid coordinates
    node_type: str  # "city", "town", "industrial", "port", "junction"
    population: int
    economic_importance: float
    connectivity_priority: int  # 1 = highest, 5 = lowest
    existing_connections: Set[str] = field(default_factory = set)
    properties: Dict[str, Any] = field(default_factory = dict)

@dataclass
class InfrastructureSegment:
    """Segment of constructed infrastructure."""
    segment_id: str
    infrastructure_type: InfrastructureType
    start_node: NetworkNode
    end_node: NetworkNode
    path: List[Tuple[int, int]]  # Grid path
    length: float  # km
    construction_cost: float
    annual_maintenance_cost: float
    capacity: float  # vehicles / day or cargo_tons / day
    current_utilization: float = 0.0
    condition: float = 1.0  # 1.0 = excellent, 0.0 = failed
    construction_date: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory = dict)

class TerrainAnalyzer:
    """Analyzes terrain for construction feasibility."""

    def __init__(self, grid_size: Tuple[int, int], cell_size_km: float = 1.0):
        self.grid_size = grid_size
        self.cell_size_km = cell_size_km
        self.terrain_grid = np.zeros(grid_size, dtype = object)
        self.elevation_map = np.zeros(grid_size)
        self.slope_map = np.zeros(grid_size)

        self._generate_terrain()

    def _generate_terrain(self):
        """Generate procedural terrain using noise functions."""
        width, height = self.grid_size

        # Generate elevation using multiple octaves of noise
        for x in range(width):
            for y in range(height):
                # Multiple noise octaves for realistic terrain
                elevation = 0
                frequency = 0.01
                amplitude = 1000

                for octave in range(4):
                    noise_val = self._perlin_noise(x * frequency, y * frequency)
                    elevation += noise_val * amplitude
                    frequency *= 2
                    amplitude *= 0.5

                # Ensure non - negative elevation
                elevation = max(0, elevation)
                self.elevation_map[x, y] = elevation

                # Calculate slope
                slope = self._calculate_slope(x, y)
                self.slope_map[x, y] = slope

                # Determine terrain type based on elevation and slope
                terrain_type = self._classify_terrain(elevation, slope, x, y)

                # Calculate construction difficulty
                difficulty = self._calculate_construction_difficulty(terrain_type, slope, elevation)

                # Create terrain cell
                self.terrain_grid[x, y] = TerrainCell(
                    x = x, y = y,
                    elevation = elevation,
                    terrain_type = terrain_type,
                    slope = slope,
                    population_density = self._estimate_population_density(terrain_type, elevation),
                    economic_value = self._estimate_economic_value(terrain_type, x, y),
                    construction_difficulty = difficulty,
                    environmental_sensitivity = self._assess_environmental_sensitivity(terrain_type)
                )

    def _perlin_noise(self, x: float, y: float) -> float:
        """Simplified Perlin noise implementation."""
        # This is a very basic implementation - in practice, use a proper noise library
        return (math.sin(x * 0.1) + math.cos(y * 0.1) +
                math.sin(x * 0.05) * math.cos(y * 0.05)) * 0.25

    def _calculate_slope(self, x: int, y: int) -> float:
        """Calculate slope at a point using neighboring elevations."""
        width, height = self.grid_size

        # Get neighboring elevations
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (dx != 0 or dy != 0):
                    neighbors.append(self.elevation_map[nx, ny])

        if not neighbors:
            return 0.0

        current_elevation = self.elevation_map[x, y]
        max_diff = max(abs(current_elevation - elev) for elev in neighbors)

        # Convert to degrees (assuming 1 cell = 1 km)
        return math.degrees(math.atan(max_diff / (self.cell_size_km * 1000)))

    def _classify_terrain(self, elevation: float, slope: float, x: int, y: int) -> TerrainType:
        """Classify terrain type based on elevation and slope."""
        if elevation > 2000:  # High elevation
            return TerrainType.MOUNTAINS if slope > 15 else TerrainType.HILLS
        elif elevation > 1000:  # Medium elevation
            return TerrainType.HILLS if slope > 10 else TerrainType.PLAINS
        elif elevation < 100:  # Low elevation
            # Check if near water (simplified)
            if x < 50 or x > self.grid_size[0] - 50:  # Near edges
                return TerrainType.COASTLINE
            else:
                return TerrainType.WETLANDS if slope < 2 else TerrainType.PLAINS
        else:  # Medium - low elevation
            return TerrainType.FORESTS if slope > 5 else TerrainType.PLAINS

    def _calculate_construction_difficulty(self, terrain_type: TerrainType,
                                        slope: float, elevation: float) -> float:
        """Calculate construction difficulty factor."""
        base_difficulty = {
            TerrainType.PLAINS: 1.0,
            TerrainType.HILLS: 1.5,
            TerrainType.MOUNTAINS: 3.0,
            TerrainType.VALLEYS: 1.3,
            TerrainType.FORESTS: 1.8,
            TerrainType.WETLANDS: 2.5,
            TerrainType.DESERT: 1.4,
            TerrainType.COASTLINE: 2.0,
            TerrainType.RIVERS: 2.8,
            TerrainType.URBAN: 2.2
        }.get(terrain_type, 1.0)

        # Slope multiplier
        slope_multiplier = 1.0 + (slope / 45.0)  # Linear increase with slope

        # Elevation factor for extreme altitudes
        elevation_factor = 1.0
        if elevation > 3000:
            elevation_factor = 1.0 + ((elevation - 3000) / 1000) * 0.5

        return base_difficulty * slope_multiplier * elevation_factor

    def _estimate_population_density(self, terrain_type: TerrainType, elevation: float) -> float:
        """Estimate population density based on terrain."""
        base_density = {
            TerrainType.PLAINS: 100,
            TerrainType.HILLS: 50,
            TerrainType.MOUNTAINS: 10,
            TerrainType.VALLEYS: 80,
            TerrainType.FORESTS: 20,
            TerrainType.WETLANDS: 15,
            TerrainType.DESERT: 5,
            TerrainType.COASTLINE: 200,
            TerrainType.RIVERS: 150,
            TerrainType.URBAN: 1000
        }.get(terrain_type, 50)

        # Elevation penalty
        if elevation > 2000:
            base_density *= 0.3
        elif elevation > 1000:
            base_density *= 0.7

        return max(1, base_density + random.uniform(-base_density * 0.3, base_density * 0.3))

    def _estimate_economic_value(self, terrain_type: TerrainType, x: int, y: int) -> float:
        """Estimate economic value for connectivity."""
        base_value = {
            TerrainType.PLAINS: 0.8,
            TerrainType.HILLS: 0.6,
            TerrainType.MOUNTAINS: 0.3,
            TerrainType.VALLEYS: 0.7,
            TerrainType.FORESTS: 0.4,
            TerrainType.WETLANDS: 0.2,
            TerrainType.DESERT: 0.3,
            TerrainType.COASTLINE: 1.0,
            TerrainType.RIVERS: 0.9,
            TerrainType.URBAN: 1.0
        }.get(terrain_type, 0.5)

        # Add some randomness and spatial clustering
        center_x, center_y = self.grid_size[0] // 2, self.grid_size[1] // 2
        distance_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        center_bonus = max(0, 0.3 * (1 - distance_from_center / max(center_x, center_y)))

        return min(1.0, base_value + center_bonus + random.uniform(-0.1, 0.1))

    def _assess_environmental_sensitivity(self, terrain_type: TerrainType) -> float:
        """Assess environmental sensitivity for construction."""
        sensitivity = {
            TerrainType.PLAINS: 0.2,
            TerrainType.HILLS: 0.3,
            TerrainType.MOUNTAINS: 0.8,
            TerrainType.VALLEYS: 0.4,
            TerrainType.FORESTS: 0.7,
            TerrainType.WETLANDS: 0.9,
            TerrainType.DESERT: 0.3,
            TerrainType.COASTLINE: 0.6,
            TerrainType.RIVERS: 0.8,
            TerrainType.URBAN: 0.1
        }.get(terrain_type, 0.3)

        return min(1.0, sensitivity + random.uniform(-0.1, 0.1))

    def get_terrain_cell(self, x: int, y: int) -> Optional[TerrainCell]:
        """Get terrain cell at coordinates."""
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            return self.terrain_grid[x, y]
        return None

    def get_construction_feasibility(self, start: Tuple[int, int], end: Tuple[int, int],
                                   infrastructure_type: InfrastructureType) -> Dict[str, Any]:
        """Analyze construction feasibility between two points."""
        path = self._find_direct_path(start, end)
        if not path:
            return {"feasible": False, "reason": "No valid path"}

        total_difficulty = 0
        environmental_issues = 0
        terrain_challenges = []

        for x, y in path:
            cell = self.get_terrain_cell(x, y)
            if not cell:
                continue

            total_difficulty += cell.construction_difficulty

            if cell.environmental_sensitivity > 0.7:
                environmental_issues += 1

            if cell.construction_difficulty > 2.5:
                terrain_challenges.append((x, y, cell.terrain_type.value))

        avg_difficulty = total_difficulty / len(path)

        return {
            "feasible": avg_difficulty < 4.0 and environmental_issues / len(path) < 0.3,
            "average_difficulty": avg_difficulty,
            "path_length": len(path),
            "environmental_concerns": environmental_issues,
            "terrain_challenges": terrain_challenges[:5],  # Top 5 challenges
            "estimated_cost_multiplier": avg_difficulty,
            "construction_time_multiplier": max(1.0, avg_difficulty - 0.5)
        }

    def _find_direct_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find direct path between two points using Bresenham's algorithm."""
        x0, y0 = start
        x1, y1 = end

        path = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else - 1
        sy = 1 if y0 < y1 else - 1
        err = dx - dy

        x, y = x0, y0

        while True:
            path.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x += sx

            if e2 < dx:
                err += dx
                y += sy

        return path

class PathfindingAlgorithm:
    """Advanced pathfinding for infrastructure routing."""

    def __init__(self, terrain_analyzer: TerrainAnalyzer):
        self.terrain_analyzer = terrain_analyzer

    def find_optimal_route(self, start: Tuple[int, int], end: Tuple[int, int],
                          infrastructure_type: InfrastructureType,
                          construction_spec: ConstructionSpec) -> Dict[str, Any]:
        """Find optimal route using A * with terrain - aware costs."""

        def heuristic(pos: Tuple[int, int]) -> float:
            return math.sqrt((pos[0] - end[0])**2 + (pos[1] - end[1])**2)

        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.terrain_analyzer.grid_size[0] and 0 <= ny < self.terrain_analyzer.grid_size[1]:
                    neighbors.append((nx, ny))
            return neighbors

        def calculate_cost(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
            cell = self.terrain_analyzer.get_terrain_cell(to_pos[0], to_pos[1])
            if not cell:
                return float('inf')

            # Base distance cost
            distance = math.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2)

            # Terrain cost multiplier
            terrain_multiplier = construction_spec.terrain_multipliers.get(
                cell.terrain_type, 2.0
            )

            # Slope penalty
            max_slope, slope_multiplier = construction_spec.slope_limits
            if cell.slope > max_slope:
                terrain_multiplier *= slope_multiplier

            # Environmental penalty
            if cell.environmental_sensitivity > 0.5:
                terrain_multiplier *= (1 + cell.environmental_sensitivity)

            return distance * terrain_multiplier * cell.construction_difficulty

        # A * algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()

                total_cost = g_score[end]
                return {
                    "success": True,
                    "path": path,
                    "total_cost": total_cost,
                    "path_length": len(path) * self.terrain_analyzer.cell_size_km,
                    "cost_per_km": total_cost / max(len(path), 1)
                }

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + calculate_cost(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor)

                    if (f_score[neighbor], neighbor) not in open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return {"success": False, "reason": "No path found"}

class InfrastructureBuilder:
    """Builds and manages infrastructure networks."""

    def __init__(self, terrain_analyzer: TerrainAnalyzer):
        self.terrain_analyzer = terrain_analyzer
        self.pathfinder = PathfindingAlgorithm(terrain_analyzer)
        self.construction_specs = self._initialize_construction_specs()
        self.built_segments = {}  # segment_id -> InfrastructureSegment
        self.network_graph = defaultdict(list)  # node_id -> [connected_node_ids]

    def _initialize_construction_specs(self) -> Dict[InfrastructureType, ConstructionSpec]:
        """Initialize construction specifications for different infrastructure types."""
        return {
            InfrastructureType.ROAD: ConstructionSpec(
                infrastructure_type = InfrastructureType.ROAD,
                base_cost_per_km = 500000,  # $500k per km
                terrain_multipliers={
                    TerrainType.PLAINS: 1.0,
                    TerrainType.HILLS: 1.5,
                    TerrainType.MOUNTAINS: 3.0,
                    TerrainType.FORESTS: 1.8,
                    TerrainType.WETLANDS: 2.5,
                    TerrainType.DESERT: 1.3,
                    TerrainType.COASTLINE: 2.0,
                    TerrainType.RIVERS: 3.5,
                    TerrainType.URBAN: 4.0
                },
                slope_limits=(15.0, 2.0),  # Max 15°, 2x cost penalty beyond
                max_elevation_change = 200,  # 200m per km
                environmental_impact = 0.3,
                construction_speed = 2.0,    # 2 km per day
                maintenance_cost_annual = 5000  # $5k per km per year
            ),

            InfrastructureType.HIGHWAY: ConstructionSpec(
                infrastructure_type = InfrastructureType.HIGHWAY,
                base_cost_per_km = 2000000,  # $2M per km
                terrain_multipliers={
                    TerrainType.PLAINS: 1.0,
                    TerrainType.HILLS: 2.0,
                    TerrainType.MOUNTAINS: 4.0,
                    TerrainType.FORESTS: 2.2,
                    TerrainType.WETLANDS: 3.0,
                    TerrainType.DESERT: 1.5,
                    TerrainType.COASTLINE: 2.5,
                    TerrainType.RIVERS: 4.5,
                    TerrainType.URBAN: 5.0
                },
                slope_limits=(8.0, 3.0),   # Max 8°, 3x cost penalty
                max_elevation_change = 100,  # 100m per km
                environmental_impact = 0.5,
                construction_speed = 1.0,    # 1 km per day
                maintenance_cost_annual = 15000  # $15k per km per year
            ),

            InfrastructureType.RAILWAY: ConstructionSpec(
                infrastructure_type = InfrastructureType.RAILWAY,
                base_cost_per_km = 3000000,  # $3M per km
                terrain_multipliers={
                    TerrainType.PLAINS: 1.0,
                    TerrainType.HILLS: 2.5,
                    TerrainType.MOUNTAINS: 5.0,
                    TerrainType.FORESTS: 2.0,
                    TerrainType.WETLANDS: 3.5,
                    TerrainType.DESERT: 1.8,
                    TerrainType.COASTLINE: 3.0,
                    TerrainType.RIVERS: 5.5,
                    TerrainType.URBAN: 6.0
                },
                slope_limits=(3.0, 4.0),   # Max 3°, 4x cost penalty (railways need gentle grades)
                max_elevation_change = 50,   # 50m per km
                environmental_impact = 0.4,
                construction_speed = 0.5,    # 0.5 km per day
                maintenance_cost_annual = 25000  # $25k per km per year
            )
        }

    def plan_network(self, nodes: List[NetworkNode],
                    infrastructure_type: InfrastructureType) -> Dict[str, Any]:
        """Plan an optimal network connecting all nodes."""
        if not nodes or len(nodes) < 2:
            return {"success": False, "reason": "Need at least 2 nodes"}

        # Use modified Minimum Spanning Tree with economic priorities
        planned_segments = []
        total_cost = 0

        # Calculate all possible connections with costs
        connections = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1:], i + 1):
                route_result = self.pathfinder.find_optimal_route(
                    node1.location, node2.location,
                    infrastructure_type, self.construction_specs[infrastructure_type]
                )

                if route_result["success"]:
                    # Adjust cost based on economic importance
                    economic_factor = (node1.economic_importance + node2.economic_importance) / 2
                    priority_factor = min(node1.connectivity_priority, node2.connectivity_priority)

                    adjusted_cost = route_result["total_cost"] / (economic_factor * (6 - priority_factor))

                    connections.append({
                        "node1": node1,
                        "node2": node2,
                        "cost": adjusted_cost,
                        "raw_cost": route_result["total_cost"],
                        "path": route_result["path"],
                        "length": route_result["path_length"]
                    })

        # Sort connections by adjusted cost
        connections.sort(key = lambda x: x["cost"])

        # Build MST using Union - Find
        node_sets = {node.node_id: {node.node_id} for node in nodes}

        for connection in connections:
            node1_id = connection["node1"].node_id
            node2_id = connection["node2"].node_id

            # Find root sets
            set1 = node_sets[node1_id]
            set2 = node_sets[node2_id]

            # If nodes are in different sets, connect them
            if set1 != set2:
                segment_id = f"{infrastructure_type.value}_{node1_id}_{node2_id}"

                segment = InfrastructureSegment(
                    segment_id = segment_id,
                    infrastructure_type = infrastructure_type,
                    start_node = connection["node1"],
                    end_node = connection["node2"],
                    path = connection["path"],
                    length = connection["length"],
                    construction_cost = connection["raw_cost"],
                    annual_maintenance_cost = connection["length"] *
                        self.construction_specs[infrastructure_type].maintenance_cost_annual,
                    capacity = self._calculate_capacity(infrastructure_type, connection["length"])
                )

                planned_segments.append(segment)
                total_cost += connection["raw_cost"]

                # Union the sets
                merged_set = set1.union(set2)
                for node_id in merged_set:
                    node_sets[node_id] = merged_set

                # Update network graph
                self.network_graph[node1_id].append(node2_id)
                self.network_graph[node2_id].append(node1_id)

                # Stop if all nodes are connected
                if len(merged_set) == len(nodes):
                    break

        # Analyze network connectivity
        connectivity_analysis = self._analyze_network_connectivity(nodes, planned_segments)

        return {
            "success": True,
            "planned_segments": planned_segments,
            "total_segments": len(planned_segments),
            "total_cost": total_cost,
            "total_length": sum(seg.length for seg in planned_segments),
            "average_cost_per_km": total_cost / max(sum(seg.length for seg in planned_segments), 1),
            "connectivity_analysis": connectivity_analysis,
            "construction_time_estimate": self._estimate_construction_time(planned_segments),
            "annual_maintenance_cost": sum(seg.annual_maintenance_cost for seg in planned_segments)
        }

    def build_segment(self, segment: InfrastructureSegment) -> Dict[str, Any]:
        """Construct an infrastructure segment."""
        spec = self.construction_specs[segment.infrastructure_type]

        # Simulate construction process
        construction_result = {
            "success": True,
            "segment_id": segment.segment_id,
            "actual_cost": segment.construction_cost,
            "construction_time_days": segment.length / spec.construction_speed,
            "environmental_permits_required": [],
            "construction_challenges": []
        }

        # Check for environmental permits needed
        for x, y in segment.path:
            cell = self.terrain_analyzer.get_terrain_cell(x, y)
            if cell and cell.environmental_sensitivity > 0.7:
                construction_result["environmental_permits_required"].append((x, y))

        # Identify construction challenges
        for x, y in segment.path:
            cell = self.terrain_analyzer.get_terrain_cell(x, y)
            if cell and cell.construction_difficulty > 3.0:
                construction_result["construction_challenges"].append({
                    "location": (x, y),
                    "terrain": cell.terrain_type.value,
                    "difficulty": cell.construction_difficulty,
                    "recommended_solution": self._recommend_construction_solution(cell)
                })

        # Add segment to built infrastructure
        self.built_segments[segment.segment_id] = segment

        # Update terrain with new infrastructure
        self._update_terrain_with_infrastructure(segment)

        return construction_result

    def _calculate_capacity(self, infrastructure_type: InfrastructureType, length: float) -> float:
        """Calculate capacity for infrastructure type."""
        base_capacities = {
            InfrastructureType.ROAD: 10000,  # vehicles per day
            InfrastructureType.HIGHWAY: 50000,
            InfrastructureType.RAILWAY: 100000,  # tons per day
            InfrastructureType.HIGH_SPEED_RAIL: 50000
        }

        return base_capacities.get(infrastructure_type, 10000)

    def _analyze_network_connectivity(self, nodes: List[NetworkNode],
                                    segments: List[InfrastructureSegment]) -> Dict[str, Any]:
        """Analyze network connectivity metrics."""
        if not nodes or not segments:
            return {"connected": False}

        # Build adjacency list
        graph = defaultdict(set)
        for segment in segments:
            graph[segment.start_node.node_id].add(segment.end_node.node_id)
            graph[segment.end_node.node_id].add(segment.start_node.node_id)

        # Check if all nodes are reachable from first node
        visited = set()
        queue = [nodes[0].node_id]
        visited.add(nodes[0].node_id)

        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        fully_connected = len(visited) == len(nodes)

        # Calculate network efficiency metrics
        total_direct_distance = 0
        total_network_distance = 0

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                # Direct distance
                direct_dist = math.sqrt(
                    (node1.location[0] - node2.location[0])**2 +
                    (node1.location[1] - node2.location[1])**2
                ) * self.terrain_analyzer.cell_size_km

                # Network distance (simplified - would use Dijkstra for accuracy)
                network_dist = self._estimate_network_distance(
                    node1.node_id, node2.node_id, graph
                )

                total_direct_distance += direct_dist
                total_network_distance += network_dist

        network_efficiency = total_direct_distance / max(total_network_distance, 1)

        return {
            "connected": fully_connected,
            "connected_nodes": len(visited),
            "total_nodes": len(nodes),
            "network_efficiency": network_efficiency,
            "average_degree": len(segments) * 2 / len(nodes),  # Average connections per node
            "redundancy_level": "low"  # Could be calculated based on alternative paths
        }

    def _estimate_construction_time(self, segments: List[InfrastructureSegment]) -> Dict[str, Any]:
        """Estimate total construction time for all segments."""
        if not segments:
            return {"total_days": 0, "total_years": 0}

        # Assume some segments can be built in parallel
        infrastructure_types = set(seg.infrastructure_type for seg in segments)
        parallel_construction_factor = 0.7  # 30% time savings from parallel construction

        total_sequential_time = 0
        for seg in segments:
            spec = self.construction_specs[seg.infrastructure_type]
            construction_days = seg.length / spec.construction_speed
            total_sequential_time += construction_days

        estimated_time = total_sequential_time * parallel_construction_factor

        return {
            "total_days": estimated_time,
            "total_years": estimated_time / 365,
            "sequential_time_days": total_sequential_time,
            "parallel_efficiency": parallel_construction_factor
        }

    def _recommend_construction_solution(self, cell: TerrainCell) -> str:
        """Recommend construction solution for challenging terrain."""
        if cell.terrain_type == TerrainType.MOUNTAINS and cell.slope > 20:
            return "Consider tunnel construction or switchback design"
        elif cell.terrain_type == TerrainType.RIVERS:
            return "Bridge construction required"
        elif cell.terrain_type == TerrainType.WETLANDS:
            return "Elevated construction or extensive drainage required"
        elif cell.environmental_sensitivity > 0.8:
            return "Environmental impact mitigation required"
        else:
            return "Standard construction with reinforced foundation"

    def _update_terrain_with_infrastructure(self, segment: InfrastructureSegment):
        """Update terrain grid with new infrastructure."""
        for x, y in segment.path:
            cell = self.terrain_analyzer.get_terrain_cell(x, y)
            if cell:
                cell.existing_infrastructure.add(segment.infrastructure_type)

    def _estimate_network_distance(self, start: str, end: str, graph: Dict[str, Set[str]]) -> float:
        """Estimate network distance between two nodes."""
        # Simplified BFS for distance estimation
        if start == end:
            return 0

        visited = set([start])
        queue = [(start, 0)]

        while queue:
            current, distance = queue.pop(0)

            for neighbor in graph[current]:
                if neighbor == end:
                    return (distance + 1) * self.terrain_analyzer.cell_size_km

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')  # Not connected

    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        if not self.built_segments:
            return {"total_segments": 0, "status": "No infrastructure built"}

        infrastructure_summary = defaultdict(lambda: {"count": 0, "total_length": 0, "total_cost": 0})

        for segment in self.built_segments.values():
            infra_type = segment.infrastructure_type.value
            infrastructure_summary[infra_type]["count"] += 1
            infrastructure_summary[infra_type]["total_length"] += segment.length
            infrastructure_summary[infra_type]["total_cost"] += segment.construction_cost

        return {
            "total_segments": len(self.built_segments),
            "infrastructure_summary": dict(infrastructure_summary),
            "network_graph_nodes": len(self.network_graph),
            "average_connectivity": np.mean([len(connections) for connections in self.network_graph.values()]) if self.network_graph else 0
        }

if __name__ == "__main__":
    # Example usage

    # Create terrain analyzer
    terrain = TerrainAnalyzer(grid_size=(200, 200), cell_size_km = 1.0)

    # Create infrastructure builder
    builder = InfrastructureBuilder(terrain)

    # Create test nodes (cities)
    nodes = [
        NetworkNode("CITY_A", "Capital City", (50, 50), "city", 1000000, 1.0, 1),
        NetworkNode("CITY_B", "Industrial Center", (150, 75), "industrial", 500000, 0.8, 2),
        NetworkNode("CITY_C", "Port City", (100, 180), "port", 300000, 0.9, 2),
        NetworkNode("TOWN_D", "Regional Town", (75, 120), "town", 50000, 0.4, 3)
    ]

    # Plan road network
    road_plan = builder.plan_network(nodes, InfrastructureType.ROAD)

    if road_plan["success"]:
        print(f"Road network planned successfully:")
        print(f"  Total segments: {road_plan['total_segments']}")
        print(f"  Total cost: ${road_plan['total_cost']:,.0f}")
        print(f"  Total length: {road_plan['total_length']:.1f} km")
        print(f"  Network connected: {road_plan['connectivity_analysis']['connected']}")
        print(f"  Construction time: {road_plan['construction_time_estimate']['total_years']:.1f} years")

        # Build a sample segment
        if road_plan["planned_segments"]:
            sample_segment = road_plan["planned_segments"][0]
            construction_result = builder.build_segment(sample_segment)

            print(f"\nSample segment construction:")
            print(f"  Segment: {construction_result['segment_id']}")
            print(f"  Construction time: {construction_result['construction_time_days']:.1f} days")
            print(f"  Challenges: {len(construction_result['construction_challenges'])}")

    # Plan railway network
    rail_plan = builder.plan_network(nodes, InfrastructureType.RAILWAY)

    if rail_plan["success"]:
        print(f"\nRailway network planned:")
        print(f"  Total cost: ${rail_plan['total_cost']:,.0f}")
        print(f"  Cost difference from roads: ${rail_plan['total_cost'] - road_plan['total_cost']:,.0f}")

    # Get network status
    status = builder.get_network_status()
    print(f"\nNetwork status: {status}")
