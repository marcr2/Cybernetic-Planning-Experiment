"""
Map-Based Economic Plan Simulator

Implements a modular, map-based simulation engine that takes high-level parameters
from a pre-defined economic plan and procedurally generates a corresponding geographical
and economic landscape. Integrates with the existing cybernetic planning system.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict
import json
from pathlib import Path

# For Perlin noise generation
try:
    from noise import pnoise2
    PERLIN_AVAILABLE = True
except ImportError:
    PERLIN_AVAILABLE = False
    print("Warning: Perlin noise not available. Using simplified terrain generation.")

class TerrainType(Enum):
    """Terrain types for map tiles."""
    FLATLAND = "flatland"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    WATER = "water"
    COASTAL = "coastal"

class SettlementType(Enum):
    """Types of settlements."""
    CITY = "city"
    TOWN = "town"
    RURAL = "rural"

class DisasterType(Enum):
    """Types of natural disasters."""
    FLOOD = "flood"
    EARTHQUAKE = "earthquake"
    DROUGHT = "drought"
    STORM = "storm"

@dataclass
class MapTile:
    """Represents a single tile on the map."""
    x: int
    y: int
    terrain_type: TerrainType
    altitude: float
    resource_level: float
    construction_difficulty: float
    population_capacity: int
    is_settlement: bool = False
    settlement_id: Optional[str] = None

@dataclass
class Settlement:
    """Represents a settlement (city, town, or rural area)."""
    id: str
    name: str
    settlement_type: SettlementType
    x: int
    y: int
    population: int
    economic_importance: float
    sectors: List[str] = field(default_factory=list)
    infrastructure_connections: List[str] = field(default_factory=list)

@dataclass
class InfrastructureSegment:
    """Represents a segment of infrastructure (road/rail)."""
    id: str
    start_settlement: str
    end_settlement: str
    path: List[Tuple[int, int]]
    cost: float
    terrain_multiplier: float
    is_damaged: bool = False

@dataclass
class DisasterEvent:
    """Represents a natural disaster event."""
    disaster_type: DisasterType
    x: int
    y: int
    radius: int
    intensity: float
    duration_days: int
    affected_tiles: List[Tuple[int, int]] = field(default_factory=list)

class MapBasedSimulator:
    """
    Main map-based economic plan simulator.
    
    Generates procedural maps and integrates with existing economic planning system.
    """
    
    def __init__(self, 
                 map_width: int = 200,
                 map_height: int = 200,
                 terrain_distribution: Optional[Dict[str, float]] = None,
                 num_cities: int = 5,
                 num_towns: int = 15,
                 total_population: int = 1000000,
                 rural_population_percent: float = 0.3,
                 urban_concentration: str = "medium",
                 log_callback: Optional[callable] = None):
        """
        Initialize the map-based simulator.
        
        Args:
            map_width: Width of the map in tiles
            map_height: Height of the map in tiles
            terrain_distribution: Target distribution of terrain types
            num_cities: Number of cities to generate
            num_towns: Number of towns to generate
            total_population: Total population to distribute
            rural_population_percent: Percentage of population in rural areas
            urban_concentration: Level of urban concentration ("high", "medium", "low")
        """
        self.map_width = map_width
        self.map_height = map_height
        self.terrain_distribution = terrain_distribution or {
            "flatland": 0.35,
            "forest": 0.25,
            "mountain": 0.15,
            "water": 0.1,
            "coastal": 0.15
        }
        
        self.num_cities = num_cities
        self.num_towns = num_towns
        self.total_population = total_population
        self.rural_population_percent = rural_population_percent
        self.urban_concentration = urban_concentration
        
        # Map data
        self.map_tiles: Dict[Tuple[int, int], MapTile] = {}
        self.settlements: Dict[str, Settlement] = {}
        self.infrastructure_segments: Dict[str, InfrastructureSegment] = {}
        self.disaster_events: List[DisasterEvent] = []
        
        # Simulation state
        self.current_time_step = 0
        self.logistics_costs: Dict[Tuple[str, str], float] = {}
        self.disaster_probability = 0.05  # 5% chance per time step
        
        # Integration with existing system
        self.economic_plan = None
        self.sector_mapping = {}
        
        # Logging callback for GUI integration
        self.log_callback = log_callback
    
    def _log(self, message: str):
        """Log message using callback if available, otherwise print."""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
        
    def generate_map(self) -> Dict[str, Any]:
        """
        Generate the procedural map based on input parameters.
        
        Returns:
            Dictionary with generation results
        """
        self._log("Generating procedural map...")
        
        # Step 1: Generate terrain using Perlin noise or simplified method
        self._generate_terrain()
        
        # Step 2: Place settlements based on terrain suitability
        self._place_settlements()
        
        # Step 3: Distribute population among settlements
        self._distribute_population()
        
        # Step 4: Generate infrastructure network
        self._generate_infrastructure()
        
        # Step 5: Calculate initial logistics costs
        self._calculate_initial_logistics()
        
        return {
            "success": True,
            "map_size": (self.map_width, self.map_height),
            "settlements_created": len(self.settlements),
            "infrastructure_segments": len(self.infrastructure_segments),
            "total_population": self.total_population,
            "terrain_distribution": self._get_actual_terrain_distribution()
        }
    
    def _generate_terrain(self):
        """Generate terrain using Perlin noise or simplified method."""
        self._log("Generating terrain...")
        
        if PERLIN_AVAILABLE:
            self._generate_terrain_perlin()
        else:
            self._generate_terrain_simplified()
    
    def _generate_terrain_perlin(self):
        """Generate terrain using Perlin noise."""
        scale = 0.1  # Controls terrain detail
        
        for x in range(self.map_width):
            for y in range(self.map_height):
                # Generate multiple noise layers for different terrain features
                height_noise = pnoise2(x * scale, y * scale, octaves=4)
                moisture_noise = pnoise2(x * scale + 1000, y * scale + 1000, octaves=2)
                
                # Determine terrain type based on noise values
                terrain_type = self._determine_terrain_type(height_noise, moisture_noise)
                
                # Calculate tile properties
                altitude = max(0, height_noise * 1000)  # Convert to altitude in meters
                resource_level = max(0, pnoise2(x * scale + 2000, y * scale + 2000))
                construction_difficulty = self._calculate_construction_difficulty(terrain_type, altitude)
                population_capacity = self._calculate_population_capacity(terrain_type, altitude)
                
                tile = MapTile(
                    x=x, y=y,
                    terrain_type=terrain_type,
                    altitude=altitude,
                    resource_level=resource_level,
                    construction_difficulty=construction_difficulty,
                    population_capacity=population_capacity
                )
                
                self.map_tiles[(x, y)] = tile
        
        # Generate coastal regions after initial terrain generation
        self._generate_coastal_regions()
    
    def _generate_terrain_simplified(self):
        """Generate terrain using simplified random method with logical coastal regions."""
        # First pass: Generate basic terrain types
        for x in range(self.map_width):
            for y in range(self.map_height):
                # Simple random terrain generation
                rand_val = random.random()
                
                if rand_val < self.terrain_distribution["water"]:
                    terrain_type = TerrainType.WATER
                elif rand_val < self.terrain_distribution["water"] + self.terrain_distribution["mountain"]:
                    terrain_type = TerrainType.MOUNTAIN
                elif rand_val < self.terrain_distribution["water"] + self.terrain_distribution["mountain"] + self.terrain_distribution["forest"]:
                    terrain_type = TerrainType.FOREST
                else:
                    terrain_type = TerrainType.FLATLAND
                
                # Calculate tile properties
                altitude = self._get_altitude_for_terrain(terrain_type)
                resource_level = random.uniform(0, 1)
                construction_difficulty = self._calculate_construction_difficulty(terrain_type, altitude)
                population_capacity = self._calculate_population_capacity(terrain_type, altitude)
                
                tile = MapTile(
                    x=x, y=y,
                    terrain_type=terrain_type,
                    altitude=altitude,
                    resource_level=resource_level,
                    construction_difficulty=construction_difficulty,
                    population_capacity=population_capacity
                )
                
                self.map_tiles[(x, y)] = tile
        
        # Second pass: Create logical coastal regions adjacent to water
        self._generate_coastal_regions()
    
    def _generate_coastal_regions(self):
        """Generate coastal regions adjacent to water bodies."""
        # Find all water tiles
        water_tiles = []
        for (x, y), tile in self.map_tiles.items():
            if tile.terrain_type == TerrainType.WATER:
                water_tiles.append((x, y))
        
        # Convert some adjacent land tiles to coastal
        coastal_candidates = []
        for water_x, water_y in water_tiles:
            # Check all 8 adjacent tiles
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    adj_x, adj_y = water_x + dx, water_y + dy
                    if (adj_x, adj_y) in self.map_tiles:
                        adj_tile = self.map_tiles[(adj_x, adj_y)]
                        # Convert suitable land tiles to coastal
                        if (adj_tile.terrain_type in [TerrainType.FLATLAND, TerrainType.FOREST] and 
                            random.random() < 0.6):  # 60% chance to become coastal
                            coastal_candidates.append((adj_x, adj_y))
        
        # Convert candidates to coastal terrain
        for x, y in coastal_candidates:
            tile = self.map_tiles[(x, y)]
            tile.terrain_type = TerrainType.COASTAL
            tile.altitude = self._get_altitude_for_terrain(TerrainType.COASTAL)
            tile.construction_difficulty = self._calculate_construction_difficulty(TerrainType.COASTAL, tile.altitude)
            tile.population_capacity = self._calculate_population_capacity(TerrainType.COASTAL, tile.altitude)
            # Coastal areas have higher resource levels (fishing, trade, etc.)
            tile.resource_level = min(1.0, tile.resource_level + 0.3)
    
    def _determine_terrain_type(self, height_noise: float, moisture_noise: float) -> TerrainType:
        """Determine terrain type based on noise values."""
        if height_noise < -0.3:
            return TerrainType.WATER
        elif height_noise > 0.3:
            return TerrainType.MOUNTAIN
        elif moisture_noise > 0.2:
            return TerrainType.FOREST
        else:
            return TerrainType.FLATLAND
    
    def _calculate_construction_difficulty(self, terrain_type: TerrainType, altitude: float) -> float:
        """Calculate construction difficulty for a terrain type."""
        base_difficulty = {
            TerrainType.FLATLAND: 1.0,
            TerrainType.FOREST: 3.0,
            TerrainType.MOUNTAIN: 10.0,
            TerrainType.WATER: float('inf'),
            TerrainType.COASTAL: 2.0
        }
        
        altitude_factor = 1.0 + (altitude / 1000) * 0.5  # Higher altitude = more difficult
        return base_difficulty[terrain_type] * altitude_factor
    
    def _calculate_population_capacity(self, terrain_type: TerrainType, altitude: float) -> int:
        """Calculate population capacity for a terrain type."""
        base_capacity = {
            TerrainType.FLATLAND: 10000,
            TerrainType.FOREST: 2000,
            TerrainType.MOUNTAIN: 500,
            TerrainType.WATER: 0,
            TerrainType.COASTAL: 15000
        }
        
        altitude_factor = max(0.1, 1.0 - (altitude / 2000))  # Higher altitude = less capacity
        return int(base_capacity[terrain_type] * altitude_factor)
    
    def _get_altitude_for_terrain(self, terrain_type: TerrainType) -> float:
        """Get altitude for terrain type in simplified generation."""
        altitude_ranges = {
            TerrainType.WATER: (0, 50),
            TerrainType.FLATLAND: (50, 200),
            TerrainType.FOREST: (200, 500),
            TerrainType.MOUNTAIN: (500, 2000),
            TerrainType.COASTAL: (0, 100)
        }
        
        min_alt, max_alt = altitude_ranges[terrain_type]
        return random.uniform(min_alt, max_alt)
    
    def _place_settlements(self):
        """Place settlements based on terrain suitability."""
        self._log("Placing settlements...")
        
        # Find suitable locations for cities (highest suitability)
        suitable_locations = self._find_suitable_locations()
        # Place cities first
        city_locations = suitable_locations[:self.num_cities]
        city_names = [
            "Springfield", "Riverside", "Oakdale", "Millbrook", "Greenfield",
            "Fairview", "Hillcrest", "Valley Falls", "Brookside", "Pineville",
            "Meadowbrook", "Sunset Hills", "Crystal Lake", "Golden Valley", "Silverton"
        ]
        
        for i, (x, y, suitability) in enumerate(city_locations):
            city_name = city_names[i % len(city_names)]
            city = Settlement(
                id=f"CITY_{i:03d}",
                name=city_name,
                settlement_type=SettlementType.CITY,
                x=x, y=y,
                population=0,  # Will be set in population distribution
                economic_importance=suitability
            )
            self.settlements[city.id] = city
            self.map_tiles[(x, y)].is_settlement = True
            self.map_tiles[(x, y)].settlement_id = city.id
        
        # Place towns based on urban concentration
        remaining_locations = suitable_locations[self.num_cities:]
        town_locations = self._select_town_locations(remaining_locations)
        
        town_names = [
            "Westfield", "Eastbrook", "Northgate", "Southport", "Central Point",
            "Newport", "Old Mill", "Rockville", "Maple Grove", "Cedar Falls",
            "Ironwood", "Stonebridge", "Willow Creek", "Pine Ridge", "Ashford",
            "Brentwood", "Clayton", "Derby", "Elmwood", "Franklin",
            "Glenwood", "Hampton", "Ivy Falls", "Jefferson", "Kingsley",
            "Lakeside", "Mountview", "Norwood", "Oxford", "Pleasant Hill"
        ]
        
        for i, (x, y, suitability) in enumerate(town_locations):
            town_name = town_names[i % len(town_names)]
            town = Settlement(
                id=f"TOWN_{i:03d}",
                name=town_name,
                settlement_type=SettlementType.TOWN,
                x=x, y=y,
                population=0,
                economic_importance=suitability * 0.7  # Towns are less important than cities
            )
            self.settlements[town.id] = town
            self.map_tiles[(x, y)].is_settlement = True
            self.map_tiles[(x, y)].settlement_id = town.id
        
    
    def _find_suitable_locations(self) -> List[Tuple[int, int, float]]:
        """Find suitable locations for settlements."""
        suitable_locations = []
        
        for (x, y), tile in self.map_tiles.items():
            if tile.terrain_type == TerrainType.WATER:
                continue  # Skip water tiles
            
            # Calculate suitability score
            suitability = (
                tile.population_capacity * 0.4 +
                tile.resource_level * 0.3 +
                (1.0 / max(1.0, tile.construction_difficulty)) * 0.3
            )
            
            # Bonus for coastal areas
            if self._is_coastal(x, y):
                suitability *= 1.5
            
            suitable_locations.append((x, y, suitability))
        
        # Sort by suitability (highest first)
        suitable_locations.sort(key=lambda x: x[2], reverse=True)
        return suitable_locations
    
    def _is_coastal(self, x: int, y: int) -> bool:
        """Check if a tile is coastal (adjacent to water)."""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.map_tiles:
                    if self.map_tiles[(nx, ny)].terrain_type == TerrainType.WATER:
                        return True
        return False
    
    def _select_town_locations(self, suitable_locations: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Select town locations based on urban concentration."""
        if self.urban_concentration == "high":
            # Towns cluster around cities
            return self._select_clustered_towns(suitable_locations)
        elif self.urban_concentration == "low":
            # Towns are evenly distributed
            return self._select_distributed_towns(suitable_locations)
        else:  # medium
            # Balanced approach
            return self._select_balanced_towns(suitable_locations)
    
    def _select_clustered_towns(self, suitable_locations: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Select towns that cluster around cities."""
        selected_towns = []
        city_positions = [(settlement.x, settlement.y) for settlement in self.settlements.values() 
                          if settlement.settlement_type == SettlementType.CITY]
        
        for location in suitable_locations:
            x, y, suitability = location
            
            # Calculate distance to nearest city
            min_distance = min(np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in city_positions)
            
            # Prefer locations closer to cities
            if min_distance < 30:  # Within reasonable distance
                selected_towns.append(location)
                if len(selected_towns) >= self.num_towns:
                    break
        
        return selected_towns
    
    def _select_distributed_towns(self, suitable_locations: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Select towns that are evenly distributed."""
        # Simple approach: take every nth suitable location
        step = max(1, len(suitable_locations) // self.num_towns)
        return suitable_locations[::step][:self.num_towns]
    
    def _select_balanced_towns(self, suitable_locations: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Select towns using a balanced approach."""
        # Mix of clustered and distributed
        clustered = self._select_clustered_towns(suitable_locations[:len(suitable_locations)//2])
        distributed = self._select_distributed_towns(suitable_locations[len(suitable_locations)//2:])
        
        return (clustered + distributed)[:self.num_towns]
    
    def _distribute_population(self):
        """Distribute population among settlements."""
        self._log("Distributing population...")
        
        urban_population = self.total_population * (1 - self.rural_population_percent)
        
        # Distribute urban population hierarchically
        cities = [s for s in self.settlements.values() if s.settlement_type == SettlementType.CITY]
        towns = [s for s in self.settlements.values() if s.settlement_type == SettlementType.TOWN]
        
        # Cities get more population based on economic importance
        city_population = urban_population * 0.7
        town_population = urban_population * 0.3
        
        # Distribute among cities
        total_city_importance = sum(city.economic_importance for city in cities)
        for city in cities:
            city.population = int(city_population * (city.economic_importance / total_city_importance))
        
        # Distribute among towns
        total_town_importance = sum(town.economic_importance for town in towns)
        for town in towns:
            town.population = int(town_population * (town.economic_importance / total_town_importance))
    
    def _generate_infrastructure(self):
        """Generate infrastructure network connecting settlements."""
        self._log("Generating infrastructure network...")
        
        settlements_list = list(self.settlements.values())
        
        # Create minimum spanning tree for basic connectivity
        mst_segments = self._create_minimum_spanning_tree(settlements_list)
        
        # Add additional connections for better connectivity
        additional_segments = self._add_additional_connections(settlements_list, mst_segments)
        
        # Store all infrastructure segments
        all_segments = mst_segments + additional_segments
        for segment in all_segments:
            self.infrastructure_segments[segment.id] = segment
    
    def _create_minimum_spanning_tree(self, settlements: List[Settlement]) -> List[InfrastructureSegment]:
        """Create minimum spanning tree connecting all settlements."""
        if len(settlements) < 2:
            return []
        
        # Use Prim's algorithm for MST
        mst_segments = []
        visited = set()
        unvisited = set(settlement.id for settlement in settlements)
        
        # Start with the first settlement
        current = settlements[0].id
        visited.add(current)
        unvisited.remove(current)
        
        while unvisited:
            min_cost = float('inf')
            min_segment = None
            
            for visited_id in visited:
                for unvisited_id in unvisited:
                    visited_settlement = self.settlements[visited_id]
                    unvisited_settlement = self.settlements[unvisited_id]
                    
                    # Calculate path and cost
                    path, cost = self._find_path(visited_settlement, unvisited_settlement)
                    
                    if cost < min_cost:
                        min_cost = cost
                        min_segment = InfrastructureSegment(
                            id=f"SEG_{len(mst_segments):03d}",
                            start_settlement=visited_id,
                            end_settlement=unvisited_id,
                            path=path,
                            cost=cost,
                            terrain_multiplier=self._calculate_terrain_multiplier(path)
                        )
            
            if min_segment:
                mst_segments.append(min_segment)
                visited.add(min_segment.end_settlement)
                unvisited.remove(min_segment.end_settlement)
        
        return mst_segments
    
    def _add_additional_connections(self, settlements: List[Settlement], 
                                  existing_segments: List[InfrastructureSegment]) -> List[InfrastructureSegment]:
        """Add additional connections for better connectivity."""
        additional_segments = []
        
        # Add connections between major cities
        cities = [s for s in settlements if s.settlement_type == SettlementType.CITY]
        
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                city1, city2 = cities[i], cities[j]
                
                # Check if connection already exists
                existing_connection = any(
                    (seg.start_settlement == city1.id and seg.end_settlement == city2.id) or
                    (seg.start_settlement == city2.id and seg.end_settlement == city1.id)
                    for seg in existing_segments
                )
                
                if not existing_connection:
                    path, cost = self._find_path(city1, city2)
                    segment = InfrastructureSegment(
                        id=f"SEG_{len(existing_segments) + len(additional_segments):03d}",
                        start_settlement=city1.id,
                        end_settlement=city2.id,
                        path=path,
                        cost=cost,
                        terrain_multiplier=self._calculate_terrain_multiplier(path)
                    )
                    additional_segments.append(segment)
        
        return additional_segments
    
    def _find_path(self, start: Settlement, end: Settlement) -> Tuple[List[Tuple[int, int]], float]:
        """Find optimal path between two settlements using A* algorithm."""
        start_pos = (start.x, start.y)
        end_pos = (end.x, end.y)
        
        # Use A* pathfinding
        path = self._astar_pathfinding(start_pos, end_pos)
        
        # Check if path is valid
        if len(path) < 2:
            # Fallback to direct line if pathfinding fails
            path = [start_pos, end_pos]
        
        # Calculate total cost
        total_cost = 0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            tile = self.map_tiles.get((x2, y2))
            if tile:
                total_cost += tile.construction_difficulty
        
        return path, total_cost
    
    def _astar_pathfinding(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm with optimizations for large maps."""
        def heuristic(pos1, pos2):
            # Use Manhattan distance for faster computation
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            # Only check cardinal directions for faster pathfinding
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    tile = self.map_tiles.get((nx, ny))
                    if tile and tile.terrain_type != TerrainType.WATER:
                        neighbors.append((nx, ny))
            return neighbors
        
        # Limit search area for very large maps
        max_distance = min(50, max(abs(start[0] - end[0]), abs(start[1] - end[1])) * 2)
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        closed_set = set()
        
        iterations = 0
        max_iterations = 1000  # Prevent infinite loops
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)[1]
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            closed_set.add(current)
            
            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tile = self.map_tiles.get(neighbor)
                if not tile:
                    continue
                
                tentative_g_score = g_score[current] + tile.construction_difficulty
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    
                    # Check if already in open set
                    in_open_set = any(neighbor == item[1] for item in open_set)
                    if not in_open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # If no path found, return direct line
        return [start, end]
    
    def _calculate_terrain_multiplier(self, path: List[Tuple[int, int]]) -> float:
        """Calculate terrain multiplier for a path."""
        if not path:
            return 1.0
        
        total_difficulty = 0
        valid_tiles = 0
        for x, y in path:
            tile = self.map_tiles.get((x, y))
            if tile:
                total_difficulty += tile.construction_difficulty
                valid_tiles += 1
        
        if valid_tiles == 0:
            return 1.0
        
        multiplier = total_difficulty / valid_tiles if valid_tiles > 0 else 1.0
        return multiplier
    
    def _calculate_initial_logistics(self):
        """Calculate initial logistics costs between settlements."""
        settlements_list = list(self.settlements.values())
        
        if len(settlements_list) < 2:
            return
        
        for i, settlement1 in enumerate(settlements_list):
            for j, settlement2 in enumerate(settlements_list):
                if i != j:
                    # Find path between settlements
                    path, cost = self._find_path(settlement1, settlement2)
                    
                    # Calculate logistics cost: BaseRate * Distance * TerrainMultiplier
                    distance = len(path)  # Simplified distance
                    terrain_multiplier = self._calculate_terrain_multiplier(path)
                    base_rate = 1.0  # Base transportation rate
                    
                    logistics_cost = base_rate * distance * terrain_multiplier
                    
                    self.logistics_costs[(settlement1.id, settlement2.id)] = logistics_cost
    
    def _get_actual_terrain_distribution(self) -> Dict[str, float]:
        """Get actual terrain distribution in the generated map."""
        terrain_counts = defaultdict(int)
        total_tiles = len(self.map_tiles)
        
        for tile in self.map_tiles.values():
            terrain_counts[tile.terrain_type.value] += 1
        
        return {terrain: count / total_tiles for terrain, count in terrain_counts.items()}
    
    def integrate_with_economic_plan(self, economic_plan: Dict[str, Any]):
        """
        Integrate the map with an existing economic plan.
        
        Args:
            economic_plan: Economic plan from the cybernetic planning system
            
        Returns:
            Dict with success status and integration details
        """
        try:
            self.economic_plan = economic_plan
            
            # Map economic sectors to settlements
            self._map_sectors_to_settlements()
            
            # Update logistics costs based on economic activity
            self._update_logistics_for_economic_plan()
            
            # Handle multi-year plan structure
            if isinstance(economic_plan, dict) and all(isinstance(k, int) for k in economic_plan.keys()):
                # Multi-year plan - use first year's data
                first_year = min(economic_plan.keys())
                year_data = economic_plan[first_year]
                sectors_count = len(year_data.get('sectors', []))
            else:
                # Single year plan
                sectors_count = len(economic_plan.get('sectors', []))
            
            settlements_count = len(self.settlements)
            
            print(f"Integrated map with economic plan containing {sectors_count} sectors")
            
            return {
                "success": True,
                "message": f"Successfully integrated {sectors_count} sectors with {settlements_count} settlements",
                "sectors_integrated": sectors_count,
                "settlements_used": settlements_count
            }
            
        except Exception as e:
            print(f"Error integrating economic plan: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _map_sectors_to_settlements(self):
        """Map economic sectors to settlements."""
        if not self.economic_plan:
            return
        
        # Handle multi-year plan structure
        if isinstance(self.economic_plan, dict) and all(isinstance(k, int) for k in self.economic_plan.keys()):
            # Multi-year plan - use first year's data
            first_year = min(self.economic_plan.keys())
            year_data = self.economic_plan[first_year]
            sectors = year_data.get('sectors', [])
        else:
            # Single year plan
            sectors = self.economic_plan.get('sectors', [])
        
        settlements_list = list(self.settlements.values())
        
        # Simple mapping: distribute sectors among settlements
        sectors_per_settlement = len(sectors) // len(settlements_list)
        remaining_sectors = len(sectors) % len(settlements_list)
        
        sector_index = 0
        for i, settlement in enumerate(settlements_list):
            # Assign sectors to this settlement
            num_sectors = sectors_per_settlement + (1 if i < remaining_sectors else 0)
            settlement.sectors = sectors[sector_index:sector_index + num_sectors]
            sector_index += num_sectors
    
    def _update_logistics_for_economic_plan(self):
        """Update logistics costs based on economic plan requirements."""
        if not self.economic_plan:
            return
        
        # Adjust logistics costs based on economic activity levels
        # Higher economic activity = higher transportation needs = higher costs
        total_economic_output = self.economic_plan.get('total_economic_output', 1.0)
        
        for (settlement1_id, settlement2_id), cost in self.logistics_costs.items():
            # Scale costs based on economic output
            scaled_cost = cost * (1.0 + total_economic_output / 1000000)  # Scale factor
            self.logistics_costs[(settlement1_id, settlement2_id)] = scaled_cost
    
    def simulate_time_step(self) -> Dict[str, Any]:
        """
        Simulate one time step of the map-based simulation.
        
        Returns:
            Dictionary with simulation results
        """
        self.current_time_step += 1
        
        # Update logistics costs (may change due to infrastructure improvements)
        self._update_logistics_costs()
        
        # Check for natural disasters
        disaster_result = self._check_for_disasters()
        
        # Calculate total logistics friction
        total_logistics_friction = self._calculate_total_logistics_friction()
        
        return {
            "time_step": self.current_time_step,
            "total_logistics_friction": total_logistics_friction,
            "disaster_events": disaster_result,
            "settlements_count": len(self.settlements),
            "infrastructure_segments": len(self.infrastructure_segments),
            "active_disasters": len([d for d in self.disaster_events if d.duration_days > 0])
        }
    
    def _update_logistics_costs(self):
        """Update logistics costs based on current conditions."""
        # Infrastructure improvements over time (simplified)
        improvement_factor = 1.0 - (self.current_time_step * 0.001)  # 0.1% improvement per time step
        improvement_factor = max(0.5, improvement_factor)  # Cap at 50% improvement
        
        for key, cost in self.logistics_costs.items():
            self.logistics_costs[key] = cost * improvement_factor
    
    def _check_for_disasters(self) -> Dict[str, Any]:
        """Check for natural disasters this time step."""
        disaster_result = {
            "new_disasters": [],
            "ongoing_disasters": [],
            "resolved_disasters": []
        }
        
        # Check for new disasters
        if random.random() < self.disaster_probability:
            disaster = self._generate_disaster()
            if disaster:
                self.disaster_events.append(disaster)
                disaster_result["new_disasters"].append(disaster)
        
        # Update ongoing disasters
        for disaster in self.disaster_events[:]:  # Copy list to avoid modification during iteration
            disaster.duration_days -= 1
            
            if disaster.duration_days <= 0:
                # Disaster resolved
                self.disaster_events.remove(disaster)
                disaster_result["resolved_disasters"].append(disaster)
            else:
                disaster_result["ongoing_disasters"].append(disaster)
        
        return disaster_result
    
    def _generate_disaster(self) -> Optional[DisasterEvent]:
        """Generate a random natural disaster."""
        # Choose disaster type based on terrain
        disaster_types = [DisasterType.FLOOD, DisasterType.EARTHQUAKE, 
                         DisasterType.DROUGHT, DisasterType.STORM]
        disaster_type = random.choice(disaster_types)
        
        # Choose location based on disaster type
        if disaster_type == DisasterType.FLOOD:
            # Floods occur near water
            suitable_tiles = [(x, y) for (x, y), tile in self.map_tiles.items()
                             if tile.terrain_type == TerrainType.WATER or self._is_coastal(x, y)]
        elif disaster_type == DisasterType.EARTHQUAKE:
            # Earthquakes occur in mountainous areas
            suitable_tiles = [(x, y) for (x, y), tile in self.map_tiles.items()
                             if tile.terrain_type == TerrainType.MOUNTAIN]
        else:
            # Other disasters can occur anywhere
            suitable_tiles = [(x, y) for (x, y), tile in self.map_tiles.items()
                             if tile.terrain_type != TerrainType.WATER]
        
        if not suitable_tiles:
            return None
        
        # Choose random location
        x, y = random.choice(suitable_tiles)
        
        # Generate disaster properties
        radius = random.randint(2, 8)
        intensity = random.uniform(0.3, 1.0)
        duration_days = random.randint(3, 10)
        
        disaster = DisasterEvent(
            disaster_type=disaster_type,
            x=x, y=y,
            radius=radius,
            intensity=intensity,
            duration_days=duration_days
        )
        
        # Calculate affected tiles
        disaster.affected_tiles = self._get_affected_tiles(disaster)
        
        return disaster
    
    def _get_affected_tiles(self, disaster: DisasterEvent) -> List[Tuple[int, int]]:
        """Get tiles affected by a disaster."""
        affected_tiles = []
        
        for dx in range(-disaster.radius, disaster.radius + 1):
            for dy in range(-disaster.radius, disaster.radius + 1):
                x, y = disaster.x + dx, disaster.y + dy
                
                # Check if tile is within radius
                distance = np.sqrt(dx**2 + dy**2)
                if distance <= disaster.radius:
                    if (x, y) in self.map_tiles:
                        affected_tiles.append((x, y))
        
        return affected_tiles
    
    def _calculate_total_logistics_friction(self) -> float:
        """Calculate total logistical friction for the entire economic plan."""
        if not self.logistics_costs:
            return 0.0
        
        # Sum all logistics costs
        total_cost = sum(self.logistics_costs.values())
        
        # Normalize by number of connections
        num_connections = len(self.logistics_costs)
        if num_connections > 0:
            return total_cost / num_connections
        
        return total_cost
    
    def get_map_summary(self) -> Dict[str, Any]:
        """Get a summary of the generated map."""
        return {
            "map_dimensions": (self.map_width, self.map_height),
            "terrain_distribution": self._get_actual_terrain_distribution(),
            "settlements": {
                "total": len(self.settlements),
                "cities": len([s for s in self.settlements.values() if s.settlement_type == SettlementType.CITY]),
                "towns": len([s for s in self.settlements.values() if s.settlement_type == SettlementType.TOWN])
            },
            "infrastructure": {
                "total_segments": len(self.infrastructure_segments),
                "total_cost": sum(seg.cost for seg in self.infrastructure_segments.values())
            },
            "population": {
                "total": sum(s.population for s in self.settlements.values()),
                "urban": sum(s.population for s in self.settlements.values() if s.settlement_type != SettlementType.RURAL),
                "rural_percent": self.rural_population_percent
            },
            "logistics": {
                "total_connections": len(self.logistics_costs),
                "average_cost": np.mean(list(self.logistics_costs.values())) if self.logistics_costs else 0,
                "total_friction": self._calculate_total_logistics_friction()
            }
        }
    
    def export_map_data(self, file_path: str):
        """Export map data to file."""
        export_data = {
            "map_config": {
                "map_width": self.map_width,
                "map_height": self.map_height,
                "terrain_distribution": self.terrain_distribution,
                "num_cities": self.num_cities,
                "num_towns": self.num_towns,
                "total_population": self.total_population,
                "rural_population_percent": self.rural_population_percent,
                "urban_concentration": self.urban_concentration
            },
            "settlements": {
                settlement.id: {
                    "id": settlement.id,
                    "name": settlement.name,
                    "settlement_type": settlement.settlement_type.value,
                    "x": settlement.x,
                    "y": settlement.y,
                    "population": settlement.population,
                    "economic_importance": settlement.economic_importance,
                    "sectors": settlement.sectors
                }
                for settlement in self.settlements.values()
            },
            "infrastructure_segments": {
                seg.id: {
                    "id": seg.id,
                    "start_settlement": seg.start_settlement,
                    "end_settlement": seg.end_settlement,
                    "path": seg.path,
                    "cost": seg.cost,
                    "terrain_multiplier": seg.terrain_multiplier,
                    "is_damaged": seg.is_damaged
                }
                for seg in self.infrastructure_segments.values()
            },
            "logistics_costs": {
                f"{start}_{end}": cost
                for (start, end), cost in self.logistics_costs.items()
            },
            "simulation_state": {
                "current_time_step": self.current_time_step,
                "disaster_probability": self.disaster_probability,
                "active_disasters": len(self.disaster_events)
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Map data exported to {file_path}")
    
    def load_map_data(self, file_path: str):
        """Load map data from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Restore configuration
        config = data["map_config"]
        self.map_width = config["map_width"]
        self.map_height = config["map_height"]
        self.terrain_distribution = config["terrain_distribution"]
        self.num_cities = config["num_cities"]
        self.num_towns = config["num_towns"]
        self.total_population = config["total_population"]
        self.rural_population_percent = config["rural_population_percent"]
        self.urban_concentration = config["urban_concentration"]
        
        # Restore settlements
        self.settlements = {}
        for settlement_data in data["settlements"].values():
            settlement = Settlement(
                id=settlement_data["id"],
                name=settlement_data["name"],
                settlement_type=SettlementType(settlement_data["settlement_type"]),
                x=settlement_data["x"],
                y=settlement_data["y"],
                population=settlement_data["population"],
                economic_importance=settlement_data["economic_importance"],
                sectors=settlement_data["sectors"]
            )
            self.settlements[settlement.id] = settlement
        
        # Restore infrastructure segments
        self.infrastructure_segments = {}
        for seg_data in data["infrastructure_segments"].values():
            segment = InfrastructureSegment(
                id=seg_data["id"],
                start_settlement=seg_data["start_settlement"],
                end_settlement=seg_data["end_settlement"],
                path=seg_data["path"],
                cost=seg_data["cost"],
                terrain_multiplier=seg_data["terrain_multiplier"],
                is_damaged=seg_data["is_damaged"]
            )
            self.infrastructure_segments[segment.id] = segment
        
        # Restore logistics costs
        self.logistics_costs = {}
        for key, cost in data["logistics_costs"].items():
            start, end = key.split("_", 1)
            self.logistics_costs[(start, end)] = cost
        
        # Restore simulation state
        state = data["simulation_state"]
        self.current_time_step = state["current_time_step"]
        self.disaster_probability = state["disaster_probability"]
        
        print(f"Map data loaded from {file_path}")
