#!/usr / bin / env python3
"""
Interactive Map Visualization Module for Cybernetic Planning System
Provides geographic visualization of the simulation environment.
"""

import folium
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import webbrowser
import tempfile

@dataclass
class GeographicFeature:
    """Represents a geographic feature on the map."""
    name: str
    coordinates: List[Tuple[float, float]]
    feature_type: str  # 'water', 'mountain', 'plain', 'forest'
    properties: Dict[str, Any]

@dataclass
class Settlement:
    """Represents a settlement (city, town, rural area)."""
    name: str
    coordinates: Tuple[float, float]
    population: int
    settlement_type: str  # 'city', 'town', 'rural'
    economic_sectors: List[str]

@dataclass
class EconomicZone:
    """Represents an economic activity zone."""
    name: str
    coordinates: List[Tuple[float, float]]
    zone_type: str  # 'industrial', 'agricultural', 'resource_extraction', 'service'
    sectors: List[str]
    production_capacity: float

@dataclass
class Infrastructure:
    """Represents infrastructure networks."""
    name: str
    coordinates: List[Tuple[float, float]]
    infrastructure_type: str  # 'road', 'railway', 'utility'
    capacity: float
    status: str  # 'operational', 'under_construction', 'damaged'

class MapGenerator:
    """Generates procedural maps for the simulation environment."""

    def __init__(self, map_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None):
        """
        Initialize map generator.

        Args:
            map_bounds: ((min_lat, min_lon), (max_lat, max_lon)) for map boundaries
        """
        if map_bounds is None:
            # Default bounds for a reasonable simulation area
            self.map_bounds = ((40.0, -80.0), (50.0, -70.0))  # Northeast US region
        else:
            self.map_bounds = map_bounds

        self.center_lat = (self.map_bounds[0][0] + self.map_bounds[1][0]) / 2
        self.center_lon = (self.map_bounds[0][1] + self.map_bounds[1][1]) / 2

        # Initialize random seed for reproducible maps
        random.seed(42)
        np.random.seed(42)

    def generate_geographic_features(self, num_features: int = 20) -> List[GeographicFeature]:
        """Generate procedural geographic features."""
        features = []

        # Generate water bodies
        for i in range(num_features // 4):
            coords = self._generate_polygon_coords(3, 8)
            features.append(GeographicFeature(
                name = f"Lake {i + 1}",
                coordinates = coords,
                feature_type="water",
                properties={"depth": random.uniform(5, 50), "area": random.uniform(10, 100)}
            ))

        # Generate mountain ranges
        for i in range(num_features // 6):
            coords = self._generate_polygon_coords(4, 12)
            features.append(GeographicFeature(
                name = f"Mountain Range {i + 1}",
                coordinates = coords,
                feature_type="mountain",
                properties={"elevation": random.uniform(500, 2000), "difficulty": random.uniform(0.5, 2.0)}
            ))

        # Generate forest areas
        for i in range(num_features // 3):
            coords = self._generate_polygon_coords(3, 10)
            features.append(GeographicFeature(
                name = f"Forest {i + 1}",
                coordinates = coords,
                feature_type="forest",
                properties={"tree_density": random.uniform(0.3, 0.9), "timber_value": random.uniform(10, 100)}
            ))

        return features

    def generate_settlements(self, num_cities: int = 3, num_towns: int = 8, num_rural: int = 15) -> List[Settlement]:
        """Generate settlements of different types."""
        settlements = []

        # Generate cities
        for i in range(num_cities):
            coords = self._random_coordinates()
            settlements.append(Settlement(
                name = f"City {i + 1}",
                coordinates = coords,
                population = random.randint(50000, 500000),
                settlement_type="city",
                economic_sectors = random.sample(["manufacturing", "services", "technology", "finance"], 3)
            ))

        # Generate towns
        for i in range(num_towns):
            coords = self._random_coordinates()
            settlements.append(Settlement(
                name = f"Town {i + 1}",
                coordinates = coords,
                population = random.randint(5000, 50000),
                settlement_type="town",
                economic_sectors = random.sample(["agriculture", "manufacturing", "services"], 2)
            ))

        # Generate rural areas
        for i in range(num_rural):
            coords = self._random_coordinates()
            settlements.append(Settlement(
                name = f"Rural Area {i + 1}",
                coordinates = coords,
                population = random.randint(100, 5000),
                settlement_type="rural",
                economic_sectors=["agriculture"]
            ))

        return settlements

    def generate_economic_zones(self, settlements: List[Settlement]) -> List[EconomicZone]:
        """Generate economic zones based on settlements."""
        zones = []

        for settlement in settlements:
            if settlement.settlement_type == "city":
                # Industrial zones around cities
                for i in range(2):
                    coords = self._generate_zone_coords(settlement.coordinates, 0.1)
                    zones.append(EconomicZone(
                        name = f"{settlement.name} Industrial Zone {i + 1}",
                        coordinates = coords,
                        zone_type="industrial",
                        sectors=["manufacturing", "heavy_industry"],
                        production_capacity = random.uniform(1000, 5000)
                    ))

            elif settlement.settlement_type == "town":
                # Mixed zones around towns
                coords = self._generate_zone_coords(settlement.coordinates, 0.05)
                zones.append(EconomicZone(
                    name = f"{settlement.name} Mixed Zone",
                    coordinates = coords,
                    zone_type="mixed",
                    sectors=["agriculture", "light_manufacturing"],
                    production_capacity = random.uniform(500, 2000)
                ))

            else:  # rural
                # Agricultural zones around rural areas
                coords = self._generate_zone_coords(settlement.coordinates, 0.2)
                zones.append(EconomicZone(
                    name = f"{settlement.name} Agricultural Zone",
                    coordinates = coords,
                    zone_type="agricultural",
                    sectors=["agriculture", "livestock"],
                    production_capacity = random.uniform(200, 1000)
                ))

        return zones

    def generate_infrastructure(self, settlements: List[Settlement]) -> List[Infrastructure]:
        """Generate infrastructure networks connecting settlements."""
        infrastructure = []

        # Generate road network
        for i, settlement1 in enumerate(settlements):
            for settlement2 in settlements[i + 1:]:
                if random.random() < 0.3:  # 30% chance of connection
                    coords = [settlement1.coordinates, settlement2.coordinates]
                    infrastructure.append(Infrastructure(
                        name = f"Road {settlement1.name}-{settlement2.name}",
                        coordinates = coords,
                        infrastructure_type="road",
                        capacity = random.uniform(100, 1000),
                        status="operational"
                    ))

        # Generate railway network (sparser)
        cities = [s for s in settlements if s.settlement_type == "city"]
        for i, city1 in enumerate(cities):
            for city2 in cities[i + 1:]:
                if random.random() < 0.5:  # 50% chance of rail connection
                    coords = [city1.coordinates, city2.coordinates]
                    infrastructure.append(Infrastructure(
                        name = f"Railway {city1.name}-{city2.name}",
                        coordinates = coords,
                        infrastructure_type="railway",
                        capacity = random.uniform(500, 2000),
                        status="operational"
                    ))

        return infrastructure

    def _random_coordinates(self) -> Tuple[float, float]:
        """Generate random coordinates within map bounds."""
        lat = random.uniform(self.map_bounds[0][0], self.map_bounds[1][0])
        lon = random.uniform(self.map_bounds[0][1], self.map_bounds[1][1])
        return (lat, lon)

    def _generate_polygon_coords(self, min_points: int, max_points: int) -> List[Tuple[float, float]]:
        """Generate coordinates for a polygon feature."""
        num_points = random.randint(min_points, max_points)
        center = self._random_coordinates()
        radius = random.uniform(0.05, 0.2)

        coords = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            lat = center[0] + radius * np.cos(angle) + random.uniform(-0.02, 0.02)
            lon = center[1] + radius * np.sin(angle) + random.uniform(-0.02, 0.02)
            coords.append((lat, lon))

        return coords

    def _generate_zone_coords(self, center: Tuple[float, float], size: float) -> List[Tuple[float, float]]:
        """Generate coordinates for an economic zone around a settlement."""
        coords = []
        for i in range(4):
            angle = np.pi * i / 2
            lat = center[0] + size * np.cos(angle) + random.uniform(-size / 4, size / 4)
            lon = center[1] + size * np.sin(angle) + random.uniform(-size / 4, size / 4)
            coords.append((lat, lon))
        return coords

class InteractiveMap:
    """Creates and manages interactive maps for the simulation."""

    def __init__(self, map_generator: MapGenerator = None):
        """Initialize the interactive map."""
        self.map_generator = map_generator or MapGenerator()
        self.map = None
        self.geographic_features = []
        self.settlements = []
        self.economic_zones = []
        self.infrastructure = []

    def generate_environment(self):
        """Generate the complete simulation environment."""
        print("Generating simulation environment...")

        # Generate all components
        self.geographic_features = self.map_generator.generate_geographic_features()
        self.settlements = self.map_generator.generate_settlements()
        self.economic_zones = self.map_generator.generate_economic_zones(self.settlements)
        self.infrastructure = self.map_generator.generate_infrastructure(self.settlements)

        print(f"Generated: {len(self.geographic_features)} geographic features, "
              f"{len(self.settlements)} settlements, "
              f"{len(self.economic_zones)} economic zones, "
              f"{len(self.infrastructure)} infrastructure elements")

    def create_map(self, zoom_start: int = 8) -> folium.Map:
        """Create the interactive map with all features."""
        # Create base map without background tiles
        self.map = folium.Map(
            location=[self.map_generator.center_lat, self.map_generator.center_lon],
            zoom_start = zoom_start,
            tiles = None  # No background tiles
        )

        # Add base terrain layer first
        self._add_base_terrain()

        # Add geographic features
        self._add_geographic_features()

        # Add settlements
        self._add_settlements()

        # Add economic zones
        self._add_economic_zones()

        # Add infrastructure
        self._add_infrastructure()

        # Add legend
        self._add_legend()

        return self.map

    def _add_base_terrain(self):
        """Add base terrain layer covering the entire map area."""
        # Create a large polygon covering the entire map area as light green (default / flatland)
        min_lat, min_lon = self.map_generator.map_bounds[0]
        max_lat, max_lon = self.map_generator.map_bounds[1]

        # Expand the bounds slightly to ensure full coverage
        padding = 0.1
        terrain_bounds = [
            [min_lat - padding, min_lon - padding],
            [max_lat + padding, min_lon - padding],
            [max_lat + padding, max_lon + padding],
            [min_lat - padding, max_lon + padding]
        ]

        folium.Polygon(
            locations = terrain_bounds,
            color='lightgreen',
            fill = True,
            fillColor='lightgreen',
            fillOpacity = 0.3,
            weight = 0,
            popup="Base Terrain (Flatland)"
        ).add_to(self.map)

    def _add_geographic_features(self):
        """Add geographic features to the map."""
        colors = {
            'water': 'blue',
            'mountain': 'brown',
            'forest': 'darkgreen',  # Dark green for forests
            'plain': 'lightgreen'   # Light green for flatland (already covered by base terrain)
        }

        for feature in self.geographic_features:
            color = colors.get(feature.feature_type, 'gray')
            folium.Polygon(
                locations = feature.coordinates,
                popup = f"<b>{feature.name}</b><br>Type: {feature.feature_type}<br>Properties: {feature.properties}",
                color = color,
                fill = True,
                fillColor = color,
                fillOpacity = 0.3,
                weight = 2
            ).add_to(self.map)

    def _add_settlements(self):
        """Add settlements to the map."""
        colors = {
            'city': 'red',
            'town': 'orange',
            'rural': 'yellow'
        }

        sizes = {
            'city': 15,
            'town': 10,
            'rural': 5
        }

        for settlement in self.settlements:
            color = colors.get(settlement.settlement_type, 'gray')
            size = sizes.get(settlement.settlement_type, 8)

            folium.CircleMarker(
                location = settlement.coordinates,
                radius = size,
                popup = f"<b>{settlement.name}</b><br>"
                      f"Type: {settlement.settlement_type}<br>"
                      f"Population: {settlement.population:,}<br>"
                      f"Sectors: {', '.join(settlement.economic_sectors)}",
                color='black',
                fillColor = color,
                fillOpacity = 0.8,
                weight = 2
            ).add_to(self.map)

    def _add_economic_zones(self):
        """Add economic zones to the map."""
        colors = {
            'industrial': 'purple',
            'agricultural': 'green',
            'mixed': 'orange',
            'resource_extraction': 'brown'
        }

        for zone in self.economic_zones:
            color = colors.get(zone.zone_type, 'gray')
            folium.Polygon(
                locations = zone.coordinates,
                popup = f"<b>{zone.name}</b><br>"
                      f"Type: {zone.zone_type}<br>"
                      f"Sectors: {', '.join(zone.sectors)}<br>"
                      f"Capacity: {zone.production_capacity:.0f}",
                color = color,
                fill = True,
                fillColor = color,
                fillOpacity = 0.2,
                weight = 1,
                dashArray='5, 5'
            ).add_to(self.map)

    def _add_infrastructure(self):
        """Add infrastructure to the map."""
        colors = {
            'road': 'black',
            'railway': 'black',  # Black for railways too
            'utility': 'gray'
        }

        weights = {
            'road': 3,
            'railway': 4,
            'utility': 2
        }

        dash_arrays = {
            'road': None,  # Solid line for roads
            'railway': '10, 5',  # Dotted line for railways
            'utility': None
        }

        for infra in self.infrastructure:
            color = colors.get(infra.infrastructure_type, 'gray')
            weight = weights.get(infra.infrastructure_type, 2)
            dash_array = dash_arrays.get(infra.infrastructure_type, None)

            folium.PolyLine(
                locations = infra.coordinates,
                popup = f"<b>{infra.name}</b><br>"
                      f"Type: {infra.infrastructure_type}<br>"
                      f"Capacity: {infra.capacity:.0f}<br>"
                      f"Status: {infra.status}",
                color = color,
                weight = weight,
                opacity = 0.8,
                dashArray = dash_array
            ).add_to(self.map)

    def _add_legend(self):
        """Add a legend to the map."""
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 200px; height: 220px;
                    background - color: white; border:2px solid grey; z - index:9999;
                    font - size:14px; padding: 10px">
        <p><b>Legend</b></p>
        <p><i class="fa fa - circle" style="color:red"></i> Cities</p>
        <p><i class="fa fa - circle" style="color:orange"></i> Towns</p>
        <p><i class="fa fa - circle" style="color:yellow"></i> Rural Areas</p>
        <p><i class="fa fa - square" style="color:blue"></i> Water Bodies</p>
        <p><i class="fa fa - square" style="color:brown"></i> Mountains</p>
        <p><i class="fa fa - square" style="color:darkgreen"></i> Forests</p>
        <p><i class="fa fa - square" style="color:lightgreen"></i> Flatland</p>
        <p><i class="fa fa - minus" style="color:black"></i> Roads</p>
        <p><i class="fa fa - minus" style="color:black; border - style: dashed"></i> Railways</p>
        </div>
        '''
        self.map.get_root().html.add_child(folium.Element(legend_html))

    def save_map(self, filename: str = "simulation_map.html"):
        """Save the map to an HTML file."""
        if self.map:
            self.map.save(filename)
            return filename
        return None

    def open_in_browser(self):
        """Open the map in the default web browser."""
        if self.map:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete = False)
            self.map.save(temp_file.name)
            temp_file.close()
            webbrowser.open(f'file://{temp_file.name}')
            return temp_file.name
        return None

    def get_environment_data(self) -> Dict[str, Any]:
        """Get the complete environment data as a dictionary."""
        return {
            'geographic_features': [
                {
                    'name': f.name,
                    'coordinates': f.coordinates,
                    'feature_type': f.feature_type,
                    'properties': f.properties
                } for f in self.geographic_features
            ],
            'settlements': [
                {
                    'name': s.name,
                    'coordinates': s.coordinates,
                    'population': s.population,
                    'settlement_type': s.settlement_type,
                    'economic_sectors': s.economic_sectors
                } for s in self.settlements
            ],
            'economic_zones': [
                {
                    'name': z.name,
                    'coordinates': z.coordinates,
                    'zone_type': z.zone_type,
                    'sectors': z.sectors,
                    'production_capacity': z.production_capacity
                } for z in self.economic_zones
            ],
            'infrastructure': [
                {
                    'name': i.name,
                    'coordinates': i.coordinates,
                    'infrastructure_type': i.infrastructure_type,
                    'capacity': i.capacity,
                    'status': i.status
                } for i in self.infrastructure
            ]
        }

def create_simulation_map(map_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None) -> InteractiveMap:
    """Create a complete simulation map with all features."""
    map_generator = MapGenerator(map_bounds)
    interactive_map = InteractiveMap(map_generator)
    interactive_map.generate_environment()
    interactive_map.create_map()
    return interactive_map

if __name__ == "__main__":
    # Test the map generation
    print("Creating simulation map...")
    sim_map = create_simulation_map()

    # Save and open the map
    filename = sim_map.save_map("test_simulation_map.html")
    print(f"Map saved to: {filename}")

    # Open in browser
    sim_map.open_in_browser()
    print("Map opened in browser")
