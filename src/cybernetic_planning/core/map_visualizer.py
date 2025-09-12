"""
Interactive Map Visualization System

Creates interactive map playback files that allow users to explore
all aspects of the map-based simulation in a browser or GUI.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import base64
from datetime import datetime

from .map_based_simulator import MapBasedSimulator, TerrainType, SettlementType, DisasterType

@dataclass
class MapVisualizationConfig:
    """Configuration for map visualization."""
    tile_size: int = 8  # Size of each tile in pixels
    show_terrain: bool = True
    show_settlements: bool = True
    show_infrastructure: bool = True
    show_disasters: bool = True
    show_logistics: bool = True
    show_population: bool = True
    show_economic_activity: bool = True
    animation_speed: float = 1.0  # Speed multiplier for time-based animations
    color_scheme: str = "default"  # "default", "terrain", "economic", "population"

class MapVisualizer:
    """
    Creates interactive map visualizations for the map-based simulator.
    
    Generates HTML/JavaScript files that can be viewed in browsers or
    integrated into desktop applications.
    """
    
    def __init__(self, simulator: MapBasedSimulator, config: Optional[MapVisualizationConfig] = None):
        """
        Initialize the map visualizer.
        
        Args:
            simulator: The map-based simulator instance
            config: Visualization configuration
        """
        self.simulator = simulator
        self.config = config or MapVisualizationConfig()
        
        # Color schemes
        self.color_schemes = {
            "default": {
                "terrain": {
                    TerrainType.FLATLAND: "#90EE90",  # Light green
                    TerrainType.FOREST: "#228B22",     # Forest green
                    TerrainType.MOUNTAIN: "#8B4513",   # Saddle brown
                    TerrainType.WATER: "#4169E1",      # Royal blue
                    TerrainType.COASTAL: "#87CEEB"     # Sky blue
                },
                "settlements": {
                    SettlementType.CITY: "#FF0000",    # Red
                    SettlementType.TOWN: "#FFA500",    # Orange
                    SettlementType.RURAL: "#FFFF00"    # Yellow
                },
                "infrastructure": "#000000",           # Black
                "disasters": {
                    DisasterType.FLOOD: "#0000FF",     # Blue
                    DisasterType.EARTHQUAKE: "#8B0000", # Dark red
                    DisasterType.DROUGHT: "#DAA520",    # Goldenrod
                    DisasterType.STORM: "#708090"      # Slate gray
                }
            },
            "terrain": {
                "terrain": {
                    TerrainType.FLATLAND: "#F4A460",   # Sandy brown
                    TerrainType.FOREST: "#006400",     # Dark green
                    TerrainType.MOUNTAIN: "#696969",   # Dim gray
                    TerrainType.WATER: "#1E90FF",      # Dodger blue
                    TerrainType.COASTAL: "#20B2AA"     # Light sea green
                },
                "settlements": {
                    SettlementType.CITY: "#DC143C",    # Crimson
                    SettlementType.TOWN: "#FF8C00",    # Dark orange
                    SettlementType.RURAL: "#FFD700"   # Gold
                },
                "infrastructure": "#2F4F4F",           # Dark slate gray
                "disasters": {
                    DisasterType.FLOOD: "#00BFFF",     # Deep sky blue
                    DisasterType.EARTHQUAKE: "#B22222", # Fire brick
                    DisasterType.DROUGHT: "#CD853F",    # Peru
                    DisasterType.STORM: "#4682B4"      # Steel blue
                }
            }
        }
        
        self.colors = self.color_schemes.get(self.config.color_scheme, self.color_schemes["default"])
        
        # Convert enum keys to strings for JSON serialization
        self.colors_serializable = {}
        for category, color_dict in self.colors.items():
            if isinstance(color_dict, dict):
                self.colors_serializable[category] = {k.value if hasattr(k, 'value') else str(k): v for k, v in color_dict.items()}
            else:
                self.colors_serializable[category] = color_dict
    
    def generate_interactive_map(self, output_path: str, include_simulation_data: bool = True) -> str:
        """
        Generate an interactive map HTML file.
        
        Args:
            output_path: Path to save the HTML file
            include_simulation_data: Whether to include simulation playback data
        
        Returns:
            Path to the generated HTML file
        """
        html_content = self._generate_html_template()
        
        # Generate map data
        map_data = self._generate_map_data()
        
        # Generate simulation data if requested
        simulation_data = {}
        if include_simulation_data:
            simulation_data = self._generate_simulation_data()
        
        # Generate JavaScript
        js_content = self._generate_javascript(map_data, simulation_data)
        
        # Combine HTML and JavaScript
        full_html = html_content.replace("<!-- MAP_DATA_PLACEHOLDER -->", json.dumps(map_data))
        full_html = full_html.replace("<!-- SIMULATION_DATA_PLACEHOLDER -->", json.dumps(simulation_data))
        full_html = full_html.replace("<!-- JAVASCRIPT_PLACEHOLDER -->", js_content)
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        return str(output_file)
    
    def _generate_html_template(self) -> str:
        """Generate the HTML template for the interactive map."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Map - Cybernetic Planning Simulation</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f0f0;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .controls {
            background: white;
            padding: 20px;
            border-bottom: 1px solid #ddd;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
            justify-content: center;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .control-group label {
            font-weight: 500;
            color: #333;
        }
        
        select, input, button {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        button {
            background: #667eea;
            color: white;
            border: none;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #5a6fd8;
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .map-container {
            display: flex;
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .map-panel {
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .map-header {
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            font-weight: 600;
            color: #333;
        }
        
        .map-canvas {
            position: relative;
            background: #e8f4f8;
            overflow: auto;
            max-height: 600px;
        }
        
        .info-panel {
            width: 300px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .info-header {
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            font-weight: 600;
            color: #333;
        }
        
        .info-content {
            padding: 20px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .info-section {
            margin-bottom: 20px;
        }
        
        .info-section h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.1em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .info-item:last-child {
            border-bottom: none;
        }
        
        .info-label {
            font-weight: 500;
            color: #666;
        }
        
        .info-value {
            color: #333;
            font-weight: 600;
        }
        
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
        }
        
        .legend-color {
            width: 16px;
            height: 16px;
            border: 1px solid #333;
            border-radius: 2px;
        }
        
        .timeline {
            margin-top: 20px;
        }
        
        .timeline-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .timeline-slider {
            flex: 1;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        
        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            max-width: 200px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 4px;
            margin: 20px;
        }
        
        @media (max-width: 768px) {
            .map-container {
                flex-direction: column;
            }
            
            .info-panel {
                width: 100%;
            }
            
            .controls {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Interactive Map Simulation</h1>
        <p>Cybernetic Planning Economic Map Visualization</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label for="viewMode">View Mode:</label>
            <select id="viewMode">
                <option value="terrain">Terrain</option>
                <option value="settlements">Settlements</option>
                <option value="infrastructure">Infrastructure</option>
                <option value="population">Population</option>
                <option value="economic">Economic Activity</option>
                <option value="disasters">Disasters</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="layerVisibility">Layers:</label>
            <select id="layerVisibility" multiple>
                <option value="terrain" selected>Terrain</option>
                <option value="settlements" selected>Settlements</option>
                <option value="infrastructure" selected>Infrastructure</option>
                <option value="disasters">Disasters</option>
                <option value="logistics">Logistics</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="animationSpeed">Speed:</label>
            <input type="range" id="animationSpeed" min="0.1" max="3" step="0.1" value="1">
            <span id="speedValue">1.0x</span>
        </div>
        
        <div class="control-group">
            <button id="playPause">Play</button>
            <button id="reset">Reset</button>
            <button id="export">Export</button>
        </div>
    </div>
    
    <div class="map-container">
        <div class="map-panel">
            <div class="map-header">
                <span id="mapTitle">Map View</span>
                <span id="mapStats" style="float: right; font-size: 0.9em; color: #666;"></span>
            </div>
            <div class="map-canvas" id="mapCanvas">
                <div class="loading">Loading map...</div>
            </div>
        </div>
        
        <div class="info-panel">
            <div class="info-header">Simulation Information</div>
            <div class="info-content" id="infoContent">
                <div class="loading">Loading information...</div>
            </div>
        </div>
    </div>
    
    <script>
        // Map data will be injected here
        const MAP_DATA = <!-- MAP_DATA_PLACEHOLDER -->;
        const SIMULATION_DATA = <!-- SIMULATION_DATA_PLACEHOLDER -->;
        
        // JavaScript code will be injected here
        <!-- JAVASCRIPT_PLACEHOLDER -->
    </script>
</body>
</html>
        """
    
    def _generate_map_data(self) -> Dict[str, Any]:
        """Generate map data for visualization."""
        map_data = {
            "dimensions": {
                "width": self.simulator.map_width,
                "height": self.simulator.map_height,
                "tile_size": self.config.tile_size
            },
            "terrain": [],
            "settlements": [],
            "infrastructure": [],
            "colors": self.colors_serializable,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "simulator_config": {
                    "num_cities": self.simulator.num_cities,
                    "num_towns": self.simulator.num_towns,
                    "total_population": self.simulator.total_population,
                    "urban_concentration": self.simulator.urban_concentration
                }
            }
        }
        
        # Generate terrain data
        for (x, y), tile in self.simulator.map_tiles.items():
            map_data["terrain"].append({
                "x": x,
                "y": y,
                "type": tile.terrain_type.value,
                "altitude": float(tile.altitude),
                "resource_level": float(tile.resource_level),
                "construction_difficulty": float(tile.construction_difficulty),
                "population_capacity": int(tile.population_capacity)
            })
        
        # Generate settlement data
        for settlement in self.simulator.settlements.values():
            map_data["settlements"].append({
                "id": settlement.id,
                "name": settlement.name,
                "type": settlement.settlement_type.value,
                "x": int(settlement.x),
                "y": int(settlement.y),
                "population": int(settlement.population),
                "economic_importance": float(settlement.economic_importance),
                "sectors": settlement.sectors
            })
        
        # Generate infrastructure data
        for segment in self.simulator.infrastructure_segments.values():
            map_data["infrastructure"].append({
                "id": segment.id,
                "start_settlement": segment.start_settlement,
                "end_settlement": segment.end_settlement,
                "path": segment.path,
                "cost": float(segment.cost),
                "terrain_multiplier": float(segment.terrain_multiplier),
                "is_damaged": bool(segment.is_damaged)
            })
        
        return map_data
    
    def _generate_simulation_data(self) -> Dict[str, Any]:
        """Generate simulation playback data."""
        simulation_data = {
            "current_time_step": self.simulator.current_time_step,
            "logistics_costs": {},
            "disaster_events": [],
            "time_series": []
        }
        
        # Convert logistics costs
        for (start, end), cost in self.simulator.logistics_costs.items():
            simulation_data["logistics_costs"][f"{start}_{end}"] = float(cost)
        
        # Convert disaster events
        for disaster in self.simulator.disaster_events:
            simulation_data["disaster_events"].append({
                "type": disaster.disaster_type.value,
                "x": int(disaster.x),
                "y": int(disaster.y),
                "radius": int(disaster.radius),
                "intensity": float(disaster.intensity),
                "duration_days": int(disaster.duration_days),
                "affected_tiles": disaster.affected_tiles
            })
        
        return simulation_data
    
    def _generate_javascript(self, map_data: Dict[str, Any], simulation_data: Dict[str, Any]) -> str:
        """Generate JavaScript code for the interactive map."""
        return """
        class InteractiveMap {
            constructor() {
                this.mapData = MAP_DATA;
                this.simulationData = SIMULATION_DATA;
                this.canvas = document.getElementById('mapCanvas');
                this.infoContent = document.getElementById('infoContent');
                this.currentView = 'terrain';
                this.isPlaying = false;
                this.currentTimeStep = 0;
                this.animationSpeed = 1.0;
                
                this.init();
            }
            
            init() {
                this.setupEventListeners();
                this.renderMap();
                this.updateInfoPanel();
                this.setupTimeline();
            }
            
            setupEventListeners() {
                // View mode change
                document.getElementById('viewMode').addEventListener('change', (e) => {
                    this.currentView = e.target.value;
                    this.renderMap();
                });
                
                // Animation speed
                const speedSlider = document.getElementById('animationSpeed');
                speedSlider.addEventListener('input', (e) => {
                    this.animationSpeed = parseFloat(e.target.value);
                    document.getElementById('speedValue').textContent = e.target.value + 'x';
                });
                
                // Play/Pause
                document.getElementById('playPause').addEventListener('click', () => {
                    this.togglePlayback();
                });
                
                // Reset
                document.getElementById('reset').addEventListener('click', () => {
                    this.resetSimulation();
                });
                
                // Export
                document.getElementById('export').addEventListener('click', () => {
                    this.exportMap();
                });
            }
            
            renderMap() {
                this.canvas.innerHTML = '';
                
                const mapWidth = this.mapData.dimensions.width * this.mapData.dimensions.tile_size;
                const mapHeight = this.mapData.dimensions.height * this.mapData.dimensions.tile_size;
                
                // Create SVG for the map
                const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                svg.setAttribute('width', mapWidth);
                svg.setAttribute('height', mapHeight);
                svg.setAttribute('viewBox', `0 0 ${mapWidth} ${mapHeight}`);
                svg.style.border = '1px solid #ddd';
                svg.style.background = '#e8f4f8';
                
                // Render terrain
                if (this.currentView === 'terrain' || this.currentView === 'settlements') {
                    this.renderTerrain(svg);
                }
                
                // Render infrastructure
                if (this.currentView === 'infrastructure') {
                    this.renderInfrastructure(svg);
                }
                
                // Render settlements
                if (this.currentView === 'settlements' || this.currentView === 'population' || this.currentView === 'economic') {
                    this.renderSettlements(svg);
                }
                
                // Render disasters
                if (this.currentView === 'disasters') {
                    this.renderDisasters(svg);
                }
                
                this.canvas.appendChild(svg);
                
                // Update map stats
                this.updateMapStats();
            }
            
            renderTerrain(svg) {
                const tileSize = this.mapData.dimensions.tile_size;
                
                this.mapData.terrain.forEach(tile => {
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', tile.x * tileSize);
                    rect.setAttribute('y', tile.y * tileSize);
                    rect.setAttribute('width', tileSize);
                    rect.setAttribute('height', tileSize);
                    rect.setAttribute('fill', this.mapData.colors.terrain[tile.type]);
                    rect.setAttribute('stroke', '#333');
                    rect.setAttribute('stroke-width', '0.5');
                    
                    // Add tooltip
                    rect.addEventListener('mouseenter', (e) => {
                        this.showTooltip(e, `Terrain: ${tile.type}\\nAltitude: ${tile.altitude.toFixed(1)}m\\nResources: ${tile.resource_level.toFixed(2)}`);
                    });
                    
                    rect.addEventListener('mouseleave', () => {
                        this.hideTooltip();
                    });
                    
                    svg.appendChild(rect);
                });
            }
            
            renderSettlements(svg) {
                const tileSize = this.mapData.dimensions.tile_size;
                
                this.mapData.settlements.forEach(settlement => {
                    // Get emoji based on settlement type
                    let emoji = '🏘️'; // Default town emoji
                    let emojiSize = 16;
                    
                    if (settlement.type === 'city') {
                        emoji = '🏙️';
                        emojiSize = 20;
                    } else if (settlement.type === 'town') {
                        emoji = '🏘️';
                        emojiSize = 16;
                    } else if (settlement.type === 'rural') {
                        emoji = '🏡';
                        emojiSize = 14;
                    }
                    
                    // Create emoji text element
                    const emojiText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    emojiText.setAttribute('x', settlement.x * tileSize + tileSize / 2);
                    emojiText.setAttribute('y', settlement.y * tileSize + tileSize / 2 + emojiSize / 3);
                    emojiText.setAttribute('text-anchor', 'middle');
                    emojiText.setAttribute('font-size', emojiSize);
                    emojiText.setAttribute('cursor', 'pointer');
                    emojiText.textContent = emoji;
                    
                    // Add click event for detailed information
                    emojiText.addEventListener('click', (e) => {
                        this.showSettlementDetails(settlement);
                    });
                    
                    // Add hover effects
                    emojiText.addEventListener('mouseenter', (e) => {
                        emojiText.setAttribute('font-size', emojiSize * 1.2);
                        this.showTooltip(e, `Click to view details for ${settlement.name}`);
                    });
                    
                    emojiText.addEventListener('mouseleave', () => {
                        emojiText.setAttribute('font-size', emojiSize);
                        this.hideTooltip();
                    });
                    
                    svg.appendChild(emojiText);
                });
            }
            
            renderInfrastructure(svg) {
                const tileSize = this.mapData.dimensions.tile_size;
                
                this.mapData.infrastructure.forEach(segment => {
                    if (segment.path.length < 2) return;
                    
                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    let pathData = '';
                    
                    segment.path.forEach((point, index) => {
                        const x = point[0] * tileSize + tileSize / 2;
                        const y = point[1] * tileSize + tileSize / 2;
                        
                        if (index === 0) {
                            pathData += `M ${x} ${y}`;
                        } else {
                            pathData += ` L ${x} ${y}`;
                        }
                    });
                    
                    path.setAttribute('d', pathData);
                    path.setAttribute('stroke', this.mapData.colors.infrastructure);
                    path.setAttribute('stroke-width', '2');
                    path.setAttribute('fill', 'none');
                    path.setAttribute('stroke-dasharray', segment.is_damaged ? '5,5' : 'none');
                    
                    // Add tooltip
                    path.addEventListener('mouseenter', (e) => {
                        this.showTooltip(e, `Infrastructure Segment\\nCost: ${segment.cost.toFixed(2)}\\nTerrain Multiplier: ${segment.terrain_multiplier.toFixed(2)}\\nStatus: ${segment.is_damaged ? 'Damaged' : 'Operational'}`);
                    });
                    
                    path.addEventListener('mouseleave', () => {
                        this.hideTooltip();
                    });
                    
                    svg.appendChild(path);
                });
            }
            
            renderDisasters(svg) {
                const tileSize = this.mapData.dimensions.tile_size;
                
                this.simulationData.disaster_events.forEach(disaster => {
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.setAttribute('cx', disaster.x * tileSize + tileSize / 2);
                    circle.setAttribute('cy', disaster.y * tileSize + tileSize / 2);
                    circle.setAttribute('r', disaster.radius * tileSize);
                    circle.setAttribute('fill', this.mapData.colors.disasters[disaster.type]);
                    circle.setAttribute('fill-opacity', '0.3');
                    circle.setAttribute('stroke', this.mapData.colors.disasters[disaster.type]);
                    circle.setAttribute('stroke-width', '2');
                    
                    // Add tooltip
                    circle.addEventListener('mouseenter', (e) => {
                        this.showTooltip(e, `Disaster: ${disaster.type}\\nIntensity: ${disaster.intensity.toFixed(2)}\\nDuration: ${disaster.duration_days} days\\nRadius: ${disaster.radius} tiles`);
                    });
                    
                    circle.addEventListener('mouseleave', () => {
                        this.hideTooltip();
                    });
                    
                    svg.appendChild(circle);
                });
            }
            
            showTooltip(event, text) {
                this.hideTooltip();
                
                const tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                tooltip.textContent = text;
                tooltip.style.left = event.pageX + 10 + 'px';
                tooltip.style.top = event.pageY - 10 + 'px';
                
                document.body.appendChild(tooltip);
            }
            
            hideTooltip() {
                const existing = document.querySelector('.tooltip');
                if (existing) {
                    existing.remove();
                }
            }
            
            showSettlementDetails(settlement) {
                // Remove any existing settlement details modal
                const existing = document.querySelector('.settlement-modal');
                if (existing) {
                    existing.remove();
                }
                
                // Create modal overlay
                const modal = document.createElement('div');
                modal.className = 'settlement-modal';
                modal.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.5);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 1000;
                `;
                
                // Create modal content
                const content = document.createElement('div');
                content.style.cssText = `
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    max-width: 500px;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                `;
                
                // Get emoji for settlement type
                let emoji = '🏘️';
                if (settlement.type === 'city') emoji = '🏙️';
                else if (settlement.type === 'rural') emoji = '🏡';
                
                // Create settlement details HTML
                content.innerHTML = `
                    <div style="text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 48px; margin-bottom: 10px;">${emoji}</div>
                        <h2 style="margin: 0; color: #333;">${settlement.name}</h2>
                        <p style="margin: 5px 0; color: #666; text-transform: capitalize;">${settlement.type.toLowerCase()} settlement</p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;">
                            <h4 style="margin: 0 0 10px 0; color: #333;">Population</h4>
                            <div style="font-size: 24px; font-weight: bold; color: #2c5aa0;">${settlement.population.toLocaleString()}</div>
                        </div>
                        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;">
                            <h4 style="margin: 0 0 10px 0; color: #333;">Economic Importance</h4>
                            <div style="font-size: 24px; font-weight: bold; color: #2c5aa0;">${settlement.economic_importance.toFixed(2)}</div>
                        </div>
                    </div>
                    
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="margin: 0 0 10px 0; color: #333;">Location</h4>
                        <p style="margin: 5px 0; color: #666;">Coordinates: (${settlement.x}, ${settlement.y})</p>
                        <p style="margin: 5px 0; color: #666;">Settlement ID: ${settlement.id}</p>
                    </div>
                    
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="margin: 0 0 10px 0; color: #333;">Settlement Type Details</h4>
                        <p style="margin: 5px 0; color: #666;">
                            ${settlement.type === 'city' ? 'Major urban center with high population density and economic activity.' : 
                              settlement.type === 'town' ? 'Medium-sized settlement serving as regional hub for surrounding areas.' : 
                              'Small rural community focused on agriculture and local services.'}
                        </p>
                    </div>
                    
                    <div style="text-align: center;">
                        <button onclick="this.closest('.settlement-modal').remove()" 
                                style="background: #2c5aa0; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px;">
                            Close
                        </button>
                    </div>
                `;
                
                modal.appendChild(content);
                document.body.appendChild(modal);
                
                // Close modal when clicking outside
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        modal.remove();
                    }
                });
            }
            
            updateMapStats() {
                const stats = document.getElementById('mapStats');
                const settlements = this.mapData.settlements.length;
                const infrastructure = this.mapData.infrastructure.length;
                const disasters = this.simulationData.disaster_events.length;
                
                stats.textContent = `${settlements} settlements, ${infrastructure} infrastructure segments, ${disasters} disasters`;
            }
            
            updateInfoPanel() {
                const content = this.infoContent;
                content.innerHTML = '';
                
                // Map overview
                const overview = this.createInfoSection('Map Overview', [
                    ['Dimensions', `${this.mapData.dimensions.width} × ${this.mapData.dimensions.height}`],
                    ['Total Tiles', (this.mapData.dimensions.width * this.mapData.dimensions.height).toLocaleString()],
                    ['Settlements', this.mapData.settlements.length],
                    ['Infrastructure Segments', this.mapData.infrastructure.length],
                    ['Current Time Step', this.simulationData.current_time_step]
                ]);
                content.appendChild(overview);
                
                // Settlement statistics
                const settlements = this.mapData.settlements;
                const totalPopulation = settlements.reduce((sum, s) => sum + s.population, 0);
                const cities = settlements.filter(s => s.type === 'city').length;
                const towns = settlements.filter(s => s.type === 'town').length;
                
                const settlementStats = this.createInfoSection('Settlement Statistics', [
                    ['Total Population', totalPopulation.toLocaleString()],
                    ['Cities', cities],
                    ['Towns', towns],
                    ['Average Population', Math.round(totalPopulation / settlements.length).toLocaleString()]
                ]);
                content.appendChild(settlementStats);
                
                // Terrain distribution
                const terrainCounts = {};
                this.mapData.terrain.forEach(tile => {
                    terrainCounts[tile.type] = (terrainCounts[tile.type] || 0) + 1;
                });
                
                const terrainStats = this.createInfoSection('Terrain Distribution', 
                    Object.entries(terrainCounts).map(([type, count]) => [
                        type.charAt(0).toUpperCase() + type.slice(1),
                        `${count} tiles (${(count / this.mapData.terrain.length * 100).toFixed(1)}%)`
                    ])
                );
                content.appendChild(terrainStats);
                
                // Legend
                const legend = this.createLegend();
                content.appendChild(legend);
            }
            
            createInfoSection(title, items) {
                const section = document.createElement('div');
                section.className = 'info-section';
                
                const header = document.createElement('h3');
                header.textContent = title;
                section.appendChild(header);
                
                items.forEach(([label, value]) => {
                    const item = document.createElement('div');
                    item.className = 'info-item';
                    
                    const labelSpan = document.createElement('span');
                    labelSpan.className = 'info-label';
                    labelSpan.textContent = label + ':';
                    
                    const valueSpan = document.createElement('span');
                    valueSpan.className = 'info-value';
                    valueSpan.textContent = value;
                    
                    item.appendChild(labelSpan);
                    item.appendChild(valueSpan);
                    section.appendChild(item);
                });
                
                return section;
            }
            
            createLegend() {
                const section = document.createElement('div');
                section.className = 'info-section';
                
                const header = document.createElement('h3');
                header.textContent = 'Legend';
                section.appendChild(header);
                
                const legend = document.createElement('div');
                legend.className = 'legend';
                
                // Terrain legend
                Object.entries(this.mapData.colors.terrain).forEach(([type, color]) => {
                    const item = document.createElement('div');
                    item.className = 'legend-item';
                    
                    const colorBox = document.createElement('div');
                    colorBox.className = 'legend-color';
                    colorBox.style.backgroundColor = color;
                    
                    const label = document.createElement('span');
                    label.textContent = type.charAt(0).toUpperCase() + type.slice(1);
                    
                    item.appendChild(colorBox);
                    item.appendChild(label);
                    legend.appendChild(item);
                });
                
                // Settlement legend
                Object.entries(this.mapData.colors.settlements).forEach(([type, color]) => {
                    const item = document.createElement('div');
                    item.className = 'legend-item';
                    
                    const colorBox = document.createElement('div');
                    colorBox.className = 'legend-color';
                    colorBox.style.backgroundColor = color;
                    
                    const label = document.createElement('span');
                    label.textContent = type.charAt(0).toUpperCase() + type.slice(1);
                    
                    item.appendChild(colorBox);
                    item.appendChild(label);
                    legend.appendChild(item);
                });
                
                section.appendChild(legend);
                return section;
            }
            
            setupTimeline() {
                // Timeline functionality would go here
                // For now, just show current time step
            }
            
            togglePlayback() {
                this.isPlaying = !this.isPlaying;
                const button = document.getElementById('playPause');
                button.textContent = this.isPlaying ? 'Pause' : 'Play';
                
                if (this.isPlaying) {
                    this.startPlayback();
                } else {
                    this.stopPlayback();
                }
            }
            
            startPlayback() {
                // Animation logic would go here
                console.log('Starting playback...');
            }
            
            stopPlayback() {
                console.log('Stopping playback...');
            }
            
            resetSimulation() {
                this.currentTimeStep = 0;
                this.isPlaying = false;
                document.getElementById('playPause').textContent = 'Play';
                this.renderMap();
            }
            
            exportMap() {
                const data = {
                    mapData: this.mapData,
                    simulationData: this.simulationData,
                    exportTime: new Date().toISOString()
                };
                
                const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `map_simulation_${new Date().toISOString().split('T')[0]}.json`;
                a.click();
                URL.revokeObjectURL(url);
            }
        }
        
        // Initialize the map when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new InteractiveMap();
        });
        """
    
    def generate_static_image(self, output_path: str, width: int = 800, height: int = 600) -> str:
        """
        Generate a static image of the map.
        
        Args:
            output_path: Path to save the image
            width: Image width in pixels
            height: Image height in pixels
        
        Returns:
            Path to the generated image
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import ListedColormap
            
            # Create figure
            fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
            
            # Calculate tile size for the image
            tile_width = width / self.simulator.map_width
            tile_height = height / self.simulator.map_height
            
            # Draw terrain
            for (x, y), tile in self.simulator.map_tiles.items():
                rect = patches.Rectangle(
                    (x * tile_width, y * tile_height),
                    tile_width, tile_height,
                    facecolor=self.colors['terrain'][tile.terrain_type],
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(rect)
            
            # Draw infrastructure
            for segment in self.simulator.infrastructure_segments.values():
                if len(segment.path) >= 2:
                    path_x = [point[0] * tile_width + tile_width/2 for point in segment.path]
                    path_y = [point[1] * tile_height + tile_height/2 for point in segment.path]
                    ax.plot(path_x, path_y, color=self.colors['infrastructure'], linewidth=2)
            
            # Draw settlements
            for settlement in self.simulator.settlements.values():
                circle = patches.Circle(
                    (settlement.x * tile_width + tile_width/2, settlement.y * tile_height + tile_height/2),
                    radius=max(2, min(8, settlement.population / 50000)),
                    facecolor=self.colors['settlements'][settlement.settlement_type],
                    edgecolor='black',
                    linewidth=2
                )
                ax.add_patch(circle)
                
                # Add settlement name
                ax.text(
                    settlement.x * tile_width + tile_width/2,
                    settlement.y * tile_height + tile_height/2 + 10,
                    settlement.name,
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold'
                )
            
            # Draw disasters
            for disaster in self.simulator.disaster_events:
                circle = patches.Circle(
                    (disaster.x * tile_width + tile_width/2, disaster.y * tile_height + tile_height/2),
                    radius=disaster.radius * min(tile_width, tile_height),
                    facecolor=self.colors['disasters'][disaster.disaster_type],
                    alpha=0.3,
                    edgecolor=self.colors['disasters'][disaster.disaster_type],
                    linewidth=2
                )
                ax.add_patch(circle)
            
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Invert Y axis to match typical map orientation
            ax.set_title('Map-Based Economic Simulation', fontsize=14, fontweight='bold')
            
            # Add legend
            legend_elements = []
            for terrain_type, color in self.colors['terrain'].items():
                legend_elements.append(patches.Patch(color=color, label=terrain_type.value.title()))
            
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except ImportError:
            raise ImportError("matplotlib is required for static image generation. Install with: pip install matplotlib")
    
    def generate_gui_integration(self) -> str:
        """
        Generate code for integrating the map visualizer into the existing GUI.
        
        Returns:
            Python code for GUI integration
        """
        return '''
def integrate_map_visualizer_with_gui(self):
    """Integrate map visualizer with the existing GUI."""
    if not hasattr(self, 'map_simulator') or not self.map_simulator:
        return
    
    # Create map visualizer
    from src.cybernetic_planning.core.map_visualizer import MapVisualizer, MapVisualizationConfig
    
    config = MapVisualizationConfig(
        tile_size=6,
        show_terrain=True,
        show_settlements=True,
        show_infrastructure=True,
        show_disasters=True,
        color_scheme="default"
    )
    
    visualizer = MapVisualizer(self.map_simulator, config)
    
    # Generate interactive map
    map_file = visualizer.generate_interactive_map("outputs/interactive_map.html")
    
    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(map_file)}")
    
    # Also generate static image for GUI display
    image_file = visualizer.generate_static_image("outputs/map_preview.png", 600, 400)
    
    return map_file, image_file
        '''
