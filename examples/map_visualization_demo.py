#!/usr/bin/env python3
"""
Map Visualization System Demo

Demonstrates how to create interactive map visualizations for the
map-based economic plan simulator.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cybernetic_planning.planning_system import CyberneticPlanningSystem

def demo_interactive_map():
    """Demonstrate interactive map generation."""
    print("=== Interactive Map Visualization Demo ===")
    
    # Create planning system and map simulation
    planning_system = CyberneticPlanningSystem()
    
    # Create synthetic data and plan
    print("Creating economic data and plan...")
    planning_system.create_synthetic_data(n_sectors=8, technology_density=0.3)
    planning_system.create_plan()
    
    # Create map-based simulation
    print("Creating map-based simulation...")
    map_result = planning_system.create_map_based_simulation(
        map_width=60,
        map_height=60,
        num_cities=3,
        num_towns=8,
        total_population=400000,
        urban_concentration="medium"
    )
    
    if not map_result["success"]:
        print(f"✗ Failed to create map: {map_result['error']}")
        return
    
    print("✓ Map simulation created successfully")
    
    # Run some simulation steps
    print("Running simulation...")
    simulation_result = planning_system.run_map_simulation(time_steps=8)
    
    if simulation_result["success"]:
        print("✓ Simulation completed")
        print(f"  - Time steps: {simulation_result['time_steps_completed']}")
        print(f"  - Total disasters: {simulation_result['summary']['total_disasters']}")
    
    # Generate interactive map visualization
    print("\\nGenerating interactive map visualization...")
    viz_result = planning_system.generate_map_visualization(
        output_path="examples/interactive_map_demo",
        visualization_type="interactive",
        config={
            "tile_size": 10,
            "color_scheme": "default",
            "show_terrain": True,
            "show_settlements": True,
            "show_infrastructure": True,
            "show_disasters": True
        }
    )
    
    if viz_result["success"]:
        print("✓ Interactive map generated successfully")
        print(f"  - File: {viz_result['files']['interactive']}")
        
        # Open in browser
        browser_result = planning_system.open_map_in_browser()
        if browser_result["success"]:
            print("✓ Map opened in browser")
        else:
            print(f"✗ Failed to open in browser: {browser_result['error']}")
    else:
        print(f"✗ Failed to generate visualization: {viz_result['error']}")

def demo_static_image():
    """Demonstrate static image generation."""
    print("\\n=== Static Image Generation Demo ===")
    
    planning_system = CyberneticPlanningSystem()
    planning_system.create_synthetic_data(n_sectors=6)
    planning_system.create_plan()
    
    # Create map simulation
    map_result = planning_system.create_map_based_simulation(
        map_width=40,
        map_height=40,
        num_cities=2,
        num_towns=6,
        total_population=250000,
        urban_concentration="high"
    )
    
    if not map_result["success"]:
        print(f"✗ Failed to create map: {map_result['error']}")
        return
    
    # Run simulation with disasters
    planning_system.map_simulator.disaster_probability = 0.2  # 20% chance
    simulation_result = planning_system.run_map_simulation(time_steps=6)
    
    # Generate static image
    print("Generating static map image...")
    viz_result = planning_system.generate_map_visualization(
        output_path="examples/static_map_demo",
        visualization_type="static",
        config={
            "color_scheme": "terrain"
        }
    )
    
    if viz_result["success"]:
        print("✓ Static image generated successfully")
        print(f"  - File: {viz_result['files']['static']}")
    else:
        print(f"✗ Failed to generate static image: {viz_result['error']}")

def demo_both_visualizations():
    """Demonstrate both interactive and static visualizations."""
    print("\\n=== Combined Visualization Demo ===")
    
    planning_system = CyberneticPlanningSystem()
    planning_system.create_synthetic_data(n_sectors=10)
    planning_system.create_plan()
    
    # Create larger map simulation
    map_result = planning_system.create_map_based_simulation(
        map_width=80,
        map_height=80,
        num_cities=4,
        num_towns=12,
        total_population=600000,
        urban_concentration="low"
    )
    
    if not map_result["success"]:
        print(f"✗ Failed to create map: {map_result['error']}")
        return
    
    # Run longer simulation
    simulation_result = planning_system.run_map_simulation(time_steps=12)
    
    # Generate both types of visualizations
    print("Generating both interactive and static visualizations...")
    viz_result = planning_system.generate_map_visualization(
        output_path="examples/combined_map_demo",
        visualization_type="both",
        config={
            "tile_size": 8,
            "color_scheme": "default",
            "show_terrain": True,
            "show_settlements": True,
            "show_infrastructure": True,
            "show_disasters": True,
            "show_logistics": True
        }
    )
    
    if viz_result["success"]:
        print("✓ Both visualizations generated successfully")
        if "interactive" in viz_result["files"]:
            print(f"  - Interactive map: {viz_result['files']['interactive']}")
        if "static" in viz_result["files"]:
            print(f"  - Static image: {viz_result['files']['static']}")
    else:
        print(f"✗ Failed to generate visualizations: {viz_result['error']}")

def demo_custom_configuration():
    """Demonstrate custom visualization configuration."""
    print("\\n=== Custom Configuration Demo ===")
    
    planning_system = CyberneticPlanningSystem()
    planning_system.create_synthetic_data(n_sectors=7)
    planning_system.create_plan()
    
    # Create map simulation
    map_result = planning_system.create_map_based_simulation(
        map_width=50,
        map_height=50,
        num_cities=3,
        num_towns=7,
        total_population=300000
    )
    
    if not map_result["success"]:
        print(f"✗ Failed to create map: {map_result['error']}")
        return
    
    # Run simulation
    simulation_result = planning_system.run_map_simulation(time_steps=5)
    
    # Generate with custom configuration
    print("Generating map with custom configuration...")
    viz_result = planning_system.generate_map_visualization(
        output_path="examples/custom_map_demo",
        visualization_type="interactive",
        config={
            "tile_size": 12,
            "color_scheme": "terrain",
            "show_terrain": True,
            "show_settlements": True,
            "show_infrastructure": False,  # Hide infrastructure
            "show_disasters": True,
            "show_logistics": False,       # Hide logistics
            "animation_speed": 2.0        # Faster animation
        }
    )
    
    if viz_result["success"]:
        print("✓ Custom visualization generated successfully")
        print(f"  - File: {viz_result['files']['interactive']}")
        print("  - Configuration: Terrain colors, no infrastructure, faster animation")
    else:
        print(f"✗ Failed to generate custom visualization: {viz_result['error']}")

def main():
    """Run all visualization demonstrations."""
    print("Map Visualization System Demonstration")
    print("=" * 60)
    
    try:
        # Create output directory
        Path("examples").mkdir(exist_ok=True)
        
        demo_interactive_map()
        demo_static_image()
        demo_both_visualizations()
        demo_custom_configuration()
        
        print("\\n" + "=" * 60)
        print("🎉 All visualization demonstrations completed!")
        print("\\nThe map visualization system provides:")
        print("- Interactive HTML maps viewable in any web browser")
        print("- Static PNG images for reports and presentations")
        print("- Customizable color schemes and display options")
        print("- Real-time simulation data integration")
        print("- Export functionality for sharing and analysis")
        print("\\nGenerated files are saved in the 'examples' directory.")
        
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
