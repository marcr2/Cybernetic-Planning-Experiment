#!/usr/bin/env python3
"""
Map-Based Economic Plan Simulator Demo

Demonstrates how to use the map-based economic plan simulator that integrates
with the existing cybernetic planning system.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cybernetic_planning.planning_system import CyberneticPlanningSystem

def demo_basic_map_simulation():
    """Demonstrate basic map-based simulation."""
    print("=== Basic Map-Based Simulation Demo ===")
    
    # Create planning system
    planning_system = CyberneticPlanningSystem()
    
    # Create synthetic economic data
    print("Creating synthetic economic data...")
    data = planning_system.create_synthetic_data(
        n_sectors=15,
        technology_density=0.4,
        resource_count=8
    )
    
    # Create economic plan
    print("Creating economic plan...")
    plan = planning_system.create_plan(
        policy_goals=["Increase industrial production", "Improve living standards"],
        use_optimization=True
    )
    
    print(f"Economic plan created with {len(plan.get('sectors', []))} sectors")
    print(f"Total economic output: {plan.get('total_economic_output', 0):.2f}")
    
    # Create map-based simulation
    print("\nCreating map-based simulation...")
    map_result = planning_system.create_map_based_simulation(
        map_width=80,
        map_height=80,
        terrain_distribution={
            "flatland": 0.4,
            "forest": 0.3,
            "mountain": 0.2,
            "water": 0.1
        },
        num_cities=3,
        num_towns=8,
        total_population=500000,
        rural_population_percent=0.25,
        urban_concentration="medium"
    )
    
    if map_result["success"]:
        print("✓ Map-based simulation created successfully")
        summary = map_result["map_summary"]
        print(f"  - Map size: {summary['map_dimensions']}")
        print(f"  - Cities: {summary['settlements']['cities']}")
        print(f"  - Towns: {summary['settlements']['towns']}")
        print(f"  - Total population: {summary['population']['total']}")
        print(f"  - Infrastructure segments: {summary['infrastructure']['total_segments']}")
        print(f"  - Terrain distribution: {summary['terrain_distribution']}")
    else:
        print(f"✗ Failed to create map simulation: {map_result['error']}")
        return
    
    # Integrate map with economic plan
    print("\nIntegrating map with economic plan...")
    integration_result = planning_system.integrate_map_with_plan()
    
    if integration_result["success"]:
        print("✓ Map integrated with economic plan")
        print(f"  - Plan sectors mapped to settlements")
        print(f"  - Logistics costs updated based on economic activity")
    else:
        print(f"✗ Integration failed: {integration_result['error']}")
        return
    
    # Run simulation
    print("\nRunning map-based simulation...")
    simulation_result = planning_system.run_map_simulation(time_steps=12)
    
    if simulation_result["success"]:
        print("✓ Simulation completed")
        summary = simulation_result["summary"]
        print(f"  - Time steps completed: {simulation_result['time_steps_completed']}")
        print(f"  - Total logistics friction: {summary['total_logistics_friction']:.2f}")
        print(f"  - Average logistics friction: {summary['average_logistics_friction']:.2f}")
        print(f"  - Total disasters: {summary['total_disasters']}")
        
        # Show some simulation details
        print("\nSimulation details:")
        for i, result in enumerate(simulation_result["simulation_results"][:5]):
            print(f"  Step {result['time_step']}: Logistics friction = {result['total_logistics_friction']:.2f}")
            if result['disaster_events']['new_disasters']:
                print(f"    New disasters: {len(result['disaster_events']['new_disasters'])}")
    else:
        print(f"✗ Simulation failed: {simulation_result['error']}")
        return
    
    # Get final status
    status = planning_system.get_map_simulation_status()
    print(f"\nFinal simulation status:")
    print(f"  - Current time step: {status['current_time_step']}")
    print(f"  - Active disasters: {status['active_disasters']}")
    print(f"  - Total settlements: {status['map_summary']['settlements']['total']}")

def demo_different_urban_concentrations():
    """Demonstrate different urban concentration patterns."""
    print("\n=== Urban Concentration Patterns Demo ===")
    
    planning_system = CyberneticPlanningSystem()
    planning_system.create_synthetic_data(n_sectors=8)
    planning_system.create_plan()
    
    concentrations = ["high", "medium", "low"]
    
    for concentration in concentrations:
        print(f"\nTesting {concentration} urban concentration...")
        
        map_result = planning_system.create_map_based_simulation(
            map_width=60,
            map_height=60,
            num_cities=2,
            num_towns=6,
            total_population=300000,
            urban_concentration=concentration
        )
        
        if map_result["success"]:
            summary = map_result["map_summary"]
            settlements = summary["settlements"]
            print(f"  - Cities: {settlements['cities']}")
            print(f"  - Towns: {settlements['towns']}")
            print(f"  - Infrastructure segments: {summary['infrastructure']['total_segments']}")
            print(f"  - Average logistics cost: {summary['logistics']['average_cost']:.2f}")

def demo_disaster_simulation():
    """Demonstrate disaster simulation."""
    print("\n=== Disaster Simulation Demo ===")
    
    planning_system = CyberneticPlanningSystem()
    planning_system.create_synthetic_data(n_sectors=6)
    planning_system.create_plan()
    
    # Create map with higher disaster probability
    map_result = planning_system.create_map_based_simulation(
        map_width=50,
        map_height=50,
        num_cities=2,
        num_towns=5,
        total_population=200000
    )
    
    if not map_result["success"]:
        print(f"✗ Failed to create map: {map_result['error']}")
        return
    
    # Increase disaster probability for demonstration
    planning_system.map_simulator.disaster_probability = 0.3  # 30% chance per step
    
    print("Running simulation with high disaster probability...")
    simulation_result = planning_system.run_map_simulation(time_steps=10)
    
    if simulation_result["success"]:
        total_disasters = simulation_result["summary"]["total_disasters"]
        print(f"✓ Simulation completed with {total_disasters} disasters")
        
        # Show disaster events
        for i, result in enumerate(simulation_result["simulation_results"]):
            disasters = result["disaster_events"]
            if disasters["new_disasters"] or disasters["ongoing_disasters"]:
                print(f"  Step {result['time_step']}:")
                if disasters["new_disasters"]:
                    print(f"    New disasters: {len(disasters['new_disasters'])}")
                if disasters["ongoing_disasters"]:
                    print(f"    Ongoing disasters: {len(disasters['ongoing_disasters'])}")

def demo_export_import():
    """Demonstrate export and import functionality."""
    print("\n=== Export/Import Demo ===")
    
    planning_system = CyberneticPlanningSystem()
    planning_system.create_synthetic_data(n_sectors=5)
    planning_system.create_plan()
    
    # Create and run simulation
    map_result = planning_system.create_map_based_simulation(
        map_width=60,
        map_height=60,
        num_cities=2,
        num_towns=5,
        total_population=200000
    )
    
    if not map_result["success"]:
        print(f"✗ Failed to create map: {map_result['error']}")
        return
    
    # Run some simulation steps
    simulation_result = planning_system.run_map_simulation(time_steps=5)
    
    # Export simulation
    export_path = "demo_map_simulation.json"
    export_result = planning_system.export_map_simulation(export_path)
    
    if export_result["success"]:
        print(f"✓ Simulation exported to {export_path}")
        
        # Create new planning system and import
        new_planning_system = CyberneticPlanningSystem()
        import_result = new_planning_system.load_map_simulation(export_path)
        
        if import_result["success"]:
            print("✓ Simulation imported successfully")
            status = new_planning_system.get_map_simulation_status()
            print(f"  - Current time step: {status['current_time_step']}")
            print(f"  - Settlements: {status['map_summary']['settlements']['total']}")
            
            # Clean up
            Path(export_path).unlink()
            print("✓ Demo file cleaned up")
        else:
            print(f"✗ Import failed: {import_result['error']}")
    else:
        print(f"✗ Export failed: {export_result['error']}")

def main():
    """Run all demonstrations."""
    print("Map-Based Economic Plan Simulator Demonstration")
    print("=" * 60)
    
    try:
        demo_basic_map_simulation()
        demo_different_urban_concentrations()
        demo_disaster_simulation()
        demo_export_import()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\nThe map-based economic plan simulator is now integrated")
        print("with the existing cybernetic planning system. Key features:")
        print("- Procedural map generation with terrain, settlements, and infrastructure")
        print("- Integration with economic plans from the planning system")
        print("- Dynamic simulation with logistics costs and natural disasters")
        print("- Export/import functionality for saving and loading simulations")
        print("- Configurable urban concentration patterns")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
