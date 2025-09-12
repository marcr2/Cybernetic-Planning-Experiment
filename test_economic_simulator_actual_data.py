#!/usr/bin/env python3
"""
Test with actual data from the unified simulation to isolate the issue.
"""

import sys
import os
import numpy as np

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cybernetic_planning.core.enhanced_simulation import EnhancedEconomicSimulation
from cybernetic_planning.core.unified_simulation_system import UnifiedSimulationSystem
from cybernetic_planning.core.unified_simulation_loop import UnifiedSimulationConfig

def test_with_actual_data():
    """Test with actual data from unified simulation."""
    print("Creating unified simulation system...")
    
    config = UnifiedSimulationConfig()
    system = UnifiedSimulationSystem(config)
    
    print("Creating unified simulation...")
    result = system.create_unified_simulation()
    
    if not result:
        print("❌ Failed to create unified simulation")
        return False
    
    print("✅ Unified simulation created")
    
    # Get the data that would be passed to the constructor
    # The synthetic data is created during create_unified_simulation
    synthetic_data = system.planning_system.current_data
    plan_result = system.planning_system.current_plan
    
    if synthetic_data is None:
        print("❌ No synthetic data available")
        return False
    
    if plan_result is None:
        print("❌ No plan result available")
        return False
    
    print("Extracting data for constructor...")
    technology_matrix = synthetic_data.get('technology_matrix')
    labor_vector = synthetic_data.get('labor_input') or synthetic_data.get('labor_vector')
    final_demand = synthetic_data.get('final_demand')
    resource_matrix = synthetic_data.get('resource_matrix')
    max_resources = synthetic_data.get('max_resources')
    sector_names = synthetic_data.get('sectors')
    
    print(f"Technology matrix type: {type(technology_matrix)}, shape: {getattr(technology_matrix, 'shape', 'N/A')}")
    print(f"Labor vector type: {type(labor_vector)}, shape: {getattr(labor_vector, 'shape', 'N/A')}")
    print(f"Final demand type: {type(final_demand)}, shape: {getattr(final_demand, 'shape', 'N/A')}")
    print(f"Sector names type: {type(sector_names)}, length: {len(sector_names) if sector_names else 'N/A'}")
    
    print("Testing constructor with actual data...")
    try:
        simulator = EnhancedEconomicSimulation(
            technology_matrix=technology_matrix,
            labor_vector=labor_vector,
            final_demand=final_demand,
            resource_matrix=resource_matrix,
            max_resources=max_resources,
            sector_names=sector_names
        )
        print("✅ Constructor succeeded with actual data!")
        return True
    except Exception as e:
        print(f"❌ Constructor failed with actual data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_actual_data()
    if success:
        print("Test passed!")
    else:
        print("Test failed!")
        sys.exit(1)
