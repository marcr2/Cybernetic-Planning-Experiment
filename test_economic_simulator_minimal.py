#!/usr/bin/env python3
"""
Minimal test to isolate the EnhancedEconomicSimulation constructor issue.
"""

import sys
import os
import numpy as np

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cybernetic_planning.core.enhanced_simulation import EnhancedEconomicSimulation

def test_minimal_constructor():
    """Test the constructor with minimal data."""
    print("Creating minimal test data...")
    
    # Create simple test data
    n_sectors = 5
    technology_matrix = np.random.rand(n_sectors, n_sectors) * 0.1
    labor_vector = np.random.rand(n_sectors)
    final_demand = np.random.rand(n_sectors) * 100
    
    print(f"Technology matrix shape: {technology_matrix.shape}")
    print(f"Labor vector shape: {labor_vector.shape}")
    print(f"Final demand shape: {final_demand.shape}")
    
    print("Creating EnhancedEconomicSimulation...")
    try:
        simulator = EnhancedEconomicSimulation(
            technology_matrix=technology_matrix,
            labor_vector=labor_vector,
            final_demand=final_demand,
            resource_matrix=None,
            max_resources=None,
            sector_names=['A', 'B', 'C', 'D', 'E']
        )
        print("✅ Constructor succeeded!")
        return True
    except Exception as e:
        print(f"❌ Constructor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_constructor()
    if success:
        print("Test passed!")
    else:
        print("Test failed!")
        sys.exit(1)
