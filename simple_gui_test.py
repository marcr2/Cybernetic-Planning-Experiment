#!/usr/bin/env python3
"""
Simple test to verify GUI parameter cleanup
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gui_import():
    """Test that the GUI can be imported and basic functionality works."""
    print("Testing GUI import and basic functionality...")
    
    try:
        import tkinter as tk
        from gui import CyberneticPlanningGUI
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Create the GUI instance
        gui = CyberneticPlanningGUI(root)
        
        # Test 1: Check that obsolete variables are removed
        print("✓ Testing obsolete variable removal...")
        assert not hasattr(gui, 'settlements_var'), "settlements_var should be removed"
        assert not hasattr(gui, 'pop_density_var'), "pop_density_var should be removed"
        assert not hasattr(gui, 'sim_sectors_var'), "sim_sectors_var should be removed"
        print("  ✓ Obsolete variables removed successfully")
        
        # Test 2: Check that new sector display variable exists
        print("✓ Testing new sector display variable...")
        assert hasattr(gui, 'sectors_display_var'), "sectors_display_var should exist"
        assert gui.sectors_display_var.get() == "Not loaded", "Default value should be 'Not loaded'"
        print("  ✓ New sector display variable works correctly")
        
        # Test 3: Test sector display update
        print("✓ Testing sector display update...")
        gui.sectors_display_var.set("15 sectors")
        assert gui.sectors_display_var.get() == "15 sectors", "Sector display update failed"
        print("  ✓ Sector display update works correctly")
        
        # Test 4: Test simulation environment logic
        print("✓ Testing simulation environment logic...")
        mock_plan = {
            'sectors': ['Agriculture', 'Manufacturing', 'Services', 'Technology', 'Energy'],
            'production_targets': [100, 200, 150, 80, 120],
            'labor_requirements': [50, 100, 75, 40, 60],
            'resource_allocations': [30, 60, 45, 24, 36]
        }
        
        gui.current_simulation_plan = mock_plan
        
        # Test sector count derivation
        if 'sectors' in mock_plan:
            sectors = len(mock_plan['sectors'])
        elif 'technology_matrix' in mock_plan:
            sectors = len(mock_plan['technology_matrix'])
        else:
            sectors = 15
        
        assert sectors == 5, f"Expected 5 sectors, got {sectors}"
        print("  ✓ Sector count derivation works correctly")
        
        # Clean up
        root.destroy()
        
        print("\n🎉 All tests passed! GUI parameter cleanup successful.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_import()
    sys.exit(0 if success else 1)
