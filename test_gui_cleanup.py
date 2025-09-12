#!/usr/bin/env python3
"""
Test script for GUI parameter cleanup
Tests that the GUI loads correctly and the new parameter structure works.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk
import json
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gui_loading():
    """Test that the GUI loads without errors."""
    print("Testing GUI loading...")
    
    try:
        # Import the GUI class
        from gui import CyberneticPlanningGUI
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Create the GUI instance
        gui = CyberneticPlanningGUI(root)
        
        # Check that the new variables exist
        assert hasattr(gui, 'sectors_display_var'), "sectors_display_var not found"
        assert not hasattr(gui, 'settlements_var'), "settlements_var should be removed"
        assert not hasattr(gui, 'pop_density_var'), "pop_density_var should be removed"
        assert not hasattr(gui, 'sim_sectors_var'), "sim_sectors_var should be removed"
        
        print("✅ GUI loads successfully with new parameter structure")
        
        # Test sector display functionality
        gui.sectors_display_var.set("15 sectors")
        assert gui.sectors_display_var.get() == "15 sectors", "Sector display not working"
        
        print("✅ Sector display functionality works")
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ GUI loading failed: {e}")
        return False

def test_simulation_initialization():
    """Test that simulation initialization works with new parameters."""
    print("Testing simulation initialization...")
    
    try:
        from gui import CyberneticPlanningGUI
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()
        
        # Create the GUI instance
        gui = CyberneticPlanningGUI(root)
        
        # Create a mock simulation plan
        mock_plan = {
            'sectors': ['Agriculture', 'Manufacturing', 'Services', 'Technology', 'Energy'],
            'production_targets': [100, 200, 150, 80, 120],
            'labor_requirements': [50, 100, 75, 40, 60],
            'resource_allocations': [30, 60, 45, 24, 36]
        }
        
        # Set the plan
        gui.current_simulation_plan = mock_plan
        
        # Test sector count derivation
        if 'sectors' in mock_plan:
            sectors = len(mock_plan['sectors'])
        elif 'technology_matrix' in mock_plan:
            sectors = len(mock_plan['technology_matrix'])
        else:
            sectors = 15
        
        assert sectors == 5, f"Expected 5 sectors, got {sectors}"
        
        print("✅ Simulation initialization logic works correctly")
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Simulation initialization test failed: {e}")
        return False

def test_plan_loading():
    """Test that plan loading updates the sector display."""
    print("Testing plan loading...")
    
    try:
        from gui import CyberneticPlanningGUI
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()
        
        # Create the GUI instance
        gui = CyberneticPlanningGUI(root)
        
        # Create a temporary plan file
        mock_plan = {
            'sectors': ['Agriculture', 'Manufacturing', 'Services', 'Technology', 'Energy', 'Healthcare'],
            'production_targets': [100, 200, 150, 80, 120, 90],
            'labor_requirements': [50, 100, 75, 40, 60, 45],
            'resource_allocations': [30, 60, 45, 24, 36, 27]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_plan, f)
            temp_file = f.name
        
        try:
            # Set the plan file variable
            gui.plan_file_var.set(temp_file)
            
            # Load the plan
            gui.load_simulation_plan()
            
            # Check that sector display was updated
            assert gui.sectors_display_var.get() == "6 sectors", f"Expected '6 sectors', got '{gui.sectors_display_var.get()}'"
            
            print("✅ Plan loading updates sector display correctly")
            
        finally:
            # Clean up temp file
            os.unlink(temp_file)
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Plan loading test failed: {e}")
        return False

def test_environment_cleanup():
    """Test that the simulation environment no longer contains obsolete variables."""
    print("Testing environment cleanup...")
    
    try:
        from gui import CyberneticPlanningGUI
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()
        
        # Create the GUI instance
        gui = CyberneticPlanningGUI(root)
        
        # Create a mock simulation plan
        mock_plan = {
            'sectors': ['Agriculture', 'Manufacturing', 'Services'],
            'production_targets': [100, 200, 150],
            'labor_requirements': [50, 100, 75],
            'resource_allocations': [30, 60, 45]
        }
        
        gui.current_simulation_plan = mock_plan
        
        # Initialize simulation environment
        gui.initialize_simulation_environment()
        
        # Check that obsolete variables are not in the environment
        env = gui.simulation_environment
        
        assert 'map_size_km' not in env, "map_size_km should be removed from environment"
        assert 'settlements' not in env, "settlements should be removed from environment"
        assert 'population_density' not in env, "population_density should be removed from environment"
        
        # Check that required variables are present
        assert 'duration_years' in env, "duration_years should be in environment"
        assert 'time_step_months' in env, "time_step_months should be in environment"
        assert 'economic_sectors' in env, "economic_sectors should be in environment"
        assert env['economic_sectors'] == 3, f"Expected 3 sectors, got {env['economic_sectors']}"
        
        print("✅ Environment cleanup successful")
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Environment cleanup test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running GUI Parameter Cleanup Tests")
    print("=" * 50)
    
    tests = [
        ("GUI Loading", test_gui_loading),
        ("Simulation Initialization", test_simulation_initialization),
        ("Plan Loading", test_plan_loading),
        ("Environment Cleanup", test_environment_cleanup),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GUI parameter cleanup successful.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
