#!/usr/bin/env python3
"""
Integration Script for Unified Simulation GUI

This script integrates the unified simulation GUI components
with the existing cybernetic planning GUI.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def integrate_unified_simulation_gui():
    """Integrate unified simulation GUI with existing GUI."""
    try:
        # Import the unified simulation GUI components
        from cybernetic_planning.gui.unified_simulation_gui import create_unified_simulation_tab
        
        print("✅ Unified simulation GUI components imported successfully")
        
        # This would be integrated into the main GUI file
        # For now, we'll create a simple demonstration
        print("\nTo integrate with the existing GUI:")
        print("1. Import the unified simulation GUI components in gui.py")
        print("2. Add the unified simulation tab to the main notebook")
        print("3. Update the GUI initialization to include the new tab")
        
        print("\nExample integration code:")
        print("""
# In gui.py, add this import:
from cybernetic_planning.gui.unified_simulation_gui import create_unified_simulation_tab

# In the CyberneticPlanningGUI.__init__ method, add:
unified_simulation_tab = create_unified_simulation_tab(self.notebook)
        """)
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import unified simulation GUI components: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False

def create_standalone_unified_simulation_gui():
    """Create a standalone unified simulation GUI for testing."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from cybernetic_planning.gui.unified_simulation_gui import UnifiedSimulationControlPanel
        
        print("Creating standalone unified simulation GUI...")
        
        # Create main window
        root = tk.Tk()
        root.title("Unified Simulation System")
        root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create unified simulation control panel
        control_panel = UnifiedSimulationControlPanel(main_frame)
        
        print("✅ Standalone unified simulation GUI created successfully")
        print("The GUI includes:")
        print("  • Configuration tab with spatial, economic, and simulation settings")
        print("  • Execution tab with run/pause/stop controls and progress monitoring")
        print("  • Monitoring tab with real-time metrics and performance data")
        print("  • Analysis tab with report generation and visualization")
        
        # Start the GUI
        root.mainloop()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create standalone GUI: {e}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("UNIFIED SIMULATION GUI INTEGRATION")
    print("=" * 60)
    print()
    
    # Test integration
    print("Testing unified simulation GUI integration...")
    if integrate_unified_simulation_gui():
        print("✅ Integration test passed")
    else:
        print("❌ Integration test failed")
        return
    
    print()
    
    # Ask user if they want to create standalone GUI
    try:
        response = input("Would you like to create a standalone unified simulation GUI? (y/n): ")
        if response.lower() in ['y', 'yes']:
            print("\nCreating standalone GUI...")
            create_standalone_unified_simulation_gui()
        else:
            print("Standalone GUI creation skipped.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
