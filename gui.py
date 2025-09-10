#!/usr / bin / env python3
"""
GUI for Cybernetic Central Planning System

A user - friendly graphical interface for the cybernetic planning system,
allowing users to create economic plans, manage data, and generate reports.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
import os
import json
from pathlib import Path
import threading
import webbrowser
import numpy as np

# Import map visualization
try:
    from src.cybernetic_planning.utils.map_visualization import InteractiveMap, MapGenerator, create_simulation_map
    MAP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Map visualization not available: {e}")
    MAP_AVAILABLE = False
from datetime import datetime
import math
import random

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from cybernetic_planning.planning_system import CyberneticPlanningSystem
except ImportError as e:
    print(f"Error importing planning system: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class WebBrowserWidget:
    """An enhanced web browser widget with HTML preview and external browser integration."""

    def __init__(self, parent, width = 800, height = 600):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_url = None

        # Create the widget frame
        self.frame = ttk.Frame(parent)

        # Create a notebook for different views
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill="both", expand = True)

        # HTML Preview tab
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="Map Preview")

        # HTML Source tab
        self.source_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.source_frame, text="HTML Source")

        # Create HTML preview area (simplified HTML renderer)
        self.preview_text = scrolledtext.ScrolledText(
            self.preview_frame,
            width = width//8,
            height = height//20,
            wrap = tk.WORD,
            font=('Arial', 10),
            bg='white',
            fg='black'
        )
        self.preview_text.pack(fill="both", expand = True, padx = 5, pady = 5)

        # Create HTML source area
        self.source_text = scrolledtext.ScrolledText(
            self.source_frame,
            width = width//8,
            height = height//20,
            wrap = tk.WORD,
            font=('Courier', 8),
            bg='#f0f0f0',
            fg='black'
        )
        self.source_text.pack(fill="both", expand = True, padx = 5, pady = 5)

        # Control buttons frame
        self.controls_frame = ttk.Frame(self.frame)
        self.controls_frame.pack(fill="x", padx = 5, pady = 5)

        # Add buttons
        self.open_button = ttk.Button(
            self.controls_frame,
            text="Open in External Browser",
            command = self.open_in_browser
        )
        self.open_button.pack(side="left", padx = 5)

        self.refresh_button = ttk.Button(
            self.controls_frame,
            text="Refresh Map",
            command = self.refresh
        )
        self.refresh_button.pack(side="left", padx = 5)

        # Add status label
        self.status_label = ttk.Label(self.controls_frame, text="No map loaded")
        self.status_label.pack(side="right", padx = 5)

    def load_html_file(self, file_path):
        """Load an HTML file into the widget."""
        try:
            self.current_url = file_path
            with open(file_path, 'r', encoding='utf - 8') as f:
                html_content = f.read()

            # Display HTML source
            self.source_text.delete("1.0", tk.END)
            self.source_text.insert("1.0", html_content)

            # Create a simplified preview
            self.create_html_preview(html_content)

            self.status_label.config(text = f"Map loaded: {os.path.basename(file_path)}")
            return True

        except Exception as e:
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert("1.0", f"Error loading map: {str(e)}")
            self.source_text.delete("1.0", tk.END)
            self.source_text.insert("1.0", f"Error loading map: {str(e)}")
            self.status_label.config(text="Error loading map")
            return False

    def create_html_preview(self, html_content):
        """Create a simplified text preview of the HTML map."""
        self.preview_text.delete("1.0", tk.END)

        # Extract key information from HTML
        preview = "🗺️  INTERACTIVE SIMULATION MAP\n"
        preview += "=" * 50 + "\n\n"

        # Count map elements from HTML
        settlements = html_content.count('CircleMarker')
        polygons = html_content.count('Polygon')
        polylines = html_content.count('PolyLine')

        preview += f"📍 Settlements: {settlements}\n"
        preview += f"🏞️  Geographic Features: {polygons}\n"
        preview += f"🛣️  Infrastructure: {polylines}\n\n"

        # Extract settlement details
        if 'City' in html_content:
            cities = html_content.count('City')
            preview += f"🏙️  Cities: {cities}\n"

        if 'Town' in html_content:
            towns = html_content.count('Town')
            preview += f"🏘️  Towns: {towns}\n"

        if 'Rural' in html_content:
            rural = html_content.count('Rural')
            preview += f"🌾 Rural Areas: {rural}\n"

        preview += "\n" + "=" * 50 + "\n"
        preview += "INTERACTIVE MAP FEATURES:\n"
        preview += "=" * 50 + "\n\n"
        preview += "🔴 Red circles = Cities\n"
        preview += "🟠 Orange circles = Towns\n"
        preview += "🟡 Yellow circles = Rural Areas\n"
        preview += "🔵 Blue areas = Water bodies\n"
        preview += "🟤 Brown areas = Mountains\n"
        preview += "🟢 Green areas = Forests\n"
        preview += "⚫ Black lines = Roads\n"
        preview += "⚫ Dashed lines = Railways\n\n"

        preview += "=" * 50 + "\n"
        preview += "HOW TO USE:\n"
        preview += "=" * 50 + "\n\n"
        preview += "1. Click 'Open in External Browser' to view the full interactive map\n"
        preview += "2. In the browser, you can:\n"
        preview += "   • Zoom in / out with mouse wheel\n"
        preview += "   • Pan by dragging\n"
        preview += "   • Click on elements for details\n"
        preview += "   • See real - time updates during simulation\n\n"
        preview += "3. The map updates automatically when simulation runs\n"
        preview += "4. Use the 'Refresh Map' button to update the display\n\n"

        preview += "=" * 50 + "\n"
        preview += "SIMULATION STATUS:\n"
        preview += "=" * 50 + "\n"
        preview += "• Map is ready for real - time simulation\n"
        preview += "• Start simulation to see dynamic changes\n"
        preview += "• Adjust speed with the dropdown above\n"

        self.preview_text.insert("1.0", preview)

    def open_in_browser(self):
        """Open the current map in the default web browser."""
        if self.current_url and os.path.exists(self.current_url):
            try:
                webbrowser.open(f'file://{os.path.abspath(self.current_url)}')
                self.status_label.config(text="Map opened in browser")
            except Exception as e:
                self.status_label.config(text = f"Error: {str(e)}")
        else:
            self.status_label.config(text="No map to open")

    def refresh(self):
        """Refresh the current map display."""
        if self.current_url:
            self.load_html_file(self.current_url)

    def pack(self, **kwargs):
        """Pack the widget frame."""
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        """Grid the widget frame."""
        self.frame.grid(**kwargs)

class CyberneticPlanningGUI:
    """Main GUI class for the Cybernetic Planning System."""

    def __init__(self, root):
        self.root = root
        self.root.title("Cybernetic Central Planning System")

        # Detect screen dimensions and calculate scaling
        self.screen_width, self.screen_height = self._get_screen_dimensions()
        self.scale_factor = self._calculate_scale_factor()
        self.window_width, self.window_height = self._calculate_window_size()

        # Set window geometry with calculated dimensions
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.minsize(int(1000 * self.scale_factor), int(700 * self.scale_factor))

        # Center the window on screen
        self._center_window()

        # Initialize planning system
        self.planning_system = CyberneticPlanningSystem()
        self.current_plan = None
        self.current_data = None

        # Create GUI elements
        self.create_widgets()
        self.setup_layout()

        # Bind cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _get_screen_dimensions(self):
        """Get the screen dimensions."""
        try:
            # Get screen dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            return screen_width, screen_height
        except Exception as e:
            print(f"Warning: Could not detect screen dimensions: {e}")
            # Fallback to common 1920x1080 resolution
            return 1920, 1080

    def _calculate_scale_factor(self):
        """Calculate appropriate scale factor based on screen size."""
        # Base resolution is 1920x1080 (16:9 aspect ratio)
        base_width, base_height = 1920, 1080

        # Calculate scale factors for width and height
        width_scale = self.screen_width / base_width
        height_scale = self.screen_height / base_height

        # Use the smaller scale factor to ensure the window fits on screen
        # but add some padding (0.8 factor) to leave room for taskbar, etc.
        scale_factor = min(width_scale, height_scale) * 0.8

        # Ensure minimum and maximum scale factors
        scale_factor = max(0.6, min(scale_factor, 2.0))

        return scale_factor

    def _calculate_window_size(self):
        """Calculate appropriate window size based on scale factor."""
        # Base window size (original design size)
        base_width, base_height = 1400, 900

        # Scale the dimensions
        window_width = int(base_width * self.scale_factor)
        window_height = int(base_height * self.scale_factor)

        # Ensure window doesn't exceed screen dimensions
        window_width = min(window_width, int(self.screen_width * 0.95))
        window_height = min(window_height, int(self.screen_height * 0.95))

        return window_width, window_height

    def _center_window(self):
        """Center the window on the screen."""
        # Calculate position to center the window
        x = (self.screen_width - self.window_width) // 2
        y = (self.screen_height - self.window_height) // 2

        # Set window position
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

    def _scale_padding(self, padding):
        """Scale padding values based on the scale factor."""
        if isinstance(padding, (int, float)):
            return int(padding * self.scale_factor)
        elif isinstance(padding, (list, tuple)) and len(padding) >= 2:
            return (int(padding[0] * self.scale_factor), int(padding[1] * self.scale_factor))
        return padding

    def _scale_font_size(self, base_size):
        """Scale font size based on the scale factor."""
        return int(base_size * self.scale_factor)

    def get_scaling_info(self):
        """Get information about current scaling settings."""
        return {
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "scale_factor": self.scale_factor,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "aspect_ratio": self.screen_width / self.screen_height
        }

    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)

        # Create tabs
        self.create_data_tab()
        self.create_planning_tab()
        self.create_results_tab()
        self.create_simulation_tab()
        self.create_export_tab()
        self.create_about_tab()

    def create_data_tab(self):
        """Create data management tab."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Management")

        # Data source selection
        source_frame = ttk.LabelFrame(self.data_frame, text="Data Source", padding = self._scale_padding(10))
        source_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        ttk.Button(source_frame, text="Load from File", command = self.load_data_from_file).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(source_frame, text="Process USA Zip File", command = self.process_usa_zip).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(source_frame, text="Generate Synthetic Data", command = self.generate_synthetic_data).pack(
            side="left", padx = self._scale_padding(5)
        )

        # Data configuration
        config_frame = ttk.LabelFrame(self.data_frame, text="Synthetic Data Configuration", padding = self._scale_padding(10))
        config_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Number of sectors
        ttk.Label(config_frame, text="Number of Sectors:").grid(row = 0, column = 0, sticky="w", padx = self._scale_padding(5))
        self.sectors_var = tk.StringVar(value="8")
        ttk.Entry(config_frame, textvariable = self.sectors_var, width = 10).grid(row = 0, column = 1, padx = self._scale_padding(5))

        # Technology density
        ttk.Label(config_frame, text="Technology Density:").grid(row = 0, column = 2, sticky="w", padx = self._scale_padding(5))
        self.density_var = tk.StringVar(value="0.4")
        ttk.Entry(config_frame, textvariable = self.density_var, width = 10).grid(row = 0, column = 3, padx = self._scale_padding(5))

        # Resource count
        ttk.Label(config_frame, text="Resource Count:").grid(row = 1, column = 0, sticky="w", padx = self._scale_padding(5))
        self.resources_var = tk.StringVar(value="3")
        ttk.Entry(config_frame, textvariable = self.resources_var, width = 10).grid(row = 1, column = 1, padx = self._scale_padding(5))

        # Data display
        display_frame = ttk.LabelFrame(self.data_frame, text="Current Data", padding = self._scale_padding(10))
        display_frame.pack(fill="both", expand = True, padx = self._scale_padding(10), pady = self._scale_padding(5))

        self.data_text = scrolledtext.ScrolledText(display_frame, height = 15, width = 80)
        self.data_text.pack(fill="both", expand = True)

        # Data status
        self.data_status = ttk.Label(display_frame, text="No data loaded", foreground="red")
        self.data_status.pack(pady = self._scale_padding(5))

    def create_planning_tab(self):
        """Create planning configuration tab."""
        self.planning_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.planning_frame, text="Planning Configuration")

        # Create a scrollable frame
        self.planning_canvas = tk.Canvas(self.planning_frame)
        self.planning_scrollbar = ttk.Scrollbar(self.planning_frame, orient="vertical", command = self.planning_canvas.yview)
        self.planning_scrollable_frame = ttk.Frame(self.planning_canvas)

        self.planning_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.planning_canvas.configure(scrollregion = self.planning_canvas.bbox("all"))
        )

        self.planning_canvas.create_window((0, 0), window = self.planning_scrollable_frame, anchor="nw")
        self.planning_canvas.configure(yscrollcommand = self.planning_scrollbar.set)

        # Pack the scrollable components
        self.planning_canvas.pack(side="left", fill="both", expand = True)
        self.planning_scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            self.planning_canvas.yview_scroll(int(-1*(event.delta / 120)), "units")
        self.planning_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Store the mousewheel binding for cleanup
        self.mousewheel_binding = _on_mousewheel

        # Policy goals
        goals_frame = ttk.LabelFrame(self.planning_scrollable_frame, text="Policy Goals", padding = 10)
        goals_frame.pack(fill="x", padx = 10, pady = 5)

        ttk.Label(goals_frame, text="Enter policy goals (one per line):").pack(anchor="w")
        self.goals_text = scrolledtext.ScrolledText(goals_frame, height = 6, width = 80)
        self.goals_text.pack(fill="x", pady = 5)

        # Default goals
        default_goals = [
            "Increase healthcare capacity by 15%",
            "Reduce carbon emissions by 20%",
            "Improve education infrastructure",
            "Ensure food security",
        ]
        self.goals_text.insert("1.0", "\n".join(default_goals))

        # Production adjustment controls
        production_frame = ttk.LabelFrame(self.planning_scrollable_frame, text="Production Adjustment", padding = 10)
        production_frame.pack(fill="x", padx = 10, pady = 5)

        # Overall production multiplier
        overall_frame = ttk.Frame(production_frame)
        overall_frame.pack(fill="x", pady = 5)

        ttk.Label(overall_frame, text="Overall Production Level:").pack(side="left")
        self.overall_production_var = tk.DoubleVar(value = 1.0)
        self.overall_production_scale = ttk.Scale(
            overall_frame,
            from_ = 0.1,
            to = 3.0,
            variable = self.overall_production_var,
            orient="horizontal",
            command = self.update_production_labels
        )
        self.overall_production_scale.pack(side="left", padx = 10, fill="x", expand = True)

        self.overall_production_label = ttk.Label(overall_frame, text="100% (Normal)")
        self.overall_production_label.pack(side="left", padx = 10)

        # Department - specific production adjustments
        dept_frame = ttk.LabelFrame(production_frame, text="Department - Specific Adjustments", padding = 5)
        dept_frame.pack(fill="x", pady = 5)

        # Department I (Means of Production)
        dept_I_frame = ttk.Frame(dept_frame)
        dept_I_frame.pack(fill="x", pady = 2)
        ttk.Label(dept_I_frame, text="Dept I (Means of Production):").pack(side="left")
        self.dept_I_production_var = tk.DoubleVar(value = 1.0)
        self.dept_I_production_scale = ttk.Scale(
            dept_I_frame,
            from_ = 0.1,
            to = 3.0,
            variable = self.dept_I_production_var,
            orient="horizontal",
            command = self.update_production_labels
        )
        self.dept_I_production_scale.pack(side="left", padx = 10, fill="x", expand = True)
        self.dept_I_production_label = ttk.Label(dept_I_frame, text="100%")
        self.dept_I_production_label.pack(side="left", padx = 10)

        # Department II (Consumer Goods)
        dept_II_frame = ttk.Frame(dept_frame)
        dept_II_frame.pack(fill="x", pady = 2)
        ttk.Label(dept_II_frame, text="Dept II (Consumer Goods):").pack(side="left")
        self.dept_II_production_var = tk.DoubleVar(value = 1.0)
        self.dept_II_production_scale = ttk.Scale(
            dept_II_frame,
            from_ = 0.1,
            to = 3.0,
            variable = self.dept_II_production_var,
            orient="horizontal",
            command = self.update_production_labels
        )
        self.dept_II_production_scale.pack(side="left", padx = 10, fill="x", expand = True)
        self.dept_II_production_label = ttk.Label(dept_II_frame, text="100%")
        self.dept_II_production_label.pack(side="left", padx = 10)

        # Department III (Services)
        dept_III_frame = ttk.Frame(dept_frame)
        dept_III_frame.pack(fill="x", pady = 2)
        ttk.Label(dept_III_frame, text="Dept III (Services):").pack(side="left")
        self.dept_III_production_var = tk.DoubleVar(value = 1.0)
        self.dept_III_production_scale = ttk.Scale(
            dept_III_frame,
            from_ = 0.1,
            to = 3.0,
            variable = self.dept_III_production_var,
            orient="horizontal",
            command = self.update_production_labels
        )
        self.dept_III_production_scale.pack(side="left", padx = 10, fill="x", expand = True)
        self.dept_III_production_label = ttk.Label(dept_III_frame, text="100%")
        self.dept_III_production_label.pack(side="left", padx = 10)

        # Apply reproduction adjustments option
        self.apply_reproduction_var = tk.BooleanVar(value = True)
        ttk.Checkbutton(production_frame, text="Apply Marxist reproduction adjustments", variable = self.apply_reproduction_var).pack(anchor="w", pady = 5)

        # Reset button
        reset_frame = ttk.Frame(production_frame)
        reset_frame.pack(fill="x", pady = 5)
        ttk.Button(reset_frame, text="Reset to Normal Production", command = self.reset_production_sliders).pack(side="left")

        # Planning options
        options_frame = ttk.LabelFrame(self.planning_scrollable_frame, text="Planning Options", padding = 10)
        options_frame.pack(fill="x", padx = 10, pady = 5)

        # Use optimization
        self.use_optimization_var = tk.BooleanVar(value = True)
        ttk.Checkbutton(options_frame, text="Use constrained optimization", variable = self.use_optimization_var).pack(
            anchor="w"
        )

        # Max iterations
        ttk.Label(options_frame, text="Max Iterations:").pack(anchor="w", pady=(10, 0))
        self.max_iterations_var = tk.StringVar(value="10")
        ttk.Entry(options_frame, textvariable = self.max_iterations_var, width = 10).pack(anchor="w")

        # Plan type
        plan_type_frame = ttk.Frame(options_frame)
        plan_type_frame.pack(fill="x", pady = 10)

        ttk.Label(plan_type_frame, text="Plan Type:").pack(side="left")
        self.plan_type_var = tk.StringVar(value="single_year")
        ttk.Radiobutton(plan_type_frame, text="Single Year", variable = self.plan_type_var, value="single_year").pack(
            side="left", padx = 10
        )
        ttk.Radiobutton(plan_type_frame, text="Five Year", variable = self.plan_type_var, value="five_year").pack(
            side="left", padx = 10
        )

        # Five - year plan options
        self.five_year_frame = ttk.Frame(options_frame)

        ttk.Label(self.five_year_frame, text="Consumption Growth Rate:").grid(row = 0, column = 0, sticky="w", padx = 5)
        self.growth_rate_var = tk.StringVar(value="0.02")
        ttk.Entry(self.five_year_frame, textvariable = self.growth_rate_var, width = 10).grid(row = 0, column = 1, padx = 5)

        ttk.Label(self.five_year_frame, text="Investment Ratio:").grid(row = 0, column = 2, sticky="w", padx = 5)
        self.investment_ratio_var = tk.StringVar(value="0.15")
        ttk.Entry(self.five_year_frame, textvariable = self.investment_ratio_var, width = 10).grid(row = 0, column = 3, padx = 5)

        # Control buttons
        control_frame = ttk.Frame(self.planning_scrollable_frame)
        control_frame.pack(fill="x", padx = 10, pady = 10)

        self.create_plan_button = ttk.Button(
            control_frame, text="Create Plan", command = self.create_plan, style="Accent.TButton"
        )
        self.create_plan_button.pack(side="left", padx = 5)

        self.progress_bar = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress_bar.pack(side="left", padx = 10, fill="x", expand = True)

        # Status
        self.planning_status = ttk.Label(control_frame, text="Ready to create plan")
        self.planning_status.pack(side="right", padx = 5)

    def create_simulation_tab(self):
        """Create simulation system tab."""
        self.simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text="Dynamic Simulation")

        # Create main horizontal layout: settings on left, map on right
        self.simulation_main_frame = ttk.Frame(self.simulation_frame)
        self.simulation_main_frame.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Left side: Settings and controls
        self.simulation_left_frame = ttk.Frame(self.simulation_main_frame)
        self.simulation_left_frame.pack(side="left", fill="both", expand = True, padx = self._scale_padding(5))

        # Right side: Map display
        self.simulation_right_frame = ttk.Frame(self.simulation_main_frame)
        self.simulation_right_frame.pack(side="right", fill="both", expand = True, padx = self._scale_padding(5))

        # Create scrollable frame for simulation controls (left side)
        self.simulation_canvas = tk.Canvas(self.simulation_left_frame)
        self.simulation_scrollbar = ttk.Scrollbar(self.simulation_left_frame, orient="vertical", command = self.simulation_canvas.yview)
        self.simulation_scrollable_frame = ttk.Frame(self.simulation_canvas)

        self.simulation_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.simulation_canvas.configure(scrollregion = self.simulation_canvas.bbox("all"))
        )

        self.simulation_canvas.create_window((0, 0), window = self.simulation_scrollable_frame, anchor="nw")
        self.simulation_canvas.configure(yscrollcommand = self.simulation_scrollbar.set)

        # Plan Loading Section
        plan_frame = ttk.LabelFrame(self.simulation_scrollable_frame, text="Plan Loading", padding = self._scale_padding(10))
        plan_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Plan file selection
        file_frame = ttk.Frame(plan_frame)
        file_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Label(file_frame, text="Load Economic Plan:").pack(side="left", padx = self._scale_padding(5))
        self.plan_file_var = tk.StringVar()
        self.plan_file_entry = ttk.Entry(file_frame, textvariable = self.plan_file_var, width = 50)
        self.plan_file_entry.pack(side="left", padx = self._scale_padding(5), fill="x", expand = True)

        ttk.Button(file_frame, text="Browse", command = self.browse_plan_file).pack(side="right", padx = self._scale_padding(5))
        ttk.Button(file_frame, text="Load Plan", command = self.load_simulation_plan).pack(side="right", padx = self._scale_padding(5))
        ttk.Button(file_frame, text="Reload Current Plan", command = self.reload_current_plan).pack(side="right", padx = self._scale_padding(5))

        # Plan status
        self.plan_status = ttk.Label(plan_frame, text="No plan loaded", foreground="red")
        self.plan_status.pack(pady = self._scale_padding(5))

        # Simulation Parameters Section
        params_frame = ttk.LabelFrame(self.simulation_scrollable_frame, text="Simulation Parameters", padding = self._scale_padding(10))
        params_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Time parameters
        time_frame = ttk.Frame(params_frame)
        time_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Label(time_frame, text="Simulation Duration (years):").pack(side="left", padx = self._scale_padding(5))
        self.sim_duration_var = tk.StringVar(value="5")
        ttk.Spinbox(time_frame, from_ = 1, to = 20, textvariable = self.sim_duration_var, width = 10).pack(side="left", padx = self._scale_padding(5))

        ttk.Label(time_frame, text="Time Step (months):").pack(side="left", padx = self._scale_padding(20))
        self.time_step_var = tk.StringVar(value="1")
        ttk.Spinbox(time_frame, from_ = 1, to = 12, textvariable = self.time_step_var, width = 10).pack(side="left", padx = self._scale_padding(5))

        # Environment parameters
        env_frame = ttk.Frame(params_frame)
        env_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Label(env_frame, text="Map Size (km):").pack(side="left", padx = self._scale_padding(5))
        self.map_size_var = tk.StringVar(value="1000")
        ttk.Spinbox(env_frame, from_ = 100, to = 10000, textvariable = self.map_size_var, width = 10).pack(side="left", padx = self._scale_padding(5))

        ttk.Label(env_frame, text="Settlements:").pack(side="left", padx = self._scale_padding(20))
        self.settlements_var = tk.StringVar(value="50")
        ttk.Spinbox(env_frame, from_ = 10, to = 500, textvariable = self.settlements_var, width = 10).pack(side="left", padx = self._scale_padding(5))

        # Economic sectors
        sectors_frame = ttk.Frame(params_frame)
        sectors_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Label(sectors_frame, text="Economic Sectors:").pack(side="left", padx = self._scale_padding(5))
        self.sectors_var = tk.StringVar(value="15")
        ttk.Spinbox(sectors_frame, from_ = 5, to = 50, textvariable = self.sectors_var, width = 10).pack(side="left", padx = self._scale_padding(5))

        ttk.Label(sectors_frame, text="Population Density:").pack(side="left", padx = self._scale_padding(20))
        self.pop_density_var = tk.StringVar(value="100")
        ttk.Spinbox(sectors_frame, from_ = 10, to = 1000, textvariable = self.pop_density_var, width = 10).pack(side="left", padx = self._scale_padding(5))

        # Stochastic Events Section
        events_frame = ttk.LabelFrame(self.simulation_scrollable_frame, text="Stochastic Events", padding = self._scale_padding(10))
        events_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Event probability controls
        prob_frame = ttk.Frame(events_frame)
        prob_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Label(prob_frame, text="Natural Disasters:").pack(side="left", padx = self._scale_padding(5))
        self.natural_disasters_var = tk.BooleanVar(value = True)
        ttk.Checkbutton(prob_frame, variable = self.natural_disasters_var).pack(side="left", padx = self._scale_padding(5))

        ttk.Label(prob_frame, text="Economic Disruptions:").pack(side="left", padx = self._scale_padding(20))
        self.economic_disruptions_var = tk.BooleanVar(value = True)
        ttk.Checkbutton(prob_frame, variable = self.economic_disruptions_var).pack(side="left", padx = self._scale_padding(5))

        ttk.Label(prob_frame, text="Infrastructure Failures:").pack(side="left", padx = self._scale_padding(20))
        self.infrastructure_failures_var = tk.BooleanVar(value = True)
        ttk.Checkbutton(prob_frame, variable = self.infrastructure_failures_var).pack(side="left", padx = self._scale_padding(5))

        # Event frequency
        freq_frame = ttk.Frame(events_frame)
        freq_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Label(freq_frame, text="Event Frequency (per year):").pack(side="left", padx = self._scale_padding(5))
        self.event_frequency_var = tk.StringVar(value="2.0")
        ttk.Spinbox(freq_frame, from_ = 0.0, to = 10.0, increment = 0.5, textvariable = self.event_frequency_var, width = 10).pack(side="left", padx = self._scale_padding(5))

        # Simulation Control Section
        control_frame = ttk.LabelFrame(self.simulation_scrollable_frame, text="Simulation Control", padding = self._scale_padding(10))
        control_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Button(button_frame, text="Initialize Simulation", command = self.initialize_simulation).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(button_frame, text="Start Simulation", command = self.start_simulation).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(button_frame, text="Pause Simulation", command = self.pause_simulation).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(button_frame, text="Stop Simulation", command = self.stop_simulation).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(button_frame, text="Reset Simulation", command = self.reset_simulation).pack(side="left", padx = self._scale_padding(5))

        # Simulation status
        self.simulation_status = ttk.Label(control_frame, text="Ready to initialize", foreground="blue")
        self.simulation_status.pack(pady = self._scale_padding(5))

        # Progress bar
        self.simulation_progress = ttk.Progressbar(control_frame, mode='determinate')
        self.simulation_progress.pack(fill="x", pady = self._scale_padding(5))

        # Simulation Results Section
        results_frame = ttk.LabelFrame(self.simulation_scrollable_frame, text="Simulation Results", padding = self._scale_padding(10))
        results_frame.pack(fill="both", expand = True, padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Create notebook for different result views
        self.simulation_notebook = ttk.Notebook(results_frame)
        self.simulation_notebook.pack(fill="both", expand = True)

        # Real - time monitoring tab
        self.monitoring_frame = ttk.Frame(self.simulation_notebook)
        self.simulation_notebook.add(self.monitoring_frame, text="Real - time Monitoring")

        # Monitoring text area
        self.monitoring_text = scrolledtext.ScrolledText(self.monitoring_frame, height = 15, width = 80)
        self.monitoring_text.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Performance metrics tab
        self.metrics_frame = ttk.Frame(self.simulation_notebook)
        self.simulation_notebook.add(self.metrics_frame, text="Performance Metrics")

        # Metrics tree view
        self.metrics_tree = ttk.Treeview(
            self.metrics_frame,
            columns=("metric", "value", "target", "status"),
            show="headings",
            height = 15
        )
        self.metrics_tree.heading("metric", text="Metric")
        self.metrics_tree.heading("value", text="Current Value")
        self.metrics_tree.heading("target", text="Target")
        self.metrics_tree.heading("status", text="Status")
        self.metrics_tree.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Event log tab
        self.events_frame = ttk.Frame(self.simulation_notebook)
        self.simulation_notebook.add(self.events_frame, text="Event Log")

        # Event log text area
        self.events_text = scrolledtext.ScrolledText(self.events_frame, height = 15, width = 80)
        self.events_text.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Interactive Map tab
        self.map_frame = ttk.Frame(self.simulation_notebook)
        self.simulation_notebook.add(self.map_frame, text="Interactive Map")

        # Map controls
        map_controls_frame = ttk.Frame(self.map_frame)
        map_controls_frame.pack(fill="x", padx = self._scale_padding(5), pady = self._scale_padding(5))

        ttk.Button(map_controls_frame, text="Generate Map", command = self.generate_simulation_map).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(map_controls_frame, text="Open in Browser", command = self.open_map_in_browser).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(map_controls_frame, text="Refresh Map", command = self.refresh_simulation_map).pack(side="left", padx = self._scale_padding(5))

        # Map status
        self.map_status = ttk.Label(map_controls_frame, text="No map generated", foreground="red")
        self.map_status.pack(side="right", padx = self._scale_padding(5))

        # Map display area (placeholder for map info)
        self.map_display_frame = ttk.LabelFrame(self.map_frame, text="Map Information", padding = self._scale_padding(10))
        self.map_display_frame.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Map info text
        self.map_info_text = scrolledtext.ScrolledText(self.map_display_frame, height = 15, width = 80)
        self.map_info_text.pack(fill="both", expand = True)

        # Initialize map variables
        self.current_map = None
        self.map_file_path = None

        # Pack canvas and scrollbar
        self.simulation_canvas.pack(side="left", fill="both", expand = True)
        self.simulation_scrollbar.pack(side="right", fill="y")

        # Right side: Real - time Map Display
        self.create_realtime_map_section()

        # Initialize simulation state
        self.simulation_state = "stopped"
        self.simulation_thread = None
        self.current_simulation = None
        self.realtime_map_thread = None
        self.map_update_active = False

    def create_realtime_map_section(self):
        """Create the real - time map display section on the right side."""
        # Map Controls Section
        map_controls_frame = ttk.LabelFrame(self.simulation_right_frame, text="Real - time Map Controls", padding = self._scale_padding(10))
        map_controls_frame.pack(fill="x", padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Generate Map Button
        ttk.Button(map_controls_frame, text="Generate Map", command = self.generate_realtime_map).pack(side="left", padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Open in Browser Button
        ttk.Button(map_controls_frame, text="Open in Browser", command = self.open_realtime_map_in_browser).pack(side="left", padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Speed Control
        speed_frame = ttk.Frame(map_controls_frame)
        speed_frame.pack(side="left", padx = self._scale_padding(10), pady = self._scale_padding(5))

        ttk.Label(speed_frame, text="Update Speed:").pack(side="left", padx = self._scale_padding(5))
        self.map_speed_var = tk.StringVar(value="1 day / sec")
        speed_combo = ttk.Combobox(speed_frame, textvariable = self.map_speed_var, width = 12, state="readonly")
        speed_combo['values'] = ("1 hour / sec", "1 day / sec", "1 month / sec", "1 year / sec")
        speed_combo.pack(side="left", padx = self._scale_padding(5))

        # Map Control Buttons
        control_frame = ttk.Frame(map_controls_frame)
        control_frame.pack(side="right", padx = self._scale_padding(5), pady = self._scale_padding(5))

        ttk.Button(control_frame, text="Start Updates", command = self.start_map_updates).pack(side="left", padx = self._scale_padding(2))
        ttk.Button(control_frame, text="Pause Updates", command = self.pause_map_updates).pack(side="left", padx = self._scale_padding(2))
        ttk.Button(control_frame, text="Stop Updates", command = self.stop_map_updates).pack(side="left", padx = self._scale_padding(2))

        # Map Status
        self.realtime_map_status = ttk.Label(map_controls_frame, text="No map generated", foreground="red")
        self.realtime_map_status.pack(side="right", padx = self._scale_padding(10))

        # Map Display Area
        self.realtime_map_display_frame = ttk.LabelFrame(self.simulation_right_frame, text="Real - time Map", padding = self._scale_padding(10))
        self.realtime_map_display_frame.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Create notebook for map display and info
        self.map_display_notebook = ttk.Notebook(self.realtime_map_display_frame)
        self.map_display_notebook.pack(fill="both", expand = True)

        # Interactive Map tab
        self.map_view_frame = ttk.Frame(self.map_display_notebook)
        self.map_display_notebook.add(self.map_view_frame, text="Interactive Map")

        # Map info tab
        self.map_info_frame = ttk.Frame(self.map_display_notebook)
        self.map_display_notebook.add(self.map_info_frame, text="Map Info")

        # Map display area (web browser widget)
        self.map_browser_widget = WebBrowserWidget(self.map_view_frame, width = 800, height = 600)
        self.map_browser_widget.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Map info display
        self.realtime_map_info_text = scrolledtext.ScrolledText(self.map_info_frame, height = 20, width = 60)
        self.realtime_map_info_text.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Initialize real - time map variables
        self.realtime_map = None
        self.realtime_map_file_path = None
        self.map_update_interval = 1.0  # seconds
        self.current_simulation_time = 0  # months
        self.map_update_timer = None

    def create_results_tab(self):
        """Create results display tab."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results & Analysis")

        # Results summary
        summary_frame = ttk.LabelFrame(self.results_frame, text="Plan Summary", padding = 10)
        summary_frame.pack(fill="x", padx = 10, pady = 5)

        self.summary_text = scrolledtext.ScrolledText(summary_frame, height = 8, width = 80)
        self.summary_text.pack(fill="both", expand = True)

        # Detailed results
        details_frame = ttk.LabelFrame(self.results_frame, text="Detailed Results", padding = 10)
        details_frame.pack(fill="both", expand = True, padx = 10, pady = 5)

        # Create notebook for different result views
        self.results_notebook = ttk.Notebook(details_frame)

        # Sector analysis
        self.sector_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.sector_frame, text="Sector Analysis")

        self.sector_tree = ttk.Treeview(
            self.sector_frame, columns=("output", "demand", "labor_value", "labor_cost"), show="headings", height = 10
        )
        self.sector_tree.heading("#0", text="Sector")
        self.sector_tree.heading("output", text="Total Output")
        self.sector_tree.heading("demand", text="Final Demand")
        self.sector_tree.heading("labor_value", text="Labor Value")
        self.sector_tree.heading("labor_cost", text="Labor Cost")

        self.sector_tree.column("#0", width = 80)
        self.sector_tree.column("output", width = 120)
        self.sector_tree.column("demand", width = 120)
        self.sector_tree.column("labor_value", width = 120)
        self.sector_tree.column("labor_cost", width = 120)

        sector_scrollbar = ttk.Scrollbar(self.sector_frame, orient="vertical", command = self.sector_tree.yview)
        self.sector_tree.configure(yscrollcommand = sector_scrollbar.set)

        self.sector_tree.pack(side="left", fill="both", expand = True)
        sector_scrollbar.pack(side="right", fill="y")

        # Report view
        self.report_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.report_frame, text="Full Report")

        self.report_text = scrolledtext.ScrolledText(self.report_frame, height = 20, width = 80)
        self.report_text.pack(fill="both", expand = True)

        self.results_notebook.pack(fill="both", expand = True)

    def create_export_tab(self):
        """Create export and save tab."""
        self.export_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.export_frame, text="Export & Save")

        # Save options
        save_frame = ttk.LabelFrame(self.export_frame, text="Save Current Plan", padding = 10)
        save_frame.pack(fill="x", padx = 10, pady = 5)

        ttk.Button(save_frame, text="Save as JSON", command = lambda: self.save_plan("json")).pack(side="left", padx = 5)
        ttk.Button(save_frame, text="Save as CSV", command = lambda: self.save_plan("csv")).pack(side="left", padx = 5)
        ttk.Button(save_frame, text="Save as Excel", command = lambda: self.save_plan("excel")).pack(side="left", padx = 5)
        ttk.Button(save_frame, text="Export for Simulation", command = self.export_plan_for_simulation).pack(side="left", padx = 5)

        # Export data
        export_frame = ttk.LabelFrame(self.export_frame, text="Export Data", padding = 10)
        export_frame.pack(fill="x", padx = 10, pady = 5)

        ttk.Button(export_frame, text="Export Current Data", command = self.export_data).pack(side="left", padx = 5)

        # Load plan
        load_frame = ttk.LabelFrame(self.export_frame, text="Load Plan", padding = 10)
        load_frame.pack(fill="x", padx = 10, pady = 5)

        ttk.Button(load_frame, text="Load Plan from File", command = self.load_plan).pack(side="left", padx = 5)

        # Status
        self.export_status = ttk.Label(self.export_frame, text="No plan to export")
        self.export_status.pack(pady = 10)

    def create_about_tab(self):
        """Create about tab."""
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")

        about_text = """
Cybernetic Central Planning System

A sophisticated economic planning system that uses Input - Output analysis
and labor - time accounting to generate comprehensive economic plans.

Features:
• Multi - agent planning system with specialized agents
• Input - Output analysis using Leontief models
• Labor value calculations and optimization
• Policy goal translation and implementation
• Resource constraint management
• Environmental impact assessment
• Comprehensive report generation
• Real - time web scraping for economic data
• Multi - country data collection support

NEW FEATURES (v2.0):
• Marxist Economic Analysis: Complete implementation of Marx's economic theory - C + V + S value composition analysis - Rate of surplus value and rate of profit calculations - Organic composition of capital analysis - Price - value transformation analysis - Simple and expanded reproduction schemas

• Cybernetic Feedback Systems: Advanced feedback control mechanisms - PID controller implementation - Circular causality and self - regulation - Requisite variety and stability analysis - Adaptive control parameters - Real - time system diagnostics

• Mathematical Validation: Comprehensive formula validation - Automatic validation of all economic formulas - Theoretical accuracy verification - Numerical precision testing - Comprehensive error reporting

Data Collection:
• USA: EIA, USGS, BLS, EPA data sources
• Russia: Government statistical agencies
• EU: Eurostat and European Commission data
• China: National Bureau of Statistics and ministries
• India: Various government departments and agencies

Agents:
• Manager Agent: Central coordination and plan orchestration
• Economics Agent: Sensitivity analysis and forecasting
• Policy Agent: Goal translation and social impact assessment
• Resource Agent: Resource optimization and environmental analysis
• Writer Agent: Report generation and documentation

The system can create both single - year and five - year economic plans,
incorporating policy goals and resource constraints to generate
optimal economic strategies using real - world data from multiple countries.
        """

        about_label = ttk.Label(self.about_frame, text = about_text, justify="left")
        about_label.pack(padx = self._scale_padding(20), pady = self._scale_padding(20))

        # Add scaling information
        scaling_info = self.get_scaling_info()
        scaling_text = f"""

DISPLAY INFORMATION:
• Screen Resolution: {scaling_info['screen_width']}x{scaling_info['screen_height']}
• Aspect Ratio: {scaling_info['aspect_ratio']:.3f}
• Scale Factor: {scaling_info['scale_factor']:.3f}
• Window Size: {scaling_info['window_width']}x{scaling_info['window_height']}
• Auto - scaling: Enabled
        """

        scaling_label = ttk.Label(self.about_frame, text = scaling_text, justify="left",
                                 font=("Arial", self._scale_font_size(9), "italic"))
        scaling_label.pack(padx = self._scale_padding(20), pady = self._scale_padding(10))

    def setup_layout(self):
        """Setup the main layout."""
        self.notebook.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))

        # Configure styles
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="blue")

    def load_data_from_file(self):
        """Load data from a file."""
        # First check if there are any processed files in the data folder
        data_folder = Path("data")
        if data_folder.exists():
            json_files = list(data_folder.glob("*.json"))
            if json_files:
                # Show a dialog to choose from existing files or browse for new ones
                choice = messagebox.askyesnocancel(
                    "Load Data",
                    f"Found {len(json_files)} processed data files in the data folder.\n\n"
                    f"Click 'Yes' to choose from existing files\n"
                    f"Click 'No' to browse for a different file\n"
                    f"Click 'Cancel' to cancel",
                )

                if choice is True:  # Choose from existing files
                    self.choose_from_existing_files(json_files)
                    return
                elif choice is False:  # Browse for new file
                    pass  # Continue with file dialog
                else:  # Cancel
                    return

        # Browse for file
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("JSON files", "*.json"),
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                # Check if it's a raw data file that needs processing
                if self.is_raw_data_file(file_path):
                    self.process_raw_data_file(file_path)
                else:
                    # Load processed data directly
                    self.planning_system.load_data_from_file(file_path)
                    self.current_data = self.planning_system.current_data

                    # Ensure data is properly converted to numpy arrays
                    self._ensure_numpy_arrays()

                    self.update_data_display()
                    self.data_status.config(text="Data loaded successfully", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
                self.data_status.config(text="Error loading data", foreground="red")

    def _ensure_numpy_arrays(self):
        """Ensure all data arrays are numpy arrays"""
        if not self.current_data:
            return

        # Convert lists to numpy arrays
        array_fields = ["technology_matrix", "final_demand", "labor_input", "resource_matrix", "max_resources"]

        for field in array_fields:
            if field in self.current_data:
                if isinstance(self.current_data[field], list):
                    self.current_data[field] = np.array(self.current_data[field])
                    print(f"✅ Converted {field} to numpy array")
                elif self.current_data[field] is None:
                    # Handle None values by creating empty arrays
                    self.current_data[field] = np.array([])
                    print(f"⚠️  {field} was None, created empty array")
                elif not hasattr(self.current_data[field], "shape"):
                    # If it doesn't have shape attribute, try to convert
                    try:
                        self.current_data[field] = np.array(self.current_data[field])
                        print(f"✅ Converted {field} to numpy array")
                    except:
                        self.current_data[field] = np.array([])
                        print(f"⚠️  {field} conversion failed, created empty array")

    def generate_synthetic_data(self):
        """Generate synthetic data based on configuration."""
        try:
            n_sectors = int(self.sectors_var.get())
            density = float(self.density_var.get())
            n_resources = int(self.resources_var.get())

            self.planning_system.create_synthetic_data(
                n_sectors = n_sectors, technology_density = density, resource_count = n_resources
            )
            self.current_data = self.planning_system.current_data

            # Ensure data is properly converted to numpy arrays
            self._ensure_numpy_arrays()

            self.update_data_display()
            self.data_status.config(text="Synthetic data generated", foreground="green")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid configuration: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")

    def check_api_keys(self):
        """Check API key status."""
        try:
            from api_keys_config import APIKeyManager

            manager = APIKeyManager()
            status = manager.check_api_key_status()

            # Update status label
            if status.get("api_manager_available", False):
                required_missing = status.get("required_keys", [])
                status.get("optional_keys", [])

                if not required_missing:
                    self.api_status.config(text="✅ All required API keys are configured", foreground="green")
                else:
                    self.api_status.config(
                        text = f"❌ Missing {len(required_missing)} required API keys", foreground="red"
                    )
            else:
                self.api_status.config(text="❌ API key manager not available", foreground="red")

            # Update information text
            self.api_info_text.delete(1.0, tk.END)

            # Get the output from the print function
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            manager.print_setup_instructions()
            sys.stdout = old_stdout
            self.api_info_text.insert(tk.END, buffer.getvalue())

        except Exception as e:
            self.api_status.config(text = f"❌ Error checking API keys: {str(e)}", foreground="red")
            messagebox.showerror("Error", f"Failed to check API keys: {str(e)}")

    def process_usa_zip(self):
        """Process USA data from zip file."""
        zip_path = filedialog.askopenfilename(
            title="Select USA Data Zip File", filetypes=[("Zip files", "*.zip"), ("All files", "*.*")]
        )

        if not zip_path:
            return

        try:
            # Import the zip processor
            from usa_zip_processor import USAZipProcessor

            # Update status
            self.data_status.config(text="Processing USA zip file...", foreground="blue")
            self.root.update()

            # Create processor and process zip
            processor = USAZipProcessor()
            result_file = processor.process_zip_file(zip_path)

            if result_file:
                # Load the processed data
                self.planning_system.load_data_from_file(result_file)
                self.current_data = self.planning_system.current_data

                # Ensure data is properly converted to numpy arrays
                self._ensure_numpy_arrays()

                self.update_data_display()
                self.data_status.config(text="USA data processed and loaded successfully", foreground="green")

                # Show success message
                messagebox.showinfo(
                    "Success",
                    f"USA data processed successfully!\n\n"
                    f"Saved to: {result_file}\n"
                    f"Sectors: {len(self.current_data.get('sectors', []))}",
                )
            else:
                self.data_status.config(text="Failed to process USA zip file", foreground="red")
                messagebox.showerror("Error", "Failed to process the USA zip file")

        except Exception as e:
            self.data_status.config(text="Error processing USA zip file", foreground="red")
            messagebox.showerror("Error", f"Failed to process USA zip file: {str(e)}")
            import traceback

            traceback.print_exc()

    def choose_from_existing_files(self, json_files):
        """Choose from existing processed files in data folder."""
        # Create a simple dialog to choose files
        choice_window = tk.Toplevel(self.root)
        choice_window.title("Choose Data File")
        choice_window.geometry("500x400")
        choice_window.transient(self.root)
        choice_window.grab_set()

        # Center the window
        choice_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))

        ttk.Label(choice_window, text="Select a processed data file:", font=("Arial", 12, "bold")).pack(pady = 10)

        # Create listbox with scrollbar
        frame = ttk.Frame(choice_window)
        frame.pack(fill="both", expand = True, padx = 10, pady = 10)

        listbox = tk.Listbox(frame, height = 15)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command = listbox.yview)
        listbox.configure(yscrollcommand = scrollbar.set)

        # Add files to listbox
        for i, file_path in enumerate(json_files):
            file_name = file_path.name
            file_size = file_path.stat().st_size
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            display_text = f"{file_name} ({file_size//1024}KB, {file_time})"
            listbox.insert(tk.END, display_text)

        listbox.pack(side="left", fill="both", expand = True)
        scrollbar.pack(side="right", fill="y")

        # Buttons
        button_frame = ttk.Frame(choice_window)
        button_frame.pack(fill="x", padx = 10, pady = 10)

        def load_selected():
            selection = listbox.curselection()
            if selection:
                selected_file = json_files[selection[0]]
                choice_window.destroy()
                self.load_specific_file(selected_file)
            else:
                messagebox.showwarning("Warning", "Please select a file")

        def cancel():
            choice_window.destroy()

        ttk.Button(button_frame, text="Load Selected", command = load_selected).pack(side="left", padx = 5)
        ttk.Button(button_frame, text="Cancel", command = cancel).pack(side="left", padx = 5)

        # Double - click to load
        listbox.bind("<Double - Button - 1>", lambda e: load_selected())

    def load_specific_file(self, file_path):
        """Load a specific file."""
        try:
            self.planning_system.load_data_from_file(str(file_path))
            self.current_data = self.planning_system.current_data
            self.update_data_display()
            self.data_status.config(text = f"Data loaded from {file_path.name}", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.data_status.config(text="Error loading data", foreground="red")

    def is_raw_data_file(self, file_path):
        """Check if the file is a raw data file that needs processing."""
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in [".xlsx", ".xls", ".csv"]

    def process_raw_data_file(self, file_path):
        """Process a raw data file and save it to the data folder."""
        try:
            # Import the data processor
            from us_gov_data_processor import process_file, detect_data_type

            # Detect the data type
            data_type = detect_data_type(file_path)

            # Create a unique filename for the processed data
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_filename = f"{base_name}_{data_type}_{timestamp}.json"
            processed_path = os.path.join("data", processed_filename)

            # Ensure data directory exists
            os.makedirs("data", exist_ok = True)

            # Process the file
            self.data_status.config(text = f"Processing {data_type} data...", foreground="blue")
            self.root.update()

            processed_data = process_file(file_path, processed_path)

            if processed_data:
                # Load the processed data into the planning system
                self.planning_system.load_data_from_file(processed_path)
                self.current_data = self.planning_system.current_data
                self.update_data_display()
                self.data_status.config(
                    text = f"Data processed and loaded successfully ({data_type})", foreground="green"
                )

                # Show success message
                messagebox.showinfo(
                    "Success",
                    f"Data processed successfully!\n\n"
                    f"Type: {data_type}\n"
                    f"Saved to: {processed_path}\n"
                    f"Sectors: {len(processed_data.get('sectors', []))}",
                )
            else:
                self.data_status.config(text="Failed to process data", foreground="red")
                messagebox.showerror("Error", "Failed to process the data file")

        except Exception as e:
            self.data_status.config(text="Error processing data", foreground="red")
            messagebox.showerror("Error", f"Failed to process data file: {str(e)}")
            import traceback

            traceback.print_exc()

    def update_data_display(self):
        """Update the data display text widget."""
        if not self.current_data:
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert("1.0", "No data loaded")
            return

        # Create a summary of the loaded data
        tech_matrix = self.current_data.get("technology_matrix", [])
        final_demand = self.current_data.get("final_demand", [])
        labor_input = self.current_data.get("labor_input", [])

        sectors = self.current_data.get('sectors', [])
        sector_count = len(sectors)

        # Safely get shapes
        tech_shape = (
            tech_matrix.shape
            if hasattr(tech_matrix, "shape")
            else f"({len(tech_matrix)}, {len(tech_matrix[0]) if tech_matrix else 0})"
        )
        final_shape = final_demand.shape if hasattr(final_demand, "shape") else f"({len(final_demand)},)"
        labor_shape = labor_input.shape if hasattr(labor_input, "shape") else f"({len(labor_input)},)"

        # Safely get matrix slice
        tech_slice = "N / A"
        if hasattr(tech_matrix, "shape"):
            # It's a numpy array
            tech_slice = str(tech_matrix[:4, :4])
        elif isinstance(tech_matrix, list) and len(tech_matrix) >= 4:
            # It's a list, get first 4x4 elements
            tech_slice = str([row[:4] for row in tech_matrix[:4]])

        # Create sector summary
        if sector_count > 0:
            first_sectors = sectors[:5]
            last_sectors = sectors[-5:] if sector_count > 10 else []
            sector_summary = f"First 5: {first_sectors}"
            if last_sectors:
                sector_summary += f"\nLast 5: {last_sectors}"
            if sector_count > 10:
                sector_summary += f"\n... and {sector_count - 10} more sectors"
        else:
            sector_summary = "No sectors available"

        summary = f"""Data Summary:
================

Sectors: {sector_count}
Technology Matrix Shape: {tech_shape}
Final Demand Shape: {final_shape}
Labor Input Shape: {labor_shape}

Sector Names:
{sector_summary}

Final Demand Values:
{final_demand}

Labor Input Values:
{labor_input}

Technology Matrix (first 4x4):
{tech_slice}
        """

        self.data_text.delete("1.0", tk.END)
        self.data_text.insert("1.0", summary)

    def create_plan(self):
        """Create an economic plan."""
        if not self.current_data:
            messagebox.showerror("Error", "Please load data first")
            return

        # Get policy goals
        goals_text = self.goals_text.get("1.0", tk.END).strip()
        policy_goals = [goal.strip() for goal in goals_text.split("\n") if goal.strip()]

        # Get planning options
        use_optimization = self.use_optimization_var.get()
        max_iterations = int(self.max_iterations_var.get())
        plan_type = self.plan_type_var.get()

        # Get production adjustment settings
        production_multipliers = {
            "overall": self.overall_production_var.get(),
            "dept_I": self.dept_I_production_var.get(),
            "dept_II": self.dept_II_production_var.get(),
            "dept_III": self.dept_III_production_var.get()
        }
        apply_reproduction = self.apply_reproduction_var.get()

        print(f"Production multipliers: {production_multipliers}")
        print(f"Apply reproduction: {apply_reproduction}")

        # Start planning in a separate thread
        self.create_plan_button.config(state="disabled")
        self.progress_bar.start()
        self.planning_status.config(text="Creating plan...")

        def plan_thread():
            try:
                print(f"Creating plan with {len(policy_goals)} policy goals")
                print(f"Data available: {bool(self.current_data)}")

                if plan_type == "single_year":
                    self.current_plan = self.planning_system.create_plan(
                        policy_goals = policy_goals,
                        use_optimization = use_optimization,
                        max_iterations = max_iterations,
                        production_multipliers = production_multipliers,
                        apply_reproduction = apply_reproduction
                    )
                    print(f"Plan created successfully: {type(self.current_plan)}")
                    if isinstance(self.current_plan, dict):
                        print(f"Plan keys: {list(self.current_plan.keys())}")
                else:  # five_year
                    growth_rate = float(self.growth_rate_var.get())
                    investment_ratio = float(self.investment_ratio_var.get())

                    self.current_plan = self.planning_system.create_five_year_plan(
                        policy_goals = policy_goals,
                        consumption_growth_rate = growth_rate,
                        investment_ratio = investment_ratio,
                        production_multipliers = production_multipliers,
                        apply_reproduction = apply_reproduction
                    )
                    print(f"Five - year plan created successfully: {type(self.current_plan)}")

                # Update UI in main thread
                self.root.after(0, self.plan_created_successfully)

            except Exception as e:
                error_msg = str(e)
                print(f"Plan creation failed: {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.plan_creation_failed(error_msg))

        threading.Thread(target = plan_thread, daemon = True).start()

    def plan_created_successfully(self):
        """Handle successful plan creation."""
        self.create_plan_button.config(state="normal")
        self.progress_bar.stop()
        self.planning_status.config(text="Plan created successfully", foreground="green")

        # Update results display
        self.update_results_display()

        # Automatically load the plan into the simulator
        self._auto_load_plan_to_simulator()

        # Switch to results tab to show the plan
        self.notebook.select(self.results_frame)

        # Update export status
        self.export_status.config(text="Plan ready for export and simulation", foreground="green")

    def _auto_load_plan_to_simulator(self):
        """Automatically load the current plan into the simulator."""
        if not self.current_plan:
            return

        try:
            # Convert the planning plan to simulation format
            simulation_plan = self._convert_planning_plan_to_simulation(self.current_plan)

            # Load it into the simulation system
            self.current_simulation_plan = simulation_plan

            # Update simulation parameters based on the plan
            if 'sectors' in simulation_plan:
                self.sectors_var.set(str(len(simulation_plan['sectors'])))

            # Update the plan status in the simulation tab
            self.plan_status.config(text="Plan auto - loaded from planning system", foreground="green")

            # Switch to simulation tab to show it's ready
            self.notebook.select(self.simulation_frame)

            print(f"✓ Plan automatically loaded into simulator with {len(simulation_plan['sectors'])} sectors")

        except Exception as e:
            print(f"Warning: Failed to auto - load plan to simulator: {e}")
            # Don't fail the plan creation if simulator loading fails
            pass

    def plan_creation_failed(self, error_msg):
        """Handle plan creation failure."""
        self.create_plan_button.config(state="normal")
        self.progress_bar.stop()
        self.planning_status.config(text="Plan creation failed", foreground="red")
        messagebox.showerror("Error", f"Failed to create plan: {error_msg}")

    def update_production_labels(self, value = None):
        """Update the production percentage labels when sliders change."""
        # Update overall production label
        overall_val = self.overall_production_var.get()
        if overall_val < 1.0:
            self.overall_production_label.config(text = f"{overall_val * 100:.0f}% (Underproduction)")
        elif overall_val > 1.0:
            self.overall_production_label.config(text = f"{overall_val * 100:.0f}% (Overproduction)")
        else:
            self.overall_production_label.config(text="100% (Normal)")

        # Update department labels
        dept_I_val = self.dept_I_production_var.get()
        self.dept_I_production_label.config(text = f"{dept_I_val * 100:.0f}%")

        dept_II_val = self.dept_II_production_var.get()
        self.dept_II_production_label.config(text = f"{dept_II_val * 100:.0f}%")

        dept_III_val = self.dept_III_production_var.get()
        self.dept_III_production_label.config(text = f"{dept_III_val * 100:.0f}%")

    def reset_production_sliders(self):
        """Reset all production sliders to normal (100%) levels."""
        self.overall_production_var.set(1.0)
        self.dept_I_production_var.set(1.0)
        self.dept_II_production_var.set(1.0)
        self.dept_III_production_var.set(1.0)
        self.update_production_labels()

    def on_closing(self):
        """Handle window closing event."""
        # Unbind mousewheel to prevent memory leaks
        if hasattr(self, 'mousewheel_binding'):
            self.planning_canvas.unbind_all("<MouseWheel>")
        self.root.destroy()

    def update_results_display(self):
        """Update the results display."""
        print(f"update_results_display called, current_plan: {bool(self.current_plan)}")

        if not self.current_plan:
            print("No current plan available for display")
            return

        # Handle different plan structures
        plan = self.current_plan
        print(f"Plan type: {type(plan)}")
        print(f"Plan keys: {list(plan.keys()) if isinstance(plan, dict) else 'Not a dict'}")

        # Check if plan has nested structure (evaluation result)
        if isinstance(plan, dict) and "plan" in plan and isinstance(plan["plan"], dict):
            plan = plan["plan"]
            print("Using nested plan structure")

        # Update summary
        if isinstance(plan, dict) and "total_output" in plan:
            # Single year plan
            summary = self.planning_system.get_plan_summary()
            summary_text = f"""Plan Summary:
================

Total Economic Output: {summary['total_economic_output']:,.2f} units
Total Labor Cost: {summary['total_labor_cost']:,.2f} person - hours
Labor Efficiency: {summary['labor_efficiency']:.2f} units / hour
Sector Count: {summary['sector_count']}
Plan Quality Score: {summary['plan_quality_score']:.2f}

Constraint Violations:
- Demand Violations: {len(summary['constraint_violations'].get('demand_violations', []))}
- Resource Violations: {len(summary['constraint_violations'].get('resource_violations', []))}
- Non - Negativity Violations: {len(summary['constraint_violations'].get('non_negativity_violations', []))}
            """
        else:
            # Five year plan
            summary_text = f"""Five - Year Plan Summary:
========================

Years Planned: {len(self.current_plan)}

Year - by - Year Summary:
"""
            for year, plan in self.current_plan.items():
                total_output = np.sum(plan["total_output"])
                labor_cost = plan["total_labor_cost"]
                summary_text += f"Year {year}: Output = {total_output:,.1f}, Labor = {labor_cost:,.1f}\n"

        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", summary_text)

        # Update sector analysis
        self.update_sector_analysis()

        # Update report
        self.update_report()

    def update_sector_analysis(self):
        """Update the sector analysis table."""
        # Clear existing items
        for item in self.sector_tree.get_children():
            self.sector_tree.delete(item)

        if not self.current_plan:
            return

        # Handle different plan structures
        plan = self.current_plan

        # Check if plan has nested structure (evaluation result)
        if isinstance(plan, dict) and "plan" in plan and isinstance(plan["plan"], dict):
            plan = plan["plan"]

        if isinstance(plan, dict) and "total_output" in plan:
            # Debug: Print plan keys and final_demand info
            print(f"DEBUG: Plan keys: {list(plan.keys())}")
            if "final_demand" in plan:
                final_demand_data = plan["final_demand"]
                print(f"DEBUG: final_demand type: {type(final_demand_data)}")
                print(f"DEBUG: final_demand shape / length: {getattr(final_demand_data, 'shape', len(final_demand_data))}")
                print(f"DEBUG: final_demand first 5 values: {final_demand_data[:5] if hasattr(final_demand_data, '__getitem__') else 'No indexing'}")
            else:
                print("DEBUG: No 'final_demand' key in plan")

            # Single year plan
            for i in range(len(plan["total_output"])):
                # Check if labor_values exists, otherwise use labor_vector
                if "labor_values" in plan:
                    labor_value = plan["labor_values"][i]
                elif "labor_vector" in plan:
                    labor_value = plan["labor_vector"][i]
                else:
                    labor_value = 0.0

                labor_cost = labor_value * plan["total_output"][i]

                # Check if final_demand exists
                final_demand_data = plan.get("final_demand", [0.0] * len(plan["total_output"]))
                # Handle both numpy arrays and lists
                if hasattr(final_demand_data, 'tolist'):
                    final_demand = final_demand_data[i]
                else:
                    final_demand = final_demand_data[i]

                self.sector_tree.insert(
                    "",
                    "end",
                    text = f"Sector {i}",
                    values=(
                        f"{plan['total_output'][i]:.2f}",
                        f"{final_demand:.2f}",
                        f"{labor_value:.4f}",
                        f"{labor_cost:.2f}",
                    ),
                )
        elif isinstance(plan, dict) and all(isinstance(k, int) for k in plan.keys()):
            # Multi - year plan (5 - year plan) - use first year's data
            first_year = min(plan.keys())
            year_data = plan[first_year]

            print(f"DEBUG: Multi - year plan detected, using year {first_year}")
            print(f"DEBUG: Year data keys: {list(year_data.keys())}")
            if "final_demand" in year_data:
                final_demand_data = year_data["final_demand"]
                print(f"DEBUG: final_demand type: {type(final_demand_data)}")
                print(f"DEBUG: final_demand shape / length: {getattr(final_demand_data, 'shape', len(final_demand_data))}")
                print(f"DEBUG: final_demand first 5 values: {final_demand_data[:5] if hasattr(final_demand_data, '__getitem__') else 'No indexing'}")
            else:
                print("DEBUG: No 'final_demand' key in year data")

            for i in range(len(year_data["total_output"])):
                # Check if labor_values exists, otherwise use labor_vector
                if "labor_values" in year_data:
                    labor_value = year_data["labor_values"][i]
                elif "labor_vector" in year_data:
                    labor_value = year_data["labor_vector"][i]
                else:
                    labor_value = 0.0

                labor_cost = labor_value * year_data["total_output"][i]

                # Check if final_demand exists
                final_demand_data = year_data.get("final_demand", [0.0] * len(year_data["total_output"]))
                # Handle both numpy arrays and lists
                if hasattr(final_demand_data, 'tolist'):
                    final_demand = final_demand_data[i]
                else:
                    final_demand = final_demand_data[i]

                self.sector_tree.insert(
                    "",
                    "end",
                    text = f"Sector {i}",
                    values=(
                        f"{year_data['total_output'][i]:.2f}",
                        f"{final_demand:.2f}",
                        f"{labor_value:.4f}",
                        f"{labor_cost:.2f}",
                    ),
                )
        else:
            # Five year plan - show first year
            first_year = min(self.current_plan.keys())
            plan = self.current_plan[first_year]
            for i in range(len(plan["total_output"])):
                # Check if labor_values exists, otherwise use labor_vector
                if "labor_values" in plan:
                    labor_value = plan["labor_values"][i]
                elif "labor_vector" in plan:
                    labor_value = plan["labor_vector"][i]
                else:
                    labor_value = 0.0

                labor_cost = labor_value * plan["total_output"][i]

                # Check if final_demand exists
                final_demand_data = plan.get("final_demand", [0.0] * len(plan["total_output"]))
                # Handle both numpy arrays and lists
                if hasattr(final_demand_data, 'tolist'):
                    final_demand = final_demand_data[i]
                else:
                    final_demand = final_demand_data[i]

                self.sector_tree.insert(
                    "",
                    "end",
                    text = f"Sector {i}",
                    values=(
                        f"{plan['total_output'][i]:.2f}",
                        f"{final_demand:.2f}",
                        f"{labor_value:.4f}",
                        f"{labor_cost:.2f}",
                    ),
                )

    def update_report(self):
        """Update the full report display."""
        if not self.current_plan:
            return

        try:
            # Handle different plan structures
            plan = self.current_plan

            # Check if plan has nested structure (evaluation result)
            if isinstance(plan, dict) and "plan" in plan and isinstance(plan["plan"], dict):
                plan = plan["plan"]

            if isinstance(plan, dict) and "total_output" in plan:
                # Single year plan
                report = self.planning_system.generate_report()
            else:
                # Five year plan - generate report for first year
                first_year = min(self.current_plan.keys())
                year_data = self.current_plan[first_year]
                print(f"DEBUG REPORT: First year data keys: {list(year_data.keys())}")
                if "final_demand" in year_data:
                    final_demand_data = year_data["final_demand"]
                    print(f"DEBUG REPORT: final_demand type: {type(final_demand_data)}")
                    print(f"DEBUG REPORT: final_demand sum: {np.sum(final_demand_data)}")
                    print(f"DEBUG REPORT: final_demand first 5 values: {final_demand_data[:5]}")
                else:
                    print("DEBUG REPORT: No 'final_demand' key in year data")
                report = self.planning_system.generate_report(year_data)

            self.report_text.delete("1.0", tk.END)
            self.report_text.insert("1.0", report)
        except Exception as e:
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert("1.0", f"Error generating report: {str(e)}")

    def save_plan(self, format_type):
        """Save the current plan."""
        if not self.current_plan:
            messagebox.showerror("Error", "No plan to save")
            return

        file_path = filedialog.asksaveasfilename(
            title = f"Save Plan as {format_type.upper()}",
            defaultextension = f".{format_type}",
            filetypes=[(f"{format_type.upper()} files", f"*.{format_type}"), ("All files", "*.*")],
        )

        if file_path:
            try:
                if isinstance(self.current_plan, dict) and "total_output" in self.current_plan:
                    # Single year plan
                    self.planning_system.save_plan(file_path, format_type)
                else:
                    # Five year plan - save as JSON for now
                    with open(file_path, "w") as f:
                        json.dump(self.current_plan, f, indent = 2, default = self.json_serializer)

                self.export_status.config(text = f"Plan saved to {file_path}", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plan: {str(e)}")

    def export_plan_for_simulation(self):
        """Export the current plan in simulation - compatible format."""
        if not self.current_plan:
            messagebox.showerror("Error", "No plan to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Plan for Simulation",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            try:
                self.planning_system.export_plan_for_simulation(file_path)
                self.export_status.config(text = f"Simulation plan exported to {file_path}", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export simulation plan: {str(e)}")

    def export_data(self):
        """Export current data."""
        if not self.current_data:
            messagebox.showerror("Error", "No data to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Data", defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.planning_system.export_data(file_path, "json")
                self.export_status.config(text = f"Data exported to {file_path}", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def load_plan(self):
        """Load a plan from file."""
        file_path = filedialog.askopenfilename(
            title="Load Plan", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.planning_system.load_plan(file_path)
                self.current_plan = self.planning_system.current_plan
                self.update_results_display()
                self.export_status.config(text="Plan loaded successfully", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load plan: {str(e)}")

    def json_serializer(self, obj):
        """JSON serializer for numpy arrays and custom objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'value'):
            # Handle enum objects like ValidationStatus FIRST
            return obj.value
        elif hasattr(obj, '__dict__'):
            # Handle dataclass objects like ValidationResult
            return {key: self.json_serializer(value) for key, value in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            # Handle lists / tuples that might contain non - serializable objects
            return [self.json_serializer(item) for item in obj]
        elif isinstance(obj, dict):
            # Handle dictionaries that might contain non - serializable objects
            return {key: self.json_serializer(value) for key, value in obj.items()}
        elif isinstance(obj, (str, int, float, bool, type(None))):
            # Handle basic JSON - serializable types
            return obj
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def export_marxist_analysis(self):
        """Export Marxist analysis results."""
        if not hasattr(self, 'marxist_text') or not self.marxist_text.get("1.0", tk.END).strip():
            messagebox.showwarning("Warning", "No analysis to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Marxist Analysis",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.marxist_text.get("1.0", tk.END))
                messagebox.showinfo("Success", f"Analysis exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export analysis: {str(e)}")

    def export_validation_report(self):
        """Export validation report."""
        if not hasattr(self, 'validation_text') or not self.validation_text.get("1.0", tk.END).strip():
            messagebox.showwarning("Warning", "No validation to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Validation Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.validation_text.get("1.0", tk.END))
                messagebox.showinfo("Success", f"Report exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")

    def _format_analysis_data(self, data, title):
        """Format analysis data for display."""
        if not data:
            return f"{title}\n\nNo data available."

        if "error" in data:
            return f"{title}\n\nError: {data['error']}"

        # Format the data nicely
        text = f"{title}\n{'='*50}\n\n"

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text += f"{key}:\n"
                    text += json.dumps(value, indent = 2, default = str) + "\n\n"
                else:
                    text += f"{key}: {value}\n"
        else:
            text += str(data)

        return text

    # Simulation System Methods
    def browse_plan_file(self):
        """Browse for a plan file to load."""
        file_path = filedialog.askopenfilename(
            title="Select Economic Plan File",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.plan_file_var.set(file_path)

    def load_simulation_plan(self):
        """Load a plan for simulation."""
        plan_file = self.plan_file_var.get()
        if not plan_file:
            messagebox.showerror("Error", "Please select a plan file first.")
            return

        try:
            with open(plan_file, 'r') as f:
                plan_data = json.load(f)

            # Check if this is a planning system plan or simulation plan
            if 'total_output' in plan_data and 'technology_matrix' in plan_data:
                # This is a planning system plan - convert it to simulation format
                plan_data = self._convert_planning_plan_to_simulation(plan_data)

            # Validate plan structure
            required_keys = ['sectors', 'production_targets', 'labor_requirements', 'resource_allocations']
            if not all(key in plan_data for key in required_keys):
                messagebox.showerror("Error", "Invalid plan file format. Missing required keys.")
                return

            self.current_simulation_plan = plan_data
            self.plan_status.config(text = f"Plan loaded: {os.path.basename(plan_file)}", foreground="green")

            # Update simulation parameters based on plan
            if 'sectors' in plan_data:
                self.sectors_var.set(str(len(plan_data['sectors'])))

            messagebox.showinfo("Success", "Plan loaded successfully for simulation.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load plan: {str(e)}")
            self.plan_status.config(text="Failed to load plan", foreground="red")

    def _convert_planning_plan_to_simulation(self, planning_plan):
        """Convert a planning system plan to simulation format."""

        # Handle multi - year plans by using the first year
        if isinstance(planning_plan, dict) and all(str(i).isdigit() for i in planning_plan.keys()):
            # This is a multi - year plan, use the first year
            first_year = min(int(k) for k in planning_plan.keys())
            plan_data = planning_plan[first_year]  # Use integer key, not string
            print(f"DEBUG: Multi - year plan detected, using year {first_year}")
        else:
            plan_data = planning_plan

        # Extract data from planning plan
        total_output = np.array(plan_data.get('total_output', []))
        labor_vector = np.array(plan_data.get('labor_vector', []))
        technology_matrix = np.array(plan_data.get('technology_matrix', []))
        final_demand = np.array(plan_data.get('final_demand', []))

        # Get sector count
        n_sectors = len(total_output)

        # Get sectors from current_data if available, otherwise create default names
        if hasattr(self, 'current_data') and 'sectors' in self.current_data:
            sectors = self.current_data['sectors']
            print(f"DEBUG: Using sectors from current_data: {len(sectors)} sectors")
        else:
            sectors = [f"Sector_{i + 1}" for i in range(n_sectors)]
            print(f"DEBUG: Using default sector names: {len(sectors)} sectors")

        # Convert to simulation format
        simulation_plan = {
            'sectors': sectors,
            'production_targets': total_output.tolist(),
            'labor_requirements': labor_vector.tolist(),
            'resource_allocations': {
                'technology_matrix': technology_matrix.tolist(),
                'final_demand': final_demand.tolist(),
                'total_labor_cost': plan_data.get('total_labor_cost', 0),
                'plan_quality_score': plan_data.get('plan_quality_score', 0)
            },
            'plan_metadata': {
                'year': plan_data.get('year', 1),
                'iteration': plan_data.get('iteration', 1),
                'status': plan_data.get('status', 'unknown'),
                'validation': plan_data.get('validation', {}),
                'constraint_violations': plan_data.get('constraint_violations', {}),
                'cybernetic_feedback': plan_data.get('cybernetic_feedback', {})
            }
        }

        return simulation_plan

    def reload_current_plan(self):
        """Reload the current planning system plan into the simulator."""
        if not self.current_plan:
            messagebox.showerror("Error", "No current plan to reload")
            return

        try:
            # Convert the current plan to simulation format
            simulation_plan = self._convert_planning_plan_to_simulation(self.current_plan)

            # Load it into the simulation system
            self.current_simulation_plan = simulation_plan

            # Update simulation parameters based on the plan
            if 'sectors' in simulation_plan:
                self.sectors_var.set(str(len(simulation_plan['sectors'])))

            # Update the plan status
            self.plan_status.config(text="Plan reloaded from planning system", foreground="green")

            messagebox.showinfo("Success", f"Plan reloaded successfully with {len(simulation_plan['sectors'])} sectors")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload plan: {str(e)}")
            self.plan_status.config(text="Failed to reload plan", foreground="red")

    def initialize_simulation(self):
        """Initialize the simulation environment."""
        if not hasattr(self, 'current_simulation_plan') or not self.current_simulation_plan:
            messagebox.showerror("Error", "Please load a plan first.")
            return

        try:
            # Get simulation parameters
            duration = int(self.sim_duration_var.get())
            time_step = int(self.time_step_var.get())
            map_size = int(self.map_size_var.get())
            settlements = int(self.settlements_var.get())
            sectors = int(self.sectors_var.get())
            pop_density = int(self.pop_density_var.get())

            # Initialize simulation environment
            self.simulation_environment = {
                'duration_years': duration,
                'time_step_months': time_step,
                'map_size_km': map_size,
                'settlements': settlements,
                'economic_sectors': sectors,
                'population_density': pop_density,
                'current_time': 0,
                'current_month': 0,
                'current_year': 0
            }

            # Initialize stochastic events
            self.stochastic_events = {
                'natural_disasters': self.natural_disasters_var.get(),
                'economic_disruptions': self.economic_disruptions_var.get(),
                'infrastructure_failures': self.infrastructure_failures_var.get(),
                'frequency_per_year': float(self.event_frequency_var.get())
            }

            # Initialize simulation state
            self.simulation_state = "initialized"
            self.simulation_status.config(text="Simulation initialized", foreground="green")

            # Clear previous results
            self.monitoring_text.delete(1.0, tk.END)
            self.events_text.delete(1.0, tk.END)
            self.metrics_tree.delete(*self.metrics_tree.get_children())

            # Add initialization message
            init_message = f"""Simulation Initialized Successfully!

Environment Parameters:
- Duration: {duration} years - Time Step: {time_step} months - Map Size: {map_size} km - Settlements: {settlements}
- Economic Sectors: {sectors}
- Population Density: {pop_density} per km²

Plan Loaded:
- Sectors: {len(self.current_simulation_plan.get('sectors', []))}
- Production Targets: {len(self.current_simulation_plan.get('production_targets', []))}

Ready to start simulation.
"""
            self.monitoring_text.insert(tk.END, init_message)

            messagebox.showinfo("Success", "Simulation initialized successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize simulation: {str(e)}")
            self.simulation_status.config(text="Initialization failed", foreground="red")

    def start_simulation(self):
        """Start the simulation."""
        if self.simulation_state != "initialized":
            messagebox.showerror("Error", "Please initialize the simulation first.")
            return

        if self.simulation_state == "running":
            messagebox.showinfo("Info", "Simulation is already running.")
            return

        try:
            self.simulation_state = "running"
            self.simulation_status.config(text="Simulation running...", foreground="blue")

            # Start simulation in a separate thread
            self.simulation_thread = threading.Thread(target = self.run_simulation, daemon = True)
            self.simulation_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {str(e)}")
            self.simulation_state = "initialized"
            self.simulation_status.config(text="Start failed", foreground="red")

    def run_simulation(self):
        """Run the simulation in a separate thread."""
        try:
            total_months = self.simulation_environment['duration_years'] * 12
            time_step = self.simulation_environment['time_step_months']

            for month in range(0, total_months, time_step):
                if self.simulation_state != "running":
                    break

                # Update time
                self.simulation_environment['current_month'] = month
                self.simulation_environment['current_year'] = month // 12
                self.simulation_environment['current_time'] = month

                # Update progress
                progress = (month / total_months) * 100
                self.root.after(0, lambda p = progress: self.simulation_progress.config(value = p))

                # Run simulation step
                self.simulate_time_step(month)

                # Check for stochastic events
                if self.should_trigger_event():
                    self.trigger_stochastic_event(month)

                # Update monitoring display
                self.update_monitoring_display(month)

                # Small delay to prevent UI freezing
                import time
                time.sleep(0.1)

            # Simulation completed
            self.root.after(0, self.simulation_completed)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.simulation_failed(error_msg))

    def simulate_time_step(self, month):
        """Simulate one time step of the simulation."""
        year = month // 12
        month_in_year = month % 12

        # Basic economic simulation based on loaded plan
        if hasattr(self, 'current_simulation_plan'):
            # Simulate production based on plan targets
            production_results = self.simulate_production(month)

            # Simulate resource allocation
            resource_results = self.simulate_resource_allocation(month)

            # Simulate labor allocation
            labor_results = self.simulate_labor_allocation(month)

            # Store results
            if not hasattr(self, 'simulation_results'):
                self.simulation_results = []

            self.simulation_results.append({
                'month': month,
                'year': year,
                'month_in_year': month_in_year,
                'production': production_results,
                'resources': resource_results,
                'labor': labor_results
            })

    def simulate_production(self, month):
        """Simulate production for the current time step."""
        # Basic production simulation based on plan targets
        production_results = {}

        if 'production_targets' in self.current_simulation_plan and 'sectors' in self.current_simulation_plan:
            targets = self.current_simulation_plan['production_targets']
            sectors = self.current_simulation_plan['sectors']

            # Add some variation based on time and stochastic events
            variation = 1.0
            if hasattr(self, 'current_events') and self.current_events:
                variation *= 0.8  # Reduce production during events

            for i, (sector, target) in enumerate(zip(sectors, targets)):
                # Seasonal variation (simplified)
                seasonal_factor = 1.0 + 0.1 * math.sin(2 * math.pi * (month % 12) / 12)

                actual_production = target * variation * seasonal_factor
                production_results[sector] = {
                    'target': target,
                    'actual': actual_production,
                    'efficiency': actual_production / target if target > 0 else 0
                }

        return production_results

    def simulate_resource_allocation(self, month):
        """Simulate resource allocation for the current time step."""
        resource_results = {}

        if 'resource_allocations' in self.current_simulation_plan:
            # resource_allocations is a dictionary, so we can iterate over it
            for resource, allocation in self.current_simulation_plan['resource_allocations'].items():
                # Simulate resource availability and distribution
                availability = 1.0
                if hasattr(self, 'current_events') and self.current_events:
                    availability *= 0.9  # Reduce availability during events

                # Handle both scalar and list allocations
                if isinstance(allocation, list):
                    actual_allocation = []
                    for a in allocation:
                        if isinstance(a, (int, float)):
                            actual_allocation.append(a * availability)
                        else:
                            actual_allocation.append(a)  # Keep non-numeric values unchanged
                else:
                    if isinstance(allocation, (int, float)):
                        actual_allocation = allocation * availability
                    else:
                        actual_allocation = allocation  # Keep non-numeric values unchanged

                resource_results[resource] = {
                    'planned': allocation,
                    'actual': actual_allocation,
                    'availability': availability
                }

        return resource_results

    def simulate_labor_allocation(self, month):
        """Simulate labor allocation for the current time step."""
        labor_results = {}

        if 'labor_requirements' in self.current_simulation_plan and 'sectors' in self.current_simulation_plan:
            requirements = self.current_simulation_plan['labor_requirements']
            sectors = self.current_simulation_plan['sectors']

            # Simulate labor productivity and availability
            productivity = 1.0
            if hasattr(self, 'current_events') and self.current_events:
                productivity *= 0.95  # Slight reduction during events

            for i, (sector, requirement) in enumerate(zip(sectors, requirements)):
                actual_labor = requirement * productivity
                labor_results[sector] = {
                    'required': requirement,
                    'allocated': actual_labor,
                    'productivity': productivity
                }

        return labor_results

    def should_trigger_event(self):
        """Determine if a stochastic event should be triggered."""
        if not self.stochastic_events['frequency_per_year']:
            return False

        # Simple probability calculation
        import random
        probability = self.stochastic_events['frequency_per_year'] / 12  # Monthly probability
        return random.random() < probability

    def trigger_stochastic_event(self, month):
        """Trigger a stochastic event."""
        if not hasattr(self, 'current_events'):
            self.current_events = []

        # Select event type
        event_types = []
        if self.stochastic_events['natural_disasters']:
            event_types.append('natural_disaster')
        if self.stochastic_events['economic_disruptions']:
            event_types.append('economic_disruption')
        if self.stochastic_events['infrastructure_failures']:
            event_types.append('infrastructure_failure')

        if not event_types:
            return

        import random
        event_type = random.choice(event_types)

        # Create event
        event = {
            'type': event_type,
            'month': month,
            'year': month // 12,
            'severity': random.uniform(0.1, 1.0),
            'duration': random.randint(1, 6)  # Duration in months
        }

        self.current_events.append(event)

        # Log event
        event_message = f"Month {month} (Year {event['year']}): {event_type.replace('_', ' ').title()} - Severity: {event['severity']:.2f}, Duration: {event['duration']} months\n"
        self.root.after(0, lambda: self.events_text.insert(tk.END, event_message))

    def update_monitoring_display(self, month):
        """Update the real - time monitoring display."""
        if not hasattr(self, 'simulation_results') or not self.simulation_results:
            return

        latest_result = self.simulation_results[-1]

        # Create monitoring message
        monitoring_message = f"""=== Simulation Update - Month {month} (Year {latest_result['year']}) ===

Production Status:
"""

        for sector, data in latest_result['production'].items():
            monitoring_message += f"  {sector}: {data['actual']:.2f} / {data['target']:.2f} ({data['efficiency']:.1%})\n"

        monitoring_message += "\nResource Allocation:\n"
        for resource, data in latest_result['resources'].items():
            # Handle both scalar and list values for actual and planned allocations
            actual_str = str(data['actual']) if isinstance(data['actual'], list) else f"{data['actual']:.2f}"
            planned_str = str(data['planned']) if isinstance(data['planned'], list) else f"{data['planned']:.2f}"
            monitoring_message += f"  {resource}: {actual_str} / {planned_str} ({data['availability']:.1%})\n"

        monitoring_message += "\nLabor Allocation:\n"
        for sector, data in latest_result['labor'].items():
            monitoring_message += f"  {sector}: {data['allocated']:.2f} / {data['required']:.2f} ({data['productivity']:.1%})\n"

        if hasattr(self, 'current_events') and self.current_events:
            monitoring_message += f"\nActive Events: {len(self.current_events)}\n"
            for event in self.current_events[-3:]:  # Show last 3 events
                monitoring_message += f"  - {event['type'].replace('_', ' ').title()} (Severity: {event['severity']:.2f})\n"

        monitoring_message += "\n" + "="*50 + "\n\n"

        # Update display
        self.root.after(0, lambda: self.monitoring_text.insert(tk.END, monitoring_message))
        self.root.after(0, lambda: self.monitoring_text.see(tk.END))

    def pause_simulation(self):
        """Pause the simulation."""
        if self.simulation_state == "running":
            self.simulation_state = "paused"
            self.simulation_status.config(text="Simulation paused", foreground="orange")

    def stop_simulation(self):
        """Stop the simulation."""
        self.simulation_state = "stopped"
        self.simulation_status.config(text="Simulation stopped", foreground="red")
        self.simulation_progress.config(value = 0)

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.simulation_state = "stopped"
        self.simulation_status.config(text="Ready to initialize", foreground="blue")
        self.simulation_progress.config(value = 0)

        # Clear results
        self.monitoring_text.delete(1.0, tk.END)
        self.events_text.delete(1.0, tk.END)
        self.metrics_tree.delete(*self.metrics_tree.get_children())

        # Reset simulation data
        if hasattr(self, 'simulation_results'):
            delattr(self, 'simulation_results')
        if hasattr(self, 'current_events'):
            delattr(self, 'current_events')

    def simulation_completed(self):
        """Handle simulation completion."""
        self.simulation_state = "completed"
        self.simulation_status.config(text="Simulation completed", foreground="green")
        self.simulation_progress.config(value = 100)

        # Update metrics display
        self.update_metrics_display()

        messagebox.showinfo("Simulation Complete", "The simulation has completed successfully!")

    def simulation_failed(self, error_msg):
        """Handle simulation failure."""
        self.simulation_state = "failed"
        self.simulation_status.config(text="Simulation failed", foreground="red")
        messagebox.showerror("Simulation Error", f"Simulation failed: {error_msg}")

    def update_metrics_display(self):
        """Update the performance metrics display."""
        if not hasattr(self, 'simulation_results') or not self.simulation_results:
            return

        # Clear existing metrics
        self.metrics_tree.delete(*self.metrics_tree.get_children())

        # Calculate overall metrics
        total_months = len(self.simulation_results)
        avg_production_efficiency = 0
        avg_resource_availability = 0
        avg_labor_productivity = 0

        for result in self.simulation_results:
            if result['production']:
                avg_production_efficiency += sum(data['efficiency'] for data in result['production'].values()) / len(result['production'])
            if result['resources']:
                avg_resource_availability += sum(data['availability'] for data in result['resources'].values()) / len(result['resources'])
            if result['labor']:
                avg_labor_productivity += sum(data['productivity'] for data in result['labor'].values()) / len(result['labor'])

        if total_months > 0:
            avg_production_efficiency /= total_months
            avg_resource_availability /= total_months
            avg_labor_productivity /= total_months

        # Add metrics to tree
        metrics = [
            ("Production Efficiency", f"{avg_production_efficiency:.1%}", "100%", "Good" if avg_production_efficiency > 0.8 else "Needs Improvement"),
            ("Resource Availability", f"{avg_resource_availability:.1%}", "100%", "Good" if avg_resource_availability > 0.9 else "Needs Improvement"),
            ("Labor Productivity", f"{avg_labor_productivity:.1%}", "100%", "Good" if avg_labor_productivity > 0.9 else "Needs Improvement"),
            ("Simulation Duration", f"{total_months} months", f"{self.simulation_environment['duration_years'] * 12} months", "Complete"),
            ("Events Triggered", f"{len(getattr(self, 'current_events', []))}", "Variable", "Normal")
        ]

        for metric, value, target, status in metrics:
            self.metrics_tree.insert("", "end", values=(metric, value, target, status))

    def generate_simulation_map(self):
        """Generate an interactive map for the simulation environment."""
        if not MAP_AVAILABLE:
            messagebox.showerror("Error", "Map visualization is not available. Please install required dependencies.")
            return

        try:
            # Get simulation parameters
            map_size = float(self.map_size_var.get())
            settlements_count = int(self.settlements_var.get())

            # Calculate map bounds based on size
            # Convert km to approximate lat / lon degrees (rough approximation)
            lat_range = map_size / 111.0  # 1 degree latitude ≈ 111 km
            lon_range = map_size / (111.0 * np.cos(np.radians(45)))  # Adjust for longitude

            # Center around a reasonable location (Northeast US)
            center_lat, center_lon = 45.0, -75.0
            map_bounds = (
                (center_lat - lat_range / 2, center_lon - lon_range / 2),
                (center_lat + lat_range / 2, center_lon + lon_range / 2)
            )

            # Generate the map
            self.current_map = create_simulation_map(map_bounds)

            # Save the map
            self.map_file_path = self.current_map.save_map("simulation_map.html")

            # Update status
            self.map_status.config(text="Map generated successfully", foreground="green")

            # Update map info display
            self.update_map_info_display()

            messagebox.showinfo("Success", f"Simulation map generated successfully!\nSaved to: {self.map_file_path}")

        except Exception as e:
            self.map_status.config(text="Map generation failed", foreground="red")
            messagebox.showerror("Error", f"Failed to generate map: {str(e)}")

    def open_map_in_browser(self):
        """Open the generated map in the default web browser."""
        if not self.current_map:
            messagebox.showwarning("Warning", "No map generated. Please generate a map first.")
            return

        try:
            if self.map_file_path and os.path.exists(self.map_file_path):
                import webbrowser
                webbrowser.open(f'file://{os.path.abspath(self.map_file_path)}')
                self.map_status.config(text="Map opened in browser", foreground="green")
            else:
                # Generate a new map if file doesn't exist
                self.generate_simulation_map()
                if self.map_file_path:
                    import webbrowser
                    webbrowser.open(f'file://{os.path.abspath(self.map_file_path)}')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open map in browser: {str(e)}")

    def refresh_simulation_map(self):
        """Refresh the current simulation map."""
        if not self.current_map:
            messagebox.showwarning("Warning", "No map to refresh. Please generate a map first.")
            return

        try:
            # Regenerate the map with current parameters
            self.generate_simulation_map()
            messagebox.showinfo("Success", "Map refreshed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh map: {str(e)}")

    def generate_realtime_map(self):
        """Generate a real - time map for the simulation environment."""
        if not MAP_AVAILABLE:
            messagebox.showerror("Error", "Map visualization is not available. Please install required dependencies.")
            return

        try:
            # Get simulation parameters
            map_size = float(self.map_size_var.get())
            settlements_count = int(self.settlements_var.get())

            # Calculate map bounds based on size
            center_lat, center_lon = 45.0, -75.0  # Default center
            lat_range = map_size / 111.0  # Approximate km per degree latitude
            lon_range = map_size / (111.0 * math.cos(math.radians(center_lat)))  # Adjust for longitude

            map_bounds = (
                (center_lat - lat_range / 2, center_lon - lon_range / 2),
                (center_lat + lat_range / 2, center_lon + lon_range / 2)
            )

            # Generate the map
            self.realtime_map = create_simulation_map(map_bounds)

            # Save the map
            self.realtime_map_file_path = "realtime_simulation_map.html"
            self.realtime_map.save_map(self.realtime_map_file_path)

            # Update status
            self.realtime_map_status.config(text="Map generated", foreground="green")

            # Update map info display
            self.update_realtime_map_info()

            # Display the map in the GUI
            self.display_map_in_gui()

            messagebox.showinfo("Success", "Real - time map generated successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate real - time map: {str(e)}")
            self.realtime_map_status.config(text="Map generation failed", foreground="red")

    def update_realtime_map_info(self):
        """Update the real - time map information display."""
        if not self.realtime_map:
            self.realtime_map_info_text.delete("1.0", tk.END)
            self.realtime_map_info_text.insert("1.0", "No map generated. Click 'Generate Map' to create a simulation environment.")
            return

        # Get environment data
        env_data = self.realtime_map.get_environment_data()

        # Create info text
        info_text = f"=== Real - time Simulation Map ===\n\n"
        info_text += f"Simulation Time: {self.current_simulation_time:.2f} months\n"
        info_text += f"Update Speed: {self.map_speed_var.get()}\n"
        info_text += f"Map Status: {'Active' if self.map_update_active else 'Inactive'}\n\n"

        info_text += f"=== Environment Statistics ===\n"
        info_text += f"Geographic Features: {len(env_data['geographic_features'])}\n"
        info_text += f"Settlements: {len(env_data['settlements'])}\n"
        info_text += f"Economic Zones: {len(env_data['economic_zones'])}\n"
        info_text += f"Infrastructure: {len(env_data['infrastructure'])}\n\n"

        # Settlement breakdown
        settlements = env_data['settlements']
        cities = [s for s in settlements if s['settlement_type'] == 'city']
        towns = [s for s in settlements if s['settlement_type'] == 'town']
        rural = [s for s in settlements if s['settlement_type'] == 'rural']

        info_text += f"=== Settlement Breakdown ===\n"
        info_text += f"Cities: {len(cities)}\n"
        info_text += f"Towns: {len(towns)}\n"
        info_text += f"Rural Areas: {len(rural)}\n"
        info_text += f"Total Population: {sum(s['population'] for s in settlements):,}\n\n"

        # Economic zones breakdown
        zones = env_data['economic_zones']
        zone_types = {}
        for zone in zones:
            zone_type = zone['zone_type']
            zone_types[zone_type] = zone_types.get(zone_type, 0) + 1

        info_text += f"=== Economic Zones ===\n"
        for zone_type, count in zone_types.items():
            info_text += f"{zone_type.title()}: {count}\n"

        info_text += f"\n=== Infrastructure ===\n"
        infra = env_data['infrastructure']
        road_count = len([i for i in infra if i['infrastructure_type'] == 'road'])
        rail_count = len([i for i in infra if i['infrastructure_type'] == 'railway'])
        info_text += f"Roads: {road_count}\n"
        info_text += f"Railways: {rail_count}\n"

        # Update the display
        self.realtime_map_info_text.delete("1.0", tk.END)
        self.realtime_map_info_text.insert("1.0", info_text)

    def display_map_in_gui(self):
        """Display the interactive map in the GUI."""
        if not self.realtime_map or not self.realtime_map_file_path:
            self.map_browser_widget.preview_text.delete("1.0", tk.END)
            self.map_browser_widget.preview_text.insert("1.0", "No map available. Click 'Generate Map' to create a simulation environment.")
            self.map_browser_widget.status_label.config(text="No map loaded")
            return

        try:
            # Load the HTML file into the web browser widget
            success = self.map_browser_widget.load_html_file(self.realtime_map_file_path)
            if success:
                self.realtime_map_status.config(text="Map displayed in GUI", foreground="green")
            else:
                self.realtime_map_status.config(text="Map display failed", foreground="red")

        except Exception as e:
            self.map_browser_widget.preview_text.delete("1.0", tk.END)
            self.map_browser_widget.preview_text.insert("1.0", f"Error loading map: {str(e)}")
            self.map_browser_widget.status_label.config(text="Error loading map")

    def refresh_map_display(self):
        """Refresh the map display with updated data."""
        if not self.realtime_map:
            return

        try:
            # Regenerate the map with updated data
            self.realtime_map.create_map()
            self.realtime_map.save_map(self.realtime_map_file_path)

            # Refresh the web browser widget
            self.map_browser_widget.refresh()

        except Exception as e:
            print(f"Error refreshing map display: {e}")

    def open_realtime_map_in_browser(self):
        """Open the real - time map in the default web browser."""
        if not self.realtime_map or not self.realtime_map_file_path:
            messagebox.showwarning("Warning", "No map generated. Please generate a map first.")
            return

        try:
            if os.path.exists(self.realtime_map_file_path):
                webbrowser.open(f'file://{os.path.abspath(self.realtime_map_file_path)}')
                self.realtime_map_status.config(text="Map opened in browser", foreground="green")
            else:
                messagebox.showerror("Error", "Map file not found. Please regenerate the map.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open map in browser: {str(e)}")

    def start_map_updates(self):
        """Start real - time map updates."""
        if not self.realtime_map:
            messagebox.showwarning("Warning", "No map generated. Please generate a map first.")
            return

        if self.map_update_active:
            messagebox.showinfo("Info", "Map updates are already running.")
            return

        # Calculate update interval based on speed setting
        speed_setting = self.map_speed_var.get()
        if speed_setting == "1 hour / sec":
            self.map_update_interval = 1.0  # 1 second = 1 hour
        elif speed_setting == "1 day / sec":
            self.map_update_interval = 1.0  # 1 second = 1 day
        elif speed_setting == "1 month / sec":
            self.map_update_interval = 1.0  # 1 second = 1 month
        elif speed_setting == "1 year / sec":
            self.map_update_interval = 1.0  # 1 second = 1 year

        self.map_update_active = True
        self.realtime_map_status.config(text="Updates active", foreground="green")

        # Start the update timer
        self.schedule_map_update()

    def pause_map_updates(self):
        """Pause real - time map updates."""
        if not self.map_update_active:
            messagebox.showinfo("Info", "Map updates are not running.")
            return

        self.map_update_active = False
        if self.map_update_timer:
            self.root.after_cancel(self.map_update_timer)
            self.map_update_timer = None

        self.realtime_map_status.config(text="Updates paused", foreground="orange")

    def stop_map_updates(self):
        """Stop real - time map updates."""
        self.map_update_active = False
        if self.map_update_timer:
            self.root.after_cancel(self.map_update_timer)
            self.map_update_timer = None

        self.realtime_map_status.config(text="Updates stopped", foreground="red")
        self.current_simulation_time = 0

    def schedule_map_update(self):
        """Schedule the next map update."""
        if not self.map_update_active:
            return

        # Update simulation time
        speed_setting = self.map_speed_var.get()
        if speed_setting == "1 hour / sec":
            self.current_simulation_time += 1 / 24  # 1 hour in months
        elif speed_setting == "1 day / sec":
            self.current_simulation_time += 1 / 30  # 1 day in months
        elif speed_setting == "1 month / sec":
            self.current_simulation_time += 1  # 1 month
        elif speed_setting == "1 year / sec":
            self.current_simulation_time += 12  # 1 year in months

        # Update the map with new data
        self.update_realtime_map_data()

        # Update the display
        self.update_realtime_map_info()

        # Refresh the map display
        self.refresh_map_display()

        # Schedule next update
        self.map_update_timer = self.root.after(int(self.map_update_interval * 1000), self.schedule_map_update)

    def update_realtime_map_data(self):
        """Update the map data based on simulation progress."""
        if not self.realtime_map:
            return

        # Simulate changes over time
        # This is where you would integrate with actual simulation data
        # For now, we'll simulate some basic changes

        # Update settlement populations (simulate growth)
        for settlement in self.realtime_map.settlements:
            # Simulate population growth
            growth_rate = 0.001  # 0.1% per month
            settlement.population = int(settlement.population * (1 + growth_rate))

        # Update economic zone production capacity
        for zone in self.realtime_map.economic_zones:
            # Simulate production changes
            change_rate = random.uniform(-0.01, 0.02)  # -1% to + 2% per month
            zone.production_capacity *= (1 + change_rate)

        # Simulate infrastructure changes
        for infra in self.realtime_map.infrastructure:
            # Simulate capacity changes
            change_rate = random.uniform(-0.005, 0.01)  # -0.5% to + 1% per month
            infra.capacity *= (1 + change_rate)

    def update_map_info_display(self):
        """Update the map information display."""
        if not self.current_map:
            self.map_info_text.delete("1.0", tk.END)
            self.map_info_text.insert("1.0", "No map generated. Click 'Generate Map' to create a simulation environment.")
            return

        # Get environment data
        env_data = self.current_map.get_environment_data()

        # Create info text
        info_text = f"""Simulation Environment Map
============================

Geographic Features: {len(env_data['geographic_features'])}
- Water Bodies: {len([f for f in env_data['geographic_features'] if f['feature_type'] == 'water'])}
- Mountain Ranges: {len([f for f in env_data['geographic_features'] if f['feature_type'] == 'mountain'])}
- Forest Areas: {len([f for f in env_data['geographic_features'] if f['feature_type'] == 'forest'])}

Settlements: {len(env_data['settlements'])}
- Cities: {len([s for s in env_data['settlements'] if s['settlement_type'] == 'city'])}
- Towns: {len([s for s in env_data['settlements'] if s['settlement_type'] == 'town'])}
- Rural Areas: {len([s for s in env_data['settlements'] if s['settlement_type'] == 'rural'])}

Economic Zones: {len(env_data['economic_zones'])}
- Industrial: {len([z for z in env_data['economic_zones'] if z['zone_type'] == 'industrial'])}
- Agricultural: {len([z for z in env_data['economic_zones'] if z['zone_type'] == 'agricultural'])}
- Mixed: {len([z for z in env_data['economic_zones'] if z['zone_type'] == 'mixed'])}

Infrastructure: {len(env_data['infrastructure'])}
- Roads: {len([i for i in env_data['infrastructure'] if i['infrastructure_type'] == 'road'])}
- Railways: {len([i for i in env_data['infrastructure'] if i['infrastructure_type'] == 'railway'])}

Map File: {self.map_file_path or 'Not saved'}

Click 'Open in Browser' to view the interactive map with all features, settlements, and infrastructure networks.
"""

        self.map_info_text.delete("1.0", tk.END)
        self.map_info_text.insert("1.0", info_text)

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    CyberneticPlanningGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
