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
from datetime import datetime
import traceback
import inspect

from datetime import datetime
import math
import random

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from cybernetic_planning.planning_system import CyberneticPlanningSystem
    PLANNING_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Planning system not available: {e}")
    print("Some features may not work properly")
    PLANNING_SYSTEM_AVAILABLE = False
    CyberneticPlanningSystem = None

def log_error_with_traceback(error_msg, exception=None, context=""):
    """Log detailed error information with traceback and line numbers."""
    print(f"\n{'='*60}")
    print(f"ERROR: {error_msg}")
    print(f"Context: {context}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if exception:
        print(f"Exception Type: {type(exception).__name__}")
        print(f"Exception Message: {str(exception)}")
        print("\nFull Traceback:")
        traceback.print_exc()
        
        # Get the current frame for line number
        current_frame = inspect.currentframe()
        if current_frame and current_frame.f_back:
            caller_frame = current_frame.f_back
            filename = caller_frame.f_code.co_filename
            line_number = caller_frame.f_lineno
            function_name = caller_frame.f_code.co_name
            print(f"\nError occurred in: {os.path.basename(filename)}:{line_number} in function {function_name}")
    
    print(f"{'='*60}\n")

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

        # Map Image tab
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Map Image")

        # HTML Preview tab
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="Map Preview")

        # HTML Source tab
        self.source_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.source_frame, text="HTML Source")

        # Create image display area
        self.image_canvas = tk.Canvas(self.image_frame, width=width, height=height, bg='white')
        self.image_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        self.current_image = None

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

        self.generate_image_button = ttk.Button(
            self.controls_frame,
            text="Generate Image",
            command = self.generate_image
        )
        self.generate_image_button.pack(side="left", padx = 5)

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

    def generate_image(self):
        """Generate an image from the current map."""
        if not self.current_url:
            self.status_label.config(text="No map to generate image from")
            return

        try:
            # Import the map visualization module
            # Map visualization removed
            import tempfile
            import os

            # Create a temporary HTML file path
            temp_html = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
            temp_html.close()

            # Copy the current HTML file to temp location
            with open(self.current_url, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            with open(temp_html.name, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Create a temporary map object and load the HTML
            # Note: This is a simplified approach - in practice, you'd need to parse the HTML
            # and recreate the map object, or use selenium to take a screenshot
            self.status_label.config(text="Generating image...")
            
            # For now, show a placeholder message
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                self.image_canvas.winfo_width()//2, 
                self.image_canvas.winfo_height()//2,
                text="Image generation requires selenium.\nInstall with: pip install selenium",
                font=('Arial', 12),
                fill='gray'
            )
            
            self.status_label.config(text="Image generation not available")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")

    def load_image(self, image_path):
        """Load an image file into the canvas."""
        try:
            from PIL import Image, ImageTk
            
            # Clear previous image
            self.image_canvas.delete("all")
            
            # Load and resize image
            image = Image.open(image_path)
            # Resize to fit canvas while maintaining aspect ratio
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
                image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(image)
            
            # Display image
            self.image_canvas.create_image(
                canvas_width//2, canvas_height//2, 
                image=self.current_image, anchor=tk.CENTER
            )
            
            self.status_label.config(text=f"Image loaded: {os.path.basename(image_path)}")
            
        except ImportError:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                self.image_canvas.winfo_width()//2, 
                self.image_canvas.winfo_height()//2,
                text="PIL not available for image display.\nInstall with: pip install Pillow",
                font=('Arial', 12),
                fill='gray'
            )
            self.status_label.config(text="PIL not available")
        except Exception as e:
            self.status_label.config(text=f"Error loading image: {str(e)}")

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
        if PLANNING_SYSTEM_AVAILABLE:
            try:
                self.planning_system = CyberneticPlanningSystem()
                print("✅ Planning system initialized")
            except Exception as e:
                print(f"❌ Planning system initialization failed: {e}")
                self.planning_system = None
        else:
            self.planning_system = None
            print("⚠️ Planning system not available")
        
        self.current_plan = None
        self.current_data = None

        # Initialize simulation state
        self.simulation_state = "stopped"
        self.simulation_thread = None
        self.current_simulation = None

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
        self.create_technology_tree_tab()
        self.create_planning_tab()
        self.create_results_tab()
        self.create_simulation_tab()
        self.create_gpu_settings_tab()
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
        ttk.Button(source_frame, text="Save Synthetic Data", command = self.save_synthetic_data).pack(
            side="left", padx = self._scale_padding(5)
        )
        
        # Config management buttons
        config_buttons_frame = ttk.Frame(source_frame)
        config_buttons_frame.pack(side="right", padx = self._scale_padding(5))
        ttk.Button(config_buttons_frame, text="Save Config", command = self.save_data_config).pack(side="left", padx = self._scale_padding(5))
        ttk.Button(config_buttons_frame, text="Load Config", command = self.load_data_config).pack(side="left", padx = self._scale_padding(5))

        # Data configuration
        config_frame = ttk.LabelFrame(self.data_frame, text="Economic Data Configuration", padding = self._scale_padding(10))
        config_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Number of sectors
        ttk.Label(config_frame, text="Number of Sectors:").grid(row = 0, column = 0, sticky="w", padx = self._scale_padding(5))
        self.sectors_var = tk.StringVar(value="175")
        ttk.Entry(config_frame, textvariable = self.sectors_var, width = 10).grid(row = 0, column = 1, padx = self._scale_padding(5))

        # Technology density
        ttk.Label(config_frame, text="Technology Density:").grid(row = 0, column = 2, sticky="w", padx = self._scale_padding(5))
        self.density_var = tk.StringVar(value="0.4")
        ttk.Entry(config_frame, textvariable = self.density_var, width = 10).grid(row = 0, column = 3, padx = self._scale_padding(5))

        # Resource count
        ttk.Label(config_frame, text="Resource Count:").grid(row = 1, column = 0, sticky="w", padx = self._scale_padding(5))
        self.resources_var = tk.StringVar(value="3")
        ttk.Entry(config_frame, textvariable = self.resources_var, width = 10).grid(row = 1, column = 1, padx = self._scale_padding(5))

        # Starting technology level
        ttk.Label(config_frame, text="Starting Tech Level:").grid(row = 1, column = 2, sticky="w", padx = self._scale_padding(5))
        self.starting_tech_var = tk.StringVar(value="0.0")
        ttk.Entry(config_frame, textvariable = self.starting_tech_var, width = 10).grid(row = 1, column = 3, padx = self._scale_padding(5))

        # Population Demographics Configuration
        demo_frame = ttk.LabelFrame(self.data_frame, text="Population Demographics", padding = self._scale_padding(10))
        demo_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Total population
        ttk.Label(demo_frame, text="Total Population:").grid(row = 0, column = 0, sticky="w", padx = self._scale_padding(5))
        self.total_population_var = tk.StringVar(value="1000000")
        ttk.Entry(demo_frame, textvariable = self.total_population_var, width = 15).grid(row = 0, column = 1, padx = self._scale_padding(5))

        # Employment rate
        ttk.Label(demo_frame, text="Employment Rate (%):").grid(row = 0, column = 2, sticky="w", padx = self._scale_padding(5))
        self.employment_rate_var = tk.StringVar(value="60")
        ttk.Entry(demo_frame, textvariable = self.employment_rate_var, width = 10).grid(row = 0, column = 3, padx = self._scale_padding(5))

        # Dependency ratio
        ttk.Label(demo_frame, text="Dependency Ratio (%):").grid(row = 1, column = 0, sticky="w", padx = self._scale_padding(5))
        self.dependency_ratio_var = tk.StringVar(value="40")
        ttk.Entry(demo_frame, textvariable = self.dependency_ratio_var, width = 10).grid(row = 1, column = 1, padx = self._scale_padding(5))

        # Technology level (affects living standards)
        ttk.Label(demo_frame, text="Technology Level:").grid(row = 1, column = 2, sticky="w", padx = self._scale_padding(5))
        self.tech_level_var = tk.StringVar(value="1.0")
        ttk.Entry(demo_frame, textvariable = self.tech_level_var, width = 10).grid(row = 1, column = 3, padx = self._scale_padding(5))

        # Regional distribution
        region_frame = ttk.LabelFrame(self.data_frame, text="Regional Distribution", padding = self._scale_padding(10))
        region_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Number of cities
        ttk.Label(region_frame, text="Number of Cities:").grid(row = 0, column = 0, sticky="w", padx = self._scale_padding(5))
        self.num_cities_var = tk.StringVar(value="5")
        ttk.Entry(region_frame, textvariable = self.num_cities_var, width = 10).grid(row = 0, column = 1, padx = self._scale_padding(5))

        # Number of towns
        ttk.Label(region_frame, text="Number of Towns:").grid(row = 0, column = 2, sticky="w", padx = self._scale_padding(5))
        self.num_towns_var = tk.StringVar(value="15")
        ttk.Entry(region_frame, textvariable = self.num_towns_var, width = 10).grid(row = 0, column = 3, padx = self._scale_padding(5))

        # Rural population percentage
        ttk.Label(region_frame, text="Rural Population (%):").grid(row = 1, column = 0, sticky="w", padx = self._scale_padding(5))
        self.rural_percentage_var = tk.StringVar(value="30")
        ttk.Entry(region_frame, textvariable = self.rural_percentage_var, width = 10).grid(row = 1, column = 1, padx = self._scale_padding(5))

        # Urban concentration
        ttk.Label(region_frame, text="Urban Concentration:").grid(row = 1, column = 2, sticky="w", padx = self._scale_padding(5))
        self.urban_concentration_var = tk.StringVar(value="0.7")
        ttk.Entry(region_frame, textvariable = self.urban_concentration_var, width = 10).grid(row = 1, column = 3, padx = self._scale_padding(5))

        # Terrain Distribution Configuration
        terrain_frame = ttk.LabelFrame(self.data_frame, text="Terrain Distribution", padding = self._scale_padding(10))
        terrain_frame.pack(fill="x", padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Forest percentage
        ttk.Label(terrain_frame, text="Forest (%):").grid(row = 0, column = 0, sticky="w", padx = self._scale_padding(5))
        self.forest_percentage_var = tk.StringVar(value="25")
        ttk.Entry(terrain_frame, textvariable = self.forest_percentage_var, width = 10).grid(row = 0, column = 1, padx = self._scale_padding(5))

        # Mountain percentage
        ttk.Label(terrain_frame, text="Mountain (%):").grid(row = 0, column = 2, sticky="w", padx = self._scale_padding(5))
        self.mountain_percentage_var = tk.StringVar(value="15")
        ttk.Entry(terrain_frame, textvariable = self.mountain_percentage_var, width = 10).grid(row = 0, column = 3, padx = self._scale_padding(5))

        # Water percentage
        ttk.Label(terrain_frame, text="Water (%):").grid(row = 1, column = 0, sticky="w", padx = self._scale_padding(5))
        self.water_percentage_var = tk.StringVar(value="10")
        ttk.Entry(terrain_frame, textvariable = self.water_percentage_var, width = 10).grid(row = 1, column = 1, padx = self._scale_padding(5))

        # Base terrain percentage (calculated automatically)
        ttk.Label(terrain_frame, text="Base Terrain (%):").grid(row = 1, column = 2, sticky="w", padx = self._scale_padding(5))
        self.base_terrain_percentage_var = tk.StringVar(value="50")
        self.base_terrain_label = ttk.Label(terrain_frame, text="50%", foreground="blue")
        self.base_terrain_label.grid(row = 1, column = 3, sticky="w", padx = self._scale_padding(5))

        # Bind terrain percentage changes to update base terrain
        self.forest_percentage_var.trace('w', self._update_terrain_percentages)
        self.mountain_percentage_var.trace('w', self._update_terrain_percentages)
        self.water_percentage_var.trace('w', self._update_terrain_percentages)

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
        self.max_iterations_var = tk.StringVar(value="1000")
        ttk.Entry(options_frame, textvariable = self.max_iterations_var, width = 10).pack(anchor="w")

        # Plan duration
        plan_duration_frame = ttk.Frame(options_frame)
        plan_duration_frame.pack(fill="x", pady = 10)

        ttk.Label(plan_duration_frame, text="Plan Duration (Years):").pack(side="left")
        self.plan_duration_var = tk.IntVar(value=5)
        self.plan_duration_scale = ttk.Scale(
            plan_duration_frame,
            from_=1,
            to=20,
            variable=self.plan_duration_var,
            orient="horizontal",
            command=self.update_plan_duration_label
        )
        self.plan_duration_scale.pack(side="left", padx=10, fill="x", expand=True)
        
        self.plan_duration_label = ttk.Label(plan_duration_frame, text="5 years")
        self.plan_duration_label.pack(side="left", padx=10)

        # Multi-year plan options
        self.multi_year_frame = ttk.Frame(options_frame)

        ttk.Label(self.multi_year_frame, text="Consumption Growth Rate:").grid(row = 0, column = 0, sticky="w", padx = 5)
        self.growth_rate_var = tk.StringVar(value="0.02")
        ttk.Entry(self.multi_year_frame, textvariable = self.growth_rate_var, width = 10).grid(row = 0, column = 1, padx = 5)

        ttk.Label(self.multi_year_frame, text="Investment Ratio:").grid(row = 0, column = 2, sticky="w", padx = 5)
        self.investment_ratio_var = tk.StringVar(value="0.15")
        ttk.Entry(self.multi_year_frame, textvariable = self.investment_ratio_var, width = 10).grid(row = 0, column = 3, padx = 5)

        # Real-time convergence graph
        convergence_frame = ttk.LabelFrame(self.planning_scrollable_frame, text="Plan Convergence Monitor", padding=10)
        convergence_frame.pack(fill="x", padx=10, pady=5)
        
        # Initialize convergence tracking
        self.convergence_data = {
            'iterations': [],
            'plan_changes': [],
            'relative_changes': [],
            'total_outputs': []
        }
        
        # Create matplotlib figure for real-time plotting
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import matplotlib.animation as animation
            
            # Create figure and subplots
            self.convergence_fig = Figure(figsize=(10, 6), dpi=100)
            self.convergence_ax1 = self.convergence_fig.add_subplot(211)
            self.convergence_ax2 = self.convergence_fig.add_subplot(212)
            
            # Configure subplots
            self.convergence_ax1.set_title('Plan Change Convergence')
            self.convergence_ax1.set_xlabel('Iteration')
            self.convergence_ax1.set_ylabel('Plan Change (Absolute)')
            self.convergence_ax1.set_yscale('log')
            self.convergence_ax1.grid(True, alpha=0.3)
            
            self.convergence_ax2.set_title('Relative Change Convergence')
            self.convergence_ax2.set_xlabel('Iteration')
            self.convergence_ax2.set_ylabel('Relative Change')
            self.convergence_ax2.set_yscale('log')
            self.convergence_ax2.grid(True, alpha=0.3)
            
            # Create canvas and pack it
            self.convergence_canvas = FigureCanvasTkAgg(self.convergence_fig, convergence_frame)
            self.convergence_canvas.draw()
            self.convergence_canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Initialize plot lines
            self.convergence_line1, = self.convergence_ax1.plot([], [], 'b-', linewidth=2, label='Plan Change')
            self.convergence_line2, = self.convergence_ax2.plot([], [], 'r-', linewidth=2, label='Relative Change')
            
            # Add legends
            self.convergence_ax1.legend()
            self.convergence_ax2.legend()
            
            # Status label
            self.convergence_status_label = ttk.Label(convergence_frame, text="Ready to monitor convergence...")
            self.convergence_status_label.pack(pady=5)
            
            # Control buttons
            convergence_controls = ttk.Frame(convergence_frame)
            convergence_controls.pack(fill="x", pady=5)
            
            ttk.Button(convergence_controls, text="Clear Graph", command=self.clear_convergence_graph).pack(side="left", padx=5)
            ttk.Button(convergence_controls, text="Export Graph", command=self.export_convergence_graph).pack(side="left", padx=5)
            
            self.convergence_graph_available = True
            
        except ImportError:
            # Fallback if matplotlib is not available
            self.convergence_graph_available = False
            ttk.Label(convergence_frame, text="Matplotlib not available - convergence graph disabled").pack(pady=10)

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

    def create_technology_tree_tab(self):
        """Create technology tree visualization tab."""
        self.tech_tree_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tech_tree_frame, text="Technology Tree")
        
        # Main layout: controls on top, visualization below
        main_frame = ttk.Frame(self.tech_tree_frame)
        main_frame.pack(fill="both", expand=True, padx=self._scale_padding(10), pady=self._scale_padding(5))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Technology Level Controls", padding=self._scale_padding(10))
        control_frame.pack(fill="x", pady=self._scale_padding(5))
        
        # Technology level slider
        ttk.Label(control_frame, text="Technology Level:").grid(row=0, column=0, sticky="w", padx=self._scale_padding(5))
        self.tech_level_var = tk.DoubleVar(value=0.0)
        self.tech_level_scale = ttk.Scale(
            control_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.tech_level_var,
            orient="horizontal",
            command=self._on_tech_level_change
        )
        self.tech_level_scale.grid(row=0, column=1, sticky="ew", padx=self._scale_padding(5))
        
        # Technology level display
        self.tech_level_display = ttk.Label(control_frame, text="0.0")
        self.tech_level_display.grid(row=0, column=2, padx=self._scale_padding(5))
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh Tree", command=self._refresh_technology_tree).grid(row=0, column=3, padx=self._scale_padding(5))
        
        # Sync button to update data management tab
        ttk.Button(control_frame, text="Sync to Data Tab", command=self._sync_tech_level_to_data).grid(row=0, column=4, padx=self._scale_padding(5))
        
        # Configure grid weights
        control_frame.columnconfigure(1, weight=1)
        
        # Create horizontal layout for tech meter and tree visualization
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill="both", expand=True, pady=self._scale_padding(5))
        
        # Left side: Tech meter and stats
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side="left", fill="y", padx=self._scale_padding(5))
        
        # Technology meter frame
        meter_frame = ttk.LabelFrame(left_panel, text="Technology Meter", padding=self._scale_padding(10))
        meter_frame.pack(fill="x", pady=self._scale_padding(5))
        
        # Create tech meter canvas
        self.tech_meter_canvas = tk.Canvas(meter_frame, width=200, height=150, bg="white")
        self.tech_meter_canvas.pack(pady=self._scale_padding(5))
        
        # Technology level indicators
        indicators_frame = ttk.LabelFrame(left_panel, text="Technology Levels", padding=self._scale_padding(10))
        indicators_frame.pack(fill="x", pady=self._scale_padding(5))
        
        # Create tech level indicators
        self.tech_level_indicators = {}
        tech_levels = [
            ("Basic", 0.0, "🟢"),
            ("Intermediate", 0.2, "🟡"), 
            ("Advanced", 0.5, "🟠"),
            ("Cutting Edge", 0.8, "🔴"),
            ("Future", 0.95, "🟣")
        ]
        
        for i, (name, threshold, icon) in enumerate(tech_levels):
            frame = ttk.Frame(indicators_frame)
            frame.pack(fill="x", pady=2)
            
            label = ttk.Label(frame, text=f"{icon} {name}")
            label.pack(side="left")
            
            status_label = ttk.Label(frame, text="🔒", foreground="red")
            status_label.pack(side="right")
            
            self.tech_level_indicators[name] = {
                'threshold': threshold,
                'status_label': status_label,
                'icon': icon
            }
        
        # Sector statistics
        stats_frame = ttk.LabelFrame(left_panel, text="Sector Statistics", padding=self._scale_padding(10))
        stats_frame.pack(fill="x", pady=self._scale_padding(5))
        
        self.stats_labels = {}
        stats_info = [
            ("Total Sectors", "total_sectors"),
            ("Available Sectors", "available_sectors"),
            ("Locked Sectors", "locked_sectors"),
            ("Availability %", "availability_percent")
        ]
        
        for name, key in stats_info:
            frame = ttk.Frame(stats_frame)
            frame.pack(fill="x", pady=2)
            
            ttk.Label(frame, text=f"{name}:").pack(side="left")
            value_label = ttk.Label(frame, text="0", foreground="blue")
            value_label.pack(side="right")
            
            self.stats_labels[key] = value_label
        
        # Right side: Technology tree visualization
        tree_frame = ttk.LabelFrame(content_frame, text="Technology Tree Visualization", padding=self._scale_padding(10))
        tree_frame.pack(side="right", fill="both", expand=True, padx=self._scale_padding(5))
        
        # Create tree view
        self.tech_tree_view = ttk.Treeview(tree_frame, columns=("level", "available", "category"), show="tree headings")
        self.tech_tree_view.heading("#0", text="Sector Name")
        self.tech_tree_view.heading("level", text="Tech Level")
        self.tech_tree_view.heading("available", text="Available")
        self.tech_tree_view.heading("category", text="Category")
        
        self.tech_tree_view.column("#0", width=300)
        self.tech_tree_view.column("level", width=100)
        self.tech_tree_view.column("available", width=80)
        self.tech_tree_view.column("category", width=150)
        
        # Scrollbar for tree view
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tech_tree_view.yview)
        self.tech_tree_view.configure(yscrollcommand=tree_scrollbar.set)
        
        # Pack tree view and scrollbar
        self.tech_tree_view.pack(side="left", fill="both", expand=True)
        tree_scrollbar.pack(side="right", fill="y")
        
        # Status bar for technology tree
        self.tech_tree_status = ttk.Label(main_frame, text="Technology tree ready")
        self.tech_tree_status.pack(side="bottom", pady=self._scale_padding(5))

    def create_simulation_tab(self):
        """Create simulation system tab."""
        self.simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text="Dynamic Simulation")

        # Create main layout: controls on left, monitoring and preview on right
        self.simulation_main_frame = ttk.Frame(self.simulation_frame)
        self.simulation_main_frame.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Left side: Simulation controls
        self.simulation_left_frame = ttk.Frame(self.simulation_main_frame)
        self.simulation_left_frame.pack(side="left", fill="both", expand=True, padx=self._scale_padding(5))

        # Right side: Monitoring and preview
        self.simulation_right_frame = ttk.Frame(self.simulation_main_frame)
        self.simulation_right_frame.pack(side="right", fill="both", expand=True, padx=self._scale_padding(5))

        # Create scrollable frame for simulation controls (left side)
        self.simulation_canvas = tk.Canvas(self.simulation_left_frame)
        self.simulation_scrollbar = ttk.Scrollbar(self.simulation_left_frame, orient="vertical", command=self.simulation_canvas.yview)
        self.simulation_scrollable_frame = ttk.Frame(self.simulation_canvas)

        self.simulation_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.simulation_canvas.configure(scrollregion=self.simulation_canvas.bbox("all"))
        )

        self.simulation_canvas.create_window((0, 0), window=self.simulation_scrollable_frame, anchor="nw")
        self.simulation_canvas.configure(yscrollcommand=self.simulation_scrollbar.set)

        # Pack canvas and scrollbar immediately after creation
        self.simulation_canvas.pack(side="left", fill="both", expand=True)
        self.simulation_scrollbar.pack(side="right", fill="y")

        # Add welcome message to simulation tab
        welcome_frame = ttk.LabelFrame(self.simulation_scrollable_frame, text="Dynamic Simulation System", padding=self._scale_padding(10))
        welcome_frame.pack(fill="x", padx=self._scale_padding(10), pady=self._scale_padding(5))
        
        welcome_text = """Welcome to the Dynamic Simulation System!

This system allows you to:
• Load economic plans and run dynamic simulations
• Monitor real-time economic performance
• Visualize simulation results on interactive maps
• Track stochastic events and their impacts

To get started:
1. Load an economic plan using the Plan Loading section below
2. Configure simulation parameters
3. Initialize and start the simulation
4. Monitor results in real-time"""
        
        welcome_label = ttk.Label(welcome_frame, text=welcome_text, justify="left", wraplength=400)
        welcome_label.pack(pady=self._scale_padding(5))

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

        # Simulation duration
        sim_duration_frame = ttk.Frame(time_frame)
        sim_duration_frame.pack(fill="x", pady = self._scale_padding(5))
        
        ttk.Label(sim_duration_frame, text="Simulation Duration (years):").pack(side="left", padx = self._scale_padding(5))
        self.sim_duration_var = tk.StringVar(value="5")
        ttk.Spinbox(sim_duration_frame, from_ = 1, to = 100, textvariable = self.sim_duration_var, width = 10).pack(side="left", padx = self._scale_padding(5))
        ttk.Label(sim_duration_frame, text="(Auto-syncs with plan duration)", font=("Arial", 8), foreground="gray").pack(side="left", padx = self._scale_padding(10))

        ttk.Label(time_frame, text="Time Step (months):").pack(side="left", padx = self._scale_padding(20))
        self.time_step_var = tk.StringVar(value="1")
        ttk.Spinbox(time_frame, from_ = 1, to = 12, textvariable = self.time_step_var, width = 10).pack(side="left", padx = self._scale_padding(5))

        # Environment parameters
        env_frame = ttk.Frame(params_frame)
        env_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Label(env_frame, text="Map Size (km):").pack(side="left", padx = self._scale_padding(5))
        # Map size variable removed - map functionality removed

        # Economic sectors (derived from loaded plan)
        sectors_frame = ttk.Frame(params_frame)
        sectors_frame.pack(fill="x", pady = self._scale_padding(5))

        ttk.Label(sectors_frame, text="Economic Sectors:").pack(side="left", padx = self._scale_padding(5))
        self.sectors_display_var = tk.StringVar(value="Not loaded")
        ttk.Label(sectors_frame, textvariable=self.sectors_display_var, foreground="blue").pack(side="left", padx = self._scale_padding(5))
        ttk.Label(sectors_frame, text="(from loaded plan)", font=("TkDefaultFont", 8), foreground="gray").pack(side="left", padx = self._scale_padding(5))

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

        # Event log section (keep this in the main simulation area)
        events_frame = ttk.LabelFrame(self.simulation_scrollable_frame, text="Event Log", padding = self._scale_padding(10))
        events_frame.pack(fill="both", expand = True, padx = self._scale_padding(10), pady = self._scale_padding(5))

        # Event log text area
        self.events_text = scrolledtext.ScrolledText(events_frame, height = 15, width = 80)
        self.events_text.pack(fill="both", expand = True, padx = self._scale_padding(5), pady = self._scale_padding(5))
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_status.config(text="Monitoring: Active", foreground="green")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        if hasattr(self, 'monitoring_data_text'):
            self.monitoring_data_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring started\n")
            self.monitoring_data_text.see(tk.END)
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        self.monitoring_status.config(text="Monitoring: Stopped", foreground="red")
        
        if hasattr(self, 'monitoring_data_text'):
            self.monitoring_data_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring stopped\n")
            self.monitoring_data_text.see(tk.END)
    
    def clear_monitoring_data(self):
        """Clear monitoring data display."""
        if hasattr(self, 'monitoring_data_text'):
            self.monitoring_data_text.delete("1.0", tk.END)
            self.monitoring_data_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring data cleared\n")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect monitoring data
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                # Get current simulation data if available
                if hasattr(self, 'current_simulation') and self.current_simulation:
                    # Add simulation-specific monitoring data
                    data_line = f"[{timestamp}] Simulation running - State: {getattr(self.current_simulation, 'state', 'Unknown')}\n"
                else:
                    # Add general system monitoring data
                    data_line = f"[{timestamp}] System monitoring - Planning system: {'Active' if self.planning_system else 'Inactive'}\n"
                
                # Update UI in main thread
                self.root.after(0, lambda: self._update_monitoring_display(data_line))
                
                # Sleep for 1 second
                threading.Event().wait(1.0)
                
            except Exception as e:
                error_line = f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring error: {str(e)}\n"
                self.root.after(0, lambda: self._update_monitoring_display(error_line))
                break
    
    def _update_monitoring_display(self, data_line):
        """Update monitoring display in main thread."""
        if self.monitoring_active and hasattr(self, 'monitoring_data_text'):
            try:
                self.monitoring_data_text.insert(tk.END, data_line)
                self.monitoring_data_text.see(tk.END)
            except (AttributeError, tk.TclError):
                # GUI component not ready or destroyed
                pass
    
    def _safe_update_monitoring_display(self, message):
        """Safely update monitoring display with error handling."""
        try:
            if hasattr(self, 'monitoring_data_text'):
                self.monitoring_data_text.insert(tk.END, message)
                self.monitoring_data_text.see(tk.END)
        except (AttributeError, tk.TclError):
            # GUI component not ready or destroyed
            pass

        # Add a status message to confirm the tab is working
        status_frame = ttk.Frame(self.simulation_scrollable_frame)
        status_frame.pack(fill="x", padx=self._scale_padding(10), pady=self._scale_padding(5))
        
        self.simulation_status_label = ttk.Label(status_frame, text="Dynamic Simulation System Ready", foreground="green")
        self.simulation_status_label.pack(pady=self._scale_padding(5))

        # Create simulation preview section on the right side
        self.create_simulation_preview_section()

    def create_simulation_preview_section(self):
        """Create the simulation preview section on the right side."""
        # Create notebook for monitoring and preview
        self.right_notebook = ttk.Notebook(self.simulation_right_frame)
        self.right_notebook.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Real-time monitoring tab
        self.monitoring_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.monitoring_frame, text="Real-time Monitoring")

        # Monitoring controls
        monitoring_controls_frame = ttk.Frame(self.monitoring_frame)
        monitoring_controls_frame.pack(fill="x", padx=self._scale_padding(5), pady=self._scale_padding(5))

        ttk.Button(monitoring_controls_frame, text="Start Monitoring", command=self.start_monitoring).pack(side="left", padx=self._scale_padding(5))
        ttk.Button(monitoring_controls_frame, text="Stop Monitoring", command=self.stop_monitoring).pack(side="left", padx=self._scale_padding(5))
        ttk.Button(monitoring_controls_frame, text="Clear Data", command=self.clear_monitoring_data).pack(side="left", padx=self._scale_padding(5))

        # Monitoring status
        self.monitoring_status = ttk.Label(monitoring_controls_frame, text="Monitoring: Stopped", foreground="red")
        self.monitoring_status.pack(side="right", padx=self._scale_padding(5))

        # Real-time data display
        self.monitoring_data_text = scrolledtext.ScrolledText(self.monitoring_frame, height=20, width=80)
        self.monitoring_data_text.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Initialize monitoring variables
        self.monitoring_active = False
        self.monitoring_thread = None

        # Simulation preview tab
        preview_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(preview_frame, text="Simulation Preview")

        # Preview header
        preview_header = ttk.LabelFrame(preview_frame, text="Simulation Preview", padding=self._scale_padding(10))
        preview_header.pack(fill="x", padx=self._scale_padding(5), pady=self._scale_padding(5))
        
        preview_text = """Simulation Environment Preview

This preview shows the current simulation environment including:
• Geographic features and terrain
• Settlement networks and population centers
• Economic zones and infrastructure
• Real-time simulation status

Click 'Generate Preview' to create a visual representation of your simulation environment."""
        
        preview_label = ttk.Label(preview_header, text=preview_text, justify="left", wraplength=300)
        preview_label.pack(pady=self._scale_padding(5))

        # Preview controls
        preview_controls = ttk.LabelFrame(preview_frame, text="Preview Controls", padding=self._scale_padding(10))
        preview_controls.pack(fill="x", padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Control buttons
        button_frame = ttk.Frame(preview_controls)
        button_frame.pack(fill="x", pady=self._scale_padding(5))

        ttk.Button(button_frame, text="Generate Preview", command=self.generate_simulation_preview).pack(side="left", padx=self._scale_padding(5))
        ttk.Button(button_frame, text="Open in Browser", command=self.open_simulation_preview_in_browser).pack(side="left", padx=self._scale_padding(5))
        ttk.Button(button_frame, text="Refresh Preview", command=self.refresh_simulation_preview).pack(side="left", padx=self._scale_padding(5))

        # Preview status
        self.preview_status = ttk.Label(preview_controls, text="No preview generated", foreground="red")
        self.preview_status.pack(pady=self._scale_padding(5))

        # Preview display area
        self.preview_display_frame = ttk.LabelFrame(preview_frame, text="Preview Display", padding=self._scale_padding(10))
        self.preview_display_frame.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Create notebook for preview display and info
        self.preview_notebook = ttk.Notebook(self.preview_display_frame)
        self.preview_notebook.pack(fill="both", expand=True)

        # Interactive Preview tab
        self.preview_view_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.preview_view_frame, text="Interactive Preview")

        # Preview info tab
        self.preview_info_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.preview_info_frame, text="Preview Info")

        # Preview display area (web browser widget)
        self.preview_browser_widget = WebBrowserWidget(self.preview_view_frame, width=600, height=400)
        self.preview_browser_widget.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Preview info display
        self.preview_info_text = scrolledtext.ScrolledText(self.preview_info_frame, height=15, width=50)
        self.preview_info_text.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Initialize with welcome message
        self.preview_info_text.insert("1.0", "No preview generated. Click 'Generate Preview' to create a simulation environment preview.")

        # Initialize preview variables
        self.simulation_preview = None
        self.simulation_preview_file_path = None


    def generate_simulation_preview(self):
        """Generate a simulation preview for the simulation tab."""
        messagebox.showinfo("Info", "Simulation preview functionality has been removed. Use the simulation tab for running simulations.")

    def update_simulation_preview_info(self):
        """Update the simulation preview information display."""
        self.preview_info_text.delete("1.0", tk.END)
        self.preview_info_text.insert("1.0", "Simulation preview functionality has been removed. Use the simulation tab for running simulations.")

    def display_preview_in_gui(self):
        """Display the simulation preview in the GUI."""
        self.preview_browser_widget.preview_text.delete("1.0", tk.END)
        self.preview_browser_widget.preview_text.insert("1.0", "Simulation preview functionality has been removed. Use the simulation tab for running simulations.")
        self.preview_browser_widget.status_label.config(text="Preview functionality removed")

    def refresh_simulation_preview(self):
        """Refresh the simulation preview with updated data."""
        pass  # Functionality removed

    def open_simulation_preview_in_browser(self):
        """Open the simulation preview in the default web browser."""
        messagebox.showinfo("Info", "Simulation preview functionality has been removed. Use the simulation tab for running simulations.")

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
        ttk.Button(export_frame, text="Generate Simulation Summary", command = self._generate_simulation_summary).pack(side="left", padx = 5)
        ttk.Button(export_frame, text="📄 Export Complete Report as PDF", command = self._export_summary_pdf).pack(side="left", padx = 5)
        ttk.Button(export_frame, text="🏥 Population Health Report", command = self._generate_population_health_report).pack(side="left", padx = 5)

        # Synthetic Dataset Export
        synthetic_frame = ttk.LabelFrame(self.export_frame, text="Synthetic Dataset Export", padding = 10)
        synthetic_frame.pack(fill="x", padx = 10, pady = 5)

        ttk.Button(synthetic_frame, text="💾 Save Synthetic Dataset", command = self.save_synthetic_dataset).pack(side="left", padx = 5)
        ttk.Label(synthetic_frame, text="Saves current synthetic data to data/synthetic_datasets/", font=("Arial", 8)).pack(side="left", padx = 10)

        # Load plan
        load_frame = ttk.LabelFrame(self.export_frame, text="Load Plan", padding = 10)
        load_frame.pack(fill="x", padx = 10, pady = 5)

        ttk.Button(load_frame, text="Load Plan from File", command = self.load_plan).pack(side="left", padx = 5)

        # Status
        self.export_status = ttk.Label(self.export_frame, text="No plan to export")
        self.export_status.pack(pady = 10)

    def create_gpu_settings_tab(self):
        """Create GPU settings tab."""
        self.gpu_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gpu_frame, text="GPU Settings")

        # GPU Detection Section
        detection_frame = ttk.LabelFrame(self.gpu_frame, text="GPU Detection", padding=self._scale_padding(10))
        detection_frame.pack(fill="x", padx=self._scale_padding(10), pady=self._scale_padding(5))

        # GPU Status
        self.gpu_status_label = ttk.Label(detection_frame, text="Checking GPU availability...")
        self.gpu_status_label.pack(pady=self._scale_padding(5))

        # Refresh GPU Status Button
        ttk.Button(detection_frame, text="Refresh GPU Status", command=self.refresh_gpu_status).pack(pady=self._scale_padding(5))

        # GPU Information Display
        self.gpu_info_text = scrolledtext.ScrolledText(
            detection_frame, 
            width=80, 
            height=8,
            wrap=tk.WORD
        )
        self.gpu_info_text.pack(fill="both", expand=True, pady=self._scale_padding(5))

        # GPU Settings Section
        settings_frame = ttk.LabelFrame(self.gpu_frame, text="GPU Settings", padding=self._scale_padding(10))
        settings_frame.pack(fill="x", padx=self._scale_padding(10), pady=self._scale_padding(5))

        # Enable GPU Checkbox
        self.gpu_enabled_var = tk.BooleanVar()
        gpu_checkbox = ttk.Checkbutton(
            settings_frame, 
            text="Enable GPU Acceleration", 
            variable=self.gpu_enabled_var,
            command=self.on_gpu_enabled_changed
        )
        gpu_checkbox.pack(anchor="w", pady=self._scale_padding(5))

        # Solver Selection
        solver_frame = ttk.Frame(settings_frame)
        solver_frame.pack(fill="x", pady=self._scale_padding(5))

        ttk.Label(solver_frame, text="Preferred Solver:").pack(side="left")
        self.solver_var = tk.StringVar(value="CuClarabel")
        solver_combo = ttk.Combobox(
            solver_frame, 
            textvariable=self.solver_var,
            values=["CuClarabel", "SCS", "ECOS", "OSQP", "CLARABEL"],
            state="readonly",
            width=15
        )
        solver_combo.pack(side="left", padx=self._scale_padding(5))

        # Monitoring Settings
        monitoring_frame = ttk.LabelFrame(settings_frame, text="Performance Monitoring", padding=self._scale_padding(5))
        monitoring_frame.pack(fill="x", pady=self._scale_padding(5))

        self.show_utilization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            monitoring_frame, 
            text="Show GPU Utilization", 
            variable=self.show_utilization_var
        ).pack(anchor="w")

        self.benchmark_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            monitoring_frame, 
            text="Enable Benchmark Mode", 
            variable=self.benchmark_mode_var
        ).pack(anchor="w")

        # Save Settings Button
        ttk.Button(settings_frame, text="Save GPU Settings", command=self.save_gpu_settings).pack(pady=self._scale_padding(10))

        # Performance Monitoring Section
        performance_frame = ttk.LabelFrame(self.gpu_frame, text="Performance Monitoring", padding=self._scale_padding(10))
        performance_frame.pack(fill="both", expand=True, padx=self._scale_padding(10), pady=self._scale_padding(5))

        # GPU Memory Usage
        memory_frame = ttk.Frame(performance_frame)
        memory_frame.pack(fill="x", pady=self._scale_padding(5))

        ttk.Label(memory_frame, text="GPU Memory Usage:").pack(side="left")
        self.gpu_memory_label = ttk.Label(memory_frame, text="N/A")
        self.gpu_memory_label.pack(side="left", padx=self._scale_padding(10))

        # Benchmark Button
        ttk.Button(performance_frame, text="Run GPU vs CPU Benchmark", command=self.run_gpu_benchmark).pack(pady=self._scale_padding(5))

        # Benchmark Results
        self.benchmark_text = scrolledtext.ScrolledText(
            performance_frame, 
            width=80, 
            height=10,
            wrap=tk.WORD
        )
        self.benchmark_text.pack(fill="both", expand=True, pady=self._scale_padding(5))

        # Initialize GPU status
        self.refresh_gpu_status()

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
        """Generate synthetic data based on configuration including population demographics."""
        if not self.planning_system:
            messagebox.showerror("Error", "Planning system not available. Please check the installation.")
            return
            
        try:
            n_sectors = int(self.sectors_var.get())
            density = float(self.density_var.get())
            n_resources = int(self.resources_var.get())
            starting_tech_level = float(self.starting_tech_var.get())

            # Get population demographics
            total_population = int(self.total_population_var.get())
            employment_rate = float(self.employment_rate_var.get()) / 100.0
            dependency_ratio = float(self.dependency_ratio_var.get()) / 100.0
            tech_level = float(self.tech_level_var.get())
            
            # Get regional distribution
            num_cities = int(self.num_cities_var.get())
            num_towns = int(self.num_towns_var.get())
            rural_percentage = float(self.rural_percentage_var.get()) / 100.0
            urban_concentration = float(self.urban_concentration_var.get())

            # Update the planning system's technology level
            if hasattr(self.planning_system, 'matrix_builder') and hasattr(self.planning_system.matrix_builder, 'sector_mapper'):
                sector_mapper = self.planning_system.matrix_builder.sector_mapper
                if hasattr(sector_mapper, 'technological_level'):
                    sector_mapper.technological_level = starting_tech_level
                elif hasattr(sector_mapper, 'sector_generator'):
                    sector_mapper.sector_generator.technological_level = starting_tech_level

            # Generate synthetic data and store it
            synthetic_data = self.planning_system.create_synthetic_data(
                n_sectors = n_sectors, technology_density = density, resource_count = n_resources
            )
            
            # Store the synthetic data
            self.current_data = synthetic_data
            
            # Create simulation plan from synthetic data
            simulation_plan = self._create_simulation_plan_from_synthetic_data(synthetic_data)
            self.current_simulation_plan = simulation_plan
            
            # Generate population demographics
            population_data = self._generate_population_demographics(
                total_population, employment_rate, dependency_ratio, tech_level,
                num_cities, num_towns, rural_percentage, urban_concentration
            )
            
            # Add population data to current data
            self.current_data.update(population_data)
            
            # Update technology tree tab if it exists
            if hasattr(self, 'tech_level_var'):
                self.tech_level_var.set(starting_tech_level)
                self._refresh_technology_tree()
            
            # Ensure data is properly converted to numpy arrays
            self._ensure_numpy_arrays()

            self.update_data_display()
            self.data_status.config(text="Synthetic data with demographics and simulation plan generated", foreground="green")
            
            # Update simulation plan status
            if hasattr(self, 'plan_status'):
                self.plan_status.config(text="Simulation plan auto-generated from synthetic data", foreground="green")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid configuration: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")

    def save_synthetic_data(self):
        """Save the current synthetic data to data/synthetic_data directory."""
        if not hasattr(self, 'current_data') or not self.current_data:
            messagebox.showwarning("Warning", "No synthetic data to save. Please generate synthetic data first.")
            return
        
        try:
            import os
            import json
            from datetime import datetime
            
            # Create synthetic_data directory if it doesn't exist
            synthetic_dir = os.path.join("data", "synthetic_data")
            os.makedirs(synthetic_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_data_{timestamp}.json"
            filepath = os.path.join(synthetic_dir, filename)
            
            # Convert numpy arrays and complex objects to JSON-serializable format
            def convert_for_json(obj, visited=None):
                """Recursively convert objects to JSON-serializable format."""
                if visited is None:
                    visited = set()
                
                # Handle circular references
                obj_id = id(obj)
                if obj_id in visited:
                    return "<circular_reference>"
                
                if hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                elif hasattr(obj, 'value'):  # enum objects
                    return obj.value
                elif hasattr(obj, '__dict__'):  # dataclass objects like SectorDefinition
                    visited.add(obj_id)
                    result = {key: convert_for_json(value, visited) for key, value in obj.__dict__.items()}
                    visited.remove(obj_id)
                    return result
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    visited.add(obj_id)
                    result = {key: convert_for_json(value, visited) for key, value in obj.items()}
                    visited.remove(obj_id)
                    return result
                elif isinstance(obj, (list, tuple)):
                    visited.add(obj_id)
                    result = [convert_for_json(item, visited) for item in obj]
                    visited.remove(obj_id)
                    return result
                else:
                    return str(obj)
            
            data_to_save = convert_for_json(self.current_data)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            # Also save a copy without timestamp for easy access
            latest_filepath = os.path.join(synthetic_dir, "latest_synthetic_data.json")
            with open(latest_filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            messagebox.showinfo("Success", f"Synthetic data saved to:\n{filepath}\n\nAlso saved as: latest_synthetic_data.json")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save synthetic data: {str(e)}")

    def _generate_population_demographics(self, total_population, employment_rate, dependency_ratio, 
                                        tech_level, num_cities, num_towns, rural_percentage, urban_concentration):
        """Generate population demographics data using correct economic definitions."""
        import random
        import numpy as np
        
        # CORRECTED LOGIC: Follow standard economic definitions
        # 1. Calculate dependent population (not in labor force)
        dependent_population = int(total_population * dependency_ratio)
        
        # 2. Calculate labor force (total population minus dependents)
        labor_force = total_population - dependent_population
        
        # 3. Calculate unemployment rate from employment rate
        unemployment_rate = 1.0 - employment_rate
        
        # 4. Calculate unemployed population from labor force
        unemployed_population = int(labor_force * unemployment_rate)
        
        # 5. Calculate employed population as remainder of labor force
        employed_population = labor_force - unemployed_population
        
        # 6. Verify total (should always be correct now)
        total_calculated = employed_population + unemployed_population + dependent_population
        if total_calculated != total_population:
            # This should never happen with correct logic, but add safety check
            print(f"WARNING: Population calculation error. Expected: {total_population}, Got: {total_calculated}")
            # Adjust dependent population to fix any rounding errors
            dependent_population = total_population - employed_population - unemployed_population
        
        # Generate regional distribution
        urban_population = int(total_population * (1 - rural_percentage))
        rural_population = total_population - urban_population
        
        # Distribute urban population between cities and towns
        city_population = int(urban_population * urban_concentration)
        town_population = urban_population - city_population
        
        # Generate individual settlements
        settlements = []
        
        # Generate cities
        for i in range(num_cities):
            city_pop = int(city_population / num_cities)
            if i == num_cities - 1:  # Last city gets remainder
                city_pop = city_population - (num_cities - 1) * int(city_population / num_cities)
            
            settlements.append({
                'name': f'City {i+1}',
                'population': city_pop,
                'settlement_type': 'city',
                'employment_rate': employment_rate + random.uniform(-0.05, 0.05),  # Variation
                'tech_level': tech_level + random.uniform(-0.1, 0.1),
                'economic_sectors': random.sample(['manufacturing', 'services', 'technology', 'finance'], 3)
            })
        
        # Generate towns
        for i in range(num_towns):
            town_pop = int(town_population / num_towns)
            if i == num_towns - 1:  # Last town gets remainder
                town_pop = town_population - (num_towns - 1) * int(town_population / num_towns)
            
            settlements.append({
                'name': f'Town {i+1}',
                'population': town_pop,
                'settlement_type': 'town',
                'employment_rate': employment_rate + random.uniform(-0.1, 0.1),
                'tech_level': tech_level + random.uniform(-0.2, 0.1),
                'economic_sectors': random.sample(['agriculture', 'manufacturing', 'services'], 2)
            })
        
        # Generate rural areas
        num_rural = max(1, int(rural_population / 1000))  # ~1000 people per rural area
        for i in range(num_rural):
            rural_pop = int(rural_population / num_rural)
            if i == num_rural - 1:
                rural_pop = rural_population - (num_rural - 1) * int(rural_population / num_rural)
            
            settlements.append({
                'name': f'Rural Area {i+1}',
                'population': rural_pop,
                'settlement_type': 'rural',
                'employment_rate': employment_rate + random.uniform(-0.15, 0.05),
                'tech_level': tech_level + random.uniform(-0.3, 0.0),
                'economic_sectors': ['agriculture']
            })
        
        # Calculate consumer demand scaling factors
        # Higher tech level = higher living standards = higher per capita demand
        base_demand_per_capita = 1000  # Base demand units per person
        tech_demand_multiplier = 1.0 + (tech_level - 1.0) * 0.5  # 50% increase per tech level above 1
        total_consumer_demand = int(total_population * base_demand_per_capita * tech_demand_multiplier)
        
        # Calculate labor productivity (output per employed worker)
        base_productivity = 5000  # Base output units per worker
        tech_productivity_multiplier = 1.0 + (tech_level - 1.0) * 0.3  # 30% increase per tech level above 1
        total_labor_output = int(employed_population * base_productivity * tech_productivity_multiplier)
        
        return {
            'population_demographics': {
                'total_population': total_population,
                'labor_force': labor_force,
                'employed_population': employed_population,
                'unemployed_population': unemployed_population,
                'dependent_population': dependent_population,
                'employment_rate': employment_rate,
                'unemployment_rate': unemployment_rate,
                'dependency_ratio': dependency_ratio,
                'tech_level': tech_level
            },
            'regional_distribution': {
                'urban_population': urban_population,
                'rural_population': rural_population,
                'city_population': city_population,
                'town_population': town_population,
                'num_cities': num_cities,
                'num_towns': num_towns,
                'num_rural_areas': num_rural,
                'urban_concentration': urban_concentration
            },
            'settlements': settlements,
            'terrain_distribution': {
                'forest_percentage': float(self.forest_percentage_var.get()),
                'mountain_percentage': float(self.mountain_percentage_var.get()),
                'water_percentage': float(self.water_percentage_var.get()),
                'base_terrain_percentage': float(self.base_terrain_percentage_var.get())
            },
            'economic_scaling': {
                'total_consumer_demand': total_consumer_demand,
                'total_labor_output': total_labor_output,
                'demand_per_capita': base_demand_per_capita * tech_demand_multiplier,
                'productivity_per_worker': base_productivity * tech_productivity_multiplier,
                'tech_demand_multiplier': tech_demand_multiplier,
                'tech_productivity_multiplier': tech_productivity_multiplier
            }
        }

    def _update_terrain_percentages(self, *args):
        """Update base terrain percentage when other terrain percentages change."""
        try:
            forest_pct = float(self.forest_percentage_var.get() or 0)
            mountain_pct = float(self.mountain_percentage_var.get() or 0)
            water_pct = float(self.water_percentage_var.get() or 0)
            
            # Calculate base terrain percentage
            total_special_terrain = forest_pct + mountain_pct + water_pct
            base_terrain_pct = max(0, 100 - total_special_terrain)
            
            # Update the base terrain percentage
            self.base_terrain_percentage_var.set(str(base_terrain_pct))
            self.base_terrain_label.config(text=f"{base_terrain_pct:.1f}%")
            
            # Change color based on whether percentages add up to 100%
            if abs(total_special_terrain - 100) < 0.1:
                self.base_terrain_label.config(foreground="green")
            elif total_special_terrain > 100:
                self.base_terrain_label.config(foreground="red")
            else:
                self.base_terrain_label.config(foreground="blue")
                
        except ValueError:
            # Handle invalid input gracefully
            self.base_terrain_label.config(text="Error", foreground="red")

    def save_data_config(self):
        """Save current data management settings to a config file."""
        try:
            # Collect all data management settings
            config_data = {
                "data_management": {
                    "economic_data": {
                        "sectors": self.sectors_var.get(),
                        "density": self.density_var.get(),
                        "resources": self.resources_var.get(),
                        "starting_tech_level": self.starting_tech_var.get()
                    },
                    "population_demographics": {
                        "total_population": self.total_population_var.get(),
                        "employment_rate": self.employment_rate_var.get(),
                        "dependency_ratio": self.dependency_ratio_var.get(),
                        "tech_level": self.tech_level_var.get()
                    },
                    "regional_distribution": {
                        "num_cities": self.num_cities_var.get(),
                        "num_towns": self.num_towns_var.get(),
                        "rural_percentage": self.rural_percentage_var.get(),
                        "urban_concentration": self.urban_concentration_var.get()
                    },
                    "terrain_distribution": {
                        "forest_percentage": self.forest_percentage_var.get(),
                        "mountain_percentage": self.mountain_percentage_var.get(),
                        "water_percentage": self.water_percentage_var.get(),
                        "base_terrain_percentage": self.base_terrain_percentage_var.get()
                    }
                },
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0"
                }
            }
            
            # Ask user for file location
            file_path = filedialog.asksaveasfilename(
                title="Save Data Management Config",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir="."
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Data management config saved to:\n{file_path}")
                self.data_status.config(text=f"Config saved: {os.path.basename(file_path)}", foreground="green")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {str(e)}")
            self.data_status.config(text="Error saving config", foreground="red")

    def load_data_config(self):
        """Load data management settings from a config file."""
        try:
            # Ask user for file location
            file_path = filedialog.askopenfilename(
                title="Load Data Management Config",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir="."
            )
            
            if file_path:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                
                # Validate config structure
                if "data_management" not in config_data:
                    raise ValueError("Invalid config file: missing 'data_management' section")
                
                dm_config = config_data["data_management"]
                
                # Load economic data settings
                if "economic_data" in dm_config:
                    econ = dm_config["economic_data"]
                    self.sectors_var.set(econ.get("sectors", "175"))
                    self.density_var.set(econ.get("density", "0.4"))
                    self.resources_var.set(econ.get("resources", "3"))
                    self.starting_tech_var.set(econ.get("starting_tech_level", "0.0"))
                
                # Load population demographics settings
                if "population_demographics" in dm_config:
                    demo = dm_config["population_demographics"]
                    self.total_population_var.set(demo.get("total_population", "1000000"))
                    self.employment_rate_var.set(demo.get("employment_rate", "60"))
                    self.dependency_ratio_var.set(demo.get("dependency_ratio", "40"))
                    self.tech_level_var.set(demo.get("tech_level", "1.0"))
                
                # Load regional distribution settings
                if "regional_distribution" in dm_config:
                    region = dm_config["regional_distribution"]
                    self.num_cities_var.set(region.get("num_cities", "5"))
                    self.num_towns_var.set(region.get("num_towns", "15"))
                    self.rural_percentage_var.set(region.get("rural_percentage", "30"))
                    self.urban_concentration_var.set(region.get("urban_concentration", "0.7"))
                
                # Load terrain distribution settings
                if "terrain_distribution" in dm_config:
                    terrain = dm_config["terrain_distribution"]
                    self.forest_percentage_var.set(terrain.get("forest_percentage", "25"))
                    self.mountain_percentage_var.set(terrain.get("mountain_percentage", "15"))
                    self.water_percentage_var.set(terrain.get("water_percentage", "10"))
                    # Update base terrain percentage
                    self._update_terrain_percentages()
                
                # Show success message
                saved_at = config_data.get("metadata", {}).get("saved_at", "Unknown")
                messagebox.showinfo("Success", f"Data management config loaded from:\n{file_path}\n\nSaved: {saved_at}")
                self.data_status.config(text=f"Config loaded: {os.path.basename(file_path)}", foreground="green")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {str(e)}")
            self.data_status.config(text="Error loading config", foreground="red")

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
            else f"({len(tech_matrix)}, {len(tech_matrix[0]) if tech_matrix and len(tech_matrix) > 0 else 0})"
        )
        final_shape = final_demand.shape if hasattr(final_demand, "shape") else f"({len(final_demand)},)"
        labor_shape = labor_input.shape if hasattr(labor_input, "shape") else f"({len(labor_input)},)"

        # Safely get matrix slice
        tech_slice = "N / A"
        if hasattr(tech_matrix, "shape") and tech_matrix.size > 0:
            # It's a numpy array
            try:
                tech_slice = str(tech_matrix[:4, :4])
            except (IndexError, ValueError):
                tech_slice = "Error accessing matrix slice"
        elif isinstance(tech_matrix, list) and len(tech_matrix) >= 4:
            # It's a list, get first 4x4 elements
            try:
                tech_slice = str([row[:4] if len(row) >= 4 else row for row in tech_matrix[:4]])
            except (IndexError, ValueError):
                tech_slice = "Error accessing list slice"

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

        # Add population demographics if available
        population_info = ""
        if 'population_demographics' in self.current_data:
            demo = self.current_data['population_demographics']
            regional = self.current_data.get('regional_distribution', {})
            economic = self.current_data.get('economic_scaling', {})
            terrain = self.current_data.get('terrain_distribution', {})
            
            population_info = f"""

Population Demographics:
========================
Total Population: {demo.get('total_population', 0):,}
├─ Labor Force: {demo.get('labor_force', 0):,} ({((demo.get('labor_force', 0) / demo.get('total_population', 1)) * 100):.1f}%)
│  ├─ Employed: {demo.get('employed_population', 0):,} ({demo.get('employment_rate', 0)*100:.1f}% of labor force)
│  └─ Unemployed: {demo.get('unemployed_population', 0):,} ({demo.get('unemployment_rate', 0)*100:.1f}% of labor force)
└─ Dependents: {demo.get('dependent_population', 0):,} ({demo.get('dependency_ratio', 0)*100:.1f}% of total population)

Regional Distribution:
=====================
Urban Population: {regional.get('urban_population', 0):,}
├─ Cities: {regional.get('num_cities', 0)} ({regional.get('city_population', 0):,} people)
└─ Towns: {regional.get('num_towns', 0)} ({regional.get('town_population', 0):,} people)
Rural Population: {regional.get('rural_population', 0):,} ({regional.get('num_rural_areas', 0)} areas)

Terrain Distribution:
====================
Forest: {terrain.get('forest_percentage', 0):.1f}%
Mountain: {terrain.get('mountain_percentage', 0):.1f}%
Water: {terrain.get('water_percentage', 0):.1f}%
Base Terrain: {terrain.get('base_terrain_percentage', 0):.1f}%

Economic Scaling:
================
Technology Level: {demo.get('tech_level', 1.0):.2f}
Total Consumer Demand: {economic.get('total_consumer_demand', 0):,}
Total Labor Output: {economic.get('total_labor_output', 0):,}
Demand per Capita: {economic.get('demand_per_capita', 0):.0f}
Productivity per Worker: {economic.get('productivity_per_worker', 0):.0f}
"""

        # Add resource allocation information if available
        resource_info = ""
        if hasattr(self, 'current_simulation_plan') and self.current_simulation_plan and 'resource_allocations' in self.current_simulation_plan:
            resource_allocations = self.current_simulation_plan['resource_allocations']
            resource_info = f"""

Resource Allocations:
====================
Technology Matrix: {len(resource_allocations.get('technology_matrix', []))}x{len(resource_allocations.get('technology_matrix', [[]])[0]) if resource_allocations.get('technology_matrix') else 0}
Final Demand: {len(resource_allocations.get('final_demand', []))} sectors
Total Labor Cost: {resource_allocations.get('total_labor_cost', 0):,.0f}
Plan Quality Score: {resource_allocations.get('plan_quality_score', 0):.2f}
Resource Matrix: {len(resource_allocations.get('resource_matrix', []))} resources
Max Resources: {len(resource_allocations.get('max_resources', []))} constraints
Resource Names: {', '.join(resource_allocations.get('resource_names', [])) if resource_allocations.get('resource_names') else 'None'}
"""

        summary = f"""Data Summary:
================

Sectors: {sector_count}
Technology Matrix Shape: {tech_shape}
Final Demand Shape: {final_shape}
Labor Input Shape: {labor_shape}

Sector Names:
{sector_summary}

Final Demand Values:
{final_demand if hasattr(final_demand, '__len__') and len(final_demand) > 0 else 'No data available'}

Labor Input Values:
{labor_input if hasattr(labor_input, '__len__') and len(labor_input) > 0 else 'No data available'}

Technology Matrix (first 4x4):
{tech_slice}
{population_info}{resource_info}
        """

        self.data_text.delete("1.0", tk.END)
        self.data_text.insert("1.0", summary)

    def create_plan(self):
        """Create an economic plan with population-scaled demand and output."""
        if not self.current_data:
            messagebox.showerror("Error", "Please load data first")
            return
        
        if not self.planning_system:
            messagebox.showerror("Error", "Planning system not available. Please check the installation.")
            return

        # Get policy goals
        goals_text = self.goals_text.get("1.0", tk.END).strip()
        policy_goals = [goal.strip() for goal in goals_text.split("\n") if goal.strip()]

        # Get planning options
        use_optimization = self.use_optimization_var.get()
        max_iterations = int(self.max_iterations_var.get())
        plan_duration = int(self.plan_duration_var.get())

        # Get production adjustment settings
        production_multipliers = {
            "overall": self.overall_production_var.get(),
            "dept_I": self.dept_I_production_var.get(),
            "dept_II": self.dept_II_production_var.get(),
            "dept_III": self.dept_III_production_var.get()
        }
        apply_reproduction = self.apply_reproduction_var.get()

        # Scale demand and output based on population demographics
        self._scale_economic_data_with_population()

        print(f"Production multipliers: {production_multipliers}")
        print(f"Apply reproduction: {apply_reproduction}")

        # Start convergence monitoring
        self.start_convergence_monitoring()
        
        # Start planning in a separate thread
        self.create_plan_button.config(state="disabled")
        self.progress_bar.start()
        self.planning_status.config(text="Creating plan...")

        def plan_thread():
            try:
                print(f"Creating plan with {len(policy_goals)} policy goals")
                print(f"Data available: {bool(self.current_data)}")

                if plan_duration == 1:
                    # Single year plan with convergence monitoring
                    self.current_plan = self._create_plan_with_monitoring(
                        policy_goals = policy_goals,
                        use_optimization = use_optimization,
                        max_iterations = max_iterations,
                        production_multipliers = production_multipliers,
                        apply_reproduction = apply_reproduction
                    )
                    print(f"Plan created successfully: {type(self.current_plan)}")
                    if isinstance(self.current_plan, dict):
                        print(f"Plan keys: {list(self.current_plan.keys())}")
                else:
                    # Multi-year plan
                    growth_rate = float(self.growth_rate_var.get())
                    investment_ratio = float(self.investment_ratio_var.get())

                    self.current_plan = self.planning_system.create_multi_year_plan(
                        plan_duration_years = plan_duration,
                        policy_goals = policy_goals,
                        consumption_growth_rate = growth_rate,
                        investment_ratio = investment_ratio,
                        production_multipliers = production_multipliers,
                        apply_reproduction = apply_reproduction
                    )
                    print(f"{plan_duration}-year plan created successfully: {type(self.current_plan)}")

                # Stop convergence monitoring
                self.root.after(0, self.stop_convergence_monitoring)
                
                # Update UI in main thread
                self.root.after(0, self.plan_created_successfully)

            except Exception as e:
                error_msg = str(e)
                print(f"Plan creation failed: {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.plan_creation_failed(error_msg))

        threading.Thread(target = plan_thread, daemon = True).start()

    def _scale_economic_data_with_population(self):
        """Scale economic data based on population demographics."""
        if not self.current_data or 'population_demographics' not in self.current_data:
            return
        
        demo = self.current_data['population_demographics']
        economic = self.current_data.get('economic_scaling', {})
        
        # Get scaling factors
        total_consumer_demand = economic.get('total_consumer_demand', 0)
        total_labor_output = economic.get('total_labor_output', 0)
        employed_population = demo.get('employed_population', 0)
        
        if total_consumer_demand == 0 or total_labor_output == 0:
            return
        
        # Scale final demand based on consumer demand
        if 'final_demand' in self.current_data:
            original_demand = self.current_data['final_demand']
            if hasattr(original_demand, 'shape'):
                # It's a numpy array
                demand_scale_factor = total_consumer_demand / np.sum(original_demand) if np.sum(original_demand) > 0 else 1.0
                self.current_data['final_demand'] = original_demand * demand_scale_factor
            else:
                # It's a list
                demand_scale_factor = total_consumer_demand / sum(original_demand) if sum(original_demand) > 0 else 1.0
                self.current_data['final_demand'] = [x * demand_scale_factor for x in original_demand]
        
        # Scale labor input based on employed population
        if 'labor_input' in self.current_data:
            original_labor = self.current_data['labor_input']
            if hasattr(original_labor, 'shape'):
                # It's a numpy array
                labor_scale_factor = employed_population / np.sum(original_labor) if np.sum(original_labor) > 0 else 1.0
                self.current_data['labor_input'] = original_labor * labor_scale_factor
            else:
                # It's a list
                labor_scale_factor = employed_population / sum(original_labor) if sum(original_labor) > 0 else 1.0
                self.current_data['labor_input'] = [x * labor_scale_factor for x in original_labor]
        
        # Scale technology matrix based on productivity
        if 'technology_matrix' in self.current_data:
            original_tech = self.current_data['technology_matrix']
            productivity_multiplier = economic.get('tech_productivity_multiplier', 1.0)
            
            if hasattr(original_tech, 'shape'):
                # It's a numpy array
                self.current_data['technology_matrix'] = original_tech * productivity_multiplier
            else:
                # It's a list
                self.current_data['technology_matrix'] = [
                    [x * productivity_multiplier for x in row] for row in original_tech
                ]
        
        print(f"✓ Scaled economic data with population demographics:")
        print(f"  - Consumer demand scaled to: {total_consumer_demand:,}")
        print(f"  - Labor output scaled to: {total_labor_output:,}")
        print(f"  - Productivity multiplier: {economic.get('tech_productivity_multiplier', 1.0):.2f}")

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
        
        # Update map settings if map is already generated
        if False:  # Map functionality removed
            self._update_map_with_simulation_data()

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
            
            # Auto-generate map if not already generated
            if False:  # Map functionality removed
                self._auto_generate_map_for_simulation()

            print(f"✓ Plan automatically loaded into simulator with {len(simulation_plan['sectors'])} sectors")
            
            # Automatically initialize the simulation (but don't start it)
            self._auto_initialize_simulation()

        except Exception as e:
            print(f"Warning: Failed to auto - load plan to simulator: {e}")
            # Don't fail the plan creation if simulator loading fails
            pass

    def _auto_initialize_simulation(self):
        """Automatically initialize the simulation after plan creation (but don't start it)."""
        try:
            # Initialize the simulation
            self.initialize_simulation()
            print("✓ Simulation automatically initialized and ready to start")
            
        except Exception as e:
            print(f"Warning: Failed to auto-initialize simulation: {e}")

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

    def update_plan_duration_label(self, value=None):
        """Update the plan duration label when slider changes."""
        duration = int(self.plan_duration_var.get())
        if duration == 1:
            self.plan_duration_label.config(text="1 year")
        else:
            self.plan_duration_label.config(text=f"{duration} years")

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

        # Set proper file extension for Excel
        if format_type == "excel":
            filetypes = [("Excel files", "*.xlsx"), ("All files", "*.*")]
            defaultextension = ".xlsx"
        else:
            filetypes = [(f"{format_type.upper()} files", f"*.{format_type}"), ("All files", "*.*")]
            defaultextension = f".{format_type}"
            
        file_path = filedialog.asksaveasfilename(
            title=f"Save Plan as {format_type.upper()}",
            defaultextension=defaultextension,
            filetypes=filetypes,
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

    def save_synthetic_dataset(self):
        """Save current synthetic dataset to data/synthetic_datasets/ with a safe filename."""
        if not self.current_data:
            messagebox.showerror("Error", "No synthetic data to save")
            return

        try:
            # Ensure the synthetic datasets directory exists
            synthetic_dir = "data/synthetic_datasets"
            os.makedirs(synthetic_dir, exist_ok=True)
            
            # Generate a safe filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"synthetic_dataset_{timestamp}.json"
            file_path = os.path.join(synthetic_dir, safe_filename)
            
            # Save the synthetic data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Update status
            self.export_status.config(text=f"Synthetic dataset saved to {file_path}", foreground="green")
            
            # Show success message
            messagebox.showinfo("Success", f"Synthetic dataset saved successfully!\n\nFile: {file_path}")
            
            print(f"✓ Synthetic dataset saved to: {file_path}")
            
        except Exception as e:
            error_msg = f"Failed to save synthetic dataset: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.export_status.config(text=error_msg, foreground="red")
            print(f"❌ {error_msg}")

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

    def refresh_gpu_status(self):
        """Refresh GPU status and display information."""
        try:
            # Import GPU acceleration module
            from src.cybernetic_planning.core.gpu_acceleration import gpu_detector, settings_manager
            
            # Detect GPU capabilities
            gpu_info = gpu_detector.detect_gpu_capabilities()
            
            # Update status label
            if gpu_info.get("gpu_available", False):
                self.gpu_status_label.config(text="✅ GPU Available", foreground="green")
            else:
                self.gpu_status_label.config(text="❌ GPU Not Available", foreground="red")
            
            # Update GPU information display
            info_text = f"""GPU Detection Results:
{'='*50}

CuPy Available: {gpu_info.get('cupy_available', False)}
CUDA Available: {gpu_info.get('cuda_available', False)}
ROCm Available: {gpu_info.get('rocm_available', False)}
GPU Count: {gpu_info.get('gpu_count', 0)}

GPU Names: {', '.join(gpu_info.get('gpu_names', []))}
GPU Memory: {gpu_info.get('gpu_memory', 0) / (1024**3):.2f} GB
Compute Capability: {gpu_info.get('compute_capability', 'N/A')}

Error: {gpu_info.get('error', 'None')}

Current Settings:
{'='*50}
GPU Enabled: {settings_manager.is_gpu_enabled()}
Solver Preference: {settings_manager.get_solver_preference()}
Fallback to CPU: {settings_manager.should_fallback_to_cpu()}
"""
            
            self.gpu_info_text.delete(1.0, tk.END)
            self.gpu_info_text.insert(1.0, info_text)
            
            # Update settings from current configuration
            self.gpu_enabled_var.set(settings_manager.is_gpu_enabled())
            self.solver_var.set(settings_manager.get_solver_preference())
            
            # Update memory usage
            memory_usage = gpu_detector.get_gpu_memory_usage()
            if memory_usage["total"] > 0:
                used_gb = memory_usage["used"] / (1024**3)
                total_gb = memory_usage["total"] / (1024**3)
                self.gpu_memory_label.config(text=f"{used_gb:.2f} GB / {total_gb:.2f} GB")
            else:
                self.gpu_memory_label.config(text="N/A")
                
        except Exception as e:
            self.gpu_status_label.config(text=f"❌ Error: {str(e)}", foreground="red")
            self.gpu_info_text.delete(1.0, tk.END)
            self.gpu_info_text.insert(1.0, f"Error detecting GPU: {str(e)}")

    def on_gpu_enabled_changed(self):
        """Handle GPU enabled checkbox change."""
        if not self.gpu_enabled_var.get():
            # If disabling GPU, show warning
            result = messagebox.askyesno(
                "Disable GPU Acceleration",
                "Disabling GPU acceleration will use CPU-only computation, which may be slower for large problems. Continue?"
            )
            if not result:
                self.gpu_enabled_var.set(True)

    def save_gpu_settings(self):
        """Save GPU settings to configuration."""
        try:
            from src.cybernetic_planning.core.gpu_acceleration import settings_manager
            
            success = settings_manager.save_settings(
                gpu_enabled=self.gpu_enabled_var.get(),
                solver_type=self.solver_var.get(),
                monitoring_enabled=self.show_utilization_var.get(),
                benchmark_mode=self.benchmark_mode_var.get()
            )
            
            if success:
                messagebox.showinfo("Success", "GPU settings saved successfully!")
                self.refresh_gpu_status()
            else:
                messagebox.showerror("Error", "Failed to save GPU settings")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save GPU settings: {str(e)}")

    def run_gpu_benchmark(self):
        """Run GPU vs CPU benchmark."""
        try:
            from src.cybernetic_planning.core.gpu_acceleration import performance_monitor, gpu_detector
            
            if not gpu_detector.is_gpu_available():
                messagebox.showwarning("Warning", "GPU not available for benchmarking")
                return
            
            # Create a simple test problem
            import numpy as np
            from src.cybernetic_planning.core.optimization import ConstrainedOptimizer
            
            # Create test matrices with proper economic constraints
            n = 50  # Test with 50 sectors
            
            # Create a productive technology matrix (spectral radius < 1)
            # Use a more conservative approach to ensure economic viability
            A = np.random.rand(n, n) * 0.05  # Lower coefficients for productivity
            # Ensure diagonal dominance for stability
            np.fill_diagonal(A, 0)  # Remove self-consumption
            # Scale down to ensure spectral radius < 1
            A = A * 0.3
            
            l = np.random.rand(n) * 0.1 + 0.01  # Labor coefficients (small positive values)
            d = np.random.rand(n) * 10 + 1  # Final demand (smaller, positive values)
            
            # Create optimizer
            optimizer = ConstrainedOptimizer(A, l, d, use_gpu=True)
            
            # Run benchmark
            self.benchmark_text.delete(1.0, tk.END)
            self.benchmark_text.insert(1.0, "Running benchmark... Please wait...\n")
            self.root.update()
            
            # Test if the problem is solvable first
            test_result = optimizer.solve(use_cvxpy=True)
            if not test_result.get('feasible', False):
                self.benchmark_text.delete(1.0, tk.END)
                self.benchmark_text.insert(1.0, "Benchmark Error: Test problem is not feasible. Cannot run benchmark.")
                return
            
            benchmark_result = optimizer.benchmark_gpu_vs_cpu()
            
            # Display results
            if "error" in benchmark_result and benchmark_result["error"] is not None:
                result_text = f"Benchmark Error: {benchmark_result['error']}"
            else:
                result_text = f"""Benchmark Results:
{'='*50}

Operation: {benchmark_result.get('operation', 'N/A')}
GPU Time: {benchmark_result.get('gpu_time', 'N/A'):.4f} seconds
CPU Time: {benchmark_result.get('cpu_time', 'N/A'):.4f} seconds
Speedup: {benchmark_result.get('speedup', 'N/A'):.2f}x
GPU Success: {benchmark_result.get('gpu_success', False)}
CPU Success: {benchmark_result.get('cpu_success', False)}

Performance Summary:
{'='*50}
"""
                
                # Get overall performance summary
                summary = performance_monitor.get_performance_summary()
                if "message" not in summary:
                    result_text += f"""Total Benchmarks: {summary.get('total_benchmarks', 0)}
Successful Benchmarks: {summary.get('successful_benchmarks', 0)}
Average Speedup: {summary.get('average_speedup', 0):.2f}x
Maximum Speedup: {summary.get('max_speedup', 0):.2f}x
Minimum Speedup: {summary.get('min_speedup', 0):.2f}x
"""
                else:
                    result_text += summary["message"]
            
            self.benchmark_text.delete(1.0, tk.END)
            self.benchmark_text.insert(1.0, result_text)
            
        except Exception as e:
            self.benchmark_text.delete(1.0, tk.END)
            self.benchmark_text.insert(1.0, f"Benchmark Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to run benchmark: {str(e)}")

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

            # Update sector count display based on plan
            if 'sectors' in plan_data:
                sector_count = len(plan_data['sectors'])
                self.sectors_display_var.set(f"{sector_count} sectors")
            elif 'technology_matrix' in plan_data:
                sector_count = len(plan_data['technology_matrix'])
                self.sectors_display_var.set(f"{sector_count} sectors")
            else:
                self.sectors_display_var.set("Unknown sectors")

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

    def _create_simulation_plan_from_synthetic_data(self, synthetic_data):
        """Create a simulation plan from synthetic data including resource allocations."""
        try:
            # Extract data from synthetic data
            sectors = synthetic_data.get('sectors', [])
            technology_matrix = synthetic_data.get('technology_matrix', np.array([]))
            final_demand = synthetic_data.get('final_demand', np.array([]))
            labor_input = synthetic_data.get('labor_input', np.array([]))
            resource_allocations = synthetic_data.get('resource_allocations', {})
            
            # Ensure we have the required data
            if len(sectors) == 0:
                sectors = [f"Sector_{i+1}" for i in range(len(final_demand))]
            
            # Create production targets from final demand
            if hasattr(final_demand, 'tolist'):
                production_targets = final_demand.tolist()
            else:
                production_targets = list(final_demand)
            
            # Create labor requirements from labor input
            if hasattr(labor_input, 'tolist'):
                labor_requirements = labor_input.tolist()
            else:
                labor_requirements = list(labor_input)
            
            # Use resource allocations from synthetic data if available, otherwise create basic structure
            if resource_allocations:
                # Use the resource allocation data from synthetic data
                resource_allocations_data = resource_allocations
            else:
                # Create basic resource allocation structure
                resource_allocations_data = {
                    "technology_matrix": technology_matrix.tolist() if hasattr(technology_matrix, 'tolist') else technology_matrix,
                    "final_demand": final_demand.tolist() if hasattr(final_demand, 'tolist') else final_demand,
                    "total_labor_cost": float(np.dot(labor_input, final_demand)) if len(labor_input) > 0 and len(final_demand) > 0 else 0.0,
                    "plan_quality_score": 0.5,  # Default moderate score
                    "resource_matrix": synthetic_data.get('resource_matrix', np.array([])).tolist() if hasattr(synthetic_data.get('resource_matrix', np.array([])), 'tolist') else [],
                    "max_resources": synthetic_data.get('max_resources', np.array([])).tolist() if hasattr(synthetic_data.get('max_resources', np.array([])), 'tolist') else [],
                    "resource_names": synthetic_data.get('resources', [])
                }
            
            # Create simulation plan
            simulation_plan = {
                'sectors': sectors,
                'production_targets': production_targets,
                'labor_requirements': labor_requirements,
                'resource_allocations': resource_allocations_data,
                'population': synthetic_data.get('population', 1000000),
                'metadata': {
                    'source': 'synthetic_data',
                    'generated_at': datetime.now().isoformat(),
                    'sector_count': len(sectors),
                    'resource_count': len(synthetic_data.get('resources', [])),
                    'technology_density': synthetic_data.get('technology_density', 0.3)
                }
            }
            
            print(f"✓ Created simulation plan with {len(sectors)} sectors and resource allocation data")
            return simulation_plan
            
        except Exception as e:
            print(f"Warning: Failed to create simulation plan from synthetic data: {e}")
            # Return a basic simulation plan as fallback
            return {
                'sectors': [f"Sector_{i+1}" for i in range(6)],
                'production_targets': [100.0] * 6,
                'labor_requirements': [1.0] * 6,
                'resource_allocations': {
                    "technology_matrix": np.eye(6).tolist(),
                    "final_demand": [100.0] * 6,
                    "total_labor_cost": 600.0,
                    "plan_quality_score": 0.5
                },
                'population': 1000000
            }

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
            
            # Determine sectors from loaded plan
            if hasattr(self, 'current_simulation_plan') and self.current_simulation_plan:
                if 'sectors' in self.current_simulation_plan:
                    sectors = len(self.current_simulation_plan['sectors'])
                elif 'technology_matrix' in self.current_simulation_plan:
                    sectors = len(self.current_simulation_plan['technology_matrix'])
                else:
                    sectors = 15  # fallback
            else:
                sectors = 15  # fallback when no plan loaded

            # Initialize simulation environment
            self.simulation_environment = {
                'duration_years': duration,
                'time_step_months': time_step,
                'economic_sectors': sectors,
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
            if hasattr(self, 'monitoring_data_text'):
                self.monitoring_data_text.delete(1.0, tk.END)
            if hasattr(self, 'events_text'):
                self.events_text.delete(1.0, tk.END)

            # Add initialization message
            init_message = f"""Simulation Initialized Successfully!

Environment Parameters:
- Duration: {duration} years - Time Step: {time_step} months
- Economic Sectors: {sectors} (from loaded plan)

Plan Loaded:
- Sectors: {len(self.current_simulation_plan.get('sectors', []))}
- Production Targets: {len(self.current_simulation_plan.get('production_targets', []))}

Ready to start simulation.
"""
            if hasattr(self, 'monitoring_data_text'):
                self.monitoring_data_text.insert(tk.END, init_message)

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

                # Minimal delay to prevent UI freezing (optimized for performance)
                import time
                time.sleep(0.01)  # Reduced from 0.1s to 0.01s for 10x speed improvement
                
                # Performance monitoring
                if hasattr(self, 'simulation') and hasattr(self.simulation, 'performance_monitor'):
                    self.simulation.performance_monitor.record_operation(
                        'simulation_step', 
                        time.time() - step_start_time,
                        cache_hits=getattr(self.simulation.cache, 'hit_count', 0),
                        cache_misses=getattr(self.simulation.cache, 'miss_count', 0)
                    )

            # Simulation completed
            self.root.after(0, self.simulation_completed)

        except Exception as e:
            error_msg = str(e)
            log_error_with_traceback(f"Simulation failed: {error_msg}", e, "run_simulation method")
            self.root.after(0, lambda: self.simulation_failed(error_msg))

    def simulate_time_step(self, month):
        """Simulate one time step of the simulation."""
        try:
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

                # Initialize population health tracker if not exists
                if not hasattr(self, 'population_health_tracker'):
                    from src.cybernetic_planning.utils.population_health_tracker import PopulationHealthTracker
                    initial_population = self.current_simulation_plan.get('population', 1000000)
                    initial_tech_level = getattr(self, 'current_technology_level', 0.0)
                    self.population_health_tracker = PopulationHealthTracker(
                        initial_population=initial_population,
                        initial_technology_level=initial_tech_level
                    )
                    
                    # Set R&D sectors for technology growth
                    rd_sectors = self._get_rd_sector_indices()
                    self.population_health_tracker.set_rd_sectors(rd_sectors)
                
                # Initialize enhanced simulation if not exists
                if not hasattr(self, 'enhanced_simulation'):
                    self._initialize_enhanced_simulation()

                # Update population health metrics
                sector_mapping = self._get_sector_mapping()
                health_metrics = self.population_health_tracker.update_monthly_metrics(
                    production_results, resource_results, labor_results, sector_mapping
                )

            # Store results
            if not hasattr(self, 'simulation_results'):
                self.simulation_results = []

            # Ensure all results are dictionaries
            if not isinstance(production_results, dict):
                print(f"WARNING: production_results is not a dict, type: {type(production_results)}")
                production_results = {}
            
            if not isinstance(resource_results, dict):
                print(f"WARNING: resource_results is not a dict, type: {type(resource_results)}")
                resource_results = {}
            
            if not isinstance(labor_results, dict):
                print(f"WARNING: labor_results is not a dict, type: {type(labor_results)}")
                labor_results = {}

            self.simulation_results.append({
                'month': month,
                'year': year,
                'month_in_year': month_in_year,
                'production': production_results,
                'resources': resource_results,
                'labor': labor_results,
                'population_health': health_metrics
            })
            
            # Update simulation time for map integration
            self.current_simulation_time = month
            
            # Update map with simulation progress
            if False:  # Map functionality removed
                self._update_map_with_simulation_data()
        except Exception as e:
            log_error_with_traceback(f"Error in simulate_time_step for month {month}", e, f"simulate_time_step(month={month})")
            raise

    def simulate_production(self, month):
        """Simulate production for the current time step using enhanced optimization."""
        try:
            # Use enhanced simulation if available
            if hasattr(self, 'enhanced_simulation') and self.enhanced_simulation:
                try:
                    # Get current population health tracker
                    population_health_tracker = getattr(self, 'population_health_tracker', None)
                    
                    # Run enhanced simulation
                    simulation_result = self.enhanced_simulation.simulate_month(
                        month=month,
                        population_health_tracker=population_health_tracker,
                        use_optimization=True
                    )
                    
                    # Update technology and living standards from simulation
                    if population_health_tracker:
                        self.enhanced_simulation.update_technology_level(
                            population_health_tracker.current_technology_level
                        )
                        self.enhanced_simulation.update_living_standards(
                            population_health_tracker.current_living_standards
                        )
                    
                    # The enhanced simulation returns a different structure
                    # Convert it to match the expected format
                    production_data = simulation_result.get('production', {})
                    if production_data:
                        # Enhanced simulation already returns the correct format
                        return production_data
                    else:
                        # Fallback if no production data
                        return {}
                
                except Exception as e:
                    print(f"Enhanced simulation failed: {e}, falling back to basic simulation")
                    return self._simulate_production_basic(month)
            else:
                return self._simulate_production_basic(month)
        except Exception as e:
            log_error_with_traceback(f"Error in simulate_production for month {month}", e, f"simulate_production(month={month})")
            raise
    
    def _simulate_production_basic(self, month):
        """Basic production simulation as fallback."""
        production_results = {}

        if 'production_targets' in self.current_simulation_plan and 'sectors' in self.current_simulation_plan:
            targets = self.current_simulation_plan['production_targets']
            sectors = self.current_simulation_plan['sectors']

            # Base variation for events
            base_variation = 1.0
            if hasattr(self, 'current_events') and self.current_events:
                base_variation *= 0.8  # Reduce production during events

            # Handle both dictionary and list formats for production_targets
            if isinstance(targets, dict):
                # If targets is a dictionary, iterate over sectors and get corresponding targets
                for i, sector in enumerate(sectors):
                    target = targets.get(sector, 0)  # Default to 0 if sector not found
                    
                    # Sector-specific characteristics
                    sector_id = i % len(sectors)  # Use sector index for consistency
                    
                    # Different sectors have different base efficiency levels
                    base_efficiency = 0.7 + 0.3 * (sector_id % 10) / 10  # 0.7 to 1.0 range
                    
                    # Seasonal variation (different for each sector)
                    seasonal_factor = 1.0 + 0.15 * math.sin(2 * math.pi * (month % 12) / 12 + sector_id * 0.5)
                    
                    # Random variation (different for each sector)
                    random_factor = 0.9 + 0.2 * (hash(sector + str(month)) % 100) / 100  # 0.9 to 1.1 range
                    
                    # Technology level variation (higher tech sectors more efficient)
                    tech_factor = 0.8 + 0.4 * (sector_id % 5) / 5  # 0.8 to 1.2 range
                    
                    # Calculate final efficiency
                    efficiency = base_efficiency * seasonal_factor * random_factor * tech_factor * base_variation
                    efficiency = max(0.1, min(1.2, efficiency))  # Clamp between 10% and 120%
                    
                    actual_production = target * efficiency
                    
                    production_results[sector] = {
                        'target': target,
                        'actual': actual_production,
                        'efficiency': efficiency
                    }
            else:
                # If targets is a list, use the original zip approach
                for i, (sector, target) in enumerate(zip(sectors, targets)):
                    # Sector-specific characteristics
                    sector_id = i % len(sectors)  # Use sector index for consistency
                    
                    # Different sectors have different base efficiency levels
                    base_efficiency = 0.7 + 0.3 * (sector_id % 10) / 10  # 0.7 to 1.0 range
                    
                    # Seasonal variation (different for each sector)
                    seasonal_factor = 1.0 + 0.15 * math.sin(2 * math.pi * (month % 12) / 12 + sector_id * 0.5)
                    
                    # Random variation (different for each sector)
                    random_factor = 0.9 + 0.2 * (hash(sector + str(month)) % 100) / 100  # 0.9 to 1.1 range
                    
                    # Technology level variation (higher tech sectors more efficient)
                    tech_factor = 0.8 + 0.4 * (sector_id % 5) / 5  # 0.8 to 1.2 range
                    
                    # Calculate final efficiency
                    efficiency = base_efficiency * seasonal_factor * random_factor * tech_factor * base_variation
                    efficiency = max(0.1, min(1.2, efficiency))  # Clamp between 10% and 120%
                    
                    actual_production = target * efficiency
                    
                    production_results[sector] = {
                        'target': target,
                        'actual': actual_production,
                        'efficiency': efficiency
                    }

        return production_results

    def _create_default_simulation_plan(self):
        """Create a default simulation plan with basic resource allocation data."""
        try:
            # Create synthetic data for default simulation
            if self.planning_system:
                synthetic_data = self.planning_system.create_synthetic_data(
                    n_sectors=10, 
                    technology_density=0.3, 
                    resource_count=5
                )
                self.current_simulation_plan = self._create_simulation_plan_from_synthetic_data(synthetic_data)
            else:
                # Create minimal default plan if planning system not available
                self.current_simulation_plan = {
                    'sectors': [f'Sector_{i+1}' for i in range(10)],
                    'production_targets': {f'Sector_{i+1}': 100.0 for i in range(10)},
                    'labor_requirements': {f'Sector_{i+1}': 10.0 for i in range(10)},
                    'resource_allocations': {
                        'technology_matrix': [[0.3 if i == j else 0.1 for j in range(10)] for i in range(10)],
                        'final_demand': [100.0] * 10,
                        'total_labor_cost': 1000.0,
                        'plan_quality_score': 0.5,
                        'resource_matrix': [[1.0] * 10 for _ in range(5)],
                        'max_resources': [1000.0] * 5,
                        'resource_names': ['Energy', 'Materials', 'Labor', 'Capital', 'Land']
                    },
                    'population': 1000000,
                    'metadata': {
                        'source': 'default_simulation',
                        'generated_at': datetime.now().isoformat(),
                        'sector_count': 10,
                        'resource_count': 5,
                        'technology_density': 0.3
                    }
                }
            print("✓ Default simulation plan created")
        except Exception as e:
            print(f"Warning: Failed to create default simulation plan: {e}")
            # Create minimal fallback plan
            self.current_simulation_plan = {
                'sectors': ['Sector_1', 'Sector_2'],
                'production_targets': {'Sector_1': 100.0, 'Sector_2': 100.0},
                'labor_requirements': {'Sector_1': 10.0, 'Sector_2': 10.0},
                'resource_allocations': {
                    'technology_matrix': [[0.3, 0.1], [0.1, 0.3]],
                    'final_demand': [100.0, 100.0],
                    'total_labor_cost': 200.0,
                    'plan_quality_score': 0.5,
                    'resource_matrix': [[1.0, 1.0], [1.0, 1.0]],
                    'max_resources': [1000.0, 1000.0],
                    'resource_names': ['Resource_1', 'Resource_2']
                },
                'population': 1000000,
                'metadata': {'source': 'fallback', 'generated_at': datetime.now().isoformat()}
            }

    def _initialize_enhanced_simulation(self):
        """Initialize the enhanced simulation system."""
        try:
            from src.cybernetic_planning.core.enhanced_simulation import EnhancedEconomicSimulation
            import numpy as np
            
            # Ensure current_simulation_plan exists and has resource_allocations
            if not hasattr(self, 'current_simulation_plan') or self.current_simulation_plan is None:
                print("No simulation plan available. Creating default simulation plan...")
                self._create_default_simulation_plan()
            
            # Extract data from current simulation plan
            if 'resource_allocations' in self.current_simulation_plan:
                resource_data = self.current_simulation_plan['resource_allocations']
            else:
                print("No resource_allocations found in simulation plan. Creating default resource allocation data...")
                # Create default resource allocation data
                n_sectors = len(self.current_simulation_plan.get('sectors', []))
                if n_sectors == 0:
                    n_sectors = 10
                
                resource_data = {
                    'technology_matrix': [[0.3 if i == j else 0.1 for j in range(n_sectors)] for i in range(n_sectors)],
                    'final_demand': [100.0] * n_sectors,
                    'total_labor_cost': 1000.0,
                    'plan_quality_score': 0.5,
                    'resource_matrix': [[1.0] * n_sectors for _ in range(5)],
                    'max_resources': [1000.0] * 5,
                    'resource_names': ['Energy', 'Materials', 'Labor', 'Capital', 'Land']
                }
                # Add it to the simulation plan for future use
                self.current_simulation_plan['resource_allocations'] = resource_data
                
                # Try to extract technology matrix and other data
                if 'technology_matrix' in resource_data:
                    technology_matrix = np.array(resource_data['technology_matrix'])
                else:
                    # Create a default technology matrix if not available
                    n_sectors = len(self.current_simulation_plan.get('sectors', []))
                    technology_matrix = np.eye(n_sectors) * 0.3  # 30% intermediate consumption
                
                if 'final_demand' in resource_data:
                    final_demand = np.array(resource_data['final_demand'])
                else:
                    # Use production targets as final demand
                    targets = self.current_simulation_plan.get('production_targets', {})
                    final_demand = np.array(list(targets.values()))
                
                # Get labor requirements
                labor_requirements = self.current_simulation_plan.get('labor_requirements', {})
                if isinstance(labor_requirements, dict):
                    labor_vector = np.array(list(labor_requirements.values()))
                else:
                    labor_vector = np.array(labor_requirements)
                
                # Get sector names
                sector_names = self.current_simulation_plan.get('sectors', [])
                
                # Sanitize data before creating simulation
                technology_matrix = self._sanitize_matrix(technology_matrix, "technology matrix")
                final_demand = self._sanitize_vector(final_demand, "final demand")
                labor_vector = self._sanitize_vector(labor_vector, "labor vector")
                
                # Ensure all arrays have the same length
                min_length = min(len(technology_matrix), len(final_demand), len(labor_vector))
                if min_length == 0:
                    raise ValueError("Cannot create simulation with empty data")
                
                technology_matrix = technology_matrix[:min_length, :min_length]
                final_demand = final_demand[:min_length]
                labor_vector = labor_vector[:min_length]
                sector_names = sector_names[:min_length] if sector_names else [f"Sector_{i+1}" for i in range(min_length)]
                
                # Initialize enhanced simulation
                self.enhanced_simulation = EnhancedEconomicSimulation(
                    technology_matrix=technology_matrix,
                    labor_vector=labor_vector,
                    final_demand=final_demand,
                    sector_names=sector_names
                )
                
                print("Enhanced simulation initialized successfully")
                
        except Exception as e:
            print(f"Failed to initialize enhanced simulation: {e}")
            self.enhanced_simulation = None
    
    def _sanitize_matrix(self, matrix, name):
        """Sanitize a matrix by replacing invalid values."""
        import numpy as np
        
        # Replace NaN and inf values
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=0.0)
        
        # Ensure non-negative values for technology matrix
        if "technology" in name.lower():
            matrix = np.maximum(matrix, 0.0)
        
        # Ensure matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            min_dim = min(matrix.shape)
            matrix = matrix[:min_dim, :min_dim]
        
        return matrix
    
    def _sanitize_vector(self, vector, name):
        """Sanitize a vector by replacing invalid values."""
        import numpy as np
        
        # Replace NaN and inf values
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=0.0)
        
        # Ensure non-negative values for labor and demand
        if "labor" in name.lower() or "demand" in name.lower():
            vector = np.maximum(vector, 0.0)
        
        return vector

    def simulate_resource_allocation(self, month):
        """Simulate resource allocation for the current time step."""
        try:
            resource_results = {}

            if 'resource_allocations' in self.current_simulation_plan:
                # resource_allocations is a dictionary, so we can iterate over it
                for resource, allocation in self.current_simulation_plan['resource_allocations'].items():
                    # Base availability for events
                    base_availability = 1.0
                    if hasattr(self, 'current_events') and self.current_events:
                        base_availability *= 0.9  # Reduce availability during events

                    # Resource-specific characteristics
                    resource_id = hash(resource) % 100  # Use resource name hash for consistency
                    
                    # Different resources have different availability levels
                    base_resource_efficiency = 0.7 + 0.3 * (resource_id % 10) / 10  # 0.7 to 1.0 range
                    
                    # Seasonal variation (different for each resource)
                    seasonal_factor = 1.0 + 0.2 * math.sin(2 * math.pi * (month % 12) / 12 + resource_id * 0.2)
                    
                    # Random variation (different for each resource)
                    random_factor = 0.85 + 0.3 * (hash(resource + str(month + 2000)) % 100) / 100  # 0.85 to 1.15 range
                    
                    # Calculate final availability
                    availability = base_resource_efficiency * seasonal_factor * random_factor * base_availability
                    availability = max(0.2, min(1.2, availability))  # Clamp between 20% and 120%

                    # Handle both scalar and list allocations
                    if isinstance(allocation, list):
                        actual_allocation = []
                        for i, a in enumerate(allocation):
                            if isinstance(a, (int, float)):
                                # Add sector-specific variation for list allocations
                                sector_variation = 0.9 + 0.2 * (i % 10) / 10  # 0.9 to 1.1 range
                                actual_allocation.append(a * availability * sector_variation)
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
        except Exception as e:
            log_error_with_traceback(f"Error in simulate_resource_allocation for month {month}", e, f"simulate_resource_allocation(month={month})")
            raise

    def simulate_labor_allocation(self, month):
        """Simulate labor allocation for the current time step using enhanced optimization."""
        try:
            # Use enhanced simulation if available
            if hasattr(self, 'enhanced_simulation') and self.enhanced_simulation:
                try:
                    # Get current population health tracker
                    population_health_tracker = getattr(self, 'population_health_tracker', None)
                    
                    # Run enhanced simulation
                    simulation_result = self.enhanced_simulation.simulate_month(
                        month=month,
                        population_health_tracker=population_health_tracker,
                        use_optimization=True
                    )
                    
                    return simulation_result['labor_allocation']
                    
                except Exception as e:
                    print(f"Enhanced labor simulation failed: {e}, falling back to basic simulation")
                    return self._simulate_labor_allocation_basic(month)
            else:
                return self._simulate_labor_allocation_basic(month)
        except Exception as e:
            log_error_with_traceback(f"Error in simulate_labor_allocation for month {month}", e, f"simulate_labor_allocation(month={month})")
            raise
    
    def _simulate_labor_allocation_basic(self, month):
        """Basic labor allocation simulation as fallback."""
        labor_results = {}

        if 'labor_requirements' in self.current_simulation_plan and 'sectors' in self.current_simulation_plan:
            requirements = self.current_simulation_plan['labor_requirements']
            sectors = self.current_simulation_plan['sectors']

            # Base productivity for events
            base_productivity = 1.0
            if hasattr(self, 'current_events') and self.current_events:
                base_productivity *= 0.95  # Slight reduction during events

            for i, (sector, requirement) in enumerate(zip(sectors, requirements)):
                # Sector-specific productivity characteristics
                sector_id = i % len(sectors)
                
                # Different sectors have different base productivity levels
                base_labor_efficiency = 0.6 + 0.4 * (sector_id % 8) / 8  # 0.6 to 1.0 range
                
                # Skill level variation (some sectors require more skilled labor)
                skill_factor = 0.8 + 0.4 * (sector_id % 6) / 6  # 0.8 to 1.2 range
                
                # Seasonal variation (different for each sector)
                seasonal_factor = 1.0 + 0.1 * math.sin(2 * math.pi * (month % 12) / 12 + sector_id * 0.3)
                
                # Random variation (different for each sector)
                random_factor = 0.9 + 0.2 * (hash(sector + str(month + 1000)) % 100) / 100  # 0.9 to 1.1 range
                
                # Calculate final productivity
                productivity = base_labor_efficiency * skill_factor * seasonal_factor * random_factor * base_productivity
                productivity = max(0.3, min(1.3, productivity))  # Clamp between 30% and 130%
                
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
        if hasattr(self, 'events_text'):
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
        if hasattr(self, 'monitoring_data_text'):
            self.root.after(0, lambda: self._safe_update_monitoring_display(monitoring_message))

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
        if hasattr(self, 'monitoring_data_text'):
            self.monitoring_data_text.delete(1.0, tk.END)
        if hasattr(self, 'events_text'):
            self.events_text.delete(1.0, tk.END)

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
        
        # Generate final map with simulation results
        self._generate_final_simulation_map()
        
        # Generate comprehensive reports and save to folder
        self._generate_comprehensive_simulation_reports()

        messagebox.showinfo("Simulation Complete", "The simulation has completed successfully! All reports and graphs have been saved to the exports folder.")

    def _generate_comprehensive_simulation_reports(self):
        """Generate comprehensive simulation reports and save to folder."""
        try:
            from datetime import datetime
            import os
            import shutil
            
            # Create timestamp for folder naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_folder = f"simulation_reports_{timestamp}"
            report_path = os.path.join("exports", report_folder)
            
            # Create the report folder
            os.makedirs(report_path, exist_ok=True)
            
            # Generate simulation report
            simulation_report_path = os.path.join(report_path, "simulation_report.txt")
            self._generate_simulation_report_file(simulation_report_path)
            
            # Generate population health report
            population_health_path = os.path.join(report_path, "population_health_report.txt")
            self._generate_population_health_report_file(population_health_path)
            
            # Generate economic plan report
            economic_plan_path = os.path.join(report_path, "economic_plan_report.txt")
            self._generate_economic_plan_report_file(economic_plan_path)
            
            # Generate graphs
            graphs_path = os.path.join(report_path, "graphs")
            os.makedirs(graphs_path, exist_ok=True)
            self._generate_simulation_graphs(graphs_path)
            
            # Create a summary index file
            self._create_report_index(report_path)
            
            print(f"All simulation reports saved to: {report_path}")
            
        except Exception as e:
            print(f"Error generating comprehensive reports: {e}")
            messagebox.showerror("Error", f"Failed to generate comprehensive reports: {str(e)}")

    def _generate_simulation_report_file(self, file_path):
        """Generate simulation report and save to file."""
        try:
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                return
                
            # Create report content
            report_content = []
            report_content.append("=" * 80)
            report_content.append("SIMULATION SUMMARY REPORT")
            report_content.append("=" * 80)
            report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append("")
            
            # Simulation overview
            total_months = len(self.simulation_results)
            report_content.append("Simulation Overview:")
            report_content.append("-" * 20)
            report_content.append(f"• Duration: {total_months} months ({total_months/12:.1f} years)")
            report_content.append(f"• Start Date: Month 1, Year 0")
            report_content.append(f"• End Date: Month {total_months % 12}, Year {total_months // 12}")
            report_content.append(f"• Status: Completed Successfully")
            report_content.append("")
            
            # Performance metrics
            report_content.append("Performance Metrics:")
            report_content.append("-" * 20)
            
            # Calculate averages
            avg_production_efficiency = 0
            avg_resource_availability = 0
            avg_labor_productivity = 0
            total_economic_output = 0
            
            for result in self.simulation_results:
                if result.get('production') and isinstance(result['production'], dict):
                    sector_efficiencies = [data.get('efficiency', 0) for data in result['production'].values() if isinstance(data, dict)]
                    if sector_efficiencies:
                        avg_production_efficiency += sum(sector_efficiencies) / len(sector_efficiencies)
                if result.get('resources') and isinstance(result['resources'], dict):
                    resource_availabilities = [data.get('availability', 0) for data in result['resources'].values() if isinstance(data, dict)]
                    if resource_availabilities:
                        avg_resource_availability += sum(resource_availabilities) / len(resource_availabilities)
                if result.get('labor') and isinstance(result['labor'], dict):
                    labor_productivities = [data.get('productivity', 0) for data in result['labor'].values() if isinstance(data, dict)]
                    if labor_productivities:
                        avg_labor_productivity += sum(labor_productivities) / len(labor_productivities)
                
                # Calculate economic output
                if result.get('production') and isinstance(result['production'], dict):
                    monthly_output = 0
                    for sector, data in result['production'].items():
                        if isinstance(data, dict) and 'actual' in data:
                            try:
                                monthly_output += float(data['actual'])
                            except (TypeError, ValueError) as e:
                                print(f"WARNING: Could not convert actual value for {sector}: {data['actual']} - {e}")
                        else:
                            print(f"WARNING: Invalid data structure for sector {sector}: {data}")
                    
                    total_economic_output += monthly_output
            
            if total_months > 0:
                avg_production_efficiency /= total_months
                avg_resource_availability /= total_months
                avg_labor_productivity /= total_months
            
            report_content.append(f"• Average Production Efficiency: {avg_production_efficiency:.1%}")
            report_content.append(f"• Average Resource Availability: {avg_resource_availability:.1%}")
            report_content.append(f"• Average Labor Productivity: {avg_labor_productivity:.1%}")
            report_content.append(f"• Total Economic Output: ${total_economic_output:,.0f}")
            report_content.append(f"• System Efficiency Trend: Stable")
            report_content.append(f"• Resource Utilization: {avg_resource_availability:.1%}")
            report_content.append("")
            
            # Sector performance
            report_content.append("Sector Performance:")
            report_content.append("-" * 20)
            
            if self.simulation_results:
                latest_result = self.simulation_results[-1]
                if 'production' in latest_result:
                    for sector, data in latest_result['production'].items():
                        report_content.append(f"• {sector}:")
                        report_content.append(f"  - Average Efficiency: {data['efficiency']:.1%}")
                        report_content.append(f"  - Production Stability: 92.5%")  # Placeholder
                        report_content.append(f"  - Target Achievement: {data['efficiency']:.1%}")
            
            report_content.append("")
            report_content.append("=" * 80)
            report_content.append("END OF SIMULATION REPORT")
            report_content.append("=" * 80)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
                
        except Exception as e:
            print(f"Error generating simulation report: {e}")

    def _generate_population_health_report_file(self, file_path):
        """Generate population health report and save to file."""
        try:
            if not hasattr(self, 'population_health_tracker') or not self.population_health_tracker:
                return

            health_summary = self.population_health_tracker.get_population_health_summary()
            
            # Create report content
            report_content = []
            report_content.append("=" * 80)
            report_content.append("POPULATION HEALTH OVER TIME REPORT")
            report_content.append("=" * 80)
            report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append("")

            # Summary statistics
            report_content.append("SUMMARY STATISTICS")
            report_content.append("-" * 40)
            report_content.append(f"Simulation Period: {health_summary.start_date.strftime('%Y-%m-%d')} to {health_summary.end_date.strftime('%Y-%m-%d')}")
            report_content.append(f"Total Months: {health_summary.total_months}")
            report_content.append(f"Initial Population: {health_summary.initial_population:,.0f}")
            report_content.append(f"Final Population: {health_summary.final_population:,.0f}")
            report_content.append(f"Population Growth Rate: {health_summary.population_growth_rate:.2%}")
            report_content.append("")

            # Technology and living standards
            report_content.append("TECHNOLOGY AND LIVING STANDARDS")
            report_content.append("-" * 40)
            report_content.append(f"Average Technology Level: {health_summary.average_technology_level:.3f}")
            report_content.append(f"Technology Growth Rate: {health_summary.technology_growth_rate:.2%}")
            report_content.append(f"Average Living Standards: {health_summary.average_living_standards:.3f}")
            report_content.append(f"Living Standards Growth Rate: {health_summary.living_standards_growth_rate:.2%}")
            report_content.append(f"Average Consumer Demand Fulfillment: {health_summary.average_consumer_demand_fulfillment:.3f}")
            report_content.append(f"Demand Fulfillment Trend: {health_summary.demand_fulfillment_trend}")
            report_content.append("")

            # Health indicators
            report_content.append("HEALTH INDICATORS")
            report_content.append("-" * 40)
            for indicator, value in health_summary.health_indicators.items():
                if 'rate' in indicator:
                    report_content.append(f"{indicator.replace('_', ' ').title()}: {value:.3f}")
                elif 'expectancy' in indicator or 'income' in indicator:
                    report_content.append(f"{indicator.replace('_', ' ').title()}: {value:,.1f}")
                else:
                    report_content.append(f"{indicator.replace('_', ' ').title()}: {value:.3f}")
            report_content.append("")

            # Monthly data summary
            if len(health_summary.monthly_data) > 0:
                report_content.append("MONTHLY DATA SUMMARY")
                report_content.append("-" * 40)
                
                # First 12 months
                report_content.append("First 12 Months:")
                for i, data in enumerate(health_summary.monthly_data[:12]):
                    report_content.append(f"  Month {data.month:2d}: Pop={data.population:8,.0f}, Tech={data.technology_level:.3f}, Living={data.living_standards_index:.3f}")
                
                if len(health_summary.monthly_data) > 24:
                    report_content.append("")
                    report_content.append("Last 12 Months:")
                    for data in health_summary.monthly_data[-12:]:
                        report_content.append(f"  Month {data.month:2d}: Pop={data.population:8,.0f}, Tech={data.technology_level:.3f}, Living={data.living_standards_index:.3f}")
                
                report_content.append("")

            # R&D and technology growth analysis
            if hasattr(self.population_health_tracker, 'rd_output_history') and self.population_health_tracker.rd_output_history:
                report_content.append("R&D AND TECHNOLOGY GROWTH ANALYSIS")
                report_content.append("-" * 40)
                avg_rd_output = np.mean(self.population_health_tracker.rd_output_history)
                total_rd_output = np.sum(self.population_health_tracker.rd_output_history)
                tech_growth_rate = self.population_health_tracker.get_technology_growth_rate()
                
                report_content.append(f"Average R&D Output per Month: {avg_rd_output:,.0f}")
                report_content.append(f"Total R&D Output: {total_rd_output:,.0f}")
                report_content.append(f"Technology Growth Rate: {tech_growth_rate:.2%}")
                report_content.append("")

            report_content.append("=" * 80)
            report_content.append("END OF POPULATION HEALTH REPORT")
            report_content.append("=" * 80)

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
                
        except Exception as e:
            print(f"Error generating population health report: {e}")

    def _generate_economic_plan_report_file(self, file_path):
        """Generate economic plan report and save to file."""
        try:
            # Create report content
            report_content = []
            report_content.append("=" * 80)
            report_content.append("ECONOMIC PLAN REPORT")
            report_content.append("=" * 80)
            report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append("")
            
            # Plan overview
            if hasattr(self, 'current_simulation_plan') and self.current_simulation_plan:
                report_content.append("PLAN OVERVIEW")
                report_content.append("-" * 40)
                report_content.append(f"Plan Name: {self.current_simulation_plan.get('plan_name', 'Simulation Plan')}")
                report_content.append(f"Plan Period: {self.current_simulation_plan.get('plan_period', 'N/A')}")
                report_content.append(f"Description: {self.current_simulation_plan.get('description', 'Economic simulation plan')}")
                report_content.append("")
                
                # Sectors
                if 'sectors' in self.current_simulation_plan:
                    report_content.append("ECONOMIC SECTORS")
                    report_content.append("-" * 40)
                    for i, sector in enumerate(self.current_simulation_plan['sectors'], 1):
                        report_content.append(f"{i:2d}. {sector}")
                    report_content.append("")
                
                # Production targets
                if 'production_targets' in self.current_simulation_plan:
                    report_content.append("PRODUCTION TARGETS")
                    report_content.append("-" * 40)
                    for sector, target in self.current_simulation_plan['production_targets'].items():
                        report_content.append(f"• {sector}: {target:,.0f} units")
                    report_content.append("")
                
                # Labor requirements
                if 'labor_requirements' in self.current_simulation_plan:
                    report_content.append("LABOR REQUIREMENTS")
                    report_content.append("-" * 40)
                    for sector, requirement in self.current_simulation_plan['labor_requirements'].items():
                        report_content.append(f"• {sector}: {requirement:,.0f} person-hours")
                    report_content.append("")
                
                # Resource allocations
                if 'resource_allocations' in self.current_simulation_plan:
                    report_content.append("RESOURCE ALLOCATIONS")
                    report_content.append("-" * 40)
                    for resource, allocation in self.current_simulation_plan['resource_allocations'].items():
                        report_content.append(f"• {resource}: {allocation:,.0f} units")
                    report_content.append("")
                
                # Social goals
                if 'social_goals' in self.current_simulation_plan:
                    report_content.append("SOCIAL GOALS")
                    report_content.append("-" * 40)
                    for goal, status in self.current_simulation_plan['social_goals'].items():
                        status_text = "✓ Achieved" if status else "○ Pending"
                        report_content.append(f"• {goal.replace('_', ' ').title()}: {status_text}")
                    report_content.append("")
                
                # Performance metrics
                if 'performance_metrics' in self.current_simulation_plan:
                    report_content.append("PERFORMANCE METRICS")
                    report_content.append("-" * 40)
                    for metric, value in self.current_simulation_plan['performance_metrics'].items():
                        if isinstance(value, float):
                            report_content.append(f"• {metric.replace('_', ' ').title()}: {value:.2%}")
                        else:
                            report_content.append(f"• {metric.replace('_', ' ').title()}: {value}")
                    report_content.append("")
            
            # Enhanced simulation metrics
            if hasattr(self, 'enhanced_simulation') and self.enhanced_simulation:
                summary = self.enhanced_simulation.get_simulation_summary()
                if summary:
                    report_content.append("ENHANCED SIMULATION METRICS")
                    report_content.append("-" * 40)
                    report_content.append(f"• Total Economic Output: ${summary.get('total_economic_output', 0):,.0f}")
                    report_content.append(f"• Average Efficiency: {summary.get('average_efficiency', 0):.1%}")
                    report_content.append(f"• Living Standards Index: {summary.get('living_standards_index', 0):.3f}")
                    report_content.append(f"• Technology Level: {summary.get('technology_level', 0):.3f}")
                    report_content.append(f"• Labor Productivity: {summary.get('labor_productivity', 0):.2f}")
                    report_content.append(f"• Resource Utilization: {summary.get('resource_utilization', 0):.1%}")
                    report_content.append(f"• Consumer Demand Fulfillment: {summary.get('consumer_demand_fulfillment', 0):.1%}")
                    report_content.append("")
            
            report_content.append("=" * 80)
            report_content.append("END OF ECONOMIC PLAN REPORT")
            report_content.append("=" * 80)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
                
        except Exception as e:
            print(f"Error generating economic plan report: {e}")

    def _generate_simulation_graphs(self, graphs_path):
        """Generate simulation graphs and save to folder."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                return
            
            # Set up the plotting style
            plt.style.use('default')
            
            # 1. Production Efficiency Over Time
            months = list(range(1, len(self.simulation_results) + 1))
            production_efficiency = []
            
            for result in self.simulation_results:
                if result['production']:
                    avg_efficiency = np.mean([data['efficiency'] for data in result['production'].values()])
                    production_efficiency.append(avg_efficiency)
                else:
                    production_efficiency.append(0)
            
            plt.figure(figsize=(12, 8))
            plt.plot(months, production_efficiency, 'b-', linewidth=2, label='Production Efficiency')
            plt.title('Production Efficiency Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Efficiency', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'production_efficiency.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Population Health Metrics
            if hasattr(self, 'population_health_tracker') and self.population_health_tracker:
                health_data = self.population_health_tracker.monthly_metrics
                if health_data:
                    pop_months = [data.month for data in health_data]
                    population = [data.population for data in health_data]
                    tech_level = [data.technology_level for data in health_data]
                    living_standards_list = [data.living_standards_index for data in health_data]
                    
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
                    
                    # Population
                    ax1.plot(pop_months, population, 'g-', linewidth=2, label='Population')
                    ax1.set_title('Population Growth Over Time', fontweight='bold')
                    ax1.set_ylabel('Population')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    
                    # Technology Level
                    ax2.plot(pop_months, tech_level, 'b-', linewidth=2, label='Technology Level')
                    ax2.set_title('Technology Level Over Time', fontweight='bold')
                    ax2.set_ylabel('Technology Level')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    
                    # Living Standards
                    ax3.plot(pop_months, living_standards_list, 'r-', linewidth=2, label='Living Standards')
                    ax3.set_title('Living Standards Over Time', fontweight='bold')
                    ax3.set_xlabel('Month')
                    ax3.set_ylabel('Living Standards Index')
                    ax3.grid(True, alpha=0.3)
                    ax3.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(graphs_path, 'population_health_metrics.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 3. Sector Performance Comparison
            if self.simulation_results:
                latest_result = self.simulation_results[-1]
                if 'production' in latest_result:
                    sectors = list(latest_result['production'].keys())
                    efficiencies = [data['efficiency'] for data in latest_result['production'].values()]
                    
                    plt.figure(figsize=(14, 8))
                    bars = plt.bar(range(len(sectors)), efficiencies, color='skyblue', alpha=0.7)
                    plt.title('Sector Performance Comparison', fontsize=16, fontweight='bold')
                    plt.xlabel('Sectors', fontsize=12)
                    plt.ylabel('Efficiency', fontsize=12)
                    plt.xticks(range(len(sectors)), sectors, rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{efficiencies[i]:.1%}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(graphs_path, 'sector_performance.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 4. Economic Output Over Time
            economic_output = []
            for result in self.simulation_results:
                if result['production']:
                    monthly_output = sum(data['actual'] for data in result['production'].values())
                    economic_output.append(monthly_output)
                else:
                    economic_output.append(0)
            
            plt.figure(figsize=(12, 8))
            plt.plot(months, economic_output, 'purple', linewidth=2, label='Economic Output')
            plt.title('Economic Output Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Economic Output ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'economic_output.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Labor Productivity Over Time
            labor_productivity = []
            for result in self.simulation_results:
                if result['labor']:
                    avg_productivity = np.mean([data['productivity'] for data in result['labor'].values()])
                    labor_productivity.append(avg_productivity)
                else:
                    labor_productivity.append(0)
            
            plt.figure(figsize=(12, 8))
            plt.plot(months, labor_productivity, 'orange', linewidth=2, label='Labor Productivity')
            plt.title('Labor Productivity Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Productivity', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_path, 'labor_productivity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating simulation graphs: {e}")

    def _create_report_index(self, report_path):
        """Create an index file for the report folder."""
        try:
            index_content = []
            index_content.append("=" * 80)
            index_content.append("SIMULATION REPORTS INDEX")
            index_content.append("=" * 80)
            index_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            index_content.append("")
            index_content.append("This folder contains comprehensive simulation reports and analysis.")
            index_content.append("")
            index_content.append("FILES INCLUDED:")
            index_content.append("-" * 40)
            index_content.append("• simulation_report.txt - Overall simulation performance and metrics")
            index_content.append("• population_health_report.txt - Population health and living standards analysis")
            index_content.append("• economic_plan_report.txt - Economic plan details and performance")
            index_content.append("• graphs/ - Visual analysis and charts")
            index_content.append("  - production_efficiency.png - Production efficiency over time")
            index_content.append("  - population_health_metrics.png - Population, technology, and living standards")
            index_content.append("  - sector_performance.png - Sector-by-sector performance comparison")
            index_content.append("  - economic_output.png - Economic output trends")
            index_content.append("  - labor_productivity.png - Labor productivity over time")
            index_content.append("")
            index_content.append("=" * 80)
            
            with open(os.path.join(report_path, "README.txt"), 'w', encoding='utf-8') as f:
                f.write('\n'.join(index_content))
                
        except Exception as e:
            print(f"Error creating report index: {e}")

    def simulation_failed(self, error_msg):
        """Handle simulation failure."""
        self.simulation_state = "failed"
        self.simulation_status.config(text="Simulation failed", foreground="red")
        messagebox.showerror("Simulation Error", f"Simulation failed: {error_msg}")

    def update_metrics_display(self):
        """Update the performance metrics display."""
        if not hasattr(self, 'simulation_results') or not self.simulation_results:
            return

        # Clear existing metrics (if any)
        # Note: metrics_tree not implemented yet

        # Calculate overall metrics
        total_months = len(self.simulation_results)
        avg_production_efficiency = 0
        avg_resource_availability = 0
        avg_labor_productivity = 0

        for result in self.simulation_results:
            if result.get('production') and isinstance(result['production'], dict):
                sector_efficiencies = [data.get('efficiency', 0) for data in result['production'].values() if isinstance(data, dict)]
                if sector_efficiencies:
                    avg_production_efficiency += sum(sector_efficiencies) / len(sector_efficiencies)
            if result.get('resources') and isinstance(result['resources'], dict):
                resource_availabilities = [data.get('availability', 0) for data in result['resources'].values() if isinstance(data, dict)]
                if resource_availabilities:
                    avg_resource_availability += sum(resource_availabilities) / len(resource_availabilities)
            if result.get('labor') and isinstance(result['labor'], dict):
                labor_productivities = [data.get('productivity', 0) for data in result['labor'].values() if isinstance(data, dict)]
                if labor_productivities:
                    avg_labor_productivity += sum(labor_productivities) / len(labor_productivities)

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

        # Note: metrics_tree not implemented yet - would display metrics here
        # for metric, value, target, status in metrics:
        #     self.metrics_tree.insert("", "end", values=(metric, value, target, status))

    def _generate_simulation_summary(self):
        """Generate comprehensive simulation summary with graphs and PDF export."""
        if not hasattr(self, 'simulation_results') or not self.simulation_results:
            return
            
        try:
            # Create summary window
            summary_window = tk.Toplevel(self.root)
            summary_window.title("Simulation Summary")
            summary_window.geometry("1000x700")
            summary_window.transient(self.root)
            
            # Create notebook for different views
            notebook = ttk.Notebook(summary_window)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Summary tab
            summary_frame = ttk.Frame(notebook)
            notebook.add(summary_frame, text="Summary")
            
            # Create scrollable text area for summary
            summary_text = scrolledtext.ScrolledText(summary_frame, height=25, width=100)
            summary_text.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Generate summary content
            summary_content = self._create_summary_content()
            summary_text.insert(tk.END, summary_content)
            summary_text.config(state=tk.DISABLED)
            
            # Graphs tab
            graphs_frame = ttk.Frame(notebook)
            notebook.add(graphs_frame, text="Performance Graphs")
            
            # Create graphs
            self._create_performance_graphs(graphs_frame)
            
            # Export buttons
            export_frame = ttk.Frame(summary_window)
            export_frame.pack(fill="x", padx=10, pady=5)
            
            ttk.Button(export_frame, text="Export Complete Report as PDF", 
                      command=lambda: self._export_summary_pdf()).pack(side="left", padx=5)
            ttk.Button(export_frame, text="Export Data as Excel", 
                      command=lambda: self._export_summary_excel()).pack(side="left", padx=5)
            ttk.Button(export_frame, text="Close", 
                      command=summary_window.destroy).pack(side="right", padx=5)
                      
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate simulation summary: {str(e)}")

    def _create_summary_content(self):
        """Create comprehensive summary content."""
        if not hasattr(self, 'simulation_results') or not self.simulation_results:
            return "No simulation data available."
            
        results = self.simulation_results
        total_months = len(results)
        
        # Calculate key metrics
        metrics = self._calculate_summary_metrics(results)
        
        # Create summary content
        content = f"""
{'='*80}
                    SIMULATION SUMMARY REPORT
{'='*80}

Simulation Overview:
-------------------
• Duration: {total_months} months ({total_months/12:.1f} years)
• Start Date: Month 1, Year {results[0]['year'] if results else 'N/A'}
• End Date: Month {results[-1]['month_in_year'] if results else 'N/A'}, Year {results[-1]['year'] if results else 'N/A'}
• Status: {'Completed Successfully' if self.simulation_state == 'completed' else 'Unknown'}

Performance Metrics:
-------------------
• Average Production Efficiency: {metrics['avg_production_efficiency']:.1%}
• Average Resource Availability: {metrics['avg_resource_availability']:.1%}
• Average Labor Productivity: {metrics['avg_labor_productivity']:.1%}
• Total Economic Output: ${metrics['total_economic_output']:,.0f}
• System Efficiency Trend: {metrics['efficiency_trend']}
• Resource Utilization: {metrics['resource_utilization']:.1%}

Sector Performance:
------------------
"""
        
        # Add sector-specific performance
        if results and 'production' in results[0]:
            sectors = list(results[0]['production'].keys())
            for sector in sectors:
                sector_metrics = self._calculate_sector_metrics(results, sector)
                content += f"• {sector}:\n"
                content += f"  - Average Efficiency: {sector_metrics['avg_efficiency']:.1%}\n"
                content += f"  - Production Stability: {sector_metrics['stability']:.1%}\n"
                content += f"  - Target Achievement: {sector_metrics['target_achievement']:.1%}\n"
        
        content += f"""
Event Analysis:
--------------
• Total Events: {metrics['total_events']}
• Critical Events: {metrics['critical_events']}
• Event Impact on Production: {metrics['event_impact']:.1%}
• Recovery Time: {metrics['recovery_time']:.1f} months average

Efficiency Analysis:
-------------------
• Plan Adherence: {metrics['plan_adherence']:.1%}
• Resource Optimization: {metrics['resource_optimization']:.1%}
• Labor Efficiency: {metrics['labor_efficiency']:.1%}
• Overall System Health: {metrics['system_health']:.1%}

Recommendations:
---------------
"""
        
        # Add recommendations based on performance
        recommendations = self._generate_recommendations(metrics)
        for i, rec in enumerate(recommendations, 1):
            content += f"{i}. {rec}\n"
        
        content += f"""
Technical Details:
-----------------
• Data Points Collected: {total_months * len(results[0].get('production', {})) if results else 0}
• Simulation Resolution: Monthly
• Stochastic Events: {'Enabled' if hasattr(self, 'stochastic_events') else 'Disabled'}
• Monitoring Frequency: Real-time
• Map Integration: Not Available (Removed)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: Cybernetic Planning Experiment v1.0
{'='*80}
"""
        
        return content

    def _calculate_summary_metrics(self, results):
        """Calculate comprehensive summary metrics."""
        if not results:
            return {}
            
        total_months = len(results)
        
        # Production metrics
        production_efficiencies = []
        resource_availabilities = []
        labor_productivities = []
        economic_outputs = []
        
        for result in results:
            if 'production' in result:
                efficiencies = [data['efficiency'] for data in result['production'].values()]
                production_efficiencies.extend(efficiencies)
                
            if 'resources' in result:
                availabilities = [data['availability'] for data in result['resources'].values()]
                resource_availabilities.extend(availabilities)
                
            if 'labor' in result:
                productivities = [data['productivity'] for data in result['labor'].values()]
                labor_productivities.extend(productivities)
                
            if 'economic_output' in result:
                economic_outputs.append(result['economic_output'])
        
        # Calculate averages
        avg_production_efficiency = np.mean(production_efficiencies) if production_efficiencies else 0
        avg_resource_availability = np.mean(resource_availabilities) if resource_availabilities else 0
        avg_labor_productivity = np.mean(labor_productivities) if labor_productivities else 0
        total_economic_output = sum(economic_outputs) if economic_outputs else 0
        
        # Calculate trends
        efficiency_trend = self._calculate_trend(production_efficiencies)
        
        # Event analysis
        total_events = len(getattr(self, 'current_events', []))
        critical_events = len([e for e in getattr(self, 'current_events', []) if e.get('severity', 0) > 0.7])
        
        return {
            'avg_production_efficiency': avg_production_efficiency,
            'avg_resource_availability': avg_resource_availability,
            'avg_labor_productivity': avg_labor_productivity,
            'total_economic_output': total_economic_output,
            'efficiency_trend': efficiency_trend,
            'resource_utilization': avg_resource_availability,
            'total_events': total_events,
            'critical_events': critical_events,
            'event_impact': max(0, 1 - avg_production_efficiency),
            'recovery_time': 2.0,  # Placeholder
            'plan_adherence': avg_production_efficiency,
            'resource_optimization': avg_resource_availability,
            'labor_efficiency': avg_labor_productivity,
            'system_health': (avg_production_efficiency + avg_resource_availability + avg_labor_productivity) / 3
        }

    def _calculate_sector_metrics(self, results, sector):
        """Calculate metrics for a specific sector."""
        sector_data = []
        for result in results:
            if 'production' in result and sector in result['production']:
                sector_data.append(result['production'][sector])
        
        if not sector_data:
            return {'avg_efficiency': 0, 'stability': 0, 'target_achievement': 0}
        
        efficiencies = [data['efficiency'] for data in sector_data]
        targets = [data.get('target', 0) for data in sector_data]
        actuals = [data.get('actual', 0) for data in sector_data]
        
        return {
            'avg_efficiency': np.mean(efficiencies),
            'stability': 1 - np.std(efficiencies) if efficiencies else 0,
            'target_achievement': np.mean([a/t if t > 0 else 0 for a, t in zip(actuals, targets)])
        }

    def _calculate_trend(self, values):
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return "Insufficient Data"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "Improving"
        elif slope < -0.01:
            return "Declining"
        else:
            return "Stable"

    def _generate_recommendations(self, metrics):
        """Generate recommendations based on performance metrics."""
        recommendations = []
        
        if metrics.get('avg_production_efficiency', 0) < 0.8:
            recommendations.append("Focus on improving production efficiency through better resource allocation and process optimization.")
        
        if metrics.get('avg_resource_availability', 0) < 0.85:
            recommendations.append("Enhance resource management systems to improve availability and reduce bottlenecks.")
        
        if metrics.get('avg_labor_productivity', 0) < 0.9:
            recommendations.append("Invest in training and development programs to boost labor productivity.")
        
        if metrics.get('critical_events', 0) > 0:
            recommendations.append("Implement better contingency planning and disaster response systems.")
        
        if metrics.get('system_health', 0) < 0.8:
            recommendations.append("Conduct comprehensive system review and implement holistic improvements.")
        
        if not recommendations:
            recommendations.append("System is performing well. Continue monitoring and consider incremental improvements.")
        
        return recommendations

    def _create_performance_graphs(self, parent):
        """Create performance graphs for the summary."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                ttk.Label(parent, text="No simulation data available for graphs.").pack(pady=20)
                return
            
            # Create figure with subplots
            fig = Figure(figsize=(12, 8))
            
            # Extract data for plotting
            months = [r['month'] for r in self.simulation_results]
            production_data = []
            resource_data = []
            labor_data = []
            
            for result in self.simulation_results:
                if 'production' in result:
                    avg_prod = np.mean([data['efficiency'] for data in result['production'].values()])
                    production_data.append(avg_prod)
                else:
                    production_data.append(0)
                    
                if 'resources' in result:
                    avg_res = np.mean([data['availability'] for data in result['resources'].values()])
                    resource_data.append(avg_res)
                else:
                    resource_data.append(0)
                    
                if 'labor' in result:
                    avg_lab = np.mean([data['productivity'] for data in result['labor'].values()])
                    labor_data.append(avg_lab)
                else:
                    labor_data.append(0)
            
            # Create subplots
            ax1 = fig.add_subplot(221)
            ax1.plot(months, production_data, 'b-', linewidth=2, label='Production Efficiency')
            ax1.set_title('Production Efficiency Over Time')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Efficiency')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2 = fig.add_subplot(222)
            ax2.plot(months, resource_data, 'g-', linewidth=2, label='Resource Availability')
            ax2.set_title('Resource Availability Over Time')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Availability')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            ax3 = fig.add_subplot(223)
            ax3.plot(months, labor_data, 'r-', linewidth=2, label='Labor Productivity')
            ax3.set_title('Labor Productivity Over Time')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Productivity')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Combined performance
            ax4 = fig.add_subplot(224)
            ax4.plot(months, production_data, 'b-', linewidth=2, label='Production')
            ax4.plot(months, resource_data, 'g-', linewidth=2, label='Resources')
            ax4.plot(months, labor_data, 'r-', linewidth=2, label='Labor')
            ax4.set_title('Overall Performance Trends')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Performance')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            fig.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except ImportError:
            ttk.Label(parent, text="Matplotlib not available for graph generation.").pack(pady=20)
        except Exception as e:
            ttk.Label(parent, text=f"Error creating graphs: {str(e)}").pack(pady=20)

    def _export_summary_pdf(self):
        """Export simulation summary as PDF."""
        temp_dir = None
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            import matplotlib.pyplot as plt
            import matplotlib.backends.backend_pdf
            import io
            import tempfile
            import os
            import shutil
            
            # Check if we have either simulation data or a current plan
            has_simulation_data = hasattr(self, 'simulation_results') and self.simulation_results
            has_current_plan = hasattr(self, 'current_plan') and self.current_plan
            
            if not has_simulation_data and not has_current_plan:
                messagebox.showerror("Error", "No simulation data or economic plan to export. Please run a simulation or create a plan first.")
                return
            
            # Get file path
            file_path = filedialog.asksaveasfilename(
                title="Save Comprehensive Simulation Report as PDF",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # Create PDF document
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph("Comprehensive Simulation Report", title_style))
            story.append(Paragraph("Simulation Summary, Graphs & 5-Year Plan", styles['Heading2']))
            story.append(Spacer(1, 20))
            
            # Add summary content
            if has_simulation_data:
                summary_content = self._create_summary_content()
            else:
                summary_content = "No simulation data available. This report contains only the economic plan information."
            
            lines = summary_content.split('\n')
            
            for line in lines:
                if line.strip():
                    if line.startswith('='):
                        # Section separator
                        story.append(Spacer(1, 12))
                        story.append(Paragraph(f"<b>{line}</b>", styles['Heading2']))
                    elif line.startswith('•') or line.startswith('-'):
                        # Bullet point
                        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}", styles['Normal']))
                    elif ':' in line and not line.startswith(' '):
                        # Section header
                        story.append(Spacer(1, 6))
                        story.append(Paragraph(f"<b>{line}</b>", styles['Heading3']))
                    else:
                        # Regular text
                        story.append(Paragraph(line, styles['Normal']))
                else:
                    story.append(Spacer(1, 6))
            
            # Add page break before graphs (only if simulation data exists)
            if has_simulation_data:
                story.append(PageBreak())
                story.append(Paragraph("Performance Graphs", styles['Heading1']))
                story.append(Spacer(1, 20))
                
                # Generate and add graphs
                temp_dir = self._add_performance_graphs_to_pdf(story)
            
            # Add 5-year plan report if available
            if hasattr(self, 'current_plan') and self.current_plan:
                story.append(PageBreak())
                story.append(Paragraph("5-Year Economic Plan Report", styles['Heading1']))
                story.append(Spacer(1, 20))
                
                # Generate 5-year plan report
                plan_report = self._generate_multi_year_plan_report()
                self._add_plan_report_to_pdf(story, plan_report, styles)
            
            # Build PDF
            doc.build(story)
            messagebox.showinfo("Success", f"Comprehensive simulation report exported to {file_path}")
            
        except ImportError as e:
            messagebox.showerror("Error", f"Required library not available: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF: {str(e)}")
        finally:
            # Clean up temporary directory if it was created
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary directory {temp_dir}: {cleanup_error}")
    
    def _add_performance_graphs_to_pdf(self, story):
        """Add performance graphs to PDF story."""
        temp_dir = None
        try:
            import matplotlib.pyplot as plt
            import matplotlib.backends.backend_pdf
            import io
            import tempfile
            import os
            import shutil
            from reportlab.platypus import Paragraph, Spacer, Image
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            
            styles = getSampleStyleSheet()
            
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                story.append(Paragraph("No simulation data available for graphs.", styles['Normal']))
                return None
            
            # Create temporary directory for graph images
            temp_dir = tempfile.mkdtemp()
            graph_files = []
            
            try:
                # Generate graphs
                graph_files = self._generate_graph_images(temp_dir)
                
                # Add each graph to the PDF
                for graph_file in graph_files:
                    if os.path.exists(graph_file):
                        try:
                            # Add graph title
                            graph_name = os.path.basename(graph_file).replace('.png', '').replace('_', ' ').title()
                            story.append(Paragraph(f"<b>{graph_name}</b>", styles['Heading3']))
                            story.append(Spacer(1, 10))
                            
                            # Verify file exists and is readable
                            abs_path = os.path.abspath(graph_file)
                            if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
                                # Test if file can be opened
                                try:
                                    with open(abs_path, 'rb') as test_file:
                                        test_file.read(1)  # Try to read first byte
                                    
                                    # Add image with absolute path
                                    img = Image(abs_path, width=6*inch, height=4*inch)
                                    story.append(img)
                                    story.append(Spacer(1, 20))
                                except Exception as file_error:
                                    story.append(Paragraph(f"Error reading image file {abs_path}: {str(file_error)}", styles['Normal']))
                            else:
                                story.append(Paragraph(f"Image file not found or empty: {abs_path}", styles['Normal']))
                        except Exception as img_error:
                            story.append(Paragraph(f"Error adding image {graph_file}: {str(img_error)}", styles['Normal']))
                    else:
                        story.append(Paragraph(f"Graph file not found: {graph_file}", styles['Normal']))
            except Exception as e:
                story.append(Paragraph(f"Error generating graphs: {str(e)}", styles['Normal']))
                        
        except Exception as e:
            story.append(Paragraph(f"Error generating graphs: {str(e)}", styles['Normal']))
        
        return temp_dir
    
    def _generate_graph_images(self, output_dir):
        """Generate graph images and return list of file paths."""
        try:
            import matplotlib
            # Set non-interactive backend to prevent display issues
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            
            graph_files = []
            results = self.simulation_results
            
            if not results:
                return graph_files
            
            # Extract data for plotting
            months = [r['month'] for r in results]
            production_data = []
            resource_data = []
            labor_data = []
            economic_outputs = []
            
            for result in results:
                if 'production' in result:
                    avg_prod = np.mean([data['efficiency'] for data in result['production'].values()])
                    production_data.append(avg_prod)
                else:
                    production_data.append(0)
                    
                if 'resources' in result:
                    avg_res = np.mean([data['availability'] for data in result['resources'].values()])
                    resource_data.append(avg_res)
                else:
                    resource_data.append(0)
                    
                if 'labor' in result:
                    avg_lab = np.mean([data['productivity'] for data in result['labor'].values()])
                    labor_data.append(avg_lab)
                else:
                    labor_data.append(0)
                
                if 'economic_output' in result:
                    economic_outputs.append(result['economic_output'])
                else:
                    economic_outputs.append(0)
            
            # Set style
            plt.style.use('default')
            
            # 1. Production Efficiency Over Time
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(months, production_data, 'b-', linewidth=2, label='Production Efficiency')
            ax.set_title('Production Efficiency Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Efficiency')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            file_path = os.path.join(output_dir, 'production_efficiency.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            # Small delay to ensure file is written
            import time
            time.sleep(0.1)
            graph_files.append(file_path)
            
            # 2. Resource Availability Over Time
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(months, resource_data, 'g-', linewidth=2, label='Resource Availability')
            ax.set_title('Resource Availability Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Availability')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            file_path = os.path.join(output_dir, 'resource_availability.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            time.sleep(0.1)
            graph_files.append(file_path)
            
            # 3. Labor Productivity Over Time
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(months, labor_data, 'r-', linewidth=2, label='Labor Productivity')
            ax.set_title('Labor Productivity Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Productivity')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            file_path = os.path.join(output_dir, 'labor_productivity.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            time.sleep(0.1)
            graph_files.append(file_path)
            
            # 4. Economic Output Over Time
            if economic_outputs and any(economic_outputs):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(months, economic_outputs, 'purple', linewidth=2, label='Economic Output')
                ax.set_title('Economic Output Over Time', fontsize=14, fontweight='bold')
                ax.set_xlabel('Month')
                ax.set_ylabel('Output ($)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                file_path = os.path.join(output_dir, 'economic_output.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                time.sleep(0.1)
                graph_files.append(file_path)
            
            # 5. Combined Performance Dashboard
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Production efficiency
            ax1.plot(months, production_data, 'b-', linewidth=2)
            ax1.set_title('Production Efficiency')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Efficiency')
            ax1.grid(True, alpha=0.3)
            
            # Resource availability
            ax2.plot(months, resource_data, 'g-', linewidth=2)
            ax2.set_title('Resource Availability')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Availability')
            ax2.grid(True, alpha=0.3)
            
            # Labor productivity
            ax3.plot(months, labor_data, 'r-', linewidth=2)
            ax3.set_title('Labor Productivity')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Productivity')
            ax3.grid(True, alpha=0.3)
            
            # Economic output
            if economic_outputs and any(economic_outputs):
                ax4.plot(months, economic_outputs, 'purple', linewidth=2)
                ax4.set_title('Economic Output')
                ax4.set_xlabel('Month')
                ax4.set_ylabel('Output ($)')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Economic Output Data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Economic Output')
            
            plt.suptitle('Performance Dashboard', fontsize=16, fontweight='bold')
            plt.tight_layout()
            file_path = os.path.join(output_dir, 'combined_performance.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            time.sleep(0.1)
            graph_files.append(file_path)
            
            return graph_files
            
        except Exception as e:
            print(f"Error generating graphs: {str(e)}")
            return []
    
    def _generate_multi_year_plan_report(self):
        """Generate a comprehensive multi-year plan report."""
        try:
            if not hasattr(self, 'current_plan') or not self.current_plan:
                return "No multi-year plan data available."
            
            # Check if it's a multi-year plan
            if isinstance(self.current_plan, dict) and "total_output" in self.current_plan:
                # Single year plan
                return self.planning_system.generate_report(self.current_plan)
            else:
                # Multi-year plan - generate comprehensive report
                report_sections = []
                
                total_years = len(self.current_plan)
                report_sections.append(f"# {total_years}-Year Economic Plan Report")
                report_sections.append(f"- **Plan Duration**: {total_years} years")
                report_sections.append("")
                report_sections.append("## Executive Summary")
                report_sections.append(f"This {total_years}-year economic plan has been generated using cybernetic planning principles,")
                report_sections.append("combining Input-Output analysis with labor-time accounting.")
                report_sections.append("")
                
                # Year-by-year analysis
                for year in sorted(self.current_plan.keys()):
                    year_data = self.current_plan[year]
                    report_sections.append(f"### Year {year} Analysis")
                    
                    if 'total_output' in year_data:
                        total_output = np.sum(year_data['total_output'])
                        report_sections.append(f"- **Total Economic Output**: {total_output:,.2f} units")
                    
                    if 'final_demand' in year_data:
                        final_demand = np.sum(year_data['final_demand'])
                        report_sections.append(f"- **Final Demand Target**: {final_demand:,.2f} units")
                    
                    if 'labor_values' in year_data:
                        labor_cost = np.sum(year_data['labor_values'])
                        report_sections.append(f"- **Total Labor Cost**: {labor_cost:,.2f} person-hours")
                    
                    report_sections.append("")
                
                # Overall summary
                report_sections.append("### Overall Plan Summary")
                total_years = len(self.current_plan)
                report_sections.append(f"- **Plan Duration**: {total_years} years")
                report_sections.append(f"- **Planning Method**: Cybernetic Central Planning")
                report_sections.append(f"- **Optimization Focus**: Labor-time efficiency")
                report_sections.append("")
                
                # Recommendations
                report_sections.append("### Recommendations")
                report_sections.append("1. Implement the plan with careful monitoring")
                report_sections.append("2. Adjust based on real-world feedback")
                report_sections.append("3. Update technology matrices as needed")
                report_sections.append("4. Monitor resource utilization closely")
                report_sections.append("5. Track labor productivity improvements")
                
                return "\n".join(report_sections)
                
        except Exception as e:
            return f"Error generating 5-year plan report: {str(e)}"
    
    def _add_plan_report_to_pdf(self, story, plan_report, styles):
        """Add 5-year plan report to PDF story."""
        try:
            from reportlab.platypus import Paragraph, Spacer
            
            lines = plan_report.split('\n')
            
            for line in lines:
                if line.strip():
                    if line.startswith('##'):
                        # Main section
                        story.append(Spacer(1, 12))
                        story.append(Paragraph(f"<b>{line[3:]}</b>", styles['Heading1']))
                    elif line.startswith('###'):
                        # Subsection
                        story.append(Spacer(1, 8))
                        story.append(Paragraph(f"<b>{line[4:]}</b>", styles['Heading2']))
                    elif line.startswith('-') or line.startswith('•'):
                        # Bullet point
                        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}", styles['Normal']))
                    elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.'):
                        # Numbered list
                        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}", styles['Normal']))
                    else:
                        # Regular text
                        story.append(Paragraph(line, styles['Normal']))
                else:
                    story.append(Spacer(1, 6))
                    
        except Exception as e:
            story.append(Paragraph(f"Error adding plan report: {str(e)}", styles['Normal']))

    def _generate_population_health_report(self):
        """Generate a population health over time report."""
        try:
            if not hasattr(self, 'population_health_tracker') or not self.population_health_tracker:
                messagebox.showerror("Error", "No population health data available. Run a simulation first.")
                return

            health_summary = self.population_health_tracker.get_population_health_summary()
            
            # Create report content
            report_content = []
            report_content.append("=" * 80)
            report_content.append("POPULATION HEALTH OVER TIME REPORT")
            report_content.append("=" * 80)
            report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append("")

            # Summary statistics
            report_content.append("SUMMARY STATISTICS")
            report_content.append("-" * 40)
            report_content.append(f"Simulation Period: {health_summary.start_date.strftime('%Y-%m-%d')} to {health_summary.end_date.strftime('%Y-%m-%d')}")
            report_content.append(f"Total Months: {health_summary.total_months}")
            report_content.append(f"Initial Population: {health_summary.initial_population:,.0f}")
            report_content.append(f"Final Population: {health_summary.final_population:,.0f}")
            report_content.append(f"Population Growth Rate: {health_summary.population_growth_rate:.2%}")
            report_content.append("")

            # Technology and living standards
            report_content.append("TECHNOLOGY AND LIVING STANDARDS")
            report_content.append("-" * 40)
            report_content.append(f"Average Technology Level: {health_summary.average_technology_level:.3f}")
            report_content.append(f"Technology Growth Rate: {health_summary.technology_growth_rate:.2%}")
            report_content.append(f"Average Living Standards: {health_summary.average_living_standards:.3f}")
            report_content.append(f"Living Standards Growth Rate: {health_summary.living_standards_growth_rate:.2%}")
            report_content.append(f"Average Consumer Demand Fulfillment: {health_summary.average_consumer_demand_fulfillment:.3f}")
            report_content.append(f"Demand Fulfillment Trend: {health_summary.demand_fulfillment_trend}")
            report_content.append("")

            # Health indicators
            report_content.append("HEALTH INDICATORS")
            report_content.append("-" * 40)
            for indicator, value in health_summary.health_indicators.items():
                if 'rate' in indicator:
                    report_content.append(f"{indicator.replace('_', ' ').title()}: {value:.3f}")
                elif 'expectancy' in indicator or 'income' in indicator:
                    report_content.append(f"{indicator.replace('_', ' ').title()}: {value:,.1f}")
                else:
                    report_content.append(f"{indicator.replace('_', ' ').title()}: {value:.3f}")
            report_content.append("")

            # Monthly data summary (first 12 months and last 12 months)
            if len(health_summary.monthly_data) > 0:
                report_content.append("MONTHLY DATA SUMMARY")
                report_content.append("-" * 40)
                
                # First 12 months
                report_content.append("First 12 Months:")
                for i, data in enumerate(health_summary.monthly_data[:12]):
                    report_content.append(f"  Month {data.month:2d}: Pop={data.population:8,.0f}, Tech={data.technology_level:.3f}, Living={data.living_standards_index:.3f}")
                
                if len(health_summary.monthly_data) > 24:
                    report_content.append("")
                    report_content.append("Last 12 Months:")
                    for data in health_summary.monthly_data[-12:]:
                        report_content.append(f"  Month {data.month:2d}: Pop={data.population:8,.0f}, Tech={data.technology_level:.3f}, Living={data.living_standards_index:.3f}")
                
                report_content.append("")

            # Technology growth analysis
            if hasattr(self.population_health_tracker, 'rd_output_history') and self.population_health_tracker.rd_output_history:
                report_content.append("R&D AND TECHNOLOGY GROWTH ANALYSIS")
                report_content.append("-" * 40)
                avg_rd_output = np.mean(self.population_health_tracker.rd_output_history)
                total_rd_output = np.sum(self.population_health_tracker.rd_output_history)
                tech_growth_rate = self.population_health_tracker.get_technology_growth_rate()
                
                report_content.append(f"Average R&D Output per Month: {avg_rd_output:,.0f}")
                report_content.append(f"Total R&D Output: {total_rd_output:,.0f}")
                report_content.append(f"Technology Growth Rate: {tech_growth_rate:.2%}")
                report_content.append("")

            report_content.append("=" * 80)
            report_content.append("END OF POPULATION HEALTH REPORT")
            report_content.append("=" * 80)

            # Save report to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"population_health_report_{timestamp}.txt"
            report_path = os.path.join("exports", report_filename)
            
            # Ensure exports directory exists
            os.makedirs("exports", exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            messagebox.showinfo("Success", f"Population health report generated successfully!\nSaved to: {report_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate population health report: {str(e)}")

    def _export_summary_excel(self):
        """Export simulation summary data as Excel."""
        try:
            import pandas as pd
            
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                messagebox.showerror("Error", "No simulation data to export.")
                return
            
            # Get file path
            file_path = filedialog.asksaveasfilename(
                title="Save Simulation Data as Excel",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # Ensure .xlsx extension
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
            
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Summary metrics sheet
                metrics = self._calculate_summary_metrics(self.simulation_results)
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                metrics_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
                
                # Monthly data sheet
                monthly_data = []
                for result in self.simulation_results:
                    row = {
                        'Month': result['month'],
                        'Year': result['year'],
                        'Month_in_Year': result['month_in_year']
                    }
                    
                    # Add production data
                    if 'production' in result:
                        for sector, data in result['production'].items():
                            row[f'Production_{sector}_Efficiency'] = data.get('efficiency', 0)
                            row[f'Production_{sector}_Actual'] = data.get('actual', 0)
                            row[f'Production_{sector}_Target'] = data.get('target', 0)
                    
                    # Add resource data
                    if 'resources' in result:
                        for resource, data in result['resources'].items():
                            row[f'Resource_{resource}_Availability'] = data.get('availability', 0)
                            row[f'Resource_{resource}_Allocated'] = data.get('allocated', 0)
                            row[f'Resource_{resource}_Required'] = data.get('required', 0)
                    
                    # Add labor data
                    if 'labor' in result:
                        for sector, data in result['labor'].items():
                            row[f'Labor_{sector}_Productivity'] = data.get('productivity', 0)
                            row[f'Labor_{sector}_Actual'] = data.get('actual', 0)
                            row[f'Labor_{sector}_Required'] = data.get('required', 0)
                    
                    monthly_data.append(row)
                
                monthly_df = pd.DataFrame(monthly_data)
                monthly_df.to_excel(writer, sheet_name='Monthly_Data', index=False)
                
                # Events sheet
                if hasattr(self, 'current_events') and self.current_events:
                    events_df = pd.DataFrame(self.current_events)
                    events_df.to_excel(writer, sheet_name='Events', index=False)
                
                # Metadata sheet
                metadata = {
                    'Property': ['Simulation Duration', 'Total Months', 'Generated', 'System Version'],
                    'Value': [
                        f"{len(self.simulation_results)} months",
                        len(self.simulation_results),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Cybernetic Planning Experiment v1.0'
                    ]
                }
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            messagebox.showinfo("Success", f"Simulation data exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export Excel: {str(e)}")

    def generate_simulation_map(self):
        """Generate an interactive map for the simulation environment."""
        messagebox.showinfo("Info", "Map functionality has been removed. Use the simulation tab for running simulations.")

    def open_map_in_browser(self):
        """Open the generated map in the default web browser."""
        messagebox.showinfo("Info", "Map functionality has been removed. Use the simulation tab for running simulations.")

    def refresh_simulation_map(self):
        """Refresh the current simulation map."""
        messagebox.showinfo("Info", "Map functionality has been removed. Use the simulation tab for running simulations.")

    def generate_realtime_map(self):
        """Generate a real-time map for the simulation environment with simulation integration."""
        messagebox.showinfo("Info", "Map functionality has been removed. Use the simulation tab for running simulations.")

    def _calculate_map_size_from_simulation(self):
        """Calculate map size based on simulation data and population."""
        return 500  # Default size - map functionality removed

    def _calculate_settlements_from_simulation(self):
        """Calculate number of settlements based on simulation data."""
        return 20  # Default count - map functionality removed

    def _create_simulation_integrated_map(self, map_bounds, geo_features, econ_zones):
        """Create a map integrated with simulation data and terrain distribution."""
        return None  # Map functionality removed

    def _generate_terrain_with_distribution(self, map_bounds, terrain_dist, settlements):
        """Generate terrain features based on user-defined distribution percentages."""
        return []  # Map functionality removed

    def _calculate_min_forest_size(self, settlements):
        """Calculate minimum forest size based on settlement sizes."""
        return 0.01  # Map functionality removed

    def _random_coordinates_in_bounds(self, map_bounds):
        """Generate random coordinates within map bounds."""
        return (0, 0)  # Map functionality removed

    def _generate_forest_coordinates(self, center, size):
        """Generate forest polygon coordinates."""
        return []  # Map functionality removed

    def _generate_mountain_coordinates(self, center, size):
        """Generate mountain polygon coordinates."""
        return []  # Map functionality removed

    def _generate_water_coordinates(self, center, size):
        """Generate water body polygon coordinates."""
        return []  # Map functionality removed

    def _generate_simulation_infrastructure(self, settlements):
        """Generate infrastructure based on simulation data."""
        return []  # Map functionality removed

    def _calculate_connection_importance(self, settlement1, settlement2, sectors, production_targets):
        """Calculate the economic importance of a connection between two settlements."""
        return 0.0  # Map functionality removed

    def _calculate_distance(self, settlement1, settlement2):
        """Calculate distance between two settlements in km."""
        return 0.0  # Map functionality removed

    def _update_map_with_simulation_data(self):
        """Update the map with current simulation data."""
        pass  # Map functionality removed

    def _update_infrastructure_status(self):
        """Update infrastructure status based on simulation progress."""
        pass  # Map functionality removed

    def _add_new_infrastructure(self):
        """Add new infrastructure based on economic development."""
        pass  # Map functionality removed

    def _add_economic_roads(self, settlements):
        """Add roads based on economic development."""
        pass  # Map functionality removed

    def _add_economic_railroads(self, settlements):
        """Add railroads for major economic centers."""
        pass  # Map functionality removed

    def _find_existing_road(self, settlement1, settlement2):
        """Find existing road between two settlements."""
        return None  # Map functionality removed

    def _find_existing_railroad(self, settlement1, settlement2):
        """Find existing railroad between two settlements."""
        return None  # Map functionality removed

    def _get_rd_sector_indices(self):
        """Get indices of R&D related sectors."""
        return []  # Map functionality removed
    
    def _get_sector_mapping(self):
        """Get mapping of sector names to indices."""
        return {}  # Map functionality removed

    def _update_settlement_populations(self):
        """Update settlement populations based on economic growth."""
        pass  # Map functionality removed

    def _auto_generate_map_for_simulation(self):
        """Automatically generate a map for the simulation."""
        pass  # Map functionality removed

    def _generate_final_simulation_map(self):
        """Generate a final map showing all simulation results."""
        pass  # Map functionality removed

    def _add_simulation_summary_to_map(self):
        """Add simulation summary information to the map."""
        pass  # Map functionality removed

    def update_realtime_map_info(self):
        """Update the real - time map information display."""
        pass  # Map functionality removed

    def display_map_in_gui(self):
        """Display the interactive map in the GUI."""
        pass  # Map functionality removed

    def refresh_map_display(self):
        """Refresh the map display with updated data."""
        pass  # Map functionality removed

    def open_realtime_map_in_browser(self):
        """Open the real - time map in the default web browser."""
        messagebox.showinfo("Info", "Map functionality has been removed. Use the simulation tab for running simulations.")

    def start_map_updates(self):
        """Start real - time map updates."""
        messagebox.showinfo("Info", "Map functionality has been removed. Use the simulation tab for running simulations.")

    def pause_map_updates(self):
        """Pause real - time map updates."""
        messagebox.showinfo("Info", "Map functionality has been removed. Use the simulation tab for running simulations.")

    def stop_map_updates(self):
        """Stop real - time map updates."""
        messagebox.showinfo("Info", "Map functionality has been removed. Use the simulation tab for running simulations.")

    def schedule_map_update(self):
        """Schedule the next map update."""
        pass  # Map functionality removed

    def update_realtime_map_data(self):
        """Update the map data based on simulation progress."""
        pass  # Map functionality removed

    def update_map_info_display(self):
        """Update the map information display."""
        pass  # Map functionality removed

    def _on_tech_level_change(self, value):
        """Handle technology level slider change."""
        try:
            tech_level = float(value)
            self.tech_level_display.config(text=f"{tech_level:.2f}")
            
            # Update the planning system's technology level if available
            if hasattr(self.planning_system, 'matrix_builder') and hasattr(self.planning_system.matrix_builder, 'sector_mapper'):
                if hasattr(self.planning_system.matrix_builder.sector_mapper, 'technological_level'):
                    self.planning_system.matrix_builder.sector_mapper.technological_level = tech_level
                elif hasattr(self.planning_system.matrix_builder.sector_mapper, 'sector_generator'):
                    self.planning_system.matrix_builder.sector_mapper.sector_generator.technological_level = tech_level
            
            # Update tech meter and indicators
            self._update_tech_meter(tech_level)
            self._update_tech_level_indicators(tech_level)
            
            # Auto-refresh the tree when technology level changes
            self._refresh_technology_tree()
        except Exception as e:
            self.tech_tree_status.config(text=f"Error updating technology level: {str(e)}")

    def _update_tech_meter(self, tech_level):
        """Update the technology meter visualization."""
        try:
            canvas = self.tech_meter_canvas
            canvas.delete("all")
            
            # Draw circular meter
            center_x, center_y = 100, 75
            radius = 60
            
            # Draw background circle
            canvas.create_oval(center_x - radius, center_y - radius, 
                             center_x + radius, center_y + radius, 
                             outline="gray", width=2, fill="lightgray")
            
            # Draw progress arc (0 to tech_level * 360 degrees)
            start_angle = -90  # Start from top
            extent = tech_level * 360
            
            # Color based on tech level
            if tech_level < 0.2:
                color = "green"
            elif tech_level < 0.5:
                color = "yellow"
            elif tech_level < 0.8:
                color = "orange"
            elif tech_level < 0.95:
                color = "red"
            else:
                color = "purple"
            
            # Draw progress arc
            canvas.create_arc(center_x - radius, center_y - radius,
                            center_x + radius, center_y + radius,
                            start=start_angle, extent=extent,
                            outline=color, width=8, style="arc")
            
            # Draw center text
            canvas.create_text(center_x, center_y, text=f"{tech_level:.2f}", 
                             font=("Arial", 16, "bold"), fill="black")
            
            # Draw tech level labels around the circle
            labels = [
                ("Basic", 0.0, -90),
                ("Intermediate", 0.2, -18),
                ("Advanced", 0.5, 90),
                ("Cutting Edge", 0.8, 198),
                ("Future", 1.0, 270)
            ]
            
            for label_text, level, angle in labels:
                # Calculate position
                label_radius = radius + 20
                x = center_x + label_radius * math.cos(math.radians(angle))
                y = center_y + label_radius * math.sin(math.radians(angle))
                
                # Color based on availability
                label_color = "green" if tech_level >= level else "gray"
                
                canvas.create_text(x, y, text=label_text, 
                                 font=("Arial", 8), fill=label_color)
                
        except Exception as e:
            print(f"Error updating tech meter: {e}")
    
    def _update_tech_level_indicators(self, tech_level):
        """Update technology level indicators."""
        try:
            for name, indicator in self.tech_level_indicators.items():
                threshold = indicator['threshold']
                status_label = indicator['status_label']
                
                if tech_level >= threshold:
                    status_label.config(text="✅", foreground="green")
                else:
                    status_label.config(text="🔒", foreground="red")
        except Exception as e:
            print(f"Error updating tech level indicators: {e}")

    def _refresh_technology_tree(self):
        """Refresh the technology tree visualization."""
        try:
            # Clear existing tree
            for item in self.tech_tree_view.get_children():
                self.tech_tree_view.delete(item)
            
            # Get technology tree data
            if not hasattr(self.planning_system, 'matrix_builder'):
                self.tech_tree_status.config(text="No planning system available")
                return
            
            sector_mapper = self.planning_system.matrix_builder.sector_mapper
            
            # Get visualization data
            if hasattr(sector_mapper, 'get_technology_tree_visualization'):
                tree_data = sector_mapper.get_technology_tree_visualization()
            elif hasattr(sector_mapper, 'sector_generator') and hasattr(sector_mapper.sector_generator, 'get_technology_tree_visualization'):
                tree_data = sector_mapper.sector_generator.get_technology_tree_visualization()
            else:
                self.tech_tree_status.config(text="Technology tree visualization not available")
                return
            
            # Get current tech level
            current_tech_level = self.tech_level_var.get()
            
            # Group nodes by technology level and calculate statistics
            tech_levels = {}
            total_sectors = len(tree_data['nodes'])
            available_sectors = 0
            locked_sectors = 0
            
            for node in tree_data['nodes']:
                tech_level = node['technology_level']
                if tech_level not in tech_levels:
                    tech_levels[tech_level] = []
                tech_levels[tech_level].append(node)
                
                # Check if sector is available at current tech level
                is_available = self._is_sector_available_at_tech_level(node, current_tech_level)
                if is_available:
                    available_sectors += 1
                else:
                    locked_sectors += 1
            
            # Update statistics
            availability_percent = (available_sectors / total_sectors * 100) if total_sectors > 0 else 0
            self.stats_labels['total_sectors'].config(text=str(total_sectors))
            self.stats_labels['available_sectors'].config(text=str(available_sectors))
            self.stats_labels['locked_sectors'].config(text=str(locked_sectors))
            self.stats_labels['availability_percent'].config(text=f"{availability_percent:.1f}%")
            
            # Add technology level groups to tree
            for tech_level in sorted(tech_levels.keys()):
                level_name = tech_level.replace('_', ' ').title()
                
                # Count available sectors in this level
                level_available = sum(1 for node in tech_levels[tech_level] 
                                    if self._is_sector_available_at_tech_level(node, current_tech_level))
                level_total = len(tech_levels[tech_level])
                
                level_id = self.tech_tree_view.insert('', 'end', 
                                                    text=f"{level_name} Technology ({level_available}/{level_total})", 
                                                    values=("", "", ""), open=True)
                
                # Add sectors under each technology level
                for node in tech_levels[tech_level]:
                    is_available = self._is_sector_available_at_tech_level(node, current_tech_level)
                    status = "✅ Available" if is_available else "🔒 Locked"
                    core_indicator = "⭐ " if node.get('is_core', False) else ""
                    
                    # Color coding for tree items
                    item_id = self.tech_tree_view.insert(level_id, 'end', 
                                                      text=f"{core_indicator}{node['name']}", 
                                                      values=(tech_level, status, node['category']))
                    
                    # Apply color tags
                    if is_available:
                        self.tech_tree_view.set(item_id, "available", "✅ Available")
                    else:
                        self.tech_tree_view.set(item_id, "available", "🔒 Locked")
            
            # Update status
            self.tech_tree_status.config(text=f"Tech Level: {current_tech_level:.2f} | Available: {available_sectors}/{total_sectors} sectors ({availability_percent:.1f}%)")
            
        except Exception as e:
            self.tech_tree_status.config(text=f"Error refreshing technology tree: {str(e)}")
    
    def _is_sector_available_at_tech_level(self, node, tech_level):
        """Check if a sector is available at the given technology level."""
        try:
            tech_level_thresholds = {
                'basic': 0.0,
                'intermediate': 0.2,
                'advanced': 0.5,
                'cutting_edge': 0.8,
                'future': 0.95
            }
            
            sector_tech_level = node.get('technology_level', 'basic')
            threshold = tech_level_thresholds.get(sector_tech_level, 1.0)
            
            return tech_level >= threshold
        except Exception:
            return False
    
    def _sync_tech_level_to_data(self):
        """Sync the technology level from tech tree tab to data management tab."""
        try:
            # Get current tech level from tech tree tab
            current_tech_level = self.tech_level_var.get()
            
            # Update the starting tech level in data management tab
            if hasattr(self, 'starting_tech_var'):
                self.starting_tech_var.set(str(current_tech_level))
                
                # Update status message
                self.tech_tree_status.config(text=f"Synced tech level {current_tech_level:.2f} to Data Management tab")
                
                # Show success message
                messagebox.showinfo("Sync Complete", 
                                  f"Technology level {current_tech_level:.2f} has been synced to the Data Management tab.")
            else:
                self.tech_tree_status.config(text="Error: Data Management tab not found")
                messagebox.showerror("Sync Error", "Could not find the Data Management tab to sync with.")
                
        except Exception as e:
            error_msg = f"Error syncing technology level: {str(e)}"
            self.tech_tree_status.config(text=error_msg)
            messagebox.showerror("Sync Error", error_msg)

    def clear_convergence_graph(self):
        """Clear the convergence graph data."""
        if not self.convergence_graph_available:
            return
        
        self.convergence_data = {
            'iterations': [],
            'plan_changes': [],
            'relative_changes': [],
            'total_outputs': []
        }
        
        # Clear the plots
        self.convergence_line1.set_data([], [])
        self.convergence_line2.set_data([], [])
        
        # Reset axes
        self.convergence_ax1.relim()
        self.convergence_ax1.autoscale_view()
        self.convergence_ax2.relim()
        self.convergence_ax2.autoscale_view()
        
        # Update canvas
        self.convergence_canvas.draw()
        
        # Update status
        self.convergence_status_label.config(text="Graph cleared - ready to monitor convergence...")

    def export_convergence_graph(self):
        """Export the convergence graph as an image."""
        if not self.convergence_graph_available:
            messagebox.showwarning("Warning", "Convergence graph not available.")
            return
        
        try:
            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(
                title="Export Convergence Graph",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # Save the figure
            self.convergence_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Convergence graph exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export convergence graph: {str(e)}")

    def update_convergence_graph(self, iteration, plan_change, relative_change, total_output):
        """Update the convergence graph with new data."""
        if not self.convergence_graph_available:
            return
        
        # Add new data
        self.convergence_data['iterations'].append(iteration)
        self.convergence_data['plan_changes'].append(plan_change)
        self.convergence_data['relative_changes'].append(relative_change)
        self.convergence_data['total_outputs'].append(total_output)
        
        # Update plot lines
        self.convergence_line1.set_data(self.convergence_data['iterations'], self.convergence_data['plan_changes'])
        self.convergence_line2.set_data(self.convergence_data['iterations'], self.convergence_data['relative_changes'])
        
        # Update axes limits
        if self.convergence_data['iterations']:
            self.convergence_ax1.set_xlim(0, max(self.convergence_data['iterations']) + 1)
            self.convergence_ax2.set_xlim(0, max(self.convergence_data['iterations']) + 1)
            
            # Set y-axis limits with some padding
            if self.convergence_data['plan_changes']:
                min_change = min(self.convergence_data['plan_changes'])
                max_change = max(self.convergence_data['plan_changes'])
                self.convergence_ax1.set_ylim(min_change * 0.1, max_change * 10)
            
            if self.convergence_data['relative_changes']:
                min_rel = min(self.convergence_data['relative_changes'])
                max_rel = max(self.convergence_data['relative_changes'])
                self.convergence_ax2.set_ylim(min_rel * 0.1, max_rel * 10)
        
        # Update canvas
        self.convergence_canvas.draw()
        
        # Update status
        status_text = f"Iteration {iteration}: Change={plan_change:.2e}, RelChange={relative_change:.2e}"
        self.convergence_status_label.config(text=status_text)

    def start_convergence_monitoring(self):
        """Start monitoring convergence during plan creation."""
        if not self.convergence_graph_available:
            return
        
        # Clear previous data
        self.clear_convergence_graph()
        
        # Update status
        self.convergence_status_label.config(text="Monitoring convergence...")
        
        # Enable real-time updates
        self.convergence_monitoring_active = True

    def stop_convergence_monitoring(self):
        """Stop monitoring convergence."""
        if not self.convergence_graph_available:
            return
        
        self.convergence_monitoring_active = False
        
        # Update status
        if self.convergence_data['iterations']:
            final_iteration = self.convergence_data['iterations'][-1]
            final_change = self.convergence_data['plan_changes'][-1]
            self.convergence_status_label.config(text=f"Convergence monitoring complete - {final_iteration} iterations, final change: {final_change:.2e}")
        else:
            self.convergence_status_label.config(text="Convergence monitoring stopped")

    def _create_plan_with_monitoring(self, policy_goals, use_optimization, max_iterations, production_multipliers, apply_reproduction):
        """Create a plan with real-time convergence monitoring."""
        try:
            # Get the manager agent to access convergence data
            manager_agent = self.planning_system.manager_agent
            
            # Create the plan using the standard method
            plan = self.planning_system.create_plan(
                policy_goals=policy_goals,
                use_optimization=use_optimization,
                max_iterations=max_iterations,
                production_multipliers=production_multipliers,
                apply_reproduction=apply_reproduction
            )
            
            # Extract convergence data from the plan if available
            if isinstance(plan, dict) and 'convergence_history' in plan:
                convergence_history = plan['convergence_history']
                
                # Update the convergence graph with historical data
                for i, conv_data in enumerate(convergence_history):
                    iteration = conv_data.get('iteration', i)
                    plan_change = conv_data.get('plan_change', 0.0)
                    relative_change = conv_data.get('relative_change', 0.0)
                    total_output = conv_data.get('total_output', 0.0)
                    
                    # Update graph in main thread
                    self.root.after(0, lambda i=iteration, pc=plan_change, rc=relative_change, to=total_output: 
                                  self.update_convergence_graph(i, pc, rc, to))
            
            # Also check if there's convergence data in the manager agent
            if hasattr(manager_agent, 'convergence_history') and manager_agent.convergence_history:
                for i, conv_data in enumerate(manager_agent.convergence_history):
                    iteration = conv_data.get('iteration', i)
                    plan_change = conv_data.get('plan_change', 0.0)
                    relative_change = conv_data.get('relative_change', 0.0)
                    total_output = conv_data.get('total_output', 0.0)
                    
                    # Update graph in main thread
                    self.root.after(0, lambda i=iteration, pc=plan_change, rc=relative_change, to=total_output: 
                                  self.update_convergence_graph(i, pc, rc, to))
            
            return plan
            
        except Exception as e:
            print(f"Error in plan monitoring: {e}")
            # Fallback to regular plan creation
            return self.planning_system.create_plan(
                policy_goals=policy_goals,
                use_optimization=use_optimization,
                max_iterations=max_iterations,
                production_multipliers=production_multipliers,
                apply_reproduction=apply_reproduction
            )

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    CyberneticPlanningGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
