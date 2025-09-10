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
from datetime import datetime
import numpy as np
import math

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from cybernetic_planning.planning_system import CyberneticPlanningSystem
except ImportError as e:
    print(f"Error importing planning system: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

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
        self.create_automatic_analyses_tab()
        self.create_performance_tab()
        self.create_web_scraper_tab()
        self.create_api_keys_tab()
        self.create_planning_tab()
        self.create_results_tab()
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
        ttk.Button(source_frame, text="Web Scraper", command = self.open_web_scraper).pack(side="left", padx = self._scale_padding(5))
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

    def create_automatic_analyses_tab(self):
        """Create automatic analyses results tab."""
        self.auto_analyses_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.auto_analyses_frame, text="Automatic Analyses")

        # Header
        header_frame = ttk.LabelFrame(self.auto_analyses_frame, text="Automatic Analysis Results", padding=self._scale_padding(10))
        header_frame.pack(fill="x", padx=self._scale_padding(10), pady=self._scale_padding(5))

        ttk.Label(header_frame, text="These analyses run automatically when data is loaded:", 
                 font=("Arial", self._scale_font_size(10), "bold")).pack(anchor="w")
        
        # Refresh button
        refresh_frame = ttk.Frame(header_frame)
        refresh_frame.pack(fill="x", pady=self._scale_padding(5))
        
        ttk.Button(refresh_frame, text="Refresh Results", 
                  command=self.refresh_automatic_analyses).pack(side="left", padx=self._scale_padding(5))
        
        # Status
        self.auto_analyses_status = ttk.Label(refresh_frame, text="No analyses available", foreground="red")
        self.auto_analyses_status.pack(side="left", padx=self._scale_padding(10))

        # Results display
        results_frame = ttk.LabelFrame(self.auto_analyses_frame, text="Analysis Results", padding=self._scale_padding(10))
        results_frame.pack(fill="both", expand=True, padx=self._scale_padding(10), pady=self._scale_padding(5))

        # Create notebook for different analysis types
        self.analyses_notebook = ttk.Notebook(results_frame)
        self.analyses_notebook.pack(fill="both", expand=True)

        # Marxist analysis tab
        self.marxist_auto_frame = ttk.Frame(self.analyses_notebook)
        self.analyses_notebook.add(self.marxist_auto_frame, text="Marxist Analysis")
        
        self.marxist_auto_text = scrolledtext.ScrolledText(self.marxist_auto_frame, height=15, width=80)
        self.marxist_auto_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Cybernetic analysis tab
        self.cybernetic_auto_frame = ttk.Frame(self.analyses_notebook)
        self.analyses_notebook.add(self.cybernetic_auto_frame, text="Cybernetic Feedback")
        
        self.cybernetic_auto_text = scrolledtext.ScrolledText(self.cybernetic_auto_frame, height=15, width=80)
        self.cybernetic_auto_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Mathematical validation tab
        self.math_auto_frame = ttk.Frame(self.analyses_notebook)
        self.analyses_notebook.add(self.math_auto_frame, text="Mathematical Validation")
        
        self.math_auto_text = scrolledtext.ScrolledText(self.math_auto_frame, height=15, width=80)
        self.math_auto_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_web_scraper_tab(self):
        """Create web scraper tab."""
        self.scraper_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.scraper_frame, text="Web Scraper")

        # Country selection
        country_frame = ttk.LabelFrame(self.scraper_frame, text="Select Country / Region", padding = 10)
        country_frame.pack(fill="x", padx = 10, pady = 5)

        self.country_var = tk.StringVar(value="USA")
        countries = ["USA", "Russia", "EU", "China", "India"]

        for i, country in enumerate(countries):
            ttk.Radiobutton(country_frame, text = country, variable = self.country_var, value = country).grid(
                row = 0, column = i, padx = 10, sticky="w"
            )

        # Data source configuration
        config_frame = ttk.LabelFrame(self.scraper_frame, text="Data Sources", padding = 10)
        config_frame.pack(fill="x", padx = 10, pady = 5)

        # Checkboxes for different data types
        self.energy_var = tk.BooleanVar(value = True)
        self.material_var = tk.BooleanVar(value = True)
        self.labor_var = tk.BooleanVar(value = True)
        self.environmental_var = tk.BooleanVar(value = True)

        ttk.Checkbutton(config_frame, text="Energy Data", variable = self.energy_var).grid(
            row = 0, column = 0, sticky="w", padx = 5
        )
        ttk.Checkbutton(config_frame, text="Material Data", variable = self.material_var).grid(
            row = 0, column = 1, sticky="w", padx = 5
        )
        ttk.Checkbutton(config_frame, text="Labor Data", variable = self.labor_var).grid(
            row = 1, column = 0, sticky="w", padx = 5
        )
        ttk.Checkbutton(config_frame, text="Environmental Data", variable = self.environmental_var).grid(
            row = 1, column = 1, sticky="w", padx = 5
        )

        # API configuration
        api_frame = ttk.LabelFrame(self.scraper_frame, text="API Configuration", padding = 10)
        api_frame.pack(fill="x", padx = 10, pady = 5)

        ttk.Label(api_frame, text="EIA API Key (optional):").grid(row = 0, column = 0, sticky="w", padx = 5)
        self.eia_api_key_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable = self.eia_api_key_var, width = 40, show="*").grid(
            row = 0, column = 1, padx = 5, sticky="ew"
        )

        ttk.Label(api_frame, text="BLS API Key (optional):").grid(row = 1, column = 0, sticky="w", padx = 5)
        self.bls_api_key_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable = self.bls_api_key_var, width = 40, show="*").grid(
            row = 1, column = 1, padx = 5, sticky="ew"
        )

        ttk.Label(api_frame, text="USGS API Key (optional):").grid(row = 2, column = 0, sticky="w", padx = 5)
        self.usgs_api_key_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable = self.usgs_api_key_var, width = 40, show="*").grid(
            row = 2, column = 1, padx = 5, sticky="ew"
        )

        ttk.Label(api_frame, text="Google API Key (optional):").grid(row = 3, column = 0, sticky="w", padx = 5)
        self.google_api_key_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable = self.google_api_key_var, width = 40, show="*").grid(
            row = 3, column = 1, padx = 5, sticky="ew"
        )

        ttk.Label(api_frame, text="BEA API Key (optional):").grid(row = 4, column = 0, sticky="w", padx = 5)
        self.bea_api_key_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable = self.bea_api_key_var, width = 40, show="*").grid(
            row = 4, column = 1, padx = 5, sticky="ew"
        )

        ttk.Label(api_frame, text="Year:").grid(row = 5, column = 0, sticky="w", padx = 5)
        self.scraper_year_var = tk.StringVar(value="2024")
        ttk.Entry(api_frame, textvariable = self.scraper_year_var, width = 10).grid(row = 5, column = 1, sticky="w", padx = 5)

        api_frame.columnconfigure(1, weight = 1)

        # Scraping controls
        control_frame = ttk.Frame(self.scraper_frame)
        control_frame.pack(fill="x", padx = 10, pady = 10)

        self.start_scraping_button = ttk.Button(
            control_frame, text="Start Data Collection", command = self.start_web_scraping, style="Accent.TButton"
        )
        self.start_scraping_button.pack(side="left", padx = 5)

        self.scraper_progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.scraper_progress.pack(side="left", padx = 10, fill="x", expand = True)

        self.scraper_status = ttk.Label(control_frame, text="Ready to collect data")
        self.scraper_status.pack(side="right", padx = 5)

        # Data sources info
        info_frame = ttk.LabelFrame(self.scraper_frame, text="Data Sources by Country", padding = 10)
        info_frame.pack(fill="both", expand = True, padx = 10, pady = 5)

        self.scraper_info_text = scrolledtext.ScrolledText(info_frame, height = 15, width = 80)
        self.scraper_info_text.pack(fill="both", expand = True)

        # Load country - specific information
        self.update_country_info()

        # Bind country selection change
        self.country_var.trace("w", self.update_country_info)

    def create_api_keys_tab(self):
        """Create API keys management tab."""
        self.api_keys_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.api_keys_frame, text="API Keys")

        # Initialize API key manager
        try:
            from api_keys_config import APIKeyManager

            self.api_manager = APIKeyManager()
        except ImportError as e:
            self.api_manager = None
            print(f"Warning: Could not import APIKeyManager: {e}")

        # Create main container with scrollbar
        main_container = ttk.Frame(self.api_keys_frame)
        main_container.pack(fill="both", expand = True, padx = 5, pady = 5)

        # Create canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command = canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion = canvas.bbox("all")))

        canvas.create_window((0, 0), window = scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand = scrollbar.set)

        # API Keys Status
        status_frame = ttk.LabelFrame(scrollable_frame, text="API Keys Status", padding = 10)
        status_frame.pack(fill="x", padx = 5, pady = 5)

        self.api_status = ttk.Label(status_frame, text="Checking API keys...", foreground="blue")
        self.api_status.pack(pady = 5)

        # API Keys Management
        management_frame = ttk.LabelFrame(scrollable_frame, text="API Keys Management", padding = 10)
        management_frame.pack(fill="x", padx = 5, pady = 5)

        # Create key management interface
        self.create_key_management_interface(management_frame)

        # API Keys Information
        info_frame = ttk.LabelFrame(scrollable_frame, text="API Keys Information", padding = 10)
        info_frame.pack(fill="both", expand = True, padx = 5, pady = 5)

        # Create scrollable text widget for API key information
        self.api_info_text = scrolledtext.ScrolledText(info_frame, height = 15, width = 80)
        self.api_info_text.pack(fill="both", expand = True)

        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill="x", padx = 5, pady = 5)

        ttk.Button(button_frame, text="Refresh Status", command = self.refresh_api_status).pack(side="left", padx = 5)
        ttk.Button(button_frame, text="Show Setup Instructions", command = self.show_api_setup_instructions).pack(
            side="left", padx = 5
        )
        ttk.Button(button_frame, text="Export Template", command = self.export_keys_template).pack(side="left", padx = 5)
        ttk.Button(button_frame, text="Save All Keys", command = self.save_all_keys).pack(side="left", padx = 5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand = True)
        scrollbar.pack(side="right", fill="y")

        # Initialize API key status
        self.refresh_api_status()

    def create_key_management_interface(self, parent):
        """Create the key management interface with input fields."""
        if not self.api_manager:
            ttk.Label(parent, text="API Key Manager not available", foreground="red").pack()
            return

        # Get keys for GUI
        self.gui_keys = self.api_manager.get_keys_for_gui()
        self.key_vars = {}
        self.key_entries = {}

        # Create a frame for the key inputs
        keys_container = ttk.Frame(parent)
        keys_container.pack(fill="both", expand = True)

        # Create a grid for the key inputs
        row = 0
        for key_name, key_value in self.gui_keys.items():
            # Key name and status
            key_frame = ttk.Frame(keys_container)
            key_frame.grid(row = row, column = 0, sticky="ew", padx = 5, pady = 2)
            keys_container.columnconfigure(0, weight = 1)

            # Key name and status
            name_frame = ttk.Frame(key_frame)
            name_frame.pack(fill="x")

            # Determine if key is set based on whether it has a value
            is_set = key_value != "" and key_value is not None
            status_icon = "✓" if is_set else "✗"
            status_text = "Configured" if is_set else "Not configured"

            ttk.Label(name_frame, text = f"{key_name}", font=("Arial", 10, "bold")).pack(side="left")

            status_color = "green" if is_set else "red"
            ttk.Label(name_frame, text = f"{status_icon} {status_text}", foreground = status_color).pack(side="right")

            # Description
            description = f"API key for {key_name.replace('_', ' ').title()}"
            ttk.Label(key_frame, text = description, font=("Arial", 8), foreground="gray").pack(anchor="w")

            # Input field
            input_frame = ttk.Frame(key_frame)
            input_frame.pack(fill="x", pady = 2)

            # Create variable for the key value
            self.key_vars[key_name] = tk.StringVar(value = key_value)

            # Entry field
            entry = ttk.Entry(input_frame, textvariable = self.key_vars[key_name], show="*", width = 60)
            entry.pack(side="left", fill="x", expand = True, padx=(0, 5))
            self.key_entries[key_name] = entry

            # Show / Hide button
            show_var = tk.BooleanVar()

            def toggle_visibility(key = key_name, var = show_var):
                if var.get():
                    self.key_entries[key].config(show="")
                else:
                    self.key_entries[key].config(show="*")

            ttk.Checkbutton(input_frame, text="Show", variable = show_var, command = toggle_visibility).pack(side="right")

            # Website link
            website_frame = ttk.Frame(key_frame)
            website_frame.pack(fill="x")

            ttk.Label(website_frame, text="Website: ", font=("Arial", 8)).pack(side="left")
            # Get website URL based on key name
            website_url = self.get_website_for_key(key_name)
            website_link = ttk.Label(
                website_frame, text = website_url, font=("Arial", 8), foreground="blue", cursor="hand2"
            )
            website_link.pack(side="left")
            website_link.bind("<Button - 1>", lambda e, url = website_url: self.open_website(url))

            # Required indicator
            if self.is_key_required(key_name):
                ttk.Label(key_frame, text="⚠️ Required", foreground="orange", font=("Arial", 8, "bold")).pack(anchor="w")

            row += 1

    def get_website_for_key(self, key_name: str) -> str:
        """Get website URL for a given API key."""
        websites = {
            "GOOGLE_API_KEY": "https://ai.google.dev/",
            "OPENAI_API_KEY": "https://platform.openai.com/",
            "EIA_API_KEY": "https://www.eia.gov / opendata/",
            "BLS_API_KEY": "https://www.bls.gov / developers/",
            "USGS_API_KEY": "https://www.usgs.gov/",
            "BEA_API_KEY": "https://www.bea.gov/",
            "EPA_API_KEY": "https://www.epa.gov/",
            "CUSTOM_API_KEY": "https://example.com/"
        }
        return websites.get(key_name, "https://example.com/")

    def is_key_required(self, key_name: str) -> bool:
        """Check if a key is required."""
        if not self.api_manager:
            return False
        return key_name in self.api_manager.required_keys

    def create_planning_tab(self):
        """Create planning configuration tab."""
        self.planning_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.planning_frame, text="Planning Configuration")
        
        # Create a scrollable frame
        self.planning_canvas = tk.Canvas(self.planning_frame)
        self.planning_scrollbar = ttk.Scrollbar(self.planning_frame, orient="vertical", command=self.planning_canvas.yview)
        self.planning_scrollable_frame = ttk.Frame(self.planning_canvas)
        
        self.planning_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.planning_canvas.configure(scrollregion=self.planning_canvas.bbox("all"))
        )
        
        self.planning_canvas.create_window((0, 0), window=self.planning_scrollable_frame, anchor="nw")
        self.planning_canvas.configure(yscrollcommand=self.planning_scrollbar.set)
        
        # Pack the scrollable components
        self.planning_canvas.pack(side="left", fill="both", expand=True)
        self.planning_scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            self.planning_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
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
        production_frame = ttk.LabelFrame(self.planning_scrollable_frame, text="Production Adjustment", padding=10)
        production_frame.pack(fill="x", padx=10, pady=5)

        # Overall production multiplier
        overall_frame = ttk.Frame(production_frame)
        overall_frame.pack(fill="x", pady=5)
        
        ttk.Label(overall_frame, text="Overall Production Level:").pack(side="left")
        self.overall_production_var = tk.DoubleVar(value=1.0)
        self.overall_production_scale = ttk.Scale(
            overall_frame, 
            from_=0.1, 
            to=3.0, 
            variable=self.overall_production_var, 
            orient="horizontal",
            command=self.update_production_labels
        )
        self.overall_production_scale.pack(side="left", padx=10, fill="x", expand=True)
        
        self.overall_production_label = ttk.Label(overall_frame, text="100% (Normal)")
        self.overall_production_label.pack(side="left", padx=10)

        # Department-specific production adjustments
        dept_frame = ttk.LabelFrame(production_frame, text="Department-Specific Adjustments", padding=5)
        dept_frame.pack(fill="x", pady=5)

        # Department I (Means of Production)
        dept_I_frame = ttk.Frame(dept_frame)
        dept_I_frame.pack(fill="x", pady=2)
        ttk.Label(dept_I_frame, text="Dept I (Means of Production):").pack(side="left")
        self.dept_I_production_var = tk.DoubleVar(value=1.0)
        self.dept_I_production_scale = ttk.Scale(
            dept_I_frame, 
            from_=0.1, 
            to=3.0, 
            variable=self.dept_I_production_var, 
            orient="horizontal",
            command=self.update_production_labels
        )
        self.dept_I_production_scale.pack(side="left", padx=10, fill="x", expand=True)
        self.dept_I_production_label = ttk.Label(dept_I_frame, text="100%")
        self.dept_I_production_label.pack(side="left", padx=10)

        # Department II (Consumer Goods)
        dept_II_frame = ttk.Frame(dept_frame)
        dept_II_frame.pack(fill="x", pady=2)
        ttk.Label(dept_II_frame, text="Dept II (Consumer Goods):").pack(side="left")
        self.dept_II_production_var = tk.DoubleVar(value=1.0)
        self.dept_II_production_scale = ttk.Scale(
            dept_II_frame, 
            from_=0.1, 
            to=3.0, 
            variable=self.dept_II_production_var, 
            orient="horizontal",
            command=self.update_production_labels
        )
        self.dept_II_production_scale.pack(side="left", padx=10, fill="x", expand=True)
        self.dept_II_production_label = ttk.Label(dept_II_frame, text="100%")
        self.dept_II_production_label.pack(side="left", padx=10)

        # Department III (Services)
        dept_III_frame = ttk.Frame(dept_frame)
        dept_III_frame.pack(fill="x", pady=2)
        ttk.Label(dept_III_frame, text="Dept III (Services):").pack(side="left")
        self.dept_III_production_var = tk.DoubleVar(value=1.0)
        self.dept_III_production_scale = ttk.Scale(
            dept_III_frame, 
            from_=0.1, 
            to=3.0, 
            variable=self.dept_III_production_var, 
            orient="horizontal",
            command=self.update_production_labels
        )
        self.dept_III_production_scale.pack(side="left", padx=10, fill="x", expand=True)
        self.dept_III_production_label = ttk.Label(dept_III_frame, text="100%")
        self.dept_III_production_label.pack(side="left", padx=10)

        # Apply reproduction adjustments option
        self.apply_reproduction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(production_frame, text="Apply Marxist reproduction adjustments", variable=self.apply_reproduction_var).pack(anchor="w", pady=5)

        # Reset button
        reset_frame = ttk.Frame(production_frame)
        reset_frame.pack(fill="x", pady=5)
        ttk.Button(reset_frame, text="Reset to Normal Production", command=self.reset_production_sliders).pack(side="left")

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
• Auto-scaling: Enabled
        """
        
        scaling_label = ttk.Label(self.about_frame, text=scaling_text, justify="left", 
                                 font=("Arial", self._scale_font_size(9), "italic"))
        scaling_label.pack(padx = self._scale_padding(20), pady = self._scale_padding(10))

    def create_performance_tab(self):
        """Create performance analysis tab."""
        self.performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_frame, text="Performance Analysis")

        # Header
        header_frame = ttk.LabelFrame(self.performance_frame, text="Industry Performance Analysis", padding=self._scale_padding(10))
        header_frame.pack(fill="x", padx=self._scale_padding(10), pady=self._scale_padding(5))

        ttk.Label(header_frame, text="Feedback-driven growth analysis based on industry performance:", 
                 font=("Arial", self._scale_font_size(10), "bold")).pack(anchor="w")
        
        # Refresh button
        refresh_frame = ttk.Frame(header_frame)
        refresh_frame.pack(fill="x", pady=self._scale_padding(5))
        
        ttk.Button(refresh_frame, text="Refresh Performance Data", 
                  command=self.refresh_performance_data).pack(side="left", padx=self._scale_padding(5))
        
        # Status
        self.performance_status = ttk.Label(refresh_frame, text="No performance data available", foreground="red")
        self.performance_status.pack(side="left", padx=self._scale_padding(10))

        # Performance display
        performance_display_frame = ttk.LabelFrame(self.performance_frame, text="Performance Metrics", padding=self._scale_padding(10))
        performance_display_frame.pack(fill="both", expand=True, padx=self._scale_padding(10), pady=self._scale_padding(5))

        # Create notebook for different performance views
        self.performance_notebook = ttk.Notebook(performance_display_frame)
        self.performance_notebook.pack(fill="both", expand=True)

        # Overall performance tab
        self.overall_performance_frame = ttk.Frame(self.performance_notebook)
        self.performance_notebook.add(self.overall_performance_frame, text="Overall Performance")

        self.overall_performance_text = scrolledtext.ScrolledText(self.overall_performance_frame, height=15, width=80)
        self.overall_performance_text.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Department performance tab
        self.dept_performance_frame = ttk.Frame(self.performance_notebook)
        self.performance_notebook.add(self.dept_performance_frame, text="Department Performance")

        self.dept_performance_text = scrolledtext.ScrolledText(self.dept_performance_frame, height=15, width=80)
        self.dept_performance_text.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

        # Growth rates tab
        self.growth_rates_frame = ttk.Frame(self.performance_notebook)
        self.performance_notebook.add(self.growth_rates_frame, text="Dynamic Growth Rates")

        self.growth_rates_text = scrolledtext.ScrolledText(self.growth_rates_frame, height=15, width=80)
        self.growth_rates_text.pack(fill="both", expand=True, padx=self._scale_padding(5), pady=self._scale_padding(5))

    def refresh_performance_data(self):
        """Refresh the performance analysis display."""
        try:
            # Get performance feedback from planning system
            feedback = self.planning_system.get_performance_feedback()
            
            if "message" in feedback:
                self.performance_status.config(text=feedback["message"], foreground="red")
                return
            
            # Update status
            self.performance_status.config(text="Performance data loaded successfully", foreground="green")
            
            # Display overall performance
            overall_text = self._format_overall_performance(feedback)
            self.overall_performance_text.delete("1.0", tk.END)
            self.overall_performance_text.insert("1.0", overall_text)
            
            # Display department performance
            dept_text = self._format_department_performance(feedback)
            self.dept_performance_text.delete("1.0", tk.END)
            self.dept_performance_text.insert("1.0", dept_text)
            
            # Display growth rates
            growth_text = self._format_growth_rates(feedback)
            self.growth_rates_text.delete("1.0", tk.END)
            self.growth_rates_text.insert("1.0", growth_text)
            
        except Exception as e:
            self.performance_status.config(text=f"Error refreshing performance data: {str(e)}", foreground="red")

    def _format_overall_performance(self, feedback):
        """Format overall performance data for display."""
        text = "OVERALL ECONOMIC PERFORMANCE\n"
        text += "=" * 50 + "\n\n"
        
        text += f"Overall Demand Fulfillment: {feedback.get('overall_fulfillment', 0):.3f}\n"
        text += f"Overall Labor Efficiency: {feedback.get('overall_efficiency', 0):.3f}\n\n"
        
        text += "DEPARTMENT PERFORMANCE\n"
        text += "-" * 30 + "\n"
        text += f"Department I (Means of Production): {feedback.get('dept_I_performance', 0):.3f}\n"
        text += f"Department II (Consumer Goods): {feedback.get('dept_II_performance', 0):.3f}\n"
        text += f"Department III (Services): {feedback.get('dept_III_performance', 0):.3f}\n\n"
        
        bottlenecks = feedback.get('bottlenecks', [])
        if bottlenecks:
            text += f"IDENTIFIED BOTTLENECKS: {bottlenecks}\n\n"
        else:
            text += "No significant bottlenecks identified.\n\n"
        
        return text

    def _format_department_performance(self, feedback):
        """Format department performance data for display."""
        text = "DEPARTMENT-SPECIFIC ANALYSIS\n"
        text += "=" * 40 + "\n\n"
        
        dept_I_perf = feedback.get('dept_I_performance', 0)
        dept_II_perf = feedback.get('dept_II_performance', 0)
        dept_III_perf = feedback.get('dept_III_performance', 0)
        
        text += "DEPARTMENT I - MEANS OF PRODUCTION\n"
        text += "-" * 35 + "\n"
        text += f"Performance Score: {dept_I_perf:.3f}\n"
        if dept_I_perf < 0.8:
            text += "Status: UNDERPERFORMING - Needs investment in capital goods\n"
        elif dept_I_perf > 1.2:
            text += "Status: OVERPERFORMING - May need demand expansion\n"
        else:
            text += "Status: BALANCED - Good performance\n"
        text += "\n"
        
        text += "DEPARTMENT II - CONSUMER GOODS\n"
        text += "-" * 30 + "\n"
        text += f"Performance Score: {dept_II_perf:.3f}\n"
        if dept_II_perf < 0.8:
            text += "Status: UNDERPERFORMING - Consumer goods shortage\n"
        elif dept_II_perf > 1.2:
            text += "Status: OVERPERFORMING - May indicate overproduction\n"
        else:
            text += "Status: BALANCED - Good performance\n"
        text += "\n"
        
        text += "DEPARTMENT III - SERVICES\n"
        text += "-" * 25 + "\n"
        text += f"Performance Score: {dept_III_perf:.3f}\n"
        if dept_III_perf < 0.8:
            text += "Status: UNDERPERFORMING - Service sector needs development\n"
        elif dept_III_perf > 1.2:
            text += "Status: OVERPERFORMING - Service sector well-developed\n"
        else:
            text += "Status: BALANCED - Good performance\n"
        
        return text

    def _format_growth_rates(self, feedback):
        """Format growth rates data for display."""
        text = "DYNAMIC GROWTH RATES\n"
        text += "=" * 25 + "\n\n"
        
        growth_rates = feedback.get('growth_rates', {})
        
        text += "CALCULATED GROWTH RATES\n"
        text += "-" * 25 + "\n"
        text += f"Population Growth: {growth_rates.get('population', 0):.3f} ({growth_rates.get('population', 0)*100:.1f}%)\n"
        text += f"Living Standards Growth: {growth_rates.get('living_standards', 0):.3f} ({growth_rates.get('living_standards', 0)*100:.1f}%)\n"
        text += f"Technology Improvement: {growth_rates.get('technology', 0):.3f} ({growth_rates.get('technology', 0)*100:.1f}%)\n"
        text += f"Capital Accumulation: {growth_rates.get('capital', 0):.3f} ({growth_rates.get('capital', 0)*100:.1f}%)\n\n"
        
        text += "GROWTH RATE ANALYSIS\n"
        text += "-" * 20 + "\n"
        
        total_growth = growth_rates.get('population', 0) + growth_rates.get('living_standards', 0)
        text += f"Total Demand Growth: {total_growth:.3f} ({total_growth*100:.1f}%)\n"
        
        if total_growth > 0.08:
            text += "Status: HIGH GROWTH - Strong economic expansion\n"
        elif total_growth > 0.04:
            text += "Status: MODERATE GROWTH - Steady economic development\n"
        elif total_growth > 0.02:
            text += "Status: LOW GROWTH - Slow economic development\n"
        else:
            text += "Status: STAGNANT - Economic growth concerns\n"
        
        return text

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
                    # Refresh automatic analyses
                    self.refresh_automatic_analyses()
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
            # Refresh automatic analyses
            self.refresh_automatic_analyses()
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

    def show_api_setup_instructions(self):
        """Show API key setup instructions."""
        try:
            from api_keys_config import APIKeyManager

            manager = APIKeyManager()

            # Clear and update information text
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
            messagebox.showerror("Error", f"Failed to show setup instructions: {str(e)}")

    def create_env_template(self):
        """Create environment template file."""
        try:
            from api_keys_config import APIKeyManager

            manager = APIKeyManager()
            manager.create_env_template()

            messagebox.showinfo(
                "Success",
                "Environment template created successfully!\n\nFile: .env.template\n\nCopy this file to .env and fill in your actual API keys.",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create environment template: {str(e)}")

    def refresh_api_status(self):
        """Refresh API key status."""
        self.check_api_keys()

    def open_website(self, url):
        """Open a website URL in the default browser."""
        import webbrowser

        webbrowser.open(url)

    def save_all_keys(self):
        """Save all API keys to the JSON file."""
        if not self.api_manager:
            messagebox.showerror("Error", "API Key Manager not available")
            return

        try:
            # Collect all key values
            keys_to_save = {}
            for key_name, key_var in self.key_vars.items():
                keys_to_save[key_name] = key_var.get()

            # Save to JSON file
            success = self.api_manager.save_keys_to_json(keys_to_save)

            if success:
                messagebox.showinfo("Success", "API keys saved successfully to keys.json!")
                # Refresh the status
                self.refresh_api_status()
            else:
                messagebox.showerror("Error", "Failed to save API keys to JSON file")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save API keys: {str(e)}")

    def export_keys_template(self):
        """Export a template JSON file with empty keys."""
        if not self.api_manager:
            messagebox.showerror("Error", "API Key Manager not available")
            return

        try:
            success = self.api_manager.export_keys_template("keys_template.json")
            if success:
                messagebox.showinfo(
                    "Success",
                    "Keys template exported to keys_template.json!\n\nCopy this file to keys.json and fill in your actual API keys.",
                )
            else:
                messagebox.showerror("Error", "Failed to export keys template")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export template: {str(e)}")

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
                # Refresh automatic analyses
                self.refresh_automatic_analyses()

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
            # Refresh automatic analyses
            self.refresh_automatic_analyses()
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

        summary = f"""Data Summary:
================

Sectors: {len(self.current_data.get('sectors', []))}
Technology Matrix Shape: {tech_shape}
Final Demand Shape: {final_shape}
Labor Input Shape: {labor_shape}

Sector Names: {self.current_data.get('sectors', ['Sector 0', 'Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Sector 5', 'Sector 6', 'Sector 7'])}

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
                    print(f"Five-year plan created successfully: {type(self.current_plan)}")

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

        # Switch to results tab to show the plan
        self.notebook.select(self.results_frame)

        # Update export status
        self.export_status.config(text="Plan ready for export", foreground="green")

    def plan_creation_failed(self, error_msg):
        """Handle plan creation failure."""
        self.create_plan_button.config(state="normal")
        self.progress_bar.stop()
        self.planning_status.config(text="Plan creation failed", foreground="red")
        messagebox.showerror("Error", f"Failed to create plan: {error_msg}")

    def update_production_labels(self, value=None):
        """Update the production percentage labels when sliders change."""
        # Update overall production label
        overall_val = self.overall_production_var.get()
        if overall_val < 1.0:
            self.overall_production_label.config(text=f"{overall_val*100:.0f}% (Underproduction)")
        elif overall_val > 1.0:
            self.overall_production_label.config(text=f"{overall_val*100:.0f}% (Overproduction)")
        else:
            self.overall_production_label.config(text="100% (Normal)")
        
        # Update department labels
        dept_I_val = self.dept_I_production_var.get()
        self.dept_I_production_label.config(text=f"{dept_I_val*100:.0f}%")
        
        dept_II_val = self.dept_II_production_var.get()
        self.dept_II_production_label.config(text=f"{dept_II_val*100:.0f}%")
        
        dept_III_val = self.dept_III_production_var.get()
        self.dept_III_production_label.config(text=f"{dept_III_val*100:.0f}%")

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
                print(f"DEBUG: final_demand shape/length: {getattr(final_demand_data, 'shape', len(final_demand_data))}")
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
            # Multi-year plan (5-year plan) - use first year's data
            first_year = min(plan.keys())
            year_data = plan[first_year]
            
            print(f"DEBUG: Multi-year plan detected, using year {first_year}")
            print(f"DEBUG: Year data keys: {list(year_data.keys())}")
            if "final_demand" in year_data:
                final_demand_data = year_data["final_demand"]
                print(f"DEBUG: final_demand type: {type(final_demand_data)}")
                print(f"DEBUG: final_demand shape/length: {getattr(final_demand_data, 'shape', len(final_demand_data))}")
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
        """JSON serializer for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def open_web_scraper(self):
        """Open the web scraper tab."""
        self.notebook.select(self.scraper_frame)

    def update_country_info(self, *args):
        """Update country - specific data source information."""
        country = self.country_var.get()

        country_info = {
            "USA": """
USA Data Sources:
================

Energy Data:
- Energy Information Administration (EIA)
  * Energy consumption by sector * Electricity generation and consumption * Renewable energy statistics * Energy intensity data

Material Data:
- US Geological Survey (USGS)
  * Mineral production and consumption * Critical materials assessment * Material flow studies * Supply chain analysis

Labor Data:
- Bureau of Labor Statistics (BLS)
  * Employment by sector and occupation * Wage and salary data * Labor productivity statistics * Occupational skills data (O * NET)

Environmental Data:
- Environmental Protection Agency (EPA)
  * Carbon emissions by sector * Water usage statistics * Waste generation data * Environmental impact assessments

Note: Some data sources may require API keys for enhanced access.
            """,
            "Russia": """
Russia Data Sources:
===================

Energy Data:
- Ministry of Energy of the Russian Federation * Energy production and consumption * Oil and gas statistics * Nuclear energy data * Renewable energy development

Material Data:
- Ministry of Natural Resources and Environment * Mineral resource statistics * Mining production data * Critical materials assessment * Resource extraction data

Labor Data:
- Federal State Statistics Service (Rosstat)
  * Employment statistics by sector * Wage data by region and sector * Labor productivity indicators * Occupational classifications

Environmental Data:
- Ministry of Natural Resources and Environment * Environmental monitoring data * Emissions statistics * Water resource usage * Waste management data

Note: Data availability may be limited due to current geopolitical situation.
            """,
            "EU": """
EU Data Sources:
===============

Energy Data:
- Eurostat Energy Statistics * Energy production and consumption * Renewable energy targets and progress * Energy efficiency indicators * Cross - border energy trade

Material Data:
- European Commission Raw Materials Information System * Critical raw materials assessment * Material flow accounts * Circular economy indicators * Supply chain mapping

Labor Data:
- Eurostat Labor Force Survey * Employment statistics by sector * Wage and income data * Labor market indicators * Skills and qualifications data

Environmental Data:
- European Environment Agency (EEA)
  * Greenhouse gas emissions * Air and water quality data * Waste generation and treatment * Environmental impact assessments

Note: Data is available for all EU member states with harmonized methodologies.
            """,
            "China": """
China Data Sources:
==================

Energy Data:
- National Energy Administration (NEA)
  * Energy production and consumption * Renewable energy development * Energy efficiency programs * Power generation statistics

Material Data:
- Ministry of Natural Resources * Mineral resource statistics * Rare earth elements data * Critical materials production * Resource utilization rates

Labor Data:
- National Bureau of Statistics (NBS)
  * Employment statistics by sector * Urban and rural employment data * Wage and income statistics * Labor productivity indicators

Environmental Data:
- Ministry of Ecology and Environment * Air quality monitoring data * Water pollution statistics * Carbon emissions data * Environmental protection measures

Note: Some data may be limited or require special access permissions.
            """,
            "India": """
India Data Sources:
==================

Energy Data:
- Ministry of Power * Electricity generation and consumption * Renewable energy capacity * Energy access statistics * Power sector reforms

Material Data:
- Ministry of Mines * Mineral production statistics * Mining sector data * Critical minerals assessment * Resource exploration data

Labor Data:
- Ministry of Labour and Employment * Employment statistics by sector * Wage and salary data * Labor force participation rates * Skill development programs

Environmental Data:
- Ministry of Environment, Forest and Climate Change * Air quality index data * Water quality monitoring * Forest cover statistics * Climate change indicators

Note: Data collection methods may vary by state and region.
            """,
        }

        info_text = country_info.get(country, "No information available for selected country.")
        self.scraper_info_text.delete("1.0", tk.END)
        self.scraper_info_text.insert("1.0", info_text)

    def start_web_scraping(self):
        """Start the web scraping process."""
        country = self.country_var.get()
        year = int(self.scraper_year_var.get())

        # Get selected data types
        data_types = []
        if self.energy_var.get():
            data_types.append("energy")
        if self.material_var.get():
            data_types.append("material")
        if self.labor_var.get():
            data_types.append("labor")
        if self.environmental_var.get():
            data_types.append("environmental")

        if not data_types:
            messagebox.showwarning("Warning", "Please select at least one data type to collect.")
            return

        # Start scraping in a separate thread
        self.start_scraping_button.config(state="disabled")
        self.scraper_progress.start()
        self.scraper_status.config(text="Collecting data...")

        def scraping_thread():
            try:
                # Import the international data collector
                from src.cybernetic_planning.data.web_scrapers.international_scrapers import InternationalDataCollector

                # Initialize the international data collector
                collector = InternationalDataCollector(cache_dir="cache", output_dir="data")

                # Collect data based on country
                if country == "USA":
                    # Use the existing USA scrapers through enhanced data loader
                    from src.cybernetic_planning.data.enhanced_data_loader import EnhancedDataLoader

                    eia_api_key = self.eia_api_key_var.get() if self.eia_api_key_var.get() else None
                    bls_api_key = self.bls_api_key_var.get() if self.bls_api_key_var.get() else None
                    usgs_api_key = self.usgs_api_key_var.get() if self.usgs_api_key_var.get() else None
                    bea_api_key = self.bea_api_key_var.get() if self.bea_api_key_var.get() else None
                    loader = EnhancedDataLoader(
                        eia_api_key = eia_api_key,
                        bls_api_key = bls_api_key,
                        usgs_api_key = usgs_api_key,
                        bea_api_key = bea_api_key,
                        data_dir="data",
                        cache_dir="cache",
                    )
                    data = loader.load_comprehensive_data(year = year, use_real_data = True)
                else:
                    # Use international data collector for other countries
                    data = collector.collect_country_data(country = country, year = year, data_types = data_types)

                # Update UI in main thread
                self.root.after(0, lambda: self.scraping_completed_successfully(data, country, year))

            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.scraping_failed(error_msg))

        threading.Thread(target = scraping_thread, daemon = True).start()

    def scraping_completed_successfully(self, data, country, year):
        """Handle successful data collection."""
        self.start_scraping_button.config(state="normal")
        self.scraper_progress.stop()
        self.scraper_status.config(text = f"Data collected successfully for {country} {year}", foreground="green")

        # Load the collected data into the planning system
        try:
            self.planning_system.load_comprehensive_data(
                year = year,
                use_real_data = True,
                eia_api_key = self.eia_api_key_var.get() if self.eia_api_key_var.get() else None,
            )
            self.current_data = self.planning_system.current_data

            # Update data display
            self.update_data_display()
            self.data_status.config(text = f"Real data loaded from {country} web scrapers", foreground="green")
            # Refresh automatic analyses
            self.refresh_automatic_analyses()

            # Show success message
            messagebox.showinfo(
                "Success",
                f"Data collection completed successfully!\n\n"
                f"Country: {country}\n"
                f"Year: {year}\n"
                f"Data loaded into planning system",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load collected data: {str(e)}")

    def scraping_completed_with_message(self, message):
        """Handle data collection with informational message."""
        self.start_scraping_button.config(state="normal")
        self.scraper_progress.stop()
        self.scraper_status.config(text="Data collection completed with message", foreground="blue")
        messagebox.showinfo("Information", message)

    def scraping_failed(self, error_msg):
        """Handle data collection failure."""
        self.start_scraping_button.config(state="normal")
        self.scraper_progress.stop()
        self.scraper_status.config(text="Data collection failed", foreground="red")
        messagebox.showerror("Error", f"Data collection failed: {error_msg}")


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

    def refresh_automatic_analyses(self):
        """Refresh the automatic analyses display."""
        try:
            # Get automatic analyses from planning system
            analyses = self.planning_system.get_automatic_analyses()
            
            if "error" in analyses:
                self.auto_analyses_status.config(text=analyses["error"], foreground="red")
                return
            
            # Update status
            self.auto_analyses_status.config(text="Analyses loaded successfully", foreground="green")
            
            # Display Marxist analysis
            marxist_data = analyses.get('marxist', {})
            marxist_text = self._format_analysis_data(marxist_data, "Marxist Economic Analysis")
            self.marxist_auto_text.delete("1.0", tk.END)
            self.marxist_auto_text.insert("1.0", marxist_text)
            
            # Display Cybernetic analysis
            cybernetic_data = analyses.get('cybernetic', {})
            cybernetic_text = self._format_analysis_data(cybernetic_data, "Cybernetic Feedback Analysis")
            self.cybernetic_auto_text.delete("1.0", tk.END)
            self.cybernetic_auto_text.insert("1.0", cybernetic_text)
            
            # Display Mathematical validation
            math_data = analyses.get('mathematical', {})
            math_text = self._format_analysis_data(math_data, "Mathematical Validation")
            self.math_auto_text.delete("1.0", tk.END)
            self.math_auto_text.insert("1.0", math_text)
            
        except Exception as e:
            self.auto_analyses_status.config(text=f"Error refreshing analyses: {str(e)}", foreground="red")

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
                    text += json.dumps(value, indent=2, default=str) + "\n\n"
                else:
                    text += f"{key}: {value}\n"
        else:
            text += str(data)
        
        return text

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    CyberneticPlanningGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
