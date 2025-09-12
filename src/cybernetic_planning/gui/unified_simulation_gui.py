"""
Unified Simulation GUI Components

GUI components for the unified simulation system that integrates
spatial and economic simulation with advanced controls and visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import webbrowser
from pathlib import Path

# Import unified simulation components
from ..core.unified_simulation_system import (
    UnifiedSimulationSystem, 
    UnifiedSimulationConfig
)
from ..core.unified_time_management import (
    UnifiedTimeManager, 
    TimeManagementConfig,
    TimeScale,
    UpdatePriority
)
from ..core.enhanced_sector_settlement_mapper import (
    EnhancedSectorSettlementMapper,
    SectorType,
    SettlementHierarchy
)
from ..core.unified_simulation_loop import (
    UnifiedSimulationLoop,
    SimulationLoopConfig
)
from ..core.unified_reporting_system import (
    UnifiedReportingSystem,
    ReportConfig
)

class UnifiedSimulationControlPanel:
    """Control panel for unified simulation system."""
    
    def __init__(self, parent_frame: ttk.Frame):
        """Initialize the unified simulation control panel."""
        self.parent_frame = parent_frame
        self.unified_system: Optional[UnifiedSimulationSystem] = None
        self.simulation_loop: Optional[UnifiedSimulationLoop] = None
        self.is_running = False
        
        # Create main notebook for different control sections
        self.notebook = ttk.Notebook(parent_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_configuration_tab()
        self._create_execution_tab()
        self._create_monitoring_tab()
        self._create_analysis_tab()
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        
        # Create status bar
        self._create_status_bar()
    
    def _create_configuration_tab(self):
        """Create configuration tab."""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")
        
        # Create scrollable frame
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Spatial Configuration
        spatial_frame = ttk.LabelFrame(scrollable_frame, text="Spatial Configuration", padding=10)
        spatial_frame.pack(fill="x", padx=5, pady=5)
        
        # Map dimensions
        ttk.Label(spatial_frame, text="Map Dimensions:").grid(row=0, column=0, sticky="w", pady=2)
        self.map_width_var = tk.IntVar(value=200)
        self.map_height_var = tk.IntVar(value=200)
        ttk.Entry(spatial_frame, textvariable=self.map_width_var, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(spatial_frame, text="x").grid(row=0, column=2)
        ttk.Entry(spatial_frame, textvariable=self.map_height_var, width=8).grid(row=0, column=3, padx=5)
        
        # Terrain distribution
        ttk.Label(spatial_frame, text="Terrain Distribution:").grid(row=1, column=0, sticky="w", pady=2)
        terrain_frame = ttk.Frame(spatial_frame)
        terrain_frame.grid(row=1, column=1, columnspan=3, sticky="w", padx=5)
        
        self.terrain_vars = {
            "flatland": tk.DoubleVar(value=0.4),
            "forest": tk.DoubleVar(value=0.3),
            "mountain": tk.DoubleVar(value=0.2),
            "water": tk.DoubleVar(value=0.1)
        }
        
        for i, (terrain, var) in enumerate(self.terrain_vars.items()):
            ttk.Label(terrain_frame, text=f"{terrain.title()}:").grid(row=0, column=i*2, padx=2)
            ttk.Entry(terrain_frame, textvariable=var, width=6).grid(row=0, column=i*2+1, padx=2)
        
        # Settlements
        ttk.Label(spatial_frame, text="Cities:").grid(row=2, column=0, sticky="w", pady=2)
        self.num_cities_var = tk.IntVar(value=5)
        ttk.Entry(spatial_frame, textvariable=self.num_cities_var, width=8).grid(row=2, column=1, padx=5)
        
        ttk.Label(spatial_frame, text="Towns:").grid(row=2, column=2, sticky="w", pady=2)
        self.num_towns_var = tk.IntVar(value=15)
        ttk.Entry(spatial_frame, textvariable=self.num_towns_var, width=8).grid(row=2, column=3, padx=5)
        
        # Population
        ttk.Label(spatial_frame, text="Total Population:").grid(row=3, column=0, sticky="w", pady=2)
        self.total_population_var = tk.IntVar(value=1000000)
        ttk.Entry(spatial_frame, textvariable=self.total_population_var, width=12).grid(row=3, column=1, padx=5)
        
        ttk.Label(spatial_frame, text="Rural %:").grid(row=3, column=2, sticky="w", pady=2)
        self.rural_population_percent_var = tk.DoubleVar(value=0.3)
        ttk.Entry(spatial_frame, textvariable=self.rural_population_percent_var, width=8).grid(row=3, column=3, padx=5)
        
        # Urban concentration
        ttk.Label(spatial_frame, text="Urban Concentration:").grid(row=4, column=0, sticky="w", pady=2)
        self.urban_concentration_var = tk.StringVar(value="medium")
        concentration_combo = ttk.Combobox(spatial_frame, textvariable=self.urban_concentration_var, 
                                         values=["low", "medium", "high"], width=8)
        concentration_combo.grid(row=4, column=1, padx=5)
        
        # Economic Configuration
        economic_frame = ttk.LabelFrame(scrollable_frame, text="Economic Configuration", padding=10)
        economic_frame.pack(fill="x", padx=5, pady=5)
        
        # Number of sectors
        ttk.Label(economic_frame, text="Number of Sectors:").grid(row=0, column=0, sticky="w", pady=2)
        self.n_sectors_var = tk.IntVar(value=15)
        ttk.Entry(economic_frame, textvariable=self.n_sectors_var, width=8).grid(row=0, column=1, padx=5)
        
        # Technology density
        ttk.Label(economic_frame, text="Technology Density:").grid(row=0, column=2, sticky="w", pady=2)
        self.technology_density_var = tk.DoubleVar(value=0.4)
        ttk.Entry(economic_frame, textvariable=self.technology_density_var, width=8).grid(row=0, column=3, padx=5)
        
        # Resource count
        ttk.Label(economic_frame, text="Resource Count:").grid(row=1, column=0, sticky="w", pady=2)
        self.resource_count_var = tk.IntVar(value=8)
        ttk.Entry(economic_frame, textvariable=self.resource_count_var, width=8).grid(row=1, column=1, padx=5)
        
        # Policy goals
        ttk.Label(economic_frame, text="Policy Goals:").grid(row=2, column=0, sticky="nw", pady=2)
        self.policy_goals_text = tk.Text(economic_frame, height=4, width=50)
        self.policy_goals_text.grid(row=2, column=1, columnspan=3, padx=5, pady=2)
        self.policy_goals_text.insert("1.0", "Increase industrial production\nImprove living standards\nDevelop infrastructure")
        
        # Simulation Configuration
        simulation_frame = ttk.LabelFrame(scrollable_frame, text="Simulation Configuration", padding=10)
        simulation_frame.pack(fill="x", padx=5, pady=5)
        
        # Duration
        ttk.Label(simulation_frame, text="Duration (months):").grid(row=0, column=0, sticky="w", pady=2)
        self.simulation_duration_var = tk.IntVar(value=60)
        ttk.Entry(simulation_frame, textvariable=self.simulation_duration_var, width=8).grid(row=0, column=1, padx=5)
        
        # Update frequencies
        ttk.Label(simulation_frame, text="Spatial Updates:").grid(row=1, column=0, sticky="w", pady=2)
        self.spatial_frequency_var = tk.StringVar(value="daily")
        spatial_combo = ttk.Combobox(simulation_frame, textvariable=self.spatial_frequency_var,
                                    values=["daily", "weekly", "monthly"], width=8)
        spatial_combo.grid(row=1, column=1, padx=5)
        
        ttk.Label(simulation_frame, text="Economic Updates:").grid(row=1, column=2, sticky="w", pady=2)
        self.economic_frequency_var = tk.StringVar(value="monthly")
        economic_combo = ttk.Combobox(simulation_frame, textvariable=self.economic_frequency_var,
                                     values=["daily", "weekly", "monthly", "quarterly", "annual"], width=8)
        economic_combo.grid(row=1, column=3, padx=5)
        
        # Disaster probability
        ttk.Label(simulation_frame, text="Disaster Probability:").grid(row=2, column=0, sticky="w", pady=2)
        self.disaster_probability_var = tk.DoubleVar(value=0.05)
        ttk.Entry(simulation_frame, textvariable=self.disaster_probability_var, width=8).grid(row=2, column=1, padx=5)
        
        # Integration options
        integration_frame = ttk.LabelFrame(scrollable_frame, text="Integration Options", padding=10)
        integration_frame.pack(fill="x", padx=5, pady=5)
        
        self.enable_bidirectional_feedback_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(integration_frame, text="Enable Bidirectional Feedback",
                       variable=self.enable_bidirectional_feedback_var).grid(row=0, column=0, sticky="w", pady=2)
        
        self.enable_spatial_constraints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(integration_frame, text="Enable Spatial Constraints",
                       variable=self.enable_spatial_constraints_var).grid(row=0, column=1, sticky="w", pady=2)
        
        self.enable_disaster_economic_impact_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(integration_frame, text="Enable Disaster Economic Impact",
                       variable=self.enable_disaster_economic_impact_var).grid(row=1, column=0, sticky="w", pady=2)
        
        self.enable_infrastructure_economic_feedback_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(integration_frame, text="Enable Infrastructure Economic Feedback",
                       variable=self.enable_infrastructure_economic_feedback_var).grid(row=1, column=1, sticky="w", pady=2)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_execution_tab(self):
        """Create execution tab."""
        execution_frame = ttk.Frame(self.notebook)
        self.notebook.add(execution_frame, text="Execution")
        
        # Control buttons
        button_frame = ttk.Frame(execution_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.create_button = ttk.Button(button_frame, text="Create Simulation", 
                                       command=self._create_simulation)
        self.create_button.pack(side="left", padx=5)
        
        self.run_button = ttk.Button(button_frame, text="Run Simulation", 
                                    command=self._run_simulation, state="disabled")
        self.run_button.pack(side="left", padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="Pause", 
                                      command=self._pause_simulation, state="disabled")
        self.pause_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self._stop_simulation, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        # Progress bar
        progress_frame = ttk.Frame(execution_frame)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(progress_frame, text="Progress:").pack(side="left")
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=300)
        self.progress_bar.pack(side="left", padx=5, fill="x", expand=True)
        
        # Log output
        log_frame = ttk.LabelFrame(execution_frame, text="Simulation Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, wrap="word")
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")
    
    def _create_monitoring_tab(self):
        """Create monitoring tab."""
        monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitoring_frame, text="Monitoring")
        
        # Create notebook for different monitoring views
        monitor_notebook = ttk.Notebook(monitoring_frame)
        monitor_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Real-time metrics
        metrics_frame = ttk.Frame(monitor_notebook)
        monitor_notebook.add(metrics_frame, text="Real-time Metrics")
        
        # Create metrics display
        self.metrics_frame = ttk.LabelFrame(metrics_frame, text="Current Metrics", padding=10)
        self.metrics_frame.pack(fill="x", padx=5, pady=5)
        
        # Spatial metrics
        spatial_metrics_frame = ttk.LabelFrame(self.metrics_frame, text="Spatial Metrics", padding=5)
        spatial_metrics_frame.pack(fill="x", pady=5)
        
        self.logistics_friction_var = tk.StringVar(value="N/A")
        self.active_disasters_var = tk.StringVar(value="N/A")
        self.settlements_count_var = tk.StringVar(value="N/A")
        
        ttk.Label(spatial_metrics_frame, text="Logistics Friction:").grid(row=0, column=0, sticky="w")
        ttk.Label(spatial_metrics_frame, textvariable=self.logistics_friction_var).grid(row=0, column=1, sticky="w", padx=10)
        
        ttk.Label(spatial_metrics_frame, text="Active Disasters:").grid(row=1, column=0, sticky="w")
        ttk.Label(spatial_metrics_frame, textvariable=self.active_disasters_var).grid(row=1, column=1, sticky="w", padx=10)
        
        ttk.Label(spatial_metrics_frame, text="Settlements:").grid(row=2, column=0, sticky="w")
        ttk.Label(spatial_metrics_frame, textvariable=self.settlements_count_var).grid(row=2, column=1, sticky="w", padx=10)
        
        # Economic metrics
        economic_metrics_frame = ttk.LabelFrame(self.metrics_frame, text="Economic Metrics", padding=5)
        economic_metrics_frame.pack(fill="x", pady=5)
        
        self.economic_output_var = tk.StringVar(value="N/A")
        self.capital_stock_var = tk.StringVar(value="N/A")
        self.sectors_active_var = tk.StringVar(value="N/A")
        
        ttk.Label(economic_metrics_frame, text="Economic Output:").grid(row=0, column=0, sticky="w")
        ttk.Label(economic_metrics_frame, textvariable=self.economic_output_var).grid(row=0, column=1, sticky="w", padx=10)
        
        ttk.Label(economic_metrics_frame, text="Capital Stock:").grid(row=1, column=0, sticky="w")
        ttk.Label(economic_metrics_frame, textvariable=self.capital_stock_var).grid(row=1, column=1, sticky="w", padx=10)
        
        ttk.Label(economic_metrics_frame, text="Active Sectors:").grid(row=2, column=0, sticky="w")
        ttk.Label(economic_metrics_frame, textvariable=self.sectors_active_var).grid(row=2, column=1, sticky="w", padx=10)
        
        # Integration metrics
        integration_metrics_frame = ttk.LabelFrame(self.metrics_frame, text="Integration Metrics", padding=5)
        integration_metrics_frame.pack(fill="x", pady=5)
        
        self.spatial_economic_efficiency_var = tk.StringVar(value="N/A")
        self.integration_stability_var = tk.StringVar(value="N/A")
        self.disaster_impact_var = tk.StringVar(value="N/A")
        
        ttk.Label(integration_metrics_frame, text="Spatial-Economic Efficiency:").grid(row=0, column=0, sticky="w")
        ttk.Label(integration_metrics_frame, textvariable=self.spatial_economic_efficiency_var).grid(row=0, column=1, sticky="w", padx=10)
        
        ttk.Label(integration_metrics_frame, text="Integration Stability:").grid(row=1, column=0, sticky="w")
        ttk.Label(integration_metrics_frame, textvariable=self.integration_stability_var).grid(row=1, column=1, sticky="w", padx=10)
        
        ttk.Label(integration_metrics_frame, text="Disaster Impact:").grid(row=2, column=0, sticky="w")
        ttk.Label(integration_metrics_frame, textvariable=self.disaster_impact_var).grid(row=2, column=1, sticky="w", padx=10)
        
        # Performance monitoring
        performance_frame = ttk.Frame(monitor_notebook)
        monitor_notebook.add(performance_frame, text="Performance")
        
        # Performance metrics
        perf_metrics_frame = ttk.LabelFrame(performance_frame, text="Performance Metrics", padding=10)
        perf_metrics_frame.pack(fill="x", padx=5, pady=5)
        
        self.simulation_time_var = tk.StringVar(value="N/A")
        self.average_step_time_var = tk.StringVar(value="N/A")
        self.error_count_var = tk.StringVar(value="N/A")
        self.memory_usage_var = tk.StringVar(value="N/A")
        
        ttk.Label(perf_metrics_frame, text="Simulation Time:").grid(row=0, column=0, sticky="w")
        ttk.Label(perf_metrics_frame, textvariable=self.simulation_time_var).grid(row=0, column=1, sticky="w", padx=10)
        
        ttk.Label(perf_metrics_frame, text="Average Step Time:").grid(row=1, column=0, sticky="w")
        ttk.Label(perf_metrics_frame, textvariable=self.average_step_time_var).grid(row=1, column=1, sticky="w", padx=10)
        
        ttk.Label(perf_metrics_frame, text="Error Count:").grid(row=2, column=0, sticky="w")
        ttk.Label(perf_metrics_frame, textvariable=self.error_count_var).grid(row=2, column=1, sticky="w", padx=10)
        
        ttk.Label(perf_metrics_frame, text="Memory Usage:").grid(row=3, column=0, sticky="w")
        ttk.Label(perf_metrics_frame, textvariable=self.memory_usage_var).grid(row=3, column=1, sticky="w", padx=10)
    
    def _create_analysis_tab(self):
        """Create analysis tab."""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")
        
        # Analysis controls
        controls_frame = ttk.Frame(analysis_frame)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(controls_frame, text="Generate Report", 
                  command=self._generate_report).pack(side="left", padx=5)
        
        ttk.Button(controls_frame, text="Export Data", 
                  command=self._export_data).pack(side="left", padx=5)
        
        ttk.Button(controls_frame, text="Open Report", 
                  command=self._open_report).pack(side="left", padx=5)
        
        # Analysis results
        results_frame = ttk.LabelFrame(analysis_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create notebook for different analysis views
        analysis_notebook = ttk.Notebook(results_frame)
        analysis_notebook.pack(fill="both", expand=True)
        
        # Summary tab
        summary_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(summary_frame, text="Summary")
        
        self.summary_text = tk.Text(summary_frame, wrap="word")
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scrollbar.pack(side="right", fill="y")
        
        # Charts tab
        charts_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(charts_frame, text="Charts")
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Recommendations tab
        recommendations_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(recommendations_frame, text="Recommendations")
        
        self.recommendations_text = tk.Text(recommendations_frame, wrap="word")
        recommendations_scrollbar = ttk.Scrollbar(recommendations_frame, orient="vertical", 
                                                command=self.recommendations_text.yview)
        self.recommendations_text.configure(yscrollcommand=recommendations_scrollbar.set)
        
        self.recommendations_text.pack(side="left", fill="both", expand=True)
        recommendations_scrollbar.pack(side="right", fill="y")
    
    def _create_status_bar(self):
        """Create status bar."""
        status_frame = ttk.Frame(self.parent_frame)
        status_frame.pack(fill="x", side="bottom")
        
        ttk.Label(status_frame, text="Status:").pack(side="left", padx=5)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left", padx=5)
        
        # Add separator
        ttk.Separator(status_frame, orient="vertical").pack(side="left", fill="y", padx=5)
        
        # Time display
        self.time_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ttk.Label(status_frame, textvariable=self.time_var).pack(side="right", padx=5)
        
        # Update time every second
        self._update_time()
    
    def _update_time(self):
        """Update time display."""
        self.time_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.parent_frame.after(1000, self._update_time)
    
    def _create_simulation(self):
        """Create unified simulation."""
        try:
            self.status_var.set("Creating simulation...")
            self._log_message("Creating unified simulation...")
            
            # Get configuration from GUI
            config = self._get_configuration()
            
            # Create unified simulation system
            self.unified_system = UnifiedSimulationSystem(config)
            
            # Create simulation in separate thread
            def create_thread():
                try:
                    result = self.unified_system.create_unified_simulation()
                    
                    if result["success"]:
                        self.parent_frame.after(0, lambda: self._on_simulation_created(result))
                    else:
                        self.parent_frame.after(0, lambda: self._on_simulation_error(result["error"]))
                        
                except Exception as e:
                    self.parent_frame.after(0, lambda: self._on_simulation_error(str(e)))
            
            threading.Thread(target=create_thread, daemon=True).start()
            
        except Exception as e:
            self._on_simulation_error(str(e))
    
    def _run_simulation(self):
        """Run unified simulation."""
        if not self.unified_system:
            messagebox.showerror("Error", "Please create simulation first")
            return
        
        try:
            self.status_var.set("Running simulation...")
            self.is_running = True
            self.run_button.config(state="disabled")
            self.pause_button.config(state="normal")
            self.stop_button.config(state="normal")
            
            self._log_message("Starting simulation...")
            
            # Create simulation loop
            loop_config = SimulationLoopConfig(
                enable_parallel_execution=True,
                enable_performance_monitoring=True,
                enable_real_time_output=True,
                output_frequency_steps=10
            )
            
            self.simulation_loop = UnifiedSimulationLoop(self.unified_system, loop_config)
            
            # Add callbacks
            self.simulation_loop.add_step_callback(self._on_simulation_step)
            self.simulation_loop.add_progress_callback(self._on_simulation_progress)
            self.simulation_loop.add_error_callback(self._on_simulation_error)
            
            # Run simulation in separate thread
            def run_thread():
                try:
                    result = self.simulation_loop.run_simulation(
                        duration_months=self.simulation_duration_var.get(),
                        spatial_frequency=self.spatial_frequency_var.get(),
                        economic_frequency=self.economic_frequency_var.get()
                    )
                    
                    self.parent_frame.after(0, lambda: self._on_simulation_completed(result))
                    
                except Exception as e:
                    self.parent_frame.after(0, lambda: self._on_simulation_error(str(e)))
            
            threading.Thread(target=run_thread, daemon=True).start()
            
        except Exception as e:
            self._on_simulation_error(str(e))
    
    def _pause_simulation(self):
        """Pause simulation."""
        if self.simulation_loop:
            result = self.simulation_loop.pause_simulation()
            if result["success"]:
                self.status_var.set("Simulation paused")
                self.pause_button.config(text="Resume", command=self._resume_simulation)
                self._log_message("Simulation paused")
    
    def _resume_simulation(self):
        """Resume simulation."""
        if self.simulation_loop:
            result = self.simulation_loop.resume_simulation()
            if result["success"]:
                self.status_var.set("Running simulation...")
                self.pause_button.config(text="Pause", command=self._pause_simulation)
                self._log_message("Simulation resumed")
    
    def _stop_simulation(self):
        """Stop simulation."""
        if self.simulation_loop:
            result = self.simulation_loop.stop_simulation()
            if result["success"]:
                self.status_var.set("Simulation stopped")
                self.is_running = False
                self.run_button.config(state="normal")
                self.pause_button.config(state="disabled", text="Pause")
                self.stop_button.config(state="disabled")
                self._log_message("Simulation stopped")
    
    def _generate_report(self):
        """Generate comprehensive report."""
        if not self.unified_system:
            messagebox.showerror("Error", "No simulation data available")
            return
        
        try:
            self.status_var.set("Generating report...")
            self._log_message("Generating comprehensive report...")
            
            # Create reporting system
            reporting_system = UnifiedReportingSystem()
            
            # Generate report in separate thread
            def generate_thread():
                try:
                    # Get simulation results (this would come from the actual simulation)
                    simulation_results = self._get_simulation_results()
                    
                    result = reporting_system.generate_comprehensive_report(
                        simulation_results,
                        {"config": self._get_configuration().__dict__}
                    )
                    
                    if result["success"]:
                        self.parent_frame.after(0, lambda: self._on_report_generated(result))
                    else:
                        self.parent_frame.after(0, lambda: self._on_report_error(result["error"]))
                        
                except Exception as e:
                    self.parent_frame.after(0, lambda: self._on_report_error(str(e)))
            
            threading.Thread(target=generate_thread, daemon=True).start()
            
        except Exception as e:
            self._on_report_error(str(e))
    
    def _export_data(self):
        """Export simulation data."""
        if not self.unified_system:
            messagebox.showerror("Error", "No simulation data available")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Simulation Data",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                self.status_var.set("Exporting data...")
                self._log_message(f"Exporting data to {file_path}")
                
                # Export data (this would use the actual simulation data)
                simulation_data = self._get_simulation_results()
                
                with open(file_path, 'w') as f:
                    json.dump(simulation_data, f, indent=2, default=str)
                
                self.status_var.set("Data exported successfully")
                self._log_message("Data export completed")
                messagebox.showinfo("Success", f"Data exported to {file_path}")
                
        except Exception as e:
            self._on_report_error(str(e))
    
    def _open_report(self):
        """Open generated report."""
        try:
            file_path = filedialog.askopenfilename(
                title="Open Report",
                filetypes=[("HTML files", "*.html"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                webbrowser.open(f"file://{Path(file_path).absolute()}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open report: {e}")
    
    def _get_configuration(self) -> UnifiedSimulationConfig:
        """Get configuration from GUI inputs."""
        # Get policy goals from text widget
        policy_goals_text = self.policy_goals_text.get("1.0", "end-1c")
        policy_goals = [goal.strip() for goal in policy_goals_text.split('\n') if goal.strip()]
        
        # Get terrain distribution
        terrain_distribution = {
            terrain: var.get() for terrain, var in self.terrain_vars.items()
        }
        
        return UnifiedSimulationConfig(
            map_width=self.map_width_var.get(),
            map_height=self.map_height_var.get(),
            terrain_distribution=terrain_distribution,
            num_cities=self.num_cities_var.get(),
            num_towns=self.num_towns_var.get(),
            total_population=self.total_population_var.get(),
            rural_population_percent=self.rural_population_percent_var.get(),
            urban_concentration=self.urban_concentration_var.get(),
            n_sectors=self.n_sectors_var.get(),
            technology_density=self.technology_density_var.get(),
            resource_count=self.resource_count_var.get(),
            policy_goals=policy_goals,
            simulation_duration_months=self.simulation_duration_var.get(),
            spatial_update_frequency=self.spatial_frequency_var.get(),
            economic_update_frequency=self.economic_frequency_var.get(),
            disaster_probability=self.disaster_probability_var.get(),
            enable_bidirectional_feedback=self.enable_bidirectional_feedback_var.get(),
            enable_spatial_constraints=self.enable_spatial_constraints_var.get(),
            enable_disaster_economic_impact=self.enable_disaster_economic_impact_var.get(),
            enable_infrastructure_economic_feedback=self.enable_infrastructure_economic_feedback_var.get()
        )
    
    def _get_simulation_results(self) -> Dict[str, Any]:
        """Get simulation results (placeholder)."""
        # This would return actual simulation results
        return {
            "spatial_metrics": {
                "total_logistics_friction": 1250.5,
                "average_logistics_friction": 45.2,
                "max_logistics_friction": 89.7,
                "total_disasters": 3,
                "average_active_disasters": 0.8
            },
            "economic_metrics": {
                "average_economic_output": 1500000.0,
                "final_economic_output": 1650000.0,
                "economic_growth_rate": 0.1,
                "average_capital_stock": 750000.0,
                "capital_accumulation_rate": 0.08
            },
            "integration_metrics": {
                "average_spatial_efficiency": 0.75,
                "integration_stability": 0.82,
                "total_disaster_economic_impact": 125000.0
            },
            "performance_metrics": {
                "total_simulation_time": 45.2,
                "average_step_time": 0.15,
                "max_step_time": 0.8,
                "memory_usage_mb": 256.5
            }
        }
    
    def _log_message(self, message: str):
        """Log message to the log text widget."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
    
    def _on_simulation_created(self, result: Dict[str, Any]):
        """Handle simulation creation success."""
        self.status_var.set("Simulation created successfully")
        self.run_button.config(state="normal")
        self._log_message("Simulation created successfully")
        
        # Update metrics display
        spatial_summary = result.get("spatial_summary", {})
        economic_summary = result.get("economic_summary", {})
        integration_summary = result.get("integration_summary", {})
        
        self.settlements_count_var.set(str(spatial_summary.get("settlements", 0)))
        self.sectors_active_var.set(str(economic_summary.get("sectors", 0)))
        
        messagebox.showinfo("Success", "Unified simulation created successfully!")
    
    def _on_simulation_error(self, error: str):
        """Handle simulation error."""
        self.status_var.set("Error")
        self.run_button.config(state="disabled")
        self.pause_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.is_running = False
        
        self._log_message(f"Error: {error}")
        messagebox.showerror("Simulation Error", error)
    
    def _on_simulation_step(self, step_result):
        """Handle simulation step callback."""
        # Update metrics
        spatial_metrics = step_result.spatial_updates
        economic_metrics = step_result.economic_updates
        integration_metrics = step_result.integration_updates
        
        if spatial_metrics:
            self.logistics_friction_var.set(f"{float(spatial_metrics.get('logistics_friction', 0)):.2f}")
            self.active_disasters_var.set(str(spatial_metrics.get('active_disasters', 0)))
        
        if economic_metrics:
            self.economic_output_var.set(f"{int(economic_metrics.get('total_economic_output', 0)):,.0f}")
            self.capital_stock_var.set(f"{int(economic_metrics.get('total_capital_stock', 0)):,.0f}")
        
        if integration_metrics:
            efficiency = integration_metrics.get('spatial_economic_efficiency', 0)
            self.spatial_economic_efficiency_var.set(f"{efficiency:.2%}")
    
    def _on_simulation_progress(self, current_step: int, total_steps: int):
        """Handle simulation progress callback."""
        progress = (current_step / total_steps) * 100 if total_steps > 0 else 0
        self.progress_var.set(progress)
        self._log_message(f"Progress: {current_step}/{total_steps} steps ({progress:.1f}%)")
    
    def _on_simulation_completed(self, result: Dict[str, Any]):
        """Handle simulation completion."""
        self.status_var.set("Simulation completed")
        self.is_running = False
        self.run_button.config(state="normal")
        self.pause_button.config(state="disabled", text="Pause")
        self.stop_button.config(state="disabled")
        
        self._log_message("Simulation completed successfully")
        
        # Update performance metrics
        self.simulation_time_var.set(f"{float(result.get('total_simulation_time', 0)):.2f}s")
        self.average_step_time_var.set(f"{float(result.get('average_step_time', 0)):.3f}s")
        self.error_count_var.set(str(result.get('error_count', 0)))
        
        messagebox.showinfo("Success", "Simulation completed successfully!")
    
    def _on_report_generated(self, result: Dict[str, Any]):
        """Handle report generation success."""
        self.status_var.set("Report generated")
        self._log_message("Report generated successfully")
        
        # Update analysis display
        self._update_analysis_display(result)
        
        messagebox.showinfo("Success", f"Report generated: {result['report_path']}")
    
    def _on_report_error(self, error: str):
        """Handle report generation error."""
        self.status_var.set("Report error")
        self._log_message(f"Report error: {error}")
        messagebox.showerror("Report Error", error)
    
    def _update_analysis_display(self, report_result: Dict[str, Any]):
        """Update analysis display with report results."""
        try:
            # Update summary
            summary_text = f"Report Generated: {report_result['report_path']}\n\n"
            summary_text += f"Sections: {', '.join(report_result['report_sections'])}\n\n"
            
            metrics = report_result.get("metrics_summary", {})
            summary_text += "Key Metrics:\n"
            summary_text += f"  Spatial Efficiency: {metrics.get('spatial_efficiency', 0):.2%}\n"
            summary_text += f"  Economic Growth: {metrics.get('economic_growth', 0):.2%}\n"
            summary_text += f"  Disaster Resilience: {metrics.get('disaster_resilience', 0):.2%}\n"
            summary_text += f"  Performance Score: {metrics.get('performance_score', 0):.2%}\n"
            
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", summary_text)
            
            # Update charts (placeholder)
            self._update_charts()
            
            # Update recommendations
            recommendations_text = "Recommendations:\n\n"
            recommendations_text += "• Optimize logistics efficiency\n"
            recommendations_text += "• Enhance disaster resilience\n"
            recommendations_text += "• Improve sector-settlement mapping\n"
            recommendations_text += "• Strengthen spatial-economic integration\n"
            
            self.recommendations_text.delete("1.0", "end")
            self.recommendations_text.insert("1.0", recommendations_text)
            
        except Exception as e:
            self._log_message(f"Error updating analysis display: {e}")
    
    def _update_charts(self):
        """Update charts with simulation data."""
        try:
            # Clear existing plots
            for ax in self.axes.flat:
                ax.clear()
            
            # Generate sample data for demonstration
            days = np.arange(1, 31)
            logistics_friction = 50 + 10 * np.sin(days * 0.2) + np.random.normal(0, 5, 30)
            economic_output = 1000000 + 50000 * days + np.random.normal(0, 10000, 30)
            disasters = np.random.poisson(0.1, 30)
            efficiency = 0.7 + 0.1 * np.sin(days * 0.1) + np.random.normal(0, 0.05, 30)
            
            # Plot logistics friction over time
            self.axes[0, 0].plot(days, logistics_friction, 'b-', linewidth=2)
            self.axes[0, 0].set_title('Logistics Friction Over Time')
            self.axes[0, 0].set_xlabel('Day')
            self.axes[0, 0].set_ylabel('Logistics Friction')
            self.axes[0, 0].grid(True, alpha=0.3)
            
            # Plot economic output over time
            self.axes[0, 1].plot(days, economic_output, 'g-', linewidth=2)
            self.axes[0, 1].set_title('Economic Output Over Time')
            self.axes[0, 1].set_xlabel('Day')
            self.axes[0, 1].set_ylabel('Economic Output')
            self.axes[0, 1].grid(True, alpha=0.3)
            
            # Plot disaster events
            self.axes[1, 0].bar(days, disasters, color='red', alpha=0.7)
            self.axes[1, 0].set_title('Disaster Events')
            self.axes[1, 0].set_xlabel('Day')
            self.axes[1, 0].set_ylabel('Number of Disasters')
            self.axes[1, 0].grid(True, alpha=0.3)
            
            # Plot efficiency over time
            self.axes[1, 1].plot(days, efficiency, 'purple', linewidth=2)
            self.axes[1, 1].set_title('Spatial-Economic Efficiency')
            self.axes[1, 1].set_xlabel('Day')
            self.axes[1, 1].set_ylabel('Efficiency')
            self.axes[1, 1].grid(True, alpha=0.3)
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            self._log_message(f"Error updating charts: {e}")

def create_unified_simulation_tab(parent_notebook: ttk.Notebook) -> ttk.Frame:
    """Create unified simulation tab for the main GUI."""
    # Create frame for unified simulation
    unified_frame = ttk.Frame(parent_notebook)
    parent_notebook.add(unified_frame, text="Unified Simulation")
    
    # Create control panel
    control_panel = UnifiedSimulationControlPanel(unified_frame)
    
    return unified_frame
