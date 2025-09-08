#!/usr/bin/env python3
"""
GUI for Cybernetic Central Planning System

A user-friendly graphical interface for the cybernetic planning system,
allowing users to create economic plans, manage data, and generate reports.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
import os
import json
import numpy as np
from pathlib import Path
import threading
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize planning system
        self.planning_system = CyberneticPlanningSystem()
        self.current_plan = None
        self.current_data = None
        
        # Create GUI elements
        self.create_widgets()
        self.setup_layout()
        
        # Load demo data by default
        self.load_demo_data()
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Create tabs
        self.create_data_tab()
        self.create_planning_tab()
        self.create_results_tab()
        self.create_export_tab()
        self.create_about_tab()
    
    def create_data_tab(self):
        """Create data management tab."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Management")
        
        # Data source selection
        source_frame = ttk.LabelFrame(self.data_frame, text="Data Source", padding=10)
        source_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(source_frame, text="Load from File", 
                  command=self.load_data_from_file).pack(side="left", padx=5)
        ttk.Button(source_frame, text="Generate Synthetic Data", 
                  command=self.generate_synthetic_data).pack(side="left", padx=5)
        ttk.Button(source_frame, text="Load Demo Data", 
                  command=self.load_demo_data).pack(side="left", padx=5)
        
        # Data configuration
        config_frame = ttk.LabelFrame(self.data_frame, text="Synthetic Data Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Number of sectors
        ttk.Label(config_frame, text="Number of Sectors:").grid(row=0, column=0, sticky="w", padx=5)
        self.sectors_var = tk.StringVar(value="8")
        ttk.Entry(config_frame, textvariable=self.sectors_var, width=10).grid(row=0, column=1, padx=5)
        
        # Technology density
        ttk.Label(config_frame, text="Technology Density:").grid(row=0, column=2, sticky="w", padx=5)
        self.density_var = tk.StringVar(value="0.4")
        ttk.Entry(config_frame, textvariable=self.density_var, width=10).grid(row=0, column=3, padx=5)
        
        # Resource count
        ttk.Label(config_frame, text="Resource Count:").grid(row=1, column=0, sticky="w", padx=5)
        self.resources_var = tk.StringVar(value="3")
        ttk.Entry(config_frame, textvariable=self.resources_var, width=10).grid(row=1, column=1, padx=5)
        
        # Data display
        display_frame = ttk.LabelFrame(self.data_frame, text="Current Data", padding=10)
        display_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.data_text = scrolledtext.ScrolledText(display_frame, height=15, width=80)
        self.data_text.pack(fill="both", expand=True)
        
        # Data status
        self.data_status = ttk.Label(display_frame, text="No data loaded", foreground="red")
        self.data_status.pack(pady=5)
    
    def create_planning_tab(self):
        """Create planning configuration tab."""
        self.planning_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.planning_frame, text="Planning Configuration")
        
        # Policy goals
        goals_frame = ttk.LabelFrame(self.planning_frame, text="Policy Goals", padding=10)
        goals_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(goals_frame, text="Enter policy goals (one per line):").pack(anchor="w")
        self.goals_text = scrolledtext.ScrolledText(goals_frame, height=6, width=80)
        self.goals_text.pack(fill="x", pady=5)
        
        # Default goals
        default_goals = [
            "Increase healthcare capacity by 15%",
            "Reduce carbon emissions by 20%",
            "Improve education infrastructure",
            "Ensure food security"
        ]
        self.goals_text.insert("1.0", "\n".join(default_goals))
        
        # Planning options
        options_frame = ttk.LabelFrame(self.planning_frame, text="Planning Options", padding=10)
        options_frame.pack(fill="x", padx=10, pady=5)
        
        # Use optimization
        self.use_optimization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use constrained optimization", 
                       variable=self.use_optimization_var).pack(anchor="w")
        
        # Max iterations
        ttk.Label(options_frame, text="Max Iterations:").pack(anchor="w", pady=(10, 0))
        self.max_iterations_var = tk.StringVar(value="10")
        ttk.Entry(options_frame, textvariable=self.max_iterations_var, width=10).pack(anchor="w")
        
        # Plan type
        plan_type_frame = ttk.Frame(options_frame)
        plan_type_frame.pack(fill="x", pady=10)
        
        ttk.Label(plan_type_frame, text="Plan Type:").pack(side="left")
        self.plan_type_var = tk.StringVar(value="single_year")
        ttk.Radiobutton(plan_type_frame, text="Single Year", variable=self.plan_type_var, 
                       value="single_year").pack(side="left", padx=10)
        ttk.Radiobutton(plan_type_frame, text="Five Year", variable=self.plan_type_var, 
                       value="five_year").pack(side="left", padx=10)
        
        # Five-year plan options
        self.five_year_frame = ttk.Frame(options_frame)
        
        ttk.Label(self.five_year_frame, text="Consumption Growth Rate:").grid(row=0, column=0, sticky="w", padx=5)
        self.growth_rate_var = tk.StringVar(value="0.02")
        ttk.Entry(self.five_year_frame, textvariable=self.growth_rate_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.five_year_frame, text="Investment Ratio:").grid(row=0, column=2, sticky="w", padx=5)
        self.investment_ratio_var = tk.StringVar(value="0.15")
        ttk.Entry(self.five_year_frame, textvariable=self.investment_ratio_var, width=10).grid(row=0, column=3, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(self.planning_frame)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        self.create_plan_button = ttk.Button(control_frame, text="Create Plan", 
                                           command=self.create_plan, style="Accent.TButton")
        self.create_plan_button.pack(side="left", padx=5)
        
        self.progress_bar = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress_bar.pack(side="left", padx=10, fill="x", expand=True)
        
        # Status
        self.planning_status = ttk.Label(control_frame, text="Ready to create plan")
        self.planning_status.pack(side="right", padx=5)
    
    def create_results_tab(self):
        """Create results display tab."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results & Analysis")
        
        # Results summary
        summary_frame = ttk.LabelFrame(self.results_frame, text="Plan Summary", padding=10)
        summary_frame.pack(fill="x", padx=10, pady=5)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=8, width=80)
        self.summary_text.pack(fill="both", expand=True)
        
        # Detailed results
        details_frame = ttk.LabelFrame(self.results_frame, text="Detailed Results", padding=10)
        details_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create notebook for different result views
        self.results_notebook = ttk.Notebook(details_frame)
        
        # Sector analysis
        self.sector_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.sector_frame, text="Sector Analysis")
        
        self.sector_tree = ttk.Treeview(self.sector_frame, columns=("output", "demand", "labor_value", "labor_cost"), 
                                       show="headings", height=10)
        self.sector_tree.heading("#0", text="Sector")
        self.sector_tree.heading("output", text="Total Output")
        self.sector_tree.heading("demand", text="Final Demand")
        self.sector_tree.heading("labor_value", text="Labor Value")
        self.sector_tree.heading("labor_cost", text="Labor Cost")
        
        self.sector_tree.column("#0", width=80)
        self.sector_tree.column("output", width=120)
        self.sector_tree.column("demand", width=120)
        self.sector_tree.column("labor_value", width=120)
        self.sector_tree.column("labor_cost", width=120)
        
        sector_scrollbar = ttk.Scrollbar(self.sector_frame, orient="vertical", command=self.sector_tree.yview)
        self.sector_tree.configure(yscrollcommand=sector_scrollbar.set)
        
        self.sector_tree.pack(side="left", fill="both", expand=True)
        sector_scrollbar.pack(side="right", fill="y")
        
        # Report view
        self.report_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.report_frame, text="Full Report")
        
        self.report_text = scrolledtext.ScrolledText(self.report_frame, height=20, width=80)
        self.report_text.pack(fill="both", expand=True)
        
        self.results_notebook.pack(fill="both", expand=True)
    
    def create_export_tab(self):
        """Create export and save tab."""
        self.export_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.export_frame, text="Export & Save")
        
        # Save options
        save_frame = ttk.LabelFrame(self.export_frame, text="Save Current Plan", padding=10)
        save_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(save_frame, text="Save as JSON", 
                  command=lambda: self.save_plan("json")).pack(side="left", padx=5)
        ttk.Button(save_frame, text="Save as CSV", 
                  command=lambda: self.save_plan("csv")).pack(side="left", padx=5)
        ttk.Button(save_frame, text="Save as Excel", 
                  command=lambda: self.save_plan("excel")).pack(side="left", padx=5)
        
        # Export data
        export_frame = ttk.LabelFrame(self.export_frame, text="Export Data", padding=10)
        export_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(export_frame, text="Export Current Data", 
                  command=self.export_data).pack(side="left", padx=5)
        
        # Load plan
        load_frame = ttk.LabelFrame(self.export_frame, text="Load Plan", padding=10)
        load_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(load_frame, text="Load Plan from File", 
                  command=self.load_plan).pack(side="left", padx=5)
        
        # Status
        self.export_status = ttk.Label(self.export_frame, text="No plan to export")
        self.export_status.pack(pady=10)
    
    def create_about_tab(self):
        """Create about tab."""
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")
        
        about_text = """
Cybernetic Central Planning System

A sophisticated economic planning system that uses Input-Output analysis 
and labor-time accounting to generate comprehensive economic plans.

Features:
• Multi-agent planning system with specialized agents
• Input-Output analysis using Leontief models
• Labor value calculations and optimization
• Policy goal translation and implementation
• Resource constraint management
• Environmental impact assessment
• Comprehensive report generation

Agents:
• Manager Agent: Central coordination and plan orchestration
• Economics Agent: Sensitivity analysis and forecasting
• Policy Agent: Goal translation and social impact assessment
• Resource Agent: Resource optimization and environmental analysis
• Writer Agent: Report generation and documentation

The system can create both single-year and five-year economic plans,
incorporating policy goals and resource constraints to generate
optimal economic strategies.
        """
        
        about_label = ttk.Label(self.about_frame, text=about_text, justify="left")
        about_label.pack(padx=20, pady=20)
    
    def setup_layout(self):
        """Setup the main layout."""
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure styles
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="blue")
    
    def load_data_from_file(self):
        """Load data from a file."""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("JSON files", "*.json"), 
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
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
                    self.update_data_display()
                    self.data_status.config(text="Data loaded successfully", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
                self.data_status.config(text="Error loading data", foreground="red")
    
    def generate_synthetic_data(self):
        """Generate synthetic data based on configuration."""
        try:
            n_sectors = int(self.sectors_var.get())
            density = float(self.density_var.get())
            n_resources = int(self.resources_var.get())
            
            self.planning_system.create_synthetic_data(
                n_sectors=n_sectors,
                technology_density=density,
                resource_count=n_resources
            )
            self.current_data = self.planning_system.current_data
            self.update_data_display()
            self.data_status.config(text="Synthetic data generated", foreground="green")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid configuration: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
    
    def load_demo_data(self):
        """Load demo data."""
        try:
            self.planning_system.create_synthetic_data(n_sectors=8, technology_density=0.4, resource_count=3)
            self.current_data = self.planning_system.current_data
            self.update_data_display()
            self.data_status.config(text="Demo data loaded", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load demo data: {str(e)}")
    
    def is_raw_data_file(self, file_path):
        """Check if the file is a raw data file that needs processing."""
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in ['.xlsx', '.xls', '.csv']
    
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
            os.makedirs("data", exist_ok=True)
            
            # Process the file
            self.data_status.config(text=f"Processing {data_type} data...", foreground="blue")
            self.root.update()
            
            processed_data = process_file(file_path, processed_path)
            
            if processed_data:
                # Load the processed data into the planning system
                self.planning_system.load_data_from_file(processed_path)
                self.current_data = self.planning_system.current_data
                self.update_data_display()
                self.data_status.config(text=f"Data processed and loaded successfully ({data_type})", foreground="green")
                
                # Show success message
                messagebox.showinfo("Success", 
                    f"Data processed successfully!\n\n"
                    f"Type: {data_type}\n"
                    f"Saved to: {processed_path}\n"
                    f"Sectors: {len(processed_data.get('sectors', []))}")
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
        summary = f"""Data Summary:
================

Sectors: {len(self.current_data.get('sectors', []))}
Technology Matrix Shape: {self.current_data['technology_matrix'].shape}
Final Demand Shape: {self.current_data['final_demand'].shape}
Labor Input Shape: {self.current_data['labor_input'].shape}

Sector Names: {self.current_data.get('sectors', ['Sector 0', 'Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Sector 5', 'Sector 6', 'Sector 7'])}

Final Demand Values:
{self.current_data['final_demand']}

Labor Input Values:
{self.current_data['labor_input']}

Technology Matrix (first 4x4):
{self.current_data['technology_matrix'][:4, :4]}
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
        policy_goals = [goal.strip() for goal in goals_text.split('\n') if goal.strip()]
        
        # Get planning options
        use_optimization = self.use_optimization_var.get()
        max_iterations = int(self.max_iterations_var.get())
        plan_type = self.plan_type_var.get()
        
        # Start planning in a separate thread
        self.create_plan_button.config(state="disabled")
        self.progress_bar.start()
        self.planning_status.config(text="Creating plan...")
        
        def plan_thread():
            try:
                if plan_type == "single_year":
                    self.current_plan = self.planning_system.create_plan(
                        policy_goals=policy_goals,
                        use_optimization=use_optimization,
                        max_iterations=max_iterations
                    )
                else:  # five_year
                    growth_rate = float(self.growth_rate_var.get())
                    investment_ratio = float(self.investment_ratio_var.get())
                    
                    self.current_plan = self.planning_system.create_five_year_plan(
                        policy_goals=policy_goals,
                        consumption_growth_rate=growth_rate,
                        investment_ratio=investment_ratio
                    )
                
                # Update UI in main thread
                self.root.after(0, self.plan_created_successfully)
                
            except Exception as e:
                self.root.after(0, lambda: self.plan_creation_failed(str(e)))
        
        threading.Thread(target=plan_thread, daemon=True).start()
    
    def plan_created_successfully(self):
        """Handle successful plan creation."""
        self.create_plan_button.config(state="normal")
        self.progress_bar.stop()
        self.planning_status.config(text="Plan created successfully", foreground="green")
        
        # Update results display
        self.update_results_display()
        
        # Update export status
        self.export_status.config(text="Plan ready for export", foreground="green")
    
    def plan_creation_failed(self, error_msg):
        """Handle plan creation failure."""
        self.create_plan_button.config(state="normal")
        self.progress_bar.stop()
        self.planning_status.config(text="Plan creation failed", foreground="red")
        messagebox.showerror("Error", f"Failed to create plan: {error_msg}")
    
    def update_results_display(self):
        """Update the results display."""
        if not self.current_plan:
            return
        
        # Update summary
        if isinstance(self.current_plan, dict) and 'total_output' in self.current_plan:
            # Single year plan
            summary = self.planning_system.get_plan_summary()
            summary_text = f"""Plan Summary:
================

Total Economic Output: {summary['total_economic_output']:,.2f} units
Total Labor Cost: {summary['total_labor_cost']:,.2f} person-hours
Labor Efficiency: {summary['labor_efficiency']:.2f} units/hour
Sector Count: {summary['sector_count']}
Plan Quality Score: {summary['plan_quality_score']:.2f}

Constraint Violations:
- Demand Violations: {len(summary['constraint_violations'].get('demand_violations', []))}
- Resource Violations: {len(summary['constraint_violations'].get('resource_violations', []))}
- Non-Negativity Violations: {len(summary['constraint_violations'].get('non_negativity_violations', []))}
            """
        else:
            # Five year plan
            summary_text = f"""Five-Year Plan Summary:
========================

Years Planned: {len(self.current_plan)}

Year-by-Year Summary:
"""
            for year, plan in self.current_plan.items():
                total_output = np.sum(plan['total_output'])
                labor_cost = plan['total_labor_cost']
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
        
        if isinstance(self.current_plan, dict) and 'total_output' in self.current_plan:
            # Single year plan
            plan = self.current_plan
            for i in range(len(plan['total_output'])):
                labor_cost = plan['labor_values'][i] * plan['total_output'][i]
                self.sector_tree.insert("", "end", 
                                      text=f"Sector {i}",
                                      values=(f"{plan['total_output'][i]:.2f}",
                                             f"{plan['final_demand'][i]:.2f}",
                                             f"{plan['labor_values'][i]:.4f}",
                                             f"{labor_cost:.2f}"))
        else:
            # Five year plan - show first year
            first_year = min(self.current_plan.keys())
            plan = self.current_plan[first_year]
            for i in range(len(plan['total_output'])):
                labor_cost = plan['labor_values'][i] * plan['total_output'][i]
                self.sector_tree.insert("", "end", 
                                      text=f"Sector {i}",
                                      values=(f"{plan['total_output'][i]:.2f}",
                                             f"{plan['final_demand'][i]:.2f}",
                                             f"{plan['labor_values'][i]:.4f}",
                                             f"{labor_cost:.2f}"))
    
    def update_report(self):
        """Update the full report display."""
        if not self.current_plan:
            return
        
        try:
            if isinstance(self.current_plan, dict) and 'total_output' in self.current_plan:
                # Single year plan
                report = self.planning_system.generate_report()
            else:
                # Five year plan - generate report for first year
                first_year = min(self.current_plan.keys())
                report = self.planning_system.generate_report(self.current_plan[first_year])
            
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
            title=f"Save Plan as {format_type.upper()}",
            defaultextension=f".{format_type}",
            filetypes=[(f"{format_type.upper()} files", f"*.{format_type}"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if isinstance(self.current_plan, dict) and 'total_output' in self.current_plan:
                    # Single year plan
                    self.planning_system.save_plan(file_path, format_type)
                else:
                    # Five year plan - save as JSON for now
                    with open(file_path, 'w') as f:
                        json.dump(self.current_plan, f, indent=2, default=self.json_serializer)
                
                self.export_status.config(text=f"Plan saved to {file_path}", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plan: {str(e)}")
    
    def export_data(self):
        """Export current data."""
        if not self.current_data:
            messagebox.showerror("Error", "No data to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.planning_system.export_data(file_path, "json")
                self.export_status.config(text=f"Data exported to {file_path}", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def load_plan(self):
        """Load a plan from file."""
        file_path = filedialog.askopenfilename(
            title="Load Plan",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
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


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = CyberneticPlanningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
