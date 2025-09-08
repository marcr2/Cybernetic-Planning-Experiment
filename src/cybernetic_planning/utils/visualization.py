"""
Visualization utilities for the cybernetic planning system.

Provides functions for creating charts and visualizations
to support economic planning analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


def create_plan_visualizations(plan_data: Dict[str, Any], 
                             output_dir: str = "outputs/visualizations") -> Dict[str, str]:
    """
    Create comprehensive visualizations for an economic plan.
    
    Args:
        plan_data: Economic plan data dictionary
        output_dir: Directory to save visualization files
        
        Returns:
            Dictionary mapping visualization names to file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizations = {}
    
    # 1. Sector Output Bar Chart
    sector_chart_path = create_sector_output_chart(plan_data, output_dir)
    visualizations['sector_output'] = sector_chart_path
    
    # 2. Labor Values Chart
    labor_chart_path = create_labor_values_chart(plan_data, output_dir)
    visualizations['labor_values'] = labor_chart_path
    
    # 3. Resource Utilization Chart
    if 'resource_usage' in plan_data and 'max_resources' in plan_data:
        resource_chart_path = create_resource_utilization_chart(plan_data, output_dir)
        visualizations['resource_utilization'] = resource_chart_path
    
    # 4. Technology Matrix Heatmap
    tech_heatmap_path = create_technology_heatmap(plan_data, output_dir)
    visualizations['technology_heatmap'] = tech_heatmap_path
    
    # 5. Plan Summary Dashboard
    dashboard_path = create_plan_dashboard(plan_data, output_dir)
    visualizations['dashboard'] = dashboard_path
    
    return visualizations


def create_sector_output_chart(plan_data: Dict[str, Any], output_dir: str) -> str:
    """Create a bar chart of sector outputs."""
    total_output = plan_data['total_output']
    n_sectors = len(total_output)
    
    plt.figure(figsize=(12, 6))
    sectors = [f"Sector {i}" for i in range(n_sectors)]
    
    bars = plt.bar(sectors, total_output, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.title('Economic Output by Sector', fontsize=16, fontweight='bold')
    plt.xlabel('Sector', fontsize=12)
    plt.ylabel('Total Output (units)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    file_path = f"{output_dir}/sector_output_chart.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path


def create_labor_values_chart(plan_data: Dict[str, Any], output_dir: str) -> str:
    """Create a chart of labor values and costs."""
    labor_values = plan_data['labor_values']
    total_output = plan_data['total_output']
    labor_costs = labor_values * total_output
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Labor values per unit
    sectors = [f"Sector {i}" for i in range(len(labor_values))]
    ax1.bar(sectors, labor_values, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax1.set_title('Labor Values per Unit Output', fontweight='bold')
    ax1.set_xlabel('Sector')
    ax1.set_ylabel('Labor Value (person-hours/unit)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Total labor costs
    ax2.bar(sectors, labor_costs, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.set_title('Total Labor Costs by Sector', fontweight='bold')
    ax2.set_xlabel('Sector')
    ax2.set_ylabel('Total Labor Cost (person-hours)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    file_path = f"{output_dir}/labor_values_chart.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path


def create_resource_utilization_chart(plan_data: Dict[str, Any], output_dir: str) -> str:
    """Create a chart of resource utilization."""
    resource_usage = plan_data['resource_usage']
    max_resources = plan_data['max_resources']
    utilization = resource_usage / (max_resources + 1e-10)
    
    plt.figure(figsize=(10, 6))
    resources = [f"Resource {i}" for i in range(len(resource_usage))]
    
    # Create bars with different colors based on utilization level
    colors = ['red' if u > 0.9 else 'orange' if u > 0.7 else 'green' for u in utilization]
    
    bars = plt.bar(resources, utilization, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, util in zip(bars, utilization):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{util:.1%}', ha='center', va='bottom')
    
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Warning Threshold (80%)')
    plt.axhline(y=0.9, color='darkred', linestyle='--', alpha=0.7, label='Critical Threshold (90%)')
    
    plt.title('Resource Utilization Rates', fontsize=16, fontweight='bold')
    plt.xlabel('Resource Type', fontsize=12)
    plt.ylabel('Utilization Rate', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    file_path = f"{output_dir}/resource_utilization_chart.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path


def create_technology_heatmap(plan_data: Dict[str, Any], output_dir: str) -> str:
    """Create a heatmap of the technology matrix."""
    tech_matrix = plan_data['technology_matrix']
    n_sectors = tech_matrix.shape[0]
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(tech_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Input Coefficient'},
                square=True)
    
    plt.title('Technology Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Input Sectors', fontsize=12)
    plt.ylabel('Output Sectors', fontsize=12)
    
    # Set tick labels
    sector_labels = [f"S{i}" for i in range(n_sectors)]
    plt.xticks(range(n_sectors), sector_labels)
    plt.yticks(range(n_sectors), sector_labels)
    
    plt.tight_layout()
    
    file_path = f"{output_dir}/technology_heatmap.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path


def create_plan_dashboard(plan_data: Dict[str, Any], output_dir: str) -> str:
    """Create a comprehensive dashboard with key metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total Output Distribution
    total_output = plan_data['total_output']
    sectors = [f"S{i}" for i in range(len(total_output))]
    
    ax1.pie(total_output, labels=sectors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Output Distribution by Sector', fontweight='bold')
    
    # 2. Labor Efficiency
    labor_values = plan_data['labor_values']
    labor_efficiency = 1 / (labor_values + 1e-10)  # Output per labor hour
    
    ax2.bar(sectors, labor_efficiency, color='lightblue', edgecolor='navy', alpha=0.7)
    ax2.set_title('Labor Efficiency by Sector', fontweight='bold')
    ax2.set_xlabel('Sector')
    ax2.set_ylabel('Output per Labor Hour')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Final Demand vs Total Output
    final_demand = plan_data['final_demand']
    x = np.arange(len(sectors))
    width = 0.35
    
    ax3.bar(x - width/2, final_demand, width, label='Final Demand', alpha=0.7)
    ax3.bar(x + width/2, total_output, width, label='Total Output', alpha=0.7)
    ax3.set_title('Final Demand vs Total Output', fontweight='bold')
    ax3.set_xlabel('Sector')
    ax3.set_ylabel('Output (units)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sectors)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Key Metrics Summary
    total_economic_output = np.sum(total_output)
    total_labor_cost = plan_data['total_labor_cost']
    labor_efficiency_overall = total_economic_output / total_labor_cost
    
    metrics_text = f"""
    Key Metrics Summary
    
    Total Economic Output: {total_economic_output:,.1f} units
    Total Labor Cost: {total_labor_cost:,.1f} person-hours
    Overall Labor Efficiency: {labor_efficiency_overall:.2f} units/hour
    Number of Sectors: {len(sectors)}
    Average Output per Sector: {np.mean(total_output):.1f} units
    Output Standard Deviation: {np.std(total_output):.1f} units
    """
    
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Plan Summary', fontweight='bold')
    
    plt.tight_layout()
    
    file_path = f"{output_dir}/plan_dashboard.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path


def create_sensitivity_analysis_chart(sensitivity_data: Dict[str, Any], 
                                    output_dir: str) -> str:
    """Create a chart for sensitivity analysis results."""
    if 'critical_sectors' not in sensitivity_data:
        return None
    
    critical_sectors = sensitivity_data['critical_sectors']
    
    if not critical_sectors:
        return None
    
    plt.figure(figsize=(10, 6))
    
    sectors = [f"Sector {cs['sector_index']}" for cs in critical_sectors]
    importance_scores = [cs['importance_score'] for cs in critical_sectors]
    
    bars = plt.bar(sectors, importance_scores, color='orange', edgecolor='darkorange', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Critical Sectors by Sensitivity Score', fontsize=16, fontweight='bold')
    plt.xlabel('Sector', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    file_path = f"{output_dir}/sensitivity_analysis_chart.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path


def create_growth_analysis_chart(growth_data: Dict[str, Any], 
                               output_dir: str) -> str:
    """Create a chart for growth analysis over time."""
    if not growth_data:
        return None
    
    years = list(growth_data.keys())
    output_growth = [growth_data[year]['output_growth'] for year in years]
    labor_growth = [growth_data[year]['labor_growth'] for year in years]
    capital_growth = [growth_data[year]['capital_growth'] for year in years]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(years))
    width = 0.25
    
    plt.bar(x - width, output_growth, width, label='Output Growth', alpha=0.7)
    plt.bar(x, labor_growth, width, label='Labor Growth', alpha=0.7)
    plt.bar(x + width, capital_growth, width, label='Capital Growth', alpha=0.7)
    
    plt.title('Economic Growth Rates Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Growth Rate', fontsize=12)
    plt.xticks(x, years)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    file_path = f"{output_dir}/growth_analysis_chart.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path
