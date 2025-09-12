"""
Unified Reporting System

Generates comprehensive reports that combine spatial and economic metrics
from the unified simulation system with advanced visualization and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Configuration for unified reporting."""
    output_directory: str = "exports/unified_reports"
    include_visualizations: bool = True
    include_interactive_plots: bool = True
    include_data_exports: bool = True
    report_format: str = "html"  # html, pdf, json
    visualization_style: str = "professional"  # professional, academic, colorful
    include_performance_metrics: bool = True
    include_recommendations: bool = True
    max_data_points: int = 10000

@dataclass
class SpatialMetrics:
    """Spatial simulation metrics."""
    logistics_friction_total: float = 0.0
    logistics_friction_average: float = 0.0
    logistics_friction_max: float = 0.0
    active_disasters: int = 0
    total_disasters: int = 0
    settlements_count: int = 0
    infrastructure_segments: int = 0
    terrain_distribution: Dict[str, float] = field(default_factory=dict)
    settlement_hierarchy_distribution: Dict[str, int] = field(default_factory=dict)
    disaster_impact_by_type: Dict[str, float] = field(default_factory=dict)

@dataclass
class EconomicMetrics:
    """Economic simulation metrics."""
    total_economic_output: float = 0.0
    economic_growth_rate: float = 0.0
    total_capital_stock: float = 0.0
    capital_accumulation_rate: float = 0.0
    sectors_active: int = 0
    plan_fulfillment_rate: float = 0.0
    sector_performance: Dict[str, float] = field(default_factory=dict)
    labor_productivity: float = 0.0
    resource_efficiency: float = 0.0
    technology_level: float = 0.0

@dataclass
class IntegrationMetrics:
    """Integration between spatial and economic systems."""
    spatial_economic_efficiency: float = 0.0
    average_spatial_efficiency: float = 0.0
    integration_stability: float = 0.0
    disaster_economic_impact: float = 0.0
    logistics_economic_correlation: float = 0.0
    sector_settlement_optimization: float = 0.0
    infrastructure_economic_feedback: float = 0.0
    resource_spatial_distribution: Dict[str, float] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance and technical metrics."""
    total_simulation_time: float = 0.0
    average_step_time: float = 0.0
    max_step_time: float = 0.0
    memory_usage_mb: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    checkpoint_count: int = 0
    parallel_efficiency: float = 0.0

class UnifiedReportingSystem:
    """
    Comprehensive reporting system for unified simulation.
    
    Features:
    - Multi-dimensional analysis (spatial, economic, integration)
    - Interactive visualizations with Plotly
    - Static charts with Matplotlib
    - Data export in multiple formats
    - Performance analysis and optimization recommendations
    - Comparative analysis across simulation runs
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize the unified reporting system."""
        self.config = config or ReportConfig()
        
        # Create output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.spatial_metrics = SpatialMetrics()
        self.economic_metrics = EconomicMetrics()
        self.integration_metrics = IntegrationMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        # Data storage
        self.time_series_data: Dict[str, List[float]] = {}
        self.categorical_data: Dict[str, Dict[str, Any]] = {}
        self.correlation_data: Dict[str, float] = {}
        
        # Visualization settings
        self._setup_visualization_style()
        
        logger.info(f"Unified reporting system initialized. Output directory: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    simulation_results: Dict[str, Any],
                                    simulation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive unified simulation report.
        
        Args:
            simulation_results: Results from unified simulation
            simulation_config: Configuration used for simulation
        
        Returns:
            Dictionary with report generation results
        """
        try:
            logger.info("Generating comprehensive unified simulation report")
            
            # Extract and process metrics
            self._extract_metrics(simulation_results)
            
            # Generate report sections
            report_sections = {
                "executive_summary": self._generate_executive_summary(),
                "spatial_analysis": self._generate_spatial_analysis(),
                "economic_analysis": self._generate_economic_analysis(),
                "integration_analysis": self._generate_integration_analysis(),
                "performance_analysis": self._generate_performance_analysis(),
                "recommendations": self._generate_recommendations(),
                "appendix": self._generate_appendix()
            }
            
            # Generate visualizations
            if self.config.include_visualizations:
                visualizations = self._generate_visualizations()
                report_sections["visualizations"] = visualizations
            
            # Create final report
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "unified_simulation_report",
                    "simulation_config": simulation_config or {},
                    "report_config": {
                        "output_directory": str(self.output_dir),
                        "include_visualizations": self.config.include_visualizations,
                        "include_interactive_plots": self.config.include_interactive_plots
                    }
                },
                "sections": report_sections
            }
            
            # Save report
            report_path = self._save_report(report)
            
            logger.info(f"Comprehensive report generated successfully: {report_path}")
            
            return {
                "success": True,
                "report_path": str(report_path),
                "report_sections": list(report_sections.keys()),
                "metrics_summary": self._get_metrics_summary()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_comparative_report(self, 
                                  simulation_runs: List[Dict[str, Any]],
                                  comparison_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comparative report across multiple simulation runs.
        
        Args:
            simulation_runs: List of simulation results to compare
            comparison_metrics: Specific metrics to compare
        
        Returns:
            Dictionary with comparative report results
        """
        try:
            logger.info(f"Generating comparative report for {len(simulation_runs)} simulation runs")
            
            if len(simulation_runs) < 2:
                return {"success": False, "error": "At least 2 simulation runs required for comparison"}
            
            # Extract metrics from all runs
            all_metrics = []
            for i, run in enumerate(simulation_runs):
                metrics = self._extract_metrics_from_run(run, run_id=f"run_{i+1}")
                all_metrics.append(metrics)
            
            # Generate comparative analysis
            comparative_analysis = self._generate_comparative_analysis(all_metrics, comparison_metrics)
            
            # Generate comparative visualizations
            comparative_visualizations = self._generate_comparative_visualizations(all_metrics)
            
            # Create comparative report
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "comparative_simulation_report",
                    "number_of_runs": len(simulation_runs),
                    "comparison_metrics": comparison_metrics or []
                },
                "comparative_analysis": comparative_analysis,
                "visualizations": comparative_visualizations,
                "run_summaries": [self._get_run_summary(metrics) for metrics in all_metrics]
            }
            
            # Save comparative report
            report_path = self._save_comparative_report(report)
            
            logger.info(f"Comparative report generated successfully: {report_path}")
            
            return {
                "success": True,
                "report_path": str(report_path),
                "comparative_analysis": comparative_analysis,
                "number_of_runs": len(simulation_runs)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comparative report: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_performance_report(self, 
                                  performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed performance analysis report.
        
        Args:
            performance_data: Performance metrics and timing data
        
        Returns:
            Dictionary with performance report results
        """
        try:
            logger.info("Generating performance analysis report")
            
            # Analyze performance data
            performance_analysis = self._analyze_performance_data(performance_data)
            
            # Generate performance visualizations
            performance_visualizations = self._generate_performance_visualizations(performance_data)
            
            # Create performance report
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "performance_analysis_report"
                },
                "performance_analysis": performance_analysis,
                "visualizations": performance_visualizations,
                "optimization_recommendations": self._generate_performance_recommendations(performance_analysis)
            }
            
            # Save performance report
            report_path = self._save_performance_report(report)
            
            logger.info(f"Performance report generated successfully: {report_path}")
            
            return {
                "success": True,
                "report_path": str(report_path),
                "performance_analysis": performance_analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"success": False, "error": str(e)}
    
    def export_data(self, 
                   simulation_results: Dict[str, Any],
                   export_formats: List[str] = None) -> Dict[str, Any]:
        """
        Export simulation data in various formats.
        
        Args:
            simulation_results: Simulation results to export
            export_formats: List of formats to export (json, csv, excel, parquet)
        
        Returns:
            Dictionary with export results
        """
        try:
            export_formats = export_formats or ["json", "csv"]
            export_paths = {}
            
            logger.info(f"Exporting simulation data in formats: {export_formats}")
            
            # Prepare data for export
            export_data = self._prepare_export_data(simulation_results)
            
            # Export in each requested format
            for format_type in export_formats:
                if format_type == "json":
                    path = self._export_json(export_data)
                elif format_type == "csv":
                    path = self._export_csv(export_data)
                elif format_type == "excel":
                    path = self._export_excel(export_data)
                elif format_type == "parquet":
                    path = self._export_parquet(export_data)
                else:
                    logger.warning(f"Unknown export format: {format_type}")
                    continue
                
                export_paths[format_type] = str(path)
            
            logger.info(f"Data exported successfully to {len(export_paths)} formats")
            
            return {
                "success": True,
                "export_paths": export_paths,
                "exported_formats": list(export_paths.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_metrics(self, simulation_results: Dict[str, Any]):
        """Extract metrics from simulation results."""
        try:
            # Helper function to safely convert to numeric type
            def safe_float(value, default=0.0):
                """Safely convert value to float, handling None and string cases."""
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=0):
                """Safely convert value to int, handling None and string cases."""
                if value is None:
                    return default
                try:
                    return int(float(value))  # Convert via float first to handle string numbers
                except (ValueError, TypeError):
                    return default
            
            # Extract spatial metrics
            spatial_data = simulation_results.get("spatial_metrics", {})
            self.spatial_metrics.logistics_friction_total = safe_float(spatial_data.get("total_logistics_friction", 0.0))
            self.spatial_metrics.logistics_friction_average = safe_float(spatial_data.get("average_logistics_friction", 0.0))
            self.spatial_metrics.logistics_friction_max = safe_float(spatial_data.get("max_logistics_friction", 0.0))
            self.spatial_metrics.total_disasters = safe_int(spatial_data.get("total_disasters", 0))
            self.spatial_metrics.average_active_disasters = safe_float(spatial_data.get("average_active_disasters", 0.0))
            
            # Extract economic metrics
            economic_data = simulation_results.get("economic_metrics", {})
            self.economic_metrics.total_economic_output = safe_float(economic_data.get("average_economic_output", 0.0))
            self.economic_metrics.economic_growth_rate = safe_float(economic_data.get("economic_growth_rate", 0.0))
            self.economic_metrics.total_capital_stock = safe_float(economic_data.get("average_capital_stock", 0.0))
            self.economic_metrics.capital_accumulation_rate = safe_float(economic_data.get("capital_accumulation_rate", 0.0))
            
            # Extract integration metrics
            integration_data = simulation_results.get("integration_metrics", {})
            self.integration_metrics.spatial_economic_efficiency = safe_float(integration_data.get("average_spatial_efficiency", 0.0))
            self.integration_metrics.integration_stability = safe_float(integration_data.get("integration_stability", 0.0))
            self.integration_metrics.disaster_economic_impact = safe_float(integration_data.get("total_disaster_economic_impact", 0.0))
            
            # Extract performance metrics
            performance_data = simulation_results.get("performance_metrics", {})
            self.performance_metrics.total_simulation_time = safe_float(performance_data.get("total_simulation_time", 0.0))
            self.performance_metrics.average_step_time = safe_float(performance_data.get("average_step_time", 0.0))
            self.performance_metrics.max_step_time = safe_float(performance_data.get("max_step_time", 0.0))
            self.performance_metrics.error_rate = safe_float(performance_data.get("error_rate", 0.0))
            
            logger.info("Metrics extracted successfully")
            
        except Exception as e:
            logger.error(f"Failed to extract metrics: {e}")
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of simulation results."""
        # Helper function to safely format numeric values
        def safe_format(value, format_str=":.2f", default="0.00"):
            """Safely format numeric values, handling None and non-numeric cases."""
            if value is None:
                return default
            try:
                return f"{float(value):{format_str}}"
            except (ValueError, TypeError):
                return default
        
        def safe_format_percent(value, default="0.00%"):
            """Safely format percentage values."""
            if value is None:
                return default
            try:
                return f"{float(value):.2%}"
            except (ValueError, TypeError):
                return default
        
        return {
            "simulation_overview": {
                "total_simulation_time": f"{safe_format(self.performance_metrics.total_simulation_time)} seconds",
                "average_step_time": f"{safe_format(self.performance_metrics.average_step_time, ':.3f', '0.000')} seconds",
                "error_rate": safe_format_percent(self.performance_metrics.error_rate)
            },
            "key_findings": {
                "spatial_efficiency": safe_format_percent(self.integration_metrics.spatial_economic_efficiency),
                "economic_growth": safe_format_percent(self.economic_metrics.economic_growth_rate),
                "disaster_resilience": f"{self.spatial_metrics.total_disasters} disasters handled",
                "logistics_optimization": f"{safe_format(self.spatial_metrics.logistics_friction_average)} average friction"
            },
            "performance_highlights": {
                "best_performing_aspect": self._identify_best_performing_aspect(),
                "areas_for_improvement": self._identify_improvement_areas(),
                "overall_simulation_quality": self._assess_simulation_quality()
            }
        }
    
    def _generate_spatial_analysis(self) -> Dict[str, Any]:
        """Generate spatial analysis section."""
        return {
            "logistics_analysis": {
                "total_logistics_friction": self.spatial_metrics.logistics_friction_total,
                "average_logistics_friction": self.spatial_metrics.logistics_friction_average,
                "max_logistics_friction": self.spatial_metrics.logistics_friction_max,
                "logistics_efficiency_trend": self._analyze_logistics_trend()
            },
            "disaster_analysis": {
                "total_disasters": self.spatial_metrics.total_disasters,
                "average_active_disasters": self.spatial_metrics.average_active_disasters,
                "disaster_impact_assessment": self._assess_disaster_impact(),
                "disaster_resilience_score": self._calculate_disaster_resilience()
            },
            "infrastructure_analysis": {
                "settlements_count": self.spatial_metrics.settlements_count,
                "infrastructure_segments": self.spatial_metrics.infrastructure_segments,
                "infrastructure_efficiency": self._calculate_infrastructure_efficiency(),
                "connectivity_analysis": self._analyze_connectivity()
            }
        }
    
    def _generate_economic_analysis(self) -> Dict[str, Any]:
        """Generate economic analysis section."""
        return {
            "output_analysis": {
                "total_economic_output": self.economic_metrics.total_economic_output,
                "economic_growth_rate": self.economic_metrics.economic_growth_rate,
                "output_trend_analysis": self._analyze_output_trend(),
                "sector_performance": self.economic_metrics.sector_performance
            },
            "capital_analysis": {
                "total_capital_stock": self.economic_metrics.total_capital_stock,
                "capital_accumulation_rate": self.economic_metrics.capital_accumulation_rate,
                "capital_efficiency": self._calculate_capital_efficiency(),
                "investment_patterns": self._analyze_investment_patterns()
            },
            "productivity_analysis": {
                "labor_productivity": self.economic_metrics.labor_productivity,
                "resource_efficiency": self.economic_metrics.resource_efficiency,
                "technology_level": self.economic_metrics.technology_level,
                "productivity_trends": self._analyze_productivity_trends()
            }
        }
    
    def _generate_integration_analysis(self) -> Dict[str, Any]:
        """Generate integration analysis section."""
        return {
            "spatial_economic_integration": {
                "overall_efficiency": self.integration_metrics.spatial_economic_efficiency,
                "integration_stability": self.integration_metrics.integration_stability,
                "integration_trends": self._analyze_integration_trends(),
                "bottlenecks_identified": self._identify_integration_bottlenecks()
            },
            "feedback_analysis": {
                "disaster_economic_impact": self.integration_metrics.disaster_economic_impact,
                "logistics_economic_correlation": self.integration_metrics.logistics_economic_correlation,
                "infrastructure_economic_feedback": self.integration_metrics.infrastructure_economic_feedback,
                "feedback_effectiveness": self._assess_feedback_effectiveness()
            },
            "optimization_analysis": {
                "sector_settlement_optimization": self.integration_metrics.sector_settlement_optimization,
                "resource_spatial_distribution": self.integration_metrics.resource_spatial_distribution,
                "optimization_opportunities": self._identify_optimization_opportunities(),
                "recommended_adjustments": self._recommend_integration_adjustments()
            }
        }
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate performance analysis section."""
        return {
            "execution_performance": {
                "total_simulation_time": self.performance_metrics.total_simulation_time,
                "average_step_time": self.performance_metrics.average_step_time,
                "max_step_time": self.performance_metrics.max_step_time,
                "performance_consistency": self._assess_performance_consistency()
            },
            "resource_utilization": {
                "memory_usage": self.performance_metrics.memory_usage_mb,
                "parallel_efficiency": self.performance_metrics.parallel_efficiency,
                "cpu_utilization": self._estimate_cpu_utilization(),
                "resource_bottlenecks": self._identify_resource_bottlenecks()
            },
            "reliability_analysis": {
                "error_count": self.performance_metrics.error_count,
                "error_rate": self.performance_metrics.error_rate,
                "checkpoint_count": self.performance_metrics.checkpoint_count,
                "reliability_score": self._calculate_reliability_score()
            }
        }
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations based on analysis."""
        recommendations = {
            "spatial_recommendations": self._generate_spatial_recommendations(),
            "economic_recommendations": self._generate_economic_recommendations(),
            "integration_recommendations": self._generate_integration_recommendations(),
            "performance_recommendations": self._generate_performance_recommendations(),
            "priority_actions": self._prioritize_recommendations()
        }
        
        return recommendations
    
    def _generate_appendix(self) -> Dict[str, Any]:
        """Generate appendix with detailed data and methodology."""
        return {
            "methodology": {
                "spatial_analysis_methods": "Procedural map generation with Perlin noise, A* pathfinding for infrastructure",
                "economic_analysis_methods": "Leontief input-output model with dynamic planning and capital accumulation",
                "integration_methods": "Bidirectional feedback loops with constraint satisfaction optimization",
                "performance_analysis_methods": "Statistical analysis of execution times and resource utilization"
            },
            "data_sources": {
                "spatial_data": "Procedurally generated terrain, settlements, and infrastructure",
                "economic_data": "Synthetic sector data with realistic input-output relationships",
                "performance_data": "Real-time execution metrics and system resource monitoring"
            },
            "assumptions": {
                "spatial_assumptions": "Terrain affects infrastructure costs, disasters impact regional production",
                "economic_assumptions": "Central planning with optimization, capital accumulation over time",
                "integration_assumptions": "Spatial constraints affect economic efficiency, economic activity drives infrastructure development"
            }
        }
    
    def _generate_visualizations(self) -> Dict[str, Any]:
        """Generate visualizations for the report."""
        visualizations = {}
        
        try:
            # Spatial visualizations
            visualizations["spatial"] = {
                "logistics_friction_over_time": self._create_logistics_friction_chart(),
                "disaster_impact_map": self._create_disaster_impact_map(),
                "infrastructure_network": self._create_infrastructure_network_chart()
            }
            
            # Economic visualizations
            visualizations["economic"] = {
                "economic_output_trend": self._create_economic_output_chart(),
                "capital_accumulation": self._create_capital_accumulation_chart(),
                "sector_performance_radar": self._create_sector_performance_radar()
            }
            
            # Integration visualizations
            visualizations["integration"] = {
                "spatial_economic_correlation": self._create_correlation_matrix(),
                "efficiency_heatmap": self._create_efficiency_heatmap(),
                "feedback_flow_diagram": self._create_feedback_flow_diagram()
            }
            
            # Performance visualizations
            if self.config.include_performance_metrics:
                visualizations["performance"] = {
                    "execution_time_trend": self._create_execution_time_chart(),
                    "resource_utilization": self._create_resource_utilization_chart(),
                    "error_analysis": self._create_error_analysis_chart()
                }
            
            logger.info("Visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            visualizations["error"] = str(e)
        
        return visualizations
    
    def _setup_visualization_style(self):
        """Setup visualization style based on configuration."""
        if self.config.visualization_style == "professional":
            plt.style.use('seaborn-v0_8-whitegrid')
            self.color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        elif self.config.visualization_style == "academic":
            plt.style.use('seaborn-v0_8-paper')
            self.color_palette = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#7209B7"]
        else:  # colorful
            plt.style.use('default')
            self.color_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    
    def _save_report(self, report: Dict[str, Any]) -> Path:
        """Save the comprehensive report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.report_format == "html":
            report_path = self.output_dir / f"unified_simulation_report_{timestamp}.html"
            self._save_html_report(report, report_path)
        elif self.config.report_format == "json":
            report_path = self.output_dir / f"unified_simulation_report_{timestamp}.json"
            self._save_json_report(report, report_path)
        else:
            report_path = self.output_dir / f"unified_simulation_report_{timestamp}.json"
            self._save_json_report(report, report_path)
        
        return report_path
    
    def _save_html_report(self, report: Dict[str, Any], report_path: Path):
        """Save report as HTML."""
        html_content = self._generate_html_content(report)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _save_json_report(self, report: Dict[str, Any], report_path: Path):
        """Save report as JSON."""
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _generate_html_content(self, report: Dict[str, Any]) -> str:
        """Generate HTML content for the report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unified Simulation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }}
                .chart {{ margin: 20px 0; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Unified Simulation Report</h1>
                <p>Generated: {report['report_metadata']['generated_at']}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                {self._format_executive_summary_html(report['sections']['executive_summary'])}
            </div>
            
            <div class="section">
                <h2>Spatial Analysis</h2>
                {self._format_spatial_analysis_html(report['sections']['spatial_analysis'])}
            </div>
            
            <div class="section">
                <h2>Economic Analysis</h2>
                {self._format_economic_analysis_html(report['sections']['economic_analysis'])}
            </div>
            
            <div class="section">
                <h2>Integration Analysis</h2>
                {self._format_integration_analysis_html(report['sections']['integration_analysis'])}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._format_recommendations_html(report['sections']['recommendations'])}
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_executive_summary_html(self, summary: Dict[str, Any]) -> str:
        """Format executive summary for HTML."""
        html = "<div class='metric'>"
        html += f"<h3>Simulation Overview</h3>"
        html += f"<p>Total simulation time: {summary['simulation_overview']['total_simulation_time']}</p>"
        html += f"<p>Average step time: {summary['simulation_overview']['average_step_time']}</p>"
        html += f"<p>Error rate: {summary['simulation_overview']['error_rate']}</p>"
        html += "</div>"
        
        html += "<div class='metric'>"
        html += f"<h3>Key Findings</h3>"
        html += f"<p>Spatial efficiency: {summary['key_findings']['spatial_efficiency']}</p>"
        html += f"<p>Economic growth: {summary['key_findings']['economic_growth']}</p>"
        html += f"<p>Disaster resilience: {summary['key_findings']['disaster_resilience']}</p>"
        html += "</div>"
        
        return html
    
    def _format_spatial_analysis_html(self, analysis: Dict[str, Any]) -> str:
        """Format spatial analysis for HTML."""
        html = "<div class='metric'>"
        html += f"<h3>Logistics Analysis</h3>"
        html += f"<p>Total logistics friction: {analysis['logistics_analysis']['total_logistics_friction']:.2f}</p>"
        html += f"<p>Average logistics friction: {analysis['logistics_analysis']['average_logistics_friction']:.2f}</p>"
        html += "</div>"
        
        html += "<div class='metric'>"
        html += f"<h3>Disaster Analysis</h3>"
        html += f"<p>Total disasters: {analysis['disaster_analysis']['total_disasters']}</p>"
        html += f"<p>Average active disasters: {analysis['disaster_analysis']['average_active_disasters']:.2f}</p>"
        html += "</div>"
        
        return html
    
    def _format_economic_analysis_html(self, analysis: Dict[str, Any]) -> str:
        """Format economic analysis for HTML."""
        html = "<div class='metric'>"
        html += f"<h3>Output Analysis</h3>"
        html += f"<p>Total economic output: {analysis['output_analysis']['total_economic_output']:.2f}</p>"
        html += f"<p>Economic growth rate: {analysis['output_analysis']['economic_growth_rate']:.2%}</p>"
        html += "</div>"
        
        html += "<div class='metric'>"
        html += f"<h3>Capital Analysis</h3>"
        html += f"<p>Total capital stock: {analysis['capital_analysis']['total_capital_stock']:.2f}</p>"
        html += f"<p>Capital accumulation rate: {analysis['capital_analysis']['capital_accumulation_rate']:.2%}</p>"
        html += "</div>"
        
        return html
    
    def _format_integration_analysis_html(self, analysis: Dict[str, Any]) -> str:
        """Format integration analysis for HTML."""
        html = "<div class='metric'>"
        html += f"<h3>Spatial-Economic Integration</h3>"
        html += f"<p>Overall efficiency: {analysis['spatial_economic_integration']['overall_efficiency']:.2%}</p>"
        html += f"<p>Integration stability: {analysis['spatial_economic_integration']['integration_stability']:.2%}</p>"
        html += "</div>"
        
        html += "<div class='metric'>"
        html += f"<h3>Feedback Analysis</h3>"
        html += f"<p>Disaster economic impact: {analysis['feedback_analysis']['disaster_economic_impact']:.2f}</p>"
        html += f"<p>Logistics-economic correlation: {analysis['feedback_analysis']['logistics_economic_correlation']:.2f}</p>"
        html += "</div>"
        
        return html
    
    def _format_recommendations_html(self, recommendations: Dict[str, Any]) -> str:
        """Format recommendations for HTML."""
        html = "<div class='metric'>"
        html += f"<h3>Priority Actions</h3>"
        priority_actions = recommendations.get('priority_actions', [])
        if priority_actions:
            html += "<ul>"
            for action in priority_actions[:5]:  # Show top 5
                html += f"<li>{action}</li>"
            html += "</ul>"
        else:
            html += "<p>No specific priority actions identified.</p>"
        html += "</div>"
        
        return html
    
    # Placeholder methods for analysis functions
    def _analyze_logistics_trend(self) -> str:
        return "Stable logistics performance with minor fluctuations"
    
    def _assess_disaster_impact(self) -> str:
        return "Moderate disaster impact with effective recovery mechanisms"
    
    def _calculate_disaster_resilience(self) -> float:
        return 0.75  # Placeholder
    
    def _calculate_infrastructure_efficiency(self) -> float:
        return 0.82  # Placeholder
    
    def _analyze_connectivity(self) -> str:
        return "Good connectivity with room for optimization"
    
    def _analyze_output_trend(self) -> str:
        return "Positive economic growth trend"
    
    def _calculate_capital_efficiency(self) -> float:
        return 0.68  # Placeholder
    
    def _analyze_investment_patterns(self) -> str:
        return "Balanced investment across sectors"
    
    def _analyze_productivity_trends(self) -> str:
        return "Improving productivity with technological advancement"
    
    def _analyze_integration_trends(self) -> str:
        return "Strengthening integration between spatial and economic systems"
    
    def _identify_integration_bottlenecks(self) -> List[str]:
        return ["Logistics costs", "Infrastructure capacity"]
    
    def _assess_feedback_effectiveness(self) -> float:
        return 0.71  # Placeholder
    
    def _identify_optimization_opportunities(self) -> List[str]:
        return ["Sector-settlement mapping", "Resource distribution"]
    
    def _recommend_integration_adjustments(self) -> List[str]:
        return ["Improve logistics efficiency", "Enhance disaster resilience"]
    
    def _assess_performance_consistency(self) -> str:
        return "Consistent performance with minor variations"
    
    def _estimate_cpu_utilization(self) -> float:
        return 0.65  # Placeholder
    
    def _identify_resource_bottlenecks(self) -> List[str]:
        return ["Memory usage", "I/O operations"]
    
    def _calculate_reliability_score(self) -> float:
        return 0.89  # Placeholder
    
    def _generate_spatial_recommendations(self) -> List[str]:
        return ["Optimize infrastructure network", "Improve disaster response"]
    
    def _generate_economic_recommendations(self) -> List[str]:
        return ["Increase capital investment", "Improve sector efficiency"]
    
    def _generate_integration_recommendations(self) -> List[str]:
        return ["Strengthen spatial-economic feedback", "Optimize sector-settlement mapping"]
    
    def _generate_performance_recommendations(self) -> Dict[str, Any]:
        return {"recommendations": ["Optimize parallel execution", "Improve memory management"]}
    
    def _prioritize_recommendations(self) -> List[str]:
        return ["Improve logistics efficiency", "Enhance disaster resilience", "Optimize sector allocation"]
    
    def _identify_best_performing_aspect(self) -> str:
        return "Economic growth and capital accumulation"
    
    def _identify_improvement_areas(self) -> List[str]:
        return ["Logistics optimization", "Disaster resilience"]
    
    def _assess_simulation_quality(self) -> str:
        return "High quality simulation with realistic outcomes"
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        return {
            "spatial_efficiency": self.integration_metrics.spatial_economic_efficiency,
            "economic_growth": self.economic_metrics.economic_growth_rate,
            "disaster_resilience": 0.75,  # Placeholder
            "performance_score": 0.85  # Placeholder
        }
    
    # Placeholder methods for visualization creation
    def _create_logistics_friction_chart(self) -> str:
        return "logistics_friction_chart.html"
    
    def _create_disaster_impact_map(self) -> str:
        return "disaster_impact_map.html"
    
    def _create_infrastructure_network_chart(self) -> str:
        return "infrastructure_network_chart.html"
    
    def _create_economic_output_chart(self) -> str:
        return "economic_output_chart.html"
    
    def _create_capital_accumulation_chart(self) -> str:
        return "capital_accumulation_chart.html"
    
    def _create_sector_performance_radar(self) -> str:
        return "sector_performance_radar.html"
    
    def _create_correlation_matrix(self) -> str:
        return "correlation_matrix.html"
    
    def _create_efficiency_heatmap(self) -> str:
        return "efficiency_heatmap.html"
    
    def _create_feedback_flow_diagram(self) -> str:
        return "feedback_flow_diagram.html"
    
    def _create_execution_time_chart(self) -> str:
        return "execution_time_chart.html"
    
    def _create_resource_utilization_chart(self) -> str:
        return "resource_utilization_chart.html"
    
    def _create_error_analysis_chart(self) -> str:
        return "error_analysis_chart.html"
    
    # Placeholder methods for data export
    def _prepare_export_data(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        return simulation_results
    
    def _export_json(self, data: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"simulation_data_{timestamp}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path
    
    def _export_csv(self, data: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"simulation_data_{timestamp}.csv"
        # Convert to DataFrame and save
        df = pd.DataFrame([data])
        df.to_csv(path, index=False)
        return path
    
    def _export_excel(self, data: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"simulation_data_{timestamp}.xlsx"
        # Convert to DataFrame and save
        df = pd.DataFrame([data])
        df.to_excel(path, index=False)
        return path
    
    def _export_parquet(self, data: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"simulation_data_{timestamp}.parquet"
        # Convert to DataFrame and save
        df = pd.DataFrame([data])
        df.to_parquet(path, index=False)
        return path
    
    # Placeholder methods for comparative analysis
    def _extract_metrics_from_run(self, run: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        return {"run_id": run_id, "metrics": run}
    
    def _generate_comparative_analysis(self, all_metrics: List[Dict[str, Any]], comparison_metrics: Optional[List[str]]) -> Dict[str, Any]:
        return {"comparative_analysis": "placeholder"}
    
    def _generate_comparative_visualizations(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"comparative_visualizations": "placeholder"}
    
    def _get_run_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {"run_summary": "placeholder"}
    
    def _save_comparative_report(self, report: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"comparative_report_{timestamp}.json"
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return path
    
    def _analyze_performance_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"performance_analysis": "placeholder"}
    
    def _generate_performance_visualizations(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"performance_visualizations": "placeholder"}
    
    def _generate_performance_recommendations(self, performance_analysis: Dict[str, Any]) -> List[str]:
        return ["Optimize parallel execution", "Improve memory management"]
    
    def _save_performance_report(self, report: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"performance_report_{timestamp}.json"
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return path
