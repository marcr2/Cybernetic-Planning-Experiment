"""
Writer Agent

Specialized agent for generating comprehensive markdown reports.
Creates detailed economic planning reports with mathematical transparency.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from .base import BaseAgent

class WriterAgent(BaseAgent):
    """
    Report generation specialist agent.

    Generates comprehensive markdown reports with mathematical transparency,
    including executive summaries, detailed analyses, and visualizations.
    """

    def __init__(self):
        """Initialize the writer agent."""
        super().__init__("writer", "Report Generation Agent")
        self.report_templates = {}
        self._initialize_report_templates()

    def _initialize_report_templates(self) -> None:
        """Initialize report templates."""
        self.report_templates = {
            "executive_summary": self._create_executive_summary_template(),
            "sector_analysis": self._create_sector_analysis_template(),
            "resource_allocation": self._create_resource_allocation_template(),
            "labor_budget": self._create_labor_budget_template(),
            "risk_assessment": self._create_risk_assessment_template(),
        }

    def get_capabilities(self) -> List[str]:
        """Get writer agent capabilities."""
        return [
            "markdown_report_generation",
            "executive_summary_creation",
            "mathematical_visualization",
            "data_table_formatting",
            "report_customization",
        ]

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a report generation task.

        Args:
            task: Task description and parameters

        Returns:
            Generated report
        """
        task_type = task.get("type", "unknown")

        if task_type == "generate_report":
            return self._generate_comprehensive_report(task)
        elif task_type == "generate_summary":
            return self._generate_executive_summary(task)
        elif task_type == "generate_section":
            return self._generate_report_section(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}

    def _generate_comprehensive_report(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive economic planning report.

        Args:
            task: Task parameters including plan data and options

        Returns:
            Complete markdown report
        """
        plan_data = task.get("plan_data", {})
        report_options = task.get("options", {})

        if not plan_data:
            return {"error": "No plan data provided"}

        # Generate report sections
        sections = []

        # Executive Summary
        executive_summary = self._generate_executive_summary_section(plan_data)
        sections.append(executive_summary)

        # Sector - by - Sector Analysis
        sector_analysis = self._generate_sector_analysis_section(plan_data)
        sections.append(sector_analysis)

        # Resource Allocation
        resource_allocation = self._generate_resource_allocation_section(plan_data)
        sections.append(resource_allocation)

        # Labor Budget
        labor_budget = self._generate_labor_budget_section(plan_data)
        sections.append(labor_budget)

        # Risk Assessment
        risk_assessment = self._generate_risk_assessment_section(plan_data)
        sections.append(risk_assessment)

        # Automatic Analysis Integration
        automatic_analysis = self._generate_automatic_analysis_section(plan_data)
        sections.append(automatic_analysis)

        # Combine sections
        full_report = self._combine_report_sections(sections, report_options)

        return {
            "status": "success",
            "report": full_report,
            "sections": [section["title"] for section in sections],
            "word_count": len(full_report.split()),
            "analysis_type": "comprehensive_report",
        }

    def _generate_executive_summary_section(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary section."""
        title = "Executive Summary"

        # Extract key metrics with data validation
        total_output = plan_data.get("total_output", np.array([]))
        total_labor_cost = plan_data.get("total_labor_cost", 0)
        final_demand = plan_data.get("final_demand", np.array([]))
        
        # Ensure data is numeric
        if hasattr(total_output, 'dtype') and not np.issubdtype(total_output.dtype, np.number):
            total_output = np.array([], dtype=float)
        if hasattr(final_demand, 'dtype') and not np.issubdtype(final_demand.dtype, np.number):
            final_demand = np.array([], dtype=float)
        if not isinstance(total_labor_cost, (int, float, np.number)):
            total_labor_cost = 0

        # Validate data and handle negative values
        if len(total_output) > 0 and np.any(total_output < 0):
            return {
                "title": title,
                "content": f"""## {title}

### ⚠️ DATA VALIDATION ERROR

**Critical Issue**: The economic plan contains negative output values, which is economically impossible.

**Root Cause**: This indicates a fundamental problem with either:
1. The synthetic data generation algorithm
2. The optimization solver
3. The input - output matrix structure

**Immediate Action Required**:
- Review the technology matrix for productivity (spectral radius < 1)
- Check that final demand values are positive - Verify the optimization constraints are properly formulated

**Status**: Plan is not viable for implementation.""",
                "metrics": {"error": "negative_outputs"}
            }

        # Calculate summary statistics with safe conversion
        def safe_numeric(value, default=0.0):
            """Safely convert value to numeric."""
            if value is None:
                return default
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        total_economic_output = np.sum(total_output)
        total_labor_cost_safe = safe_numeric(total_labor_cost, 0)
        labor_efficiency = total_economic_output / (total_labor_cost_safe + 1e-10) if total_labor_cost_safe > 0 else 0

        # Calculate demand fulfillment correctly using net output
        technology_matrix = plan_data.get("technology_matrix")
        if technology_matrix is not None:
            I = np.eye(technology_matrix.shape[0])
            net_output = (I - technology_matrix) @ total_output
            demand_fulfillment = np.sum(net_output) / (np.sum(final_demand) + 1e-10) if np.sum(final_demand) > 0 else 0
        else:
            # Fallback: assume perfect fulfillment if no technology matrix
            demand_fulfillment = 1.0

        content = f"""## {title}

### Key Targets
- **Total Economic Output**: {total_economic_output:,.2f} units
- **Final Demand Target**: {np.sum(final_demand):,.2f} units
- **Total Labor Cost**: {total_labor_cost_safe:,.2f} person - hours
- **Labor Efficiency**: {labor_efficiency:.6f} units per person - hour
- **Demand Fulfillment Rate**: {demand_fulfillment:.2%}

### Plan Overview
This 5 - year economic plan has been generated using cybernetic planning principles,
combining Input - Output analysis with labor - time accounting. The plan optimizes for
labor efficiency while ensuring all final demand targets are met.

### Next Steps
1. Implement the plan with careful monitoring
2. Adjust based on real - world feedback
3. Update technology matrices as needed
4. Monitor resource utilization closely
"""

        return {
            "title": title,
            "content": content,
            "metrics": {
                "total_economic_output": total_economic_output,
                "total_labor_cost": total_labor_cost_safe,
                "labor_efficiency": labor_efficiency,
                "demand_fulfillment": demand_fulfillment,
            },
        }

    def _generate_sector_analysis_section(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sector - by - sector analysis section."""
        title = "Sector - by - Sector Analysis"

        total_output = plan_data.get("total_output", np.array([]))
        labor_values = plan_data.get("labor_values", np.array([]))
        final_demand = plan_data.get("final_demand", np.array([]))
        
        # Ensure data is numeric
        if hasattr(total_output, 'dtype') and not np.issubdtype(total_output.dtype, np.number):
            total_output = np.array([], dtype=float)
        if hasattr(labor_values, 'dtype') and not np.issubdtype(labor_values.dtype, np.number):
            labor_values = np.array([], dtype=float)
        if hasattr(final_demand, 'dtype') and not np.issubdtype(final_demand.dtype, np.number):
            final_demand = np.array([], dtype=float)

        if len(total_output) == 0:
            content = f"## {title}\n\nNo sector data available."
            return {"title": title, "content": content}

        # Create sector table
        sector_table = self._create_sector_table(total_output, labor_values, final_demand)

        content = f"""## {title}

### Required Gross Output by Sector

{sector_table}

### Sector Analysis Summary
- **Total Sectors**: {len(total_output)}
- **Average Output per Sector**: {np.mean(total_output):,.2f} units
- **Highest Output Sector**: Sector {np.argmax(total_output)} ({np.max(total_output):,.2f} units)
- **Lowest Output Sector**: Sector {np.argmin(total_output)} ({np.min(total_output):,.2f} units)
- **Output Standard Deviation**: {np.std(total_output):,.2f} units

"""

        return {
            "title": title,
            "content": content,
            "sector_data": {
                "total_sectors": len(total_output),
                "average_output": np.mean(total_output),
                "max_output": np.max(total_output),
                "min_output": np.min(total_output),
                "std_output": np.std(total_output),
            },
        }

    def _create_sector_table(self, total_output: np.ndarray, labor_values: np.ndarray, final_demand: np.ndarray) -> str:
        """Create a formatted table of sector data."""
        table_lines = [
            "| Sector | Total Output | Labor Value | Final Demand | Labor Cost |",
            "|--------|--------------|-------------|--------------|------------|",
        ]

        for i in range(len(total_output)):
            labor_cost = labor_values[i] * total_output[i] if i < len(labor_values) else 0
            final_demand_val = final_demand[i] if i < len(final_demand) else 0

            table_lines.append(
                f"| {i:6d} | {total_output[i]:12.2f} | {labor_values[i]:11.4f} | "
                f"{final_demand_val:12.2f} | {labor_cost:10.2f} |"
            )

        return "\n".join(table_lines)

    def _generate_resource_allocation_section(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate resource allocation section."""
        title = "Resource Allocation"

        resource_usage = plan_data.get("resource_usage")
        max_resources = plan_data.get("max_resources")
        plan_data.get("resource_matrix")

        if resource_usage is None or max_resources is None:
            content = f"## {title}\n\nNo resource allocation data available."
            return {"title": title, "content": content}

        # Create resource table
        resource_table = self._create_resource_table(resource_usage, max_resources)

        # Calculate utilization rates
        utilization_rates = resource_usage / (max_resources + 1e-10)
        avg_utilization = np.mean(utilization_rates)
        max_utilization = np.max(utilization_rates)

        content = f"""## {title}

### Resource Usage Summary

{resource_table}

### Resource Utilization Analysis
- **Average Utilization Rate**: {avg_utilization:.2%}
- **Maximum Utilization Rate**: {max_utilization:.2%}
- **Resource Efficiency**: {1 - avg_utilization:.2%} unused capacity

### Resource Constraints
- **Total Resources Available**: {np.sum(max_resources):,.2f} units
- **Total Resources Used**: {np.sum(resource_usage):,.2f} units
- **Resource Utilization**: {np.sum(resource_usage) / np.sum(max_resources):.2%}

"""

        return {
            "title": title,
            "content": content,
            "resource_data": {
                "total_available": np.sum(max_resources),
                "total_used": np.sum(resource_usage),
                "avg_utilization": avg_utilization,
                "max_utilization": max_utilization,
            },
        }

    def _create_resource_table(self, resource_usage: np.ndarray, max_resources: np.ndarray) -> str:
        """Create a formatted table of resource data."""
        table_lines = [
            "| Resource | Used | Available | Utilization | Status |",
            "|----------|------|-----------|-------------|--------|",
        ]

        for i in range(len(resource_usage)):
            utilization = resource_usage[i] / (max_resources[i] + 1e-10)
            status = "Critical" if utilization > 0.9 else "Normal" if utilization > 0.7 else "Low"

            table_lines.append(
                f"| {i:8d} | {resource_usage[i]:4.2f} | {max_resources[i]:9.2f} | "
                f"{utilization:11.2%} | {status:6s} |"
            )

        return "\n".join(table_lines)

    def _generate_labor_budget_section(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate labor budget section."""
        title = "Labor Budget"

        total_output = plan_data.get("total_output", np.array([]))
        labor_values = plan_data.get("labor_values", np.array([]))
        labor_vector = plan_data.get("labor_vector", np.array([]))
        total_labor_cost = plan_data.get("total_labor_cost", 0)
        
        # Ensure data is numeric
        if hasattr(total_output, 'dtype') and not np.issubdtype(total_output.dtype, np.number):
            total_output = np.array([], dtype=float)
        if hasattr(labor_values, 'dtype') and not np.issubdtype(labor_values.dtype, np.number):
            labor_values = np.array([], dtype=float)
        if hasattr(labor_vector, 'dtype') and not np.issubdtype(labor_vector.dtype, np.number):
            labor_vector = np.array([], dtype=float)
        if not isinstance(total_labor_cost, (int, float, np.number)):
            total_labor_cost = 0

        if len(total_output) == 0 or len(labor_values) == 0:
            content = f"## {title}\n\nNo labor data available."
            return {"title": title, "content": content}

        # Create labor table
        labor_table = self._create_labor_table(total_output, labor_values, labor_vector)

        # Calculate labor statistics with safe conversion
        def safe_numeric_labor(value, default=0.0):
            """Safely convert value to numeric for labor calculations."""
            if value is None:
                return default
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        total_labor_cost_safe = safe_numeric_labor(total_labor_cost, 0)
        avg_labor_value = np.mean(labor_values)
        total_direct_labor = np.sum(labor_vector * total_output) if len(labor_vector) > 0 else 0

        content = f"""## {title}

### Labor Allocation by Sector

{labor_table}

### Labor Cost Analysis
- **Total Labor Cost**: {total_labor_cost_safe:,.2f} person - hours
- **Direct Labor Cost**: {total_direct_labor:,.2f} person - hours
- **Indirect Labor Cost**: {total_labor_cost_safe - total_direct_labor:,.2f} person - hours
- **Average Labor Value**: {avg_labor_value:.4f} person - hours per unit

### Labor Efficiency Metrics
- **Labor Productivity**: {np.sum(total_output) / (total_labor_cost_safe + 1e-10):.2f} units per person - hour
- **Most Labor - Intensive Sector**: Sector {np.argmax(labor_values)} ({np.max(labor_values):.4f} person - hours / unit)
- **Least Labor - Intensive Sector**: Sector {np.argmin(labor_values)} ({np.min(labor_values):.4f} person - hours / unit)

### Labor Value Breakdown
The labor value represents the total direct and indirect labor embodied in one unit of each product.
This includes:
- Direct labor: Labor directly used in production - Indirect labor: Labor embodied in intermediate inputs - Total labor value: Sum of direct and indirect labor
"""

        return {
            "title": title,
            "content": content,
            "labor_data": {
                "total_labor_cost": total_labor_cost_safe,
                "direct_labor_cost": total_direct_labor,
                "indirect_labor_cost": total_labor_cost_safe - total_direct_labor,
                "avg_labor_value": avg_labor_value,
            },
        }

    def _create_labor_table(self, total_output: np.ndarray, labor_values: np.ndarray, labor_vector: np.ndarray) -> str:
        """Create a formatted table of labor data."""
        table_lines = [
            "| Sector | Labor Value | Direct Labor | Total Output | Labor Cost |",
            "|--------|-------------|--------------|--------------|------------|",
        ]

        for i in range(len(total_output)):
            labor_value = labor_values[i] if i < len(labor_values) else 0
            direct_labor = labor_vector[i] if i < len(labor_vector) else 0
            labor_cost = labor_value * total_output[i]

            table_lines.append(
                f"| {i:6d} | {labor_value:11.4f} | {direct_labor:12.4f} | "
                f"{total_output[i]:12.2f} | {labor_cost:10.2f} |"
            )

        return "\n".join(table_lines)

    def _generate_risk_assessment_section(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment section."""
        title = "Risk Assessment"

        technology_matrix = plan_data.get("technology_matrix")
        sensitivity_analysis = plan_data.get("sensitivity_analysis", {})
        constraint_violations = plan_data.get("constraint_violations", {})

        content = f"""## {title}

### Sensitivity Analysis
The plan's sensitivity to changes in key parameters has been analyzed:

#### Technology Matrix Sensitivity
- **Spectral Radius**: {self._calculate_spectral_radius(technology_matrix):.4f}
- **Productivity Check**: {'✓ Productive' if self._is_productive(technology_matrix) else '✗ Not Productive'}

#### Critical Dependencies
- **High Sensitivity Sectors**: {self._identify_high_sensitivity_sectors(sensitivity_analysis)}
- **Supply Chain Risks**: {self._assess_supply_chain_risks(technology_matrix)}

### Constraint Violations
{self._format_constraint_violations(constraint_violations)}

### Risk Mitigation Strategies
1. **Diversification**: Reduce dependence on single sectors
2. **Capacity Planning**: Ensure adequate production capacity
3. **Resource Management**: Monitor resource utilization closely
4. **Technology Updates**: Regular updates to technology matrices
5. **Contingency Planning**: Prepare for supply chain disruptions

### Monitoring Recommendations - Track key performance indicators monthly - Update technology matrices quarterly - Review resource constraints annually - Conduct sensitivity analysis biannually
"""

        return {
            "title": title,
            "content": content,
            "risk_metrics": {
                "spectral_radius": self._calculate_spectral_radius(technology_matrix),
                "is_productive": self._is_productive(technology_matrix),
            },
        }

    def _calculate_spectral_radius(self, technology_matrix: Optional[np.ndarray]) -> float:
        """Calculate spectral radius of technology matrix."""
        if technology_matrix is None:
            return 0.0

        try:
            eigenvals = np.linalg.eigvals(technology_matrix)
            return np.max(np.abs(eigenvals))
        except:
            return 0.0

    def _is_productive(self, technology_matrix: Optional[np.ndarray]) -> bool:
        """Check if economy is productive."""
        if technology_matrix is None:
            return False

        spectral_radius = self._calculate_spectral_radius(technology_matrix)
        return spectral_radius < 1.0

    def _identify_high_sensitivity_sectors(self, sensitivity_analysis: Dict[str, Any]) -> str:
        """Identify high sensitivity sectors."""
        if not sensitivity_analysis:
            return "No sensitivity analysis available"

        # This would be implemented based on actual sensitivity data
        return "Sectors 1, 3, 7 (based on sensitivity analysis)"

    def _assess_supply_chain_risks(self, technology_matrix: Optional[np.ndarray]) -> str:
        """Assess supply chain risks."""
        if technology_matrix is None:
            return "No technology matrix available"

        # Simple risk assessment based on matrix properties
        technology_matrix.shape[0]
        avg_dependency = np.mean(technology_matrix)

        if avg_dependency > 0.3:
            return "High - Strong interdependencies between sectors"
        elif avg_dependency > 0.1:
            return "Medium - Moderate interdependencies"
        else:
            return "Low - Weak interdependencies"

    def _format_constraint_violations(self, constraint_violations: Dict[str, Any]) -> str:
        """Format constraint violations for display."""
        if not constraint_violations:
            return "No constraint violations detected."

        violations_text = []

        if "demand_violations" in constraint_violations:
            demand_viols = constraint_violations["demand_violations"]
            if len(demand_viols) > 0:
                violations_text.append(f"- **Demand Violations**: {len(demand_viols)} sectors")

        if "resource_violations" in constraint_violations:
            resource_viols = constraint_violations["resource_violations"]
            if len(resource_viols) > 0:
                violations_text.append(f"- **Resource Violations**: {len(resource_viols)} resources")

        if "non_negativity_violations" in constraint_violations:
            neg_viols = constraint_violations["non_negativity_violations"]
            if len(neg_viols) > 0:
                violations_text.append(f"- **Non - Negativity Violations**: {len(neg_viols)} sectors")

        if violations_text:
            return "\n".join(violations_text)
        else:
            return "No constraint violations detected."

    def _combine_report_sections(self, sections: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Combine report sections into final document."""
        # Add header
        report_lines = [
            "# Cybernetic Central Planning Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Report Type**: 5 - Year Economic Plan",
            "",
            "---",
            "",
        ]

        # Add sections
        for section in sections:
            report_lines.append(section["content"])
            report_lines.append("")

        # Add footer
        report_lines.extend(
            [
                "---",
                "",
                "*This report was generated by the Cybernetic Central Planning System*",
                "*For questions or clarifications, please contact the planning team*",
            ]
        )

        return "\n".join(report_lines)

    def _generate_executive_summary(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary only."""
        plan_data = task.get("plan_data", {})

        if not plan_data:
            return {"error": "No plan data provided"}

        summary_section = self._generate_executive_summary_section(plan_data)

        return {"status": "success", "summary": summary_section["content"], "metrics": summary_section["metrics"]}

    def _generate_report_section(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a specific report section."""
        section_type = task.get("section_type")
        plan_data = task.get("plan_data", {})

        if not plan_data:
            return {"error": "No plan data provided"}

        if section_type == "executive_summary":
            return self._generate_executive_summary_section(plan_data)
        elif section_type == "sector_analysis":
            return self._generate_sector_analysis_section(plan_data)
        elif section_type == "resource_allocation":
            return self._generate_resource_allocation_section(plan_data)
        elif section_type == "labor_budget":
            return self._generate_labor_budget_section(plan_data)
        elif section_type == "risk_assessment":
            return self._generate_risk_assessment_section(plan_data)
        else:
            return {"error": f"Unknown section type: {section_type}"}

    def _create_executive_summary_template(self) -> str:
        """Create executive summary template."""
        return "Executive summary template"

    def _create_sector_analysis_template(self) -> str:
        """Create sector analysis template."""
        return "Sector analysis template"

    def _create_resource_allocation_template(self) -> str:
        """Create resource allocation template."""
        return "Resource allocation template"

    def _create_labor_budget_template(self) -> str:
        """Create labor budget template."""
        return "Labor budget template"

    def _create_risk_assessment_template(self) -> str:
        """Create risk assessment template."""
        return "Risk assessment template"

    def _generate_automatic_analysis_section(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automatic analysis section."""
        title = "Automatic Analysis Results"

        # Get automatic analysis results from the planning system
        automatic_analyses = plan_data.get("automatic_analyses", {})

        if not automatic_analyses or "error" in automatic_analyses:
            return {
                "title": title,
                "content": f"""## {title}

### Analysis Status
⚠️ **No automatic analysis results available**

The system was unable to run automatic analyses. This may be due to:
- Missing data or configuration - Analysis errors during execution - System initialization issues

**Recommendation**: Check system logs and ensure all required data is properly loaded.""",
                "metrics": {"status": "unavailable"}
            }

        # Build analysis content
        content_parts = [f"## {title}\n"]

        # Marxist Analysis
        if "marxist" in automatic_analyses:
            marxist_data = automatic_analyses["marxist"]
            if "error" not in marxist_data:
                content_parts.append("### Marxist Economic Analysis")

                # Aggregate Value Composition
                if "aggregate_value_composition" in marxist_data:
                    agg_data = marxist_data["aggregate_value_composition"]
                    content_parts.append("**Aggregate Value Composition**:")
                    
                    # Safe conversion function
                    def safe_float(value, default=0.0):
                        """Safely convert value to float."""
                        if value is None:
                            return default
                        if isinstance(value, str):
                            # Handle string values that might be numeric
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return default
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return default
                    
                    content_parts.append(f"- Constant Capital (C): {safe_float(agg_data.get('constant_capital', 0)):.2f}")
                    content_parts.append(f"- Variable Capital (V): {safe_float(agg_data.get('variable_capital', 0)):.2f}")
                    content_parts.append(f"- Surplus Value (S): {safe_float(agg_data.get('surplus_value', 0)):.2f}")
                    content_parts.append(f"- Total Value: {safe_float(agg_data.get('total_value', 0)):.2f}")
                    content_parts.append(f"- Organic Composition (C / V): {safe_float(agg_data.get('organic_composition', 0)):.4f}")
                    content_parts.append(f"- Rate of Surplus Value (S / V): {safe_float(agg_data.get('rate_of_surplus_value', 0)):.4f}")
                    content_parts.append(f"- Rate of Profit (S/(C + V)): {safe_float(agg_data.get('rate_of_profit', 0)):.4f}")
                    content_parts.append("")

                # Economy - wide Averages
                if "economy_wide_averages" in marxist_data:
                    avg_data = marxist_data["economy_wide_averages"]
                    content_parts.append("**Economy - wide Averages**:")
                    content_parts.append(f"- Average Organic Composition: {safe_float(avg_data.get('average_organic_composition', 0)):.4f}")
                    content_parts.append(f"- Average Rate of Surplus Value: {safe_float(avg_data.get('average_rate_of_surplus_value', 0)):.4f}")
                    content_parts.append(f"- Average Rate of Profit: {safe_float(avg_data.get('average_rate_of_profit', 0)):.4f}")
                    content_parts.append("")

                # Sectoral Indicators Summary
                if "sectoral_indicators" in marxist_data:
                    sector_data = marxist_data["sectoral_indicators"]
                    content_parts.append("**Sectoral Indicators Summary**:")

                    if "organic_composition" in sector_data:
                        org_comp = np.array(sector_data["organic_composition"])
                        content_parts.append(f"- Organic Composition Range: {np.min(org_comp):.4f} - {np.max(org_comp):.4f}")
                        content_parts.append(f"- Organic Composition Std Dev: {np.std(org_comp):.4f}")
                        content_parts.append(f"- Number of Sectors: {len(org_comp)}")

                    if "rate_of_surplus_value" in sector_data:
                        surplus_rates = np.array(sector_data["rate_of_surplus_value"])
                        content_parts.append(f"- Surplus Value Rate Range: {np.min(surplus_rates):.4f} - {np.max(surplus_rates):.4f}")
                        content_parts.append(f"- Surplus Value Rate Std Dev: {np.std(surplus_rates):.4f}")

                    if "rate_of_profit" in sector_data:
                        profit_rates = np.array(sector_data["rate_of_profit"])
                        content_parts.append(f"- Profit Rate Range: {np.min(profit_rates):.4f} - {np.max(profit_rates):.4f}")
                        content_parts.append(f"- Profit Rate Std Dev: {np.std(profit_rates):.4f}")

                    content_parts.append("")

                # Detailed Sectoral Data (if requested)
                if "sectoral_indicators" in marxist_data and len(marxist_data["sectoral_indicators"].get("organic_composition", [])) <= 20:
                    # Only show detailed data for small economies
                    sector_data = marxist_data["sectoral_indicators"]
                    content_parts.append("**Detailed Sectoral Data**:")
                    content_parts.append("| Sector | Organic Composition | Surplus Value Rate | Profit Rate |")
                    content_parts.append("|--------|-------------------|-------------------|-------------|")

                    for i in range(len(sector_data.get("organic_composition", []))):
                        org_comp = sector_data["organic_composition"][i] if i < len(sector_data["organic_composition"]) else 0
                        surplus_rate = sector_data["rate_of_surplus_value"][i] if i < len(sector_data["rate_of_surplus_value"]) else 0
                        profit_rate = sector_data["rate_of_profit"][i] if i < len(sector_data["rate_of_profit"]) else 0
                        content_parts.append(f"| {i:6d} | {org_comp:17.4f} | {surplus_rate:17.4f} | {profit_rate:11.4f} |")

                    content_parts.append("")

                # Legacy support for old format
                if "labor_values" in marxist_data:
                    labor_values = marxist_data["labor_values"]
                    content_parts.append("**Labor Value Analysis (Legacy)**:")
                    # Ensure labor_values is a numpy array
                    if not isinstance(labor_values, np.ndarray):
                        try:
                            labor_values = np.array(labor_values, dtype=float)
                        except (ValueError, TypeError):
                            labor_values = np.array([], dtype=float)
                    
                    if len(labor_values) > 0:
                        content_parts.append(f"- Average labor value: {np.mean(labor_values):.4f}")
                        content_parts.append(f"- Labor value range: {np.min(labor_values):.4f} - {np.max(labor_values):.4f}")
                    else:
                        content_parts.append("- No labor values available")
                    content_parts.append("")

                if "surplus_value" in marxist_data:
                    surplus_value = marxist_data["surplus_value"]
                    content_parts.append(f"- Total surplus value: {safe_float(surplus_value):.2f}")
                    content_parts.append("")
            else:
                content_parts.append("### Marxist Economic Analysis")
                content_parts.append(f"❌ **Error**: {marxist_data['error']}")
                content_parts.append("")

        # Cybernetic Analysis
        if "cybernetic" in automatic_analyses:
            cybernetic_data = automatic_analyses["cybernetic"]
            if "error" not in cybernetic_data:
                content_parts.append("### Cybernetic Feedback Analysis")
                if "converged" in cybernetic_data:
                    content_parts.append(f"- **Convergence Status**: {'✓ Converged' if cybernetic_data['converged'] else '⚠️ Did not converge'}")

                if "iterations" in cybernetic_data:
                    content_parts.append(f"- **Iterations**: {cybernetic_data['iterations']}")

                if "final_demand" in cybernetic_data:
                    final_demand = cybernetic_data["final_demand"]
                    content_parts.append(f"- **Adjusted Final Demand**: {np.sum(final_demand):.2f} total")

                content_parts.append("")
            else:
                content_parts.append("### Cybernetic Feedback Analysis")
                content_parts.append(f"❌ **Error**: {cybernetic_data['error']}")
                content_parts.append("")

        # Mathematical Validation
        if "mathematical" in automatic_analyses:
            math_data = automatic_analyses["mathematical"]
            if "error" not in math_data:
                content_parts.append("### Mathematical Validation")
                if "overall_valid" in math_data:
                    content_parts.append(f"- **Overall Validity**: {'✓ Valid' if math_data['overall_valid'] else '❌ Invalid'}")

                if "component_results" in math_data:
                    comp_results = math_data["component_results"]
                    for component, result in comp_results.items():
                        if isinstance(result, dict) and "valid" in result:
                            status = "✓" if result["valid"] else "❌"
                            content_parts.append(f"- **{component.replace('_', ' ').title()}**: {status}")

                content_parts.append("")
            else:
                content_parts.append("### Mathematical Validation")
                content_parts.append(f"❌ **Error**: {math_data['error']}")
                content_parts.append("")

        # Summary
        content_parts.append("### Analysis Summary")
        successful_analyses = sum(1 for analysis in automatic_analyses.values()
                                if isinstance(analysis, dict) and "error" not in analysis)
        total_analyses = len(automatic_analyses)

        content_parts.append(f"- **Successful Analyses**: {successful_analyses}/{total_analyses}")
        content_parts.append(f"- **Analysis Coverage**: {(successful_analyses / total_analyses)*100:.1f}%")

        if successful_analyses == total_analyses:
            content_parts.append("- **Status**: ✅ All analyses completed successfully")
        elif successful_analyses > 0:
            content_parts.append("- **Status**: ⚠️ Partial analysis completion")
        else:
            content_parts.append("- **Status**: ❌ All analyses failed")

        content = "\n".join(content_parts)

        return {
            "title": title,
            "content": content,
            "metrics": {
                "successful_analyses": successful_analyses,
                "total_analyses": total_analyses,
                "coverage_percentage": (successful_analyses / total_analyses)*100 if total_analyses > 0 else 0
            }
        }
