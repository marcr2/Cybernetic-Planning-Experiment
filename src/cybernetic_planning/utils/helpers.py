"""
Helper utilities for the cybernetic planning system.

Provides formatting, data manipulation, and other utility functions.
"""

from typing import Dict, Any, Union

def format_number(value: Union[float, int], decimals: int = 2) -> str:
    """
    Format a number with appropriate thousands separators and decimal places.

    Args:
        value: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    if isinstance(value, (int, float)):
        if abs(value) >= 1e6:
            return f"{value / 1e6:.{decimals}f}M"
        elif abs(value) >= 1e3:
            return f"{value / 1e3:.{decimals}f}K"
        else:
            return f"{value:.{decimals}f}"
    return str(value)

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as a percentage.

    Args:
        value: Value to format (0 - 1 range)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"

def create_summary_table(data: Dict[str, Any], title: str = "Summary Table") -> str:
    """
    Create a formatted summary table from data.

    Args:
        data: Dictionary containing summary data
        title: Table title

    Returns:
        Formatted table string
    """
    table_lines = [f"## {title}", ""]

    for key, value in data.items():
        if isinstance(value, (int, float)):
            formatted_value = format_number(value)
        else:
            formatted_value = str(value)

        table_lines.append(f"- **{key.replace('_', ' ').title()}**: {formatted_value}")

    return "\n".join(table_lines)

def calculate_economic_indicators(plan_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate key economic indicators from plan data.

    Args:
        plan_data: Economic plan data

    Returns:
        Dictionary of economic indicators
    """
    total_output = plan_data["total_output"]
    total_labor_cost = plan_data["total_labor_cost"]
    final_demand = plan_data["final_demand"]

    indicators = {
        "total_economic_output": np.sum(total_output),
        "total_labor_cost": total_labor_cost,
        "labor_efficiency": np.sum(total_output) / (total_labor_cost + 1e - 10),
        "demand_fulfillment_rate": np.sum(final_demand) / (np.sum(total_output) + 1e - 10),
        "output_inequality": np.std(total_output) / (np.mean(total_output) + 1e - 10),
        "average_sector_output": np.mean(total_output),
        "max_sector_output": np.max(total_output),
        "min_sector_output": np.min(total_output),
    }

    return indicators

def validate_plan_consistency(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate consistency of plan data.

    Args:
        plan_data: Economic plan data

    Returns:
        Validation results
    """
    results = {"consistent": True, "warnings": [], "errors": []}

    # Check for required fields
    required_fields = ["total_output", "final_demand", "labor_values", "total_labor_cost"]
    for field in required_fields:
        if field not in plan_data:
            results["errors"].append(f"Missing required field: {field}")
            results["consistent"] = False

    if not results["consistent"]:
        return results

    # Check dimension consistency
    total_output = plan_data["total_output"]
    final_demand = plan_data["final_demand"]
    labor_values = plan_data["labor_values"]

    if len(final_demand) != len(total_output):
        results["errors"].append("Final demand length doesn't match total output length")
        results["consistent"] = False

    if len(labor_values) != len(total_output):
        results["errors"].append("Labor values length doesn't match total output length")
        results["consistent"] = False

    # Check for negative values
    if np.any(total_output < 0):
        results["warnings"].append("Total output contains negative values")

    if np.any(final_demand < 0):
        results["warnings"].append("Final demand contains negative values")

    if np.any(labor_values < 0):
        results["warnings"].append("Labor values contain negative values")

    # Check labor cost calculation
    calculated_labor_cost = np.sum(labor_values * total_output)
    actual_labor_cost = plan_data["total_labor_cost"]

    if not np.isclose(calculated_labor_cost, actual_labor_cost, rtol = 1e - 6):
        results["warnings"].append(
            f"Labor cost mismatch: calculated {calculated_labor_cost:.2f}, " f"stored {actual_labor_cost:.2f}"
        )

    return results
