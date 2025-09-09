"""Utility functions and helpers for the cybernetic planning system."""

# Import helpers (these don't require external dependencies)

# Import visualization functions only if seaborn is available
try:
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    # Create a dummy function for when visualization is not available
    def create_plan_visualizations(*args, **kwargs):
        raise ImportError("Visualization functions require seaborn and matplotlib. Please install dependencies first.")

__all__ = [
    "format_number",
    "format_percentage",
    "create_summary_table",
]

if VISUALIZATION_AVAILABLE:
    __all__.append("create_plan_visualizations")
