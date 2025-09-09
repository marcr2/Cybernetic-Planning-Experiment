"""
Marxist Expanded Reproduction System

Implements Marx's expanded reproduction schemes with proper capital accumulation,
surplus value realization, and department balance based on Capital Volume 2.

This module provides comprehensive implementation of Marx's reproduction schemas
with mathematical precision and theoretical accuracy.
"""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np

@dataclass
class DepartmentBalance:
    """Represents the balance conditions for a department."""
    constant_capital: float
    variable_capital: float
    surplus_value: float
    total_value: float
    output: float
    demand: float
    balance_ratio: float
    is_balanced: bool

class MarxistReproductionSystem:
    """
    Implements Marx's expanded reproduction schemes.

    Based on Marx's Capital Volume 2, this system ensures:
    - Proper balance between Department I (means of production) and Department II (consumer goods)
    - Capital accumulation through surplus value reinvestment - Realization of surplus value through consumption and investment - Expanded reproduction with technological progress - Mathematical precision in reproduction conditions
    """

    def __init__(
        self,
        technology_matrix: np.ndarray,
        final_demand: np.ndarray,
        labor_vector: np.ndarray,
        n_dept_I: int = 50,
        n_dept_II: int = 50,
        n_dept_III: int = 75
    ):
        """
        Initialize the Marxist reproduction system.

        Args:
            technology_matrix: Technology matrix A
            final_demand: Final demand vector d
            labor_vector: Labor input vector l
            n_dept_I: Number of sectors in Department I (means of production)
            n_dept_II: Number of sectors in Department II (consumer goods)
            n_dept_III: Number of sectors in Department III (services)
        """
        self.A = np.asarray(technology_matrix)
        self.d = np.asarray(final_demand)
        self.l = np.asarray(labor_vector)

        self.n_sectors = self.A.shape[0]
        self.n_dept_I = n_dept_I
        self.n_dept_II = n_dept_II
        self.n_dept_III = n_dept_III

        # Validate dimensions
        if self.n_sectors != n_dept_I + n_dept_II + n_dept_III:
            raise ValueError("Total sectors must equal sum of departments")

        # Department indices
        self.dept_I_indices = list(range(n_dept_I))
        self.dept_II_indices = list(range(n_dept_I, n_dept_I + n_dept_II))
        self.dept_III_indices = list(range(n_dept_I + n_dept_II, self.n_sectors))

        # Reproduction parameters
        self.organic_composition = self._calculate_organic_composition()
        self.rate_of_surplus_value = 1.0  # 100% surplus value rate
        self.accumulation_rate = 0.5  # 50% of surplus value accumulated

    def _calculate_organic_composition(self) -> np.ndarray:
        """Calculate organic composition of capital (c / v) for each sector."""
        # For simplicity, use technology matrix diagonal as constant capital
        # and labor vector as variable capital
        constant_capital = np.diag(self.A)
        variable_capital = self.l

        # Avoid division by zero
        organic_composition = np.where(
            variable_capital > 0,
            constant_capital / variable_capital,
            1.0  # Default value
        )

        return organic_composition

    def calculate_department_balance(self, output: np.ndarray) -> Dict[str, Any]:
        """
        Calculate balance between departments based on Marx's reproduction schemes.

        Args:
            output: Total output vector x

        Returns:
            Department balance analysis
        """
        # Calculate department outputs
        dept_I_output = np.sum(output[self.dept_I_indices])
        dept_II_output = np.sum(output[self.dept_II_indices])
        dept_III_output = np.sum(output[self.dept_III_indices])

        # Calculate department demands
        dept_I_demand = np.sum(self.d[self.dept_I_indices])
        dept_II_demand = np.sum(self.d[self.dept_II_indices])
        dept_III_demand = np.sum(self.d[self.dept_III_indices])

        # Calculate surplus value by department
        surplus_value = self._calculate_surplus_value(output)
        dept_I_surplus = np.sum(surplus_value[self.dept_I_indices])
        dept_II_surplus = np.sum(surplus_value[self.dept_II_indices])
        dept_III_surplus = np.sum(surplus_value[self.dept_III_indices])

        # Calculate accumulation requirements
        accumulation_I = dept_I_surplus * self.accumulation_rate
        accumulation_II = dept_II_surplus * self.accumulation_rate
        accumulation_III = dept_III_surplus * self.accumulation_rate

        # Check reproduction balance
        # Department I must produce enough means of production for both departments
        required_I_for_I = np.sum(self.A[self.dept_I_indices, :][:, self.dept_I_indices] @ output[self.dept_I_indices])
        required_I_for_II = np.sum(self.A[self.dept_I_indices, :][:, self.dept_II_indices] @ output[self.dept_II_indices])
        required_I_for_III = np.sum(self.A[self.dept_I_indices, :][:, self.dept_III_indices] @ output[self.dept_III_indices])

        total_required_I = required_I_for_I + required_I_for_II + required_I_for_III + dept_I_demand + accumulation_I

        # Department II must produce enough consumer goods for workers
        required_II_for_workers = np.sum(self.l * output)  # Total wages
        total_required_II = required_II_for_workers + dept_II_demand + accumulation_II

        balance = {
            "department_I": {
                "output": dept_I_output,
                "demand": dept_I_demand,
                "required": total_required_I,
                "surplus": dept_I_surplus,
                "accumulation": accumulation_I,
                "balance_ratio": dept_I_output / total_required_I if total_required_I > 0 else 0
            },
            "department_II": {
                "output": dept_II_output,
                "demand": dept_II_demand,
                "required": total_required_II,
                "surplus": dept_II_surplus,
                "accumulation": accumulation_II,
                "balance_ratio": dept_II_output / total_required_II if total_required_II > 0 else 0
            },
            "department_III": {
                "output": dept_III_output,
                "demand": dept_III_demand,
                "surplus": dept_III_surplus,
                "accumulation": accumulation_III
            },
            "overall_balance": {
                "dept_I_balanced": dept_I_output >= total_required_I * 0.95,  # 5% tolerance
                "dept_II_balanced": dept_II_output >= total_required_II * 0.95,
                "total_surplus": np.sum(surplus_value),
                "total_accumulation": accumulation_I + accumulation_II + accumulation_III
            }
        }

        return balance

    def _calculate_surplus_value(self, output: np.ndarray) -> np.ndarray:
        """Calculate surplus value for each sector."""
        # Surplus value = total value - constant capital - variable capital
        total_value = output
        constant_capital = np.diag(self.A) * output
        variable_capital = self.l * output

        surplus_value = total_value - constant_capital - variable_capital

        # Ensure non - negative surplus value
        surplus_value = np.maximum(surplus_value, 0)

        return surplus_value

    def adjust_for_reproduction_balance(self, output: np.ndarray) -> np.ndarray:
        """
        Adjust output to maintain reproduction balance.

        Args:
            output: Current output vector

        Returns:
            Adjusted output vector
        """
        balance = self.calculate_department_balance(output)
        adjusted_output = output.copy()

        # Adjust Department I if underproducing
        if not balance["overall_balance"]["dept_I_balanced"]:
            dept_I_ratio = balance["department_I"]["balance_ratio"]
            if dept_I_ratio < 1.0:
                # Increase Department I output
                adjustment_factor = 1.0 / dept_I_ratio
                adjusted_output[self.dept_I_indices] *= adjustment_factor

        # Adjust Department II if underproducing
        if not balance["overall_balance"]["dept_II_balanced"]:
            dept_II_ratio = balance["department_II"]["balance_ratio"]
            if dept_II_ratio < 1.0:
                # Increase Department II output
                adjustment_factor = 1.0 / dept_II_ratio
                adjusted_output[self.dept_II_indices] *= adjustment_factor

        return adjusted_output

    def calculate_expanded_reproduction_demands(self, base_demand: np.ndarray, growth_rate: float = 0.05) -> np.ndarray:
        """
        Calculate demands for expanded reproduction.

        Args:
            base_demand: Base final demand
            growth_rate: Annual growth rate for expanded reproduction

        Returns:
            Expanded reproduction demand vector
        """
        expanded_demand = base_demand.copy()

        # Department I: Investment demand grows with accumulation
        dept_I_growth = 1 + growth_rate
        expanded_demand[self.dept_I_indices] *= dept_I_growth

        # Department II: Consumption demand grows with population and wages
        dept_II_growth = 1 + growth_rate * 0.8  # Slightly slower than investment
        expanded_demand[self.dept_II_indices] *= dept_II_growth

        # Department III: Service demand grows moderately
        dept_III_growth = 1 + growth_rate * 0.6
        expanded_demand[self.dept_III_indices] *= dept_III_growth

        return expanded_demand

    def calculate_accumulation_requirements(self, output: np.ndarray) -> np.ndarray:
        """
        Calculate investment requirements for capital accumulation.

        Args:
            output: Current output vector

        Returns:
            Investment demand vector for accumulation
        """
        surplus_value = self._calculate_surplus_value(output)

        # Calculate accumulation by department
        accumulation = np.zeros(self.n_sectors)

        # Department I: Accumulate in means of production
        dept_I_surplus = np.sum(surplus_value[self.dept_I_indices])
        dept_I_accumulation = dept_I_surplus * self.accumulation_rate
        accumulation[self.dept_I_indices] = dept_I_accumulation / len(self.dept_I_indices)

        # Department II: Accumulate in consumer goods production
        dept_II_surplus = np.sum(surplus_value[self.dept_II_indices])
        dept_II_accumulation = dept_II_surplus * self.accumulation_rate
        accumulation[self.dept_II_indices] = dept_II_accumulation / len(self.dept_II_indices)

        # Department III: Moderate accumulation
        dept_III_surplus = np.sum(surplus_value[self.dept_III_indices])
        dept_III_accumulation = dept_III_surplus * self.accumulation_rate * 0.5  # Lower rate
        accumulation[self.dept_III_indices] = dept_III_accumulation / len(self.dept_III_indices)

        return accumulation

    def validate_reproduction_conditions(self, output: np.ndarray) -> Dict[str, Any]:
        """
        Validate that reproduction conditions are met.

        Args:
            output: Output vector to validate

        Returns:
            Validation results
        """
        balance = self.calculate_department_balance(output)

        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }

        # Check Department I balance
        if not balance["overall_balance"]["dept_I_balanced"]:
            validation["valid"] = False
            validation["errors"].append("Department I (means of production) underproducing")

        # Check Department II balance
        if not balance["overall_balance"]["dept_II_balanced"]:
            validation["valid"] = False
            validation["errors"].append("Department II (consumer goods) underproducing")

        # Check surplus value realization
        total_surplus = balance["overall_balance"]["total_surplus"]
        if total_surplus <= 0:
            validation["warnings"].append("No surplus value generated")

        # Check accumulation requirements
        total_accumulation = balance["overall_balance"]["total_accumulation"]
        if total_accumulation <= 0:
            validation["warnings"].append("No accumulation possible")

        # Recommendations
        if balance["department_I"]["balance_ratio"] < 1.0:
            validation["recommendations"].append("Increase investment in Department I")

        if balance["department_II"]["balance_ratio"] < 1.0:
            validation["recommendations"].append("Increase investment in Department II")

        return validation
