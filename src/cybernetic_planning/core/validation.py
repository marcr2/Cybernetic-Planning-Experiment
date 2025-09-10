"""
Economic Planning Validation System

Validates economic plans according to cybernetic planning principles
and Paul Cockshott's labor theory of value.
"""

from typing import Dict, Any, List, Tuple

class EconomicPlanValidator:
    """
    Validates economic plans for theoretical and practical viability.

    Ensures plans conform to:
    - Paul Cockshott's labor theory of value - Cybernetic planning principles - Economic viability constraints
    """

    def __init__(self):
        """Initialize the validator."""
        self.validation_results = {}
        self.errors = []
        self.warnings = []

    def validate_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a complete economic plan.

        Args:
            plan_data: Dictionary containing plan data

        Returns:
            Validation results with errors, warnings, and recommendations
        """
        self.errors = []
        self.warnings = []
        self.validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "theoretical_compliance": {},
            "economic_viability": {}
        }

        # Extract data
        technology_matrix = plan_data.get("technology_matrix")
        final_demand = plan_data.get("final_demand")
        labor_vector = plan_data.get("labor_vector")
        total_output = plan_data.get("total_output")
        labor_values = plan_data.get("labor_values")

        # Validate each component
        self._validate_technology_matrix(technology_matrix)
        self._validate_final_demand(final_demand)
        self._validate_labor_vector(labor_vector)
        self._validate_total_output(total_output)
        self._validate_labor_values(labor_values)

        # Validate economic relationships
        if all(x is not None for x in [technology_matrix, final_demand, total_output]):
            self._validate_demand_fulfillment(technology_matrix, final_demand, total_output)

        if all(x is not None for x in [labor_vector, total_output]):
            self._validate_labor_calculation(labor_vector, total_output, plan_data.get("total_labor_cost"))

        # Check theoretical compliance
        self._check_theoretical_compliance(plan_data)

        # Compile results
        self.validation_results["errors"] = self.errors
        self.validation_results["warnings"] = self.warnings
        self.validation_results["is_valid"] = len(self.errors) == 0

        return self.validation_results

    def _validate_technology_matrix(self, A: np.ndarray) -> None:
        """Validate technology matrix."""
        if A is None:
            self.errors.append("Technology matrix is missing")
            return

        # Check dimensions
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            self.errors.append("Technology matrix must be square")
            return

        # Check for negative values
        if np.any(A < 0):
            self.errors.append("Technology matrix contains negative values - economically impossible")

        # Check for values >= 1 (input coefficients should be < 1)
        if np.any(A >= 1):
            self.warnings.append("Some input coefficients are >= 1 - may indicate data quality issues")

        # Check productivity (spectral radius < 1)
        eigenvals = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigenvals))

        if spectral_radius >= 1:
            self.errors.append(f"Economy is not productive (spectral radius = {spectral_radius:.4f} >= 1)")
        else:
            self.validation_results["economic_viability"]["spectral_radius"] = spectral_radius
            self.validation_results["economic_viability"]["is_productive"] = True

    def _validate_final_demand(self, d: np.ndarray) -> None:
        """Validate final demand vector."""
        if d is None:
            self.errors.append("Final demand vector is missing")
            return

        # Check dimensions
        if d.ndim != 1:
            self.errors.append("Final demand must be a vector")

        # Check for negative values
        if np.any(d < 0):
            self.errors.append("Final demand contains negative values - economically impossible")

        # Check for zero values
        if np.any(d == 0):
            self.warnings.append("Some sectors have zero final demand - may indicate unused sectors")

        # Check total demand
        total_demand = np.sum(d)
        if total_demand <= 0:
            self.errors.append("Total final demand must be positive")

    def _validate_labor_vector(self, l: np.ndarray) -> None:
        """Validate labor input vector."""
        if l is None:
            self.errors.append("Labor input vector is missing")
            return

        # Check dimensions
        if l.ndim != 1:
            self.errors.append("Labor input must be a vector")

        # Check for negative values
        if np.any(l < 0):
            self.errors.append("Labor input contains negative values - economically impossible")

        # Check for zero values
        if np.any(l == 0):
            self.warnings.append("Some sectors have zero labor input - may indicate automated sectors")

    def _validate_total_output(self, x: np.ndarray) -> None:
        """Validate total output vector."""
        if x is None:
            self.errors.append("Total output vector is missing")
            return

        # Check dimensions
        if x.ndim != 1:
            self.errors.append("Total output must be a vector")

        # Check for negative values (critical error)
        if np.any(x < 0):
            self.errors.append("Total output contains negative values - economically impossible")
            return

        # Check for zero values
        if np.any(x == 0):
            self.warnings.append("Some sectors have zero output - may indicate unused sectors")

        # Check total output
        total_output = np.sum(x)
        if total_output <= 0:
            self.errors.append("Total output must be positive")

    def _validate_labor_values(self, v: np.ndarray) -> None:
        """Validate labor values vector."""
        if v is None:
            self.warnings.append("Labor values vector is missing")
            return

        # Check dimensions
        if v.ndim != 1:
            self.errors.append("Labor values must be a vector")

        # Check for negative values (violates labor theory of value)
        if np.any(v < 0):
            self.errors.append("Labor values contain negative values - violates labor theory of value")

        # Check for zero values
        if np.any(v == 0):
            self.warnings.append("Some sectors have zero labor values - may indicate free goods")

    def _validate_demand_fulfillment(self, A: np.ndarray, d: np.ndarray, x: np.ndarray) -> None:
        """Validate that demand is exactly fulfilled."""
        if A is None or d is None or x is None:
            return

        # Calculate net output: (I - A)x
        I = np.eye(A.shape[0])
        net_output = (I - A) @ x

        # Check if demand is exactly fulfilled
        demand_error = np.abs(net_output - d)
        max_error = np.max(demand_error)
        relative_error = max_error / (np.max(d) + 1e - 10)

        if relative_error > 1e - 3:  # Allow for reasonable numerical errors
            self.errors.append(f"Demand not exactly fulfilled (max error: {max_error:.2e}, relative: {relative_error:.2e})")

        self.validation_results["economic_viability"]["demand_fulfillment_error"] = relative_error

    def _validate_labor_calculation(self, l: np.ndarray, x: np.ndarray, total_labor_cost: float) -> None:
        """Validate labor cost calculation."""
        if l is None or x is None:
            return

        # Calculate expected total labor cost
        expected_labor_cost = np.dot(l, x)

        if total_labor_cost is not None:
            # Check if calculated labor cost matches expected
            error = abs(total_labor_cost - expected_labor_cost)
            relative_error = error / (expected_labor_cost + 1e - 10)

            if relative_error > 1e - 6:
                self.warnings.append(f"Labor cost calculation mismatch (error: {error:.2e})")

        self.validation_results["economic_viability"]["total_labor_cost"] = expected_labor_cost

    def _check_theoretical_compliance(self, plan_data: Dict[str, Any]) -> None:
        """Check compliance with cybernetic planning theory."""
        compliance = {}

        # Check Cockshott's labor theory of value
        labor_values = plan_data.get("labor_values")
        if labor_values is not None and np.all(labor_values > 0):
            compliance["labor_theory_of_value"] = "compliant"
        else:
            compliance["labor_theory_of_value"] = "non_compliant"

        # Check input - output analysis validity
        technology_matrix = plan_data.get("technology_matrix")
        if technology_matrix is not None:
            eigenvals = np.linalg.eigvals(technology_matrix)
            spectral_radius = np.max(np.abs(eigenvals))
            if spectral_radius < 1:
                compliance["input_output_analysis"] = "compliant"
            else:
                compliance["input_output_analysis"] = "non_compliant"

        # Check optimization objective (minimize labor)
        total_labor_cost = plan_data.get("total_labor_cost")
        if total_labor_cost is not None and total_labor_cost > 0:
            compliance["labor_minimization"] = "compliant"
        else:
            compliance["labor_minimization"] = "non_compliant"

        self.validation_results["theoretical_compliance"] = compliance

    def get_recommendations(self) -> List[str]:
        """Get recommendations for improving the plan."""
        recommendations = []

        if not self.validation_results.get("is_valid", False):
            recommendations.append("Fix all validation errors before proceeding with plan implementation")

        if self.validation_results.get("economic_viability", {}).get("spectral_radius", 0) > 0.9:
            recommendations.append("Consider reducing input coefficients to improve economic stability")

        if any("negative" in error.lower() for error in self.errors):
            recommendations.append("Review data generation algorithms to ensure positive values")

        if not self.validation_results.get("theoretical_compliance", {}).get("labor_theory_of_value") == "compliant":
            recommendations.append("Ensure labor values are calculated according to Cockshott's theory")

        return recommendations
