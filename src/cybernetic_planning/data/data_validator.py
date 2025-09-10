"""
Data Validator

Validates economic data for consistency, completeness, and mathematical properties.
Ensures data quality for the cybernetic planning system.
"""

from typing import Dict, Any
import numpy as np

class DataValidator:
    """
    Validator for economic data used in cybernetic planning.

    Provides comprehensive validation of matrices, vectors, and economic relationships
    to ensure data quality and mathematical consistency.
    """

    def __init__(self):
        """Initialize the data validator."""
        self.validation_rules = {}
        self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> None:
        """Initialize default validation rules."""
        self.validation_rules = {
            "technology_matrix": {
                "must_be_square": True,
                "must_be_non_negative": False,  # Allow negative values with warning
                "spectral_radius_max": 0.99,
                "diagonal_elements_max": 0.5,
            },
            "final_demand": {
                "must_be_non_negative": False,  # Allow negative values with warning
                "must_have_positive_sum": True,
            },
            "labor_input": {"must_be_non_negative": True, "must_have_positive_sum": True},
            "resource_matrix": {"must_be_non_negative": True, "must_have_positive_elements": True},
            "max_resources": {"must_be_non_negative": True, "must_have_positive_elements": True},
        }

    def validate_technology_matrix(self, matrix: np.ndarray, strict: bool = False) -> Dict[str, Any]:
        """
        Validate technology matrix for economic planning.

        Args:
            matrix: Technology matrix to validate
            strict: Whether to use strict validation rules

        Returns:
            Validation results dictionary
        """
        results = {"valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Check if matrix is numpy array
        if not isinstance(matrix, np.ndarray):
            results["errors"].append("Technology matrix must be a numpy array")
            results["valid"] = False
            return results

        # Check dimensions
        if matrix.ndim != 2:
            results["errors"].append("Technology matrix must be 2 - dimensional")
            results["valid"] = False
            return results

        if matrix.shape[0] != matrix.shape[1]:
            results["errors"].append("Technology matrix must be square")
            results["valid"] = False
            return results

        n = matrix.shape[0]
        results["metrics"]["size"] = n

        # Check for NaN or infinite values
        if np.any(np.isnan(matrix)):
            results["errors"].append("Technology matrix contains NaN values")
            results["valid"] = False

        if np.any(np.isinf(matrix)):
            results["errors"].append("Technology matrix contains infinite values")
            results["valid"] = False

        # Check for negative values
        negative_count = np.sum(matrix < 0)
        if negative_count > 0:
            if strict or self.validation_rules["technology_matrix"]["must_be_non_negative"]:
                results["errors"].append(f"Technology matrix contains {negative_count} negative values")
                results["valid"] = False
            else:
                results["warnings"].append(f"Technology matrix contains {negative_count} negative values")

        # Check diagonal elements
        diagonal_elements = np.diag(matrix)
        max_diagonal = np.max(diagonal_elements)
        if max_diagonal > self.validation_rules["technology_matrix"]["diagonal_elements_max"]:
            results["warnings"].append(f"Maximum diagonal element ({max_diagonal:.4f}) exceeds recommended threshold")

        # Check spectral radius
        try:
            eigenvals = np.linalg.eigvals(matrix)
            spectral_radius = np.max(np.abs(eigenvals))
            results["metrics"]["spectral_radius"] = spectral_radius

            if spectral_radius >= 1.0:
                results["errors"].append(f"Spectral radius ({spectral_radius:.4f}) >= 1, economy is not productive")
                results["valid"] = False
            elif spectral_radius > self.validation_rules["technology_matrix"]["spectral_radius_max"]:
                results["warnings"].append(
                    f"Spectral radius ({spectral_radius:.4f}) is close to 1, economy may be unstable"
                )
        except np.linalg.LinAlgError:
            results["warnings"].append("Could not calculate spectral radius")

        # Check matrix properties
        results["metrics"]["density"] = np.count_nonzero(matrix) / (n * n)
        results["metrics"]["max_value"] = np.max(matrix)
        results["metrics"]["min_value"] = np.min(matrix)
        results["metrics"]["mean_value"] = np.mean(matrix)
        results["metrics"]["std_value"] = np.std(matrix)

        return results

    def validate_final_demand(self, vector: np.ndarray, strict: bool = False) -> Dict[str, Any]:
        """
        Validate final demand vector.

        Args:
            vector: Final demand vector to validate
            strict: Whether to use strict validation rules

        Returns:
            Validation results dictionary
        """
        results = {"valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Check if vector is numpy array
        if not isinstance(vector, np.ndarray):
            results["errors"].append("Final demand vector must be a numpy array")
            results["valid"] = False
            return results

        # Check dimensions
        if vector.ndim != 1:
            results["errors"].append("Final demand vector must be 1 - dimensional")
            results["valid"] = False
            return results

        n = len(vector)
        results["metrics"]["length"] = n

        # Check for NaN or infinite values
        if np.any(np.isnan(vector)):
            results["errors"].append("Final demand vector contains NaN values")
            results["valid"] = False

        if np.any(np.isinf(vector)):
            results["errors"].append("Final demand vector contains infinite values")
            results["valid"] = False

        # Check for negative values
        negative_count = np.sum(vector < 0)
        if negative_count > 0:
            if strict or self.validation_rules["final_demand"]["must_be_non_negative"]:
                results["errors"].append(f"Final demand vector contains {negative_count} negative values")
                results["valid"] = False
            else:
                results["warnings"].append(f"Final demand vector contains {negative_count} negative values")

        # Check sum
        total_demand = np.sum(vector)
        results["metrics"]["total_demand"] = total_demand

        if total_demand <= 0:
            if self.validation_rules["final_demand"]["must_have_positive_sum"]:
                results["errors"].append("Final demand vector sum must be positive")
                results["valid"] = False
            else:
                results["warnings"].append("Final demand vector sum is not positive")

        # Calculate additional metrics
        results["metrics"]["max_value"] = np.max(vector)
        results["metrics"]["min_value"] = np.min(vector)
        results["metrics"]["mean_value"] = np.mean(vector)
        results["metrics"]["std_value"] = np.std(vector)
        results["metrics"]["zero_count"] = np.sum(vector == 0)

        return results

    def validate_labor_input(self, vector: np.ndarray, strict: bool = False) -> Dict[str, Any]:
        """
        Validate labor input vector.

        Args:
            vector: Labor input vector to validate
            strict: Whether to use strict validation rules

        Returns:
            Validation results dictionary
        """
        results = {"valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Check if vector is numpy array
        if not isinstance(vector, np.ndarray):
            results["errors"].append("Labor input vector must be a numpy array")
            results["valid"] = False
            return results

        # Check dimensions
        if vector.ndim != 1:
            results["errors"].append("Labor input vector must be 1 - dimensional")
            results["valid"] = False
            return results

        n = len(vector)
        results["metrics"]["length"] = n

        # Check for NaN or infinite values
        if np.any(np.isnan(vector)):
            results["errors"].append("Labor input vector contains NaN values")
            results["valid"] = False

        if np.any(np.isinf(vector)):
            results["errors"].append("Labor input vector contains infinite values")
            results["valid"] = False

        # Check for negative values
        negative_count = np.sum(vector < 0)
        if negative_count > 0:
            if strict or self.validation_rules["labor_input"]["must_be_non_negative"]:
                results["errors"].append(f"Labor input vector contains {negative_count} negative values")
                results["valid"] = False
            else:
                results["warnings"].append(f"Labor input vector contains {negative_count} negative values")

        # Check sum
        total_labor = np.sum(vector)
        results["metrics"]["total_labor"] = total_labor

        if total_labor <= 0:
            if self.validation_rules["labor_input"]["must_have_positive_sum"]:
                results["errors"].append("Labor input vector sum must be positive")
                results["valid"] = False
            else:
                results["warnings"].append("Labor input vector sum is not positive")

        # Calculate additional metrics
        results["metrics"]["max_value"] = np.max(vector)
        results["metrics"]["min_value"] = np.min(vector)
        results["metrics"]["mean_value"] = np.mean(vector)
        results["metrics"]["std_value"] = np.std(vector)
        results["metrics"]["zero_count"] = np.sum(vector == 0)

        return results

    def validate_resource_matrix(self, matrix: np.ndarray, strict: bool = False) -> Dict[str, Any]:
        """
        Validate resource constraint matrix.

        Args:
            matrix: Resource matrix to validate
            strict: Whether to use strict validation rules

        Returns:
            Validation results dictionary
        """
        results = {"valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Check if matrix is numpy array
        if not isinstance(matrix, np.ndarray):
            results["errors"].append("Resource matrix must be a numpy array")
            results["valid"] = False
            return results

        # Check dimensions
        if matrix.ndim != 2:
            results["errors"].append("Resource matrix must be 2 - dimensional")
            results["valid"] = False
            return results

        m, n = matrix.shape
        results["metrics"]["shape"] = (m, n)

        # Check for NaN or infinite values
        if np.any(np.isnan(matrix)):
            results["errors"].append("Resource matrix contains NaN values")
            results["valid"] = False

        if np.any(np.isinf(matrix)):
            results["errors"].append("Resource matrix contains infinite values")
            results["valid"] = False

        # Check for negative values
        negative_count = np.sum(matrix < 0)
        if negative_count > 0:
            if strict or self.validation_rules["resource_matrix"]["must_be_non_negative"]:
                results["errors"].append(f"Resource matrix contains {negative_count} negative values")
                results["valid"] = False
            else:
                results["warnings"].append(f"Resource matrix contains {negative_count} negative values")

        # Check for positive elements
        positive_count = np.sum(matrix > 0)
        if positive_count == 0:
            if self.validation_rules["resource_matrix"]["must_have_positive_elements"]:
                results["errors"].append("Resource matrix must have at least one positive element")
                results["valid"] = False
            else:
                results["warnings"].append("Resource matrix has no positive elements")

        # Calculate additional metrics
        results["metrics"]["density"] = positive_count / (m * n)
        results["metrics"]["max_value"] = np.max(matrix)
        results["metrics"]["min_value"] = np.min(matrix)
        results["metrics"]["mean_value"] = np.mean(matrix)
        results["metrics"]["std_value"] = np.std(matrix)

        return results

    def validate_max_resources(self, vector: np.ndarray, strict: bool = False) -> Dict[str, Any]:
        """
        Validate maximum resources vector.

        Args:
            vector: Maximum resources vector to validate
            strict: Whether to use strict validation rules

        Returns:
            Validation results dictionary
        """
        results = {"valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Check if vector is numpy array
        if not isinstance(vector, np.ndarray):
            results["errors"].append("Maximum resources vector must be a numpy array")
            results["valid"] = False
            return results

        # Check dimensions
        if vector.ndim != 1:
            results["errors"].append("Maximum resources vector must be 1 - dimensional")
            results["valid"] = False
            return results

        n = len(vector)
        results["metrics"]["length"] = n

        # Check for NaN or infinite values
        if np.any(np.isnan(vector)):
            results["errors"].append("Maximum resources vector contains NaN values")
            results["valid"] = False

        if np.any(np.isinf(vector)):
            results["errors"].append("Maximum resources vector contains infinite values")
            results["valid"] = False

        # Check for negative values
        negative_count = np.sum(vector < 0)
        if negative_count > 0:
            if strict or self.validation_rules["max_resources"]["must_be_non_negative"]:
                results["errors"].append(f"Maximum resources vector contains {negative_count} negative values")
                results["valid"] = False
            else:
                results["warnings"].append(f"Maximum resources vector contains {negative_count} negative values")

        # Check for positive elements
        positive_count = np.sum(vector > 0)
        if positive_count == 0:
            if self.validation_rules["max_resources"]["must_have_positive_elements"]:
                results["errors"].append("Maximum resources vector must have at least one positive element")
                results["valid"] = False
            else:
                results["warnings"].append("Maximum resources vector has no positive elements")

        # Calculate additional metrics
        results["metrics"]["total_resources"] = np.sum(vector)
        results["metrics"]["max_value"] = np.max(vector)
        results["metrics"]["min_value"] = np.min(vector)
        results["metrics"]["mean_value"] = np.mean(vector)
        results["metrics"]["std_value"] = np.std(vector)
        results["metrics"]["zero_count"] = np.sum(vector == 0)

        return results

    def validate_data_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate consistency between different data components.

        Args:
            data: Dictionary containing all economic data

        Returns:
            Validation results dictionary
        """
        results = {"valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Check if required components are present
        required_components = ["technology_matrix", "final_demand", "labor_input"]
        for component in required_components:
            if component not in data:
                results["errors"].append(f"Missing required component: {component}")
                results["valid"] = False

        if not results["valid"]:
            return results

        # Get dimensions
        tech_matrix = data["technology_matrix"]
        final_demand = data["final_demand"]
        labor_input = data["labor_input"]

        n_sectors = tech_matrix.shape[0]

        # Check dimension consistency
        if len(final_demand) != n_sectors:
            results["errors"].append(
                f"Final demand length ({len(final_demand)}) must match technology matrix size ({n_sectors})"
            )
            results["valid"] = False

        if len(labor_input) != n_sectors:
            results["errors"].append(
                f"Labor input length ({len(labor_input)}) must match technology matrix size ({n_sectors})"
            )
            results["valid"] = False

        # Check resource constraints if present
        if "resource_matrix" in data and "max_resources" in data:
            resource_matrix = data["resource_matrix"]
            max_resources = data["max_resources"]

            if resource_matrix.shape[1] != n_sectors:
                results["errors"].append(
                    f"Resource matrix columns ({resource_matrix.shape[1]}) must match technology matrix size ({n_sectors})"
                )
                results["valid"] = False

            if len(max_resources) != resource_matrix.shape[0]:
                results["errors"].append(
                    f"Max resources length ({len(max_resources)}) must match resource matrix rows ({resource_matrix.shape[0]})"
                )
                results["valid"] = False

        # Check economic feasibility
        if results["valid"]:
            feasibility_check = self._check_economic_feasibility(data)
            results.update(feasibility_check)

        return results

    def _check_economic_feasibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check economic feasibility of the data.

        Args:
            data: Dictionary containing all economic data

        Returns:
            Feasibility check results
        """
        results = {"feasible": True, "warnings": [], "metrics": {}}

        tech_matrix = data["technology_matrix"]
        final_demand = data["final_demand"]

        try:
            # Check if Leontief inverse exists
            I = np.eye(tech_matrix.shape[0])
            leontief_inverse = np.linalg.inv(I - tech_matrix)

            # Check if total output is positive
            total_output = leontief_inverse @ final_demand

            if np.any(total_output < 0):
                results["feasible"] = False
                results["warnings"].append("Some sectors have negative output requirements")

            results["metrics"]["total_output"] = total_output
            results["metrics"]["min_output"] = np.min(total_output)
            results["metrics"]["max_output"] = np.max(total_output)

        except np.linalg.LinAlgError:
            results["feasible"] = False
            results["warnings"].append("Cannot compute Leontief inverse - economy may not be productive")

        return results

    def validate_plan_feasibility(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate feasibility of an economic plan.

        Args:
            plan: Economic plan dictionary

        Returns:
            Feasibility validation results
        """
        results = {"feasible": True, "errors": [], "warnings": [], "metrics": {}}

        # Check if plan has required components
        required_components = ["total_output", "total_labor_cost"]
        for component in required_components:
            if component not in plan:
                results["errors"].append(f"Missing required plan component: {component}")
                results["feasible"] = False

        if not results["feasible"]:
            return results

        total_output = plan["total_output"]
        total_labor_cost = plan["total_labor_cost"]

        # Check output non - negativity
        if np.any(total_output < 0):
            results["feasible"] = False
            results["errors"].append("Plan contains negative output values")

        # Check labor cost positivity
        if total_labor_cost <= 0:
            results["warnings"].append("Total labor cost is not positive")

        # Check resource constraints if present
        if "resource_usage" in plan and "max_resources" in plan:
            resource_usage = plan["resource_usage"]
            max_resources = plan["max_resources"]

            violations = resource_usage - max_resources
            if np.any(violations > 0):
                results["feasible"] = False
                results["errors"].append("Plan violates resource constraints")
                results["warnings"].append(f"Resource violations: {np.sum(violations[violations > 0]):.2f}")

        # Calculate plan metrics
        results["metrics"]["total_output_value"] = np.sum(total_output)
        results["metrics"]["labor_efficiency"] = np.sum(total_output) / (total_labor_cost + 1e-10)
        results["metrics"]["output_inequality"] = np.std(total_output) / (np.mean(total_output) + 1e-10)

        return results

    def set_validation_rule(self, component: str, rule: str, value: Any) -> None:
        """
        Set a validation rule for a component.

        Args:
            component: Component name (e.g., 'technology_matrix')
            rule: Rule name (e.g., 'must_be_non_negative')
            value: Rule value
        """
        if component not in self.validation_rules:
            self.validation_rules[component] = {}

        self.validation_rules[component][rule] = value

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules."""
        return self.validation_rules.copy()

    def validate_all(self, data: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        """
        Validate all components in the data dictionary.

        Args:
            data: Dictionary containing all economic data
            strict: Whether to use strict validation rules

        Returns:
            Comprehensive validation results
        """
        results = {"overall_valid": True, "component_results": {}, "consistency_results": {}, "summary": {}}

        # Validate individual components
        if "technology_matrix" in data:
            results["component_results"]["technology_matrix"] = self.validate_technology_matrix(
                data["technology_matrix"], strict
            )

        if "final_demand" in data:
            results["component_results"]["final_demand"] = self.validate_final_demand(data["final_demand"], strict)

        if "labor_input" in data:
            results["component_results"]["labor_input"] = self.validate_labor_input(data["labor_input"], strict)

        if "resource_matrix" in data:
            results["component_results"]["resource_matrix"] = self.validate_resource_matrix(
                data["resource_matrix"], strict
            )

        if "max_resources" in data:
            results["component_results"]["max_resources"] = self.validate_max_resources(data["max_resources"], strict)

        # Validate consistency
        results["consistency_results"] = self.validate_data_consistency(data)

        # Determine overall validity
        for component_result in results["component_results"].values():
            if not component_result["valid"]:
                results["overall_valid"] = False

        if not results["consistency_results"]["valid"]:
            results["overall_valid"] = False

        # Create summary
        results["summary"] = {
            "total_errors": sum(len(r.get("errors", [])) for r in results["component_results"].values())
            + len(results["consistency_results"].get("errors", [])),
            "total_warnings": sum(len(r.get("warnings", [])) for r in results["component_results"].values())
            + len(results["consistency_results"].get("warnings", [])),
            "components_validated": len(results["component_results"]),
            "consistency_valid": results["consistency_results"]["valid"],
        }

        return results
