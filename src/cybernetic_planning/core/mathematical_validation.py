"""
Mathematical Validation Module

Validates all economic formulas against theoretical sources and ensures
mathematical accuracy of calculations.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ValidationStatus(Enum):
    """Status of mathematical validation."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationResult:
    """Result of mathematical validation."""
    status: ValidationStatus
    message: str
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    tolerance: Optional[float] = None
    formula: Optional[str] = None

class MathematicalValidator:
    """
    Validates mathematical formulas against theoretical sources.

    Ensures accuracy of:
    - Marxist economic calculations - Leontief input - output analysis - Cybernetic feedback mechanisms - Optimization algorithms
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the mathematical validator.

        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        self.validation_results = []

    def validate_marxist_formulas(self, calculator) -> List[ValidationResult]:
        """
        Validate Marxist economic formulas.

        Args:
            calculator: MarxistEconomicCalculator instance

        Returns:
            List of validation results
        """
        results = []

        # Validate value composition formula: W = C + V + S
        compositions = calculator.get_value_compositions()
        for i, comp in enumerate(compositions):
            expected_total = comp.constant_capital + comp.variable_capital + comp.surplus_value
            actual_total = comp.total_value

            if abs(expected_total - actual_total) < self.tolerance:
                results.append(ValidationResult(
                    status = ValidationStatus.VALID,
                    message = f"Value composition formula valid for sector {i}",
                    expected_value = expected_total,
                    actual_value = actual_total,
                    formula="W = C + V + S"
                ))
            else:
                results.append(ValidationResult(
                    status = ValidationStatus.ERROR,
                    message = f"Value composition formula invalid for sector {i}",
                    expected_value = expected_total,
                    actual_value = actual_total,
                    tolerance = self.tolerance,
                    formula="W = C + V + S"
                ))

        # Validate rate of surplus value: s' = S / V
        for i, comp in enumerate(compositions):
            if comp.variable_capital > 0:
                expected_rate = comp.surplus_value / comp.variable_capital
                actual_rate = comp.rate_of_surplus_value

                if abs(expected_rate - actual_rate) < self.tolerance:
                    results.append(ValidationResult(
                        status = ValidationStatus.VALID,
                        message = f"Rate of surplus value valid for sector {i}",
                        expected_value = expected_rate,
                        actual_value = actual_rate,
                        formula="s' = S / V"
                    ))
                else:
                    results.append(ValidationResult(
                        status = ValidationStatus.ERROR,
                        message = f"Rate of surplus value invalid for sector {i}",
                        expected_value = expected_rate,
                        actual_value = actual_rate,
                        tolerance = self.tolerance,
                        formula="s' = S / V"
                    ))

        # Validate rate of profit: p' = S/(C + V)
        for i, comp in enumerate(compositions):
            if (comp.constant_capital + comp.variable_capital) > 0:
                expected_rate = comp.surplus_value / (comp.constant_capital + comp.variable_capital)
                actual_rate = comp.rate_of_profit

                if abs(expected_rate - actual_rate) < self.tolerance:
                    results.append(ValidationResult(
                        status = ValidationStatus.VALID,
                        message = f"Rate of profit valid for sector {i}",
                        expected_value = expected_rate,
                        actual_value = actual_rate,
                        formula="p' = S/(C + V)"
                    ))
                else:
                    results.append(ValidationResult(
                        status = ValidationStatus.ERROR,
                        message = f"Rate of profit invalid for sector {i}",
                        expected_value = expected_rate,
                        actual_value = actual_rate,
                        tolerance = self.tolerance,
                        formula="p' = S/(C + V)"
                    ))

        # Validate organic composition: OCC = C / V
        for i, comp in enumerate(compositions):
            if comp.variable_capital > 0:
                expected_occ = comp.constant_capital / comp.variable_capital
                actual_occ = comp.organic_composition

                if abs(expected_occ - actual_occ) < self.tolerance:
                    results.append(ValidationResult(
                        status = ValidationStatus.VALID,
                        message = f"Organic composition valid for sector {i}",
                        expected_value = expected_occ,
                        actual_value = actual_occ,
                        formula="OCC = C / V"
                    ))
                else:
                    results.append(ValidationResult(
                        status = ValidationStatus.ERROR,
                        message = f"Organic composition invalid for sector {i}",
                        expected_value = expected_occ,
                        actual_value = actual_occ,
                        tolerance = self.tolerance,
                        formula="OCC = C / V"
                    ))

        return results

    def validate_leontief_formulas(self, model) -> List[ValidationResult]:
        """
        Validate Leontief input - output formulas.

        Args:
            model: LeontiefModel instance

        Returns:
            List of validation results
        """
        results = []

        # Validate fundamental equation: x = Ax + d
        total_output = model.compute_total_output()
        intermediate_demand = model.compute_intermediate_demand(total_output)
        expected_output = intermediate_demand + model.d
        actual_output = total_output

        error = np.linalg.norm(expected_output - actual_output)
        if error < self.tolerance:
            results.append(ValidationResult(
                status = ValidationStatus.VALID,
                message="Leontief fundamental equation valid",
                expected_value = np.sum(expected_output),
                actual_value = np.sum(actual_output),
                formula="x = Ax + d"
            ))
        else:
            results.append(ValidationResult(
                status = ValidationStatus.ERROR,
                message="Leontief fundamental equation invalid",
                expected_value = np.sum(expected_output),
                actual_value = np.sum(actual_output),
                tolerance = self.tolerance,
                formula="x = Ax + d"
            ))

        # Validate Leontief inverse: (I - A)^(-1)
        leontief_inverse = model.get_leontief_inverse()
        I = np.eye(model.A.shape[0])
        identity_check = (I - model.A) @ leontief_inverse

        error = np.linalg.norm(identity_check - I)
        if error < self.tolerance:
            results.append(ValidationResult(
                status = ValidationStatus.VALID,
                message="Leontief inverse calculation valid",
                expected_value = 1.0,
                actual_value = np.trace(identity_check) / model.A.shape[0],
                formula="(I - A)^(-1)"
            ))
        else:
            results.append(ValidationResult(
                status = ValidationStatus.ERROR,
                message="Leontief inverse calculation invalid",
                expected_value = 1.0,
                actual_value = np.trace(identity_check) / model.A.shape[0],
                tolerance = self.tolerance,
                formula="(I - A)^(-1)"
            ))

        # Validate value added calculation: (I - A)x = d
        value_added = model.compute_value_added(total_output)
        error = np.linalg.norm(value_added - model.d)
        if error < self.tolerance:
            results.append(ValidationResult(
                status = ValidationStatus.VALID,
                message="Value added calculation valid",
                expected_value = np.sum(model.d),
                actual_value = np.sum(value_added),
                formula="(I - A)x = d"
            ))
        else:
            results.append(ValidationResult(
                status = ValidationStatus.ERROR,
                message="Value added calculation invalid",
                expected_value = np.sum(model.d),
                actual_value = np.sum(value_added),
                tolerance = self.tolerance,
                formula="(I - A)x = d"
            ))

        return results

    def validate_cybernetic_formulas(self, system, result) -> List[ValidationResult]:
        """
        Validate cybernetic feedback formulas.

        Args:
            system: CyberneticFeedbackSystem instance
            result: Result from apply_cybernetic_feedback

        Returns:
            List of validation results
        """
        results = []

        # Validate PID controller formula
        if "feedback_history" in result and len(result["feedback_history"]) > 0:
            last_feedback = result["feedback_history"][-1]

            # Check that feedback signal is finite
            if np.all(np.isfinite(last_feedback.get("feedback_signal", []))):
                results.append(ValidationResult(
                    status = ValidationStatus.VALID,
                    message="PID controller produces finite values",
                    formula="u(t) = Kp * e(t) + Ki*∫e(t)dt + Kd * de(t)/dt"
                ))
            else:
                results.append(ValidationResult(
                    status = ValidationStatus.ERROR,
                    message="PID controller produces infinite values",
                    formula="u(t) = Kp * e(t) + Ki*∫e(t)dt + Kd * de(t)/dt"
                ))

        # Validate convergence
        if result.get("converged", False):
            results.append(ValidationResult(
                status = ValidationStatus.VALID,
                message="Cybernetic feedback converged",
                formula="Convergence condition"
            ))
        else:
            results.append(ValidationResult(
                status = ValidationStatus.WARNING,
                message="Cybernetic feedback did not converge",
                formula="Convergence condition"
            ))

        # Validate stability (output should be finite and positive)
        final_output = result.get("final_output", [])
        if np.all(np.isfinite(final_output)) and np.all(final_output >= 0):
            results.append(ValidationResult(
                status = ValidationStatus.VALID,
                message="Cybernetic feedback produces stable output",
                formula="Stability condition"
            ))
        else:
            results.append(ValidationResult(
                status = ValidationStatus.ERROR,
                message="Cybernetic feedback produces unstable output",
                formula="Stability condition"
            ))

        return results

    def validate_optimization_formulas(self, optimizer, result) -> List[ValidationResult]:
        """
        Validate optimization formulas.

        Args:
            optimizer: ConstrainedOptimizer instance
            result: Result from solve method

        Returns:
            List of validation results
        """
        results = []

        if result.get("feasible", False) and result.get("solution") is not None:
            solution = result["solution"]

            # Validate constraint satisfaction
            constraint_check = optimizer.check_constraints(solution)

            if constraint_check["all_constraints_satisfied"]:
                results.append(ValidationResult(
                    status = ValidationStatus.VALID,
                    message="All constraints satisfied",
                    formula="Constraint satisfaction"
                ))
            else:
                results.append(ValidationResult(
                    status = ValidationStatus.ERROR,
                    message="Some constraints violated",
                    formula="Constraint satisfaction"
                ))

            # Validate objective function
            if result.get("objective_value") is not None:
                expected_objective = np.dot(optimizer.l, solution)
                actual_objective = result["objective_value"]

                if abs(expected_objective - actual_objective) < self.tolerance:
                    results.append(ValidationResult(
                        status = ValidationStatus.VALID,
                        message="Objective function calculation valid",
                        expected_value = expected_objective,
                        actual_value = actual_objective,
                        formula="minimize l^T * x"
                    ))
                else:
                    results.append(ValidationResult(
                        status = ValidationStatus.ERROR,
                        message="Objective function calculation invalid",
                        expected_value = expected_objective,
                        actual_value = actual_objective,
                        tolerance = self.tolerance,
                        formula="minimize l^T * x"
                    ))
        else:
            results.append(ValidationResult(
                status = ValidationStatus.WARNING,
                message="Optimization problem not feasible",
                formula="Feasibility condition"
            ))

        return results

    def validate_economic_consistency(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate economic consistency across all models.

        Args:
            data: Dictionary containing economic data

        Returns:
            List of validation results
        """
        results = []

        # Check that all matrices have compatible dimensions
        tech_matrix = data.get("technology_matrix")
        final_demand = data.get("final_demand")
        labor_vector = data.get("labor_vector")

        if all(x is not None for x in [tech_matrix, final_demand, labor_vector]):
            n_sectors = tech_matrix.shape[0]

            if tech_matrix.shape[1] != n_sectors:
                results.append(ValidationResult(
                    status = ValidationStatus.ERROR,
                    message="Technology matrix is not square",
                    formula="Matrix dimensions"
                ))
            elif final_demand.shape[0] != n_sectors:
                results.append(ValidationResult(
                    status = ValidationStatus.ERROR,
                    message="Final demand dimension mismatch",
                    formula="Matrix dimensions"
                ))
            elif labor_vector.shape[0] != n_sectors:
                results.append(ValidationResult(
                    status = ValidationStatus.ERROR,
                    message="Labor vector dimension mismatch",
                    formula="Matrix dimensions"
                ))
            else:
                results.append(ValidationResult(
                    status = ValidationStatus.VALID,
                    message="All dimensions compatible",
                    formula="Matrix dimensions"
                ))

        # Check for negative values
        if tech_matrix is not None and np.any(tech_matrix < 0):
            results.append(ValidationResult(
                status = ValidationStatus.ERROR,
                message="Technology matrix contains negative values",
                formula="Economic validity"
            ))

        if final_demand is not None and np.any(final_demand < 0):
            results.append(ValidationResult(
                status = ValidationStatus.ERROR,
                message="Final demand contains negative values",
                formula="Economic validity"
            ))

        if labor_vector is not None and np.any(labor_vector < 0):
            results.append(ValidationResult(
                status = ValidationStatus.ERROR,
                message="Labor vector contains negative values",
                formula="Economic validity"
            ))

        return results

    def validate_all(self, system_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all system components.

        Args:
            system_components: Dictionary containing all system components

        Returns:
            Comprehensive validation results
        """
        all_results = {
            "marxist_validation": [],
            "leontief_validation": [],
            "cybernetic_validation": [],
            "optimization_validation": [],
            "consistency_validation": [],
            "overall_status": ValidationStatus.VALID,
            "summary": {}
        }

        # Validate Marxist economics
        if "marxist_calculator" in system_components:
            marxist_results = self.validate_marxist_formulas(system_components["marxist_calculator"])
            all_results["marxist_validation"] = marxist_results

        # Validate Leontief model
        if "leontief_model" in system_components:
            leontief_results = self.validate_leontief_formulas(system_components["leontief_model"])
            all_results["leontief_validation"] = leontief_results

        # Validate cybernetic feedback
        if "cybernetic_system" in system_components and "cybernetic_result" in system_components:
            cybernetic_results = self.validate_cybernetic_formulas(
                system_components["cybernetic_system"],
                system_components["cybernetic_result"]
            )
            all_results["cybernetic_validation"] = cybernetic_results

        # Validate optimization
        if "optimizer" in system_components and "optimization_result" in system_components:
            optimization_results = self.validate_optimization_formulas(
                system_components["optimizer"],
                system_components["optimization_result"]
            )
            all_results["optimization_validation"] = optimization_results

        # Validate consistency
        if "data" in system_components:
            consistency_results = self.validate_economic_consistency(system_components["data"])
            all_results["consistency_validation"] = consistency_results

        # Determine overall status
        all_validation_results = []
        for category in ["marxist_validation", "leontief_validation", "cybernetic_validation",
                        "optimization_validation", "consistency_validation"]:
            all_validation_results.extend(all_results[category])

        error_count = sum(1 for r in all_validation_results if r.status == ValidationStatus.ERROR)
        warning_count = sum(1 for r in all_validation_results if r.status == ValidationStatus.WARNING)

        if error_count > 0:
            all_results["overall_status"] = ValidationStatus.ERROR
        elif warning_count > 0:
            all_results["overall_status"] = ValidationStatus.WARNING
        else:
            all_results["overall_status"] = ValidationStatus.VALID

        # Create summary
        all_results["summary"] = {
            "total_validations": len(all_validation_results),
            "valid": sum(1 for r in all_validation_results if r.status == ValidationStatus.VALID),
            "warnings": warning_count,
            "errors": error_count,
            "overall_status": all_results["overall_status"].value
        }

        return all_results

    def get_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human - readable validation report.

        Args:
            validation_results: Results from validate_all

        Returns:
            Formatted validation report
        """
        report = ["MATHEMATICAL VALIDATION REPORT", "=" * 50, ""]

        # Overall status
        status = validation_results["overall_status"]
        report.append(f"Overall Status: {status.value.upper()}")
        report.append("")

        # Summary
        summary = validation_results["summary"]
        report.append("SUMMARY:")
        report.append(f"  Total Validations: {summary['total_validations']}")
        report.append(f"  Valid: {summary['valid']}")
        report.append(f"  Warnings: {summary['warnings']}")
        report.append(f"  Errors: {summary['errors']}")
        report.append("")

        # Detailed results by category
        categories = [
            ("Marxist Economics", "marxist_validation"),
            ("Leontief Model", "leontief_validation"),
            ("Cybernetic Feedback", "cybernetic_validation"),
            ("Optimization", "optimization_validation"),
            ("Economic Consistency", "consistency_validation")
        ]

        for category_name, category_key in categories:
            if category_key in validation_results and validation_results[category_key]:
                report.append(f"{category_name.upper()}:")
                report.append("-" * len(category_name))

                for result in validation_results[category_key]:
                    status_symbol = {
                        ValidationStatus.VALID: "✓",
                        ValidationStatus.WARNING: "⚠",
                        ValidationStatus.ERROR: "✗"
                    }[result.status]

                    report.append(f"  {status_symbol} {result.message}")
                    if result.formula:
                        report.append(f"    Formula: {result.formula}")
                    if result.expected_value is not None and result.actual_value is not None:
                        report.append(f"    Expected: {result.expected_value:.6f}")
                        report.append(f"    Actual: {result.actual_value:.6f}")
                    if result.tolerance:
                        report.append(f"    Tolerance: {result.tolerance:.2e}")
                report.append("")

        return "\n".join(report)

    def validate_all_formulas(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all formulas with the current data.

        Args:
            data: Economic data dictionary

        Returns:
            Validation results dictionary
        """
        # Create a simple validation for the data
        results = {
            "data_validation": [],
            "overall_status": "valid",
            "summary": {}
        }

        # Check data consistency
        consistency_results = self.validate_economic_consistency(data)
        results["data_validation"] = consistency_results

        # Count results
        error_count = sum(1 for r in consistency_results if r.status == ValidationStatus.ERROR)
        warning_count = sum(1 for r in consistency_results if r.status == ValidationStatus.WARNING)

        if error_count > 0:
            results["overall_status"] = "error"
        elif warning_count > 0:
            results["overall_status"] = "warning"

        results["summary"] = {
            "total_validations": len(consistency_results),
            "valid": sum(1 for r in consistency_results if r.status == ValidationStatus.VALID),
            "warnings": warning_count,
            "errors": error_count
        }

        return results
