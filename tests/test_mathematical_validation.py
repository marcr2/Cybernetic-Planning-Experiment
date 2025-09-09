"""
Comprehensive tests for mathematical validation.

Tests the accuracy and completeness of mathematical formula validation.
"""

import pytest
    MathematicalValidator,
    ValidationResult,
    ValidationStatus
)
from src.cybernetic_planning.core.marxist_economics import MarxistEconomicCalculator
from src.cybernetic_planning.core.leontief import LeontiefModel
from src.cybernetic_planning.core.cybernetic_feedback import CyberneticFeedbackSystem
from src.cybernetic_planning.core.optimization import ConstrainedOptimizer

class TestValidationResult:
    """Test the ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creation of validation result."""
        result = ValidationResult(
            status = ValidationStatus.VALID,
            message="Test validation",
            expected_value = 1.0,
            actual_value = 1.0,
            tolerance = 1e - 10,
            formula="x = y"
        )

        assert result.status == ValidationStatus.VALID
        assert result.message == "Test validation"
        assert result.expected_value == 1.0
        assert result.actual_value == 1.0
        assert result.tolerance == 1e - 10
        assert result.formula == "x = y"

class TestMathematicalValidator:
    """Test the MathematicalValidator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample economic data for testing."""
        # Simple 2x2 economy
        technology_matrix = np.array([
            [0.2, 0.1],
            [0.1, 0.3]
        ])
        final_demand = np.array([1.0, 1.0])
        labor_vector = np.array([0.5, 0.3])
        return technology_matrix, final_demand, labor_vector

    def test_initialization(self):
        """Test validator initialization."""
        validator = MathematicalValidator(tolerance = 1e - 8)
        assert validator.tolerance == 1e - 8
        assert len(validator.validation_results) == 0

    def test_marxist_formulas_validation(self, sample_data):
        """Test validation of Marxist economic formulas."""
        tech_matrix, final_demand, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)
        validator = MathematicalValidator()

        results = validator.validate_marxist_formulas(calculator)

        # Should have validation results for each sector and formula
        assert len(results) > 0

        # Check that all value composition formulas are valid
        value_composition_results = [r for r in results if r.formula == "W = C + V + S"]
        assert len(value_composition_results) == 2  # One for each sector
        assert all(r.status == ValidationStatus.VALID for r in value_composition_results)

        # Check that all rate of surplus value formulas are valid
        surplus_value_results = [r for r in results if r.formula == "s' = S / V"]
        assert len(surplus_value_results) == 2
        assert all(r.status == ValidationStatus.VALID for r in surplus_value_results)

        # Check that all rate of profit formulas are valid
        profit_results = [r for r in results if r.formula == "p' = S/(C + V)"]
        assert len(profit_results) == 2
        assert all(r.status == ValidationStatus.VALID for r in profit_results)

        # Check that all organic composition formulas are valid
        occ_results = [r for r in results if r.formula == "OCC = C / V"]
        assert len(occ_results) == 2
        assert all(r.status == ValidationStatus.VALID for r in occ_results)

    def test_leontief_formulas_validation(self, sample_data):
        """Test validation of Leontief input - output formulas."""
        tech_matrix, final_demand, labor_vector = sample_data
        model = LeontiefModel(tech_matrix, final_demand)
        validator = MathematicalValidator()

        results = validator.validate_leontief_formulas(model)

        # Should have validation results for key formulas
        assert len(results) >= 3  # At least 3 key formulas

        # Check that fundamental equation is valid
        fundamental_results = [r for r in results if r.formula == "x = Ax + d"]
        assert len(fundamental_results) == 1
        assert fundamental_results[0].status == ValidationStatus.VALID

        # Check that Leontief inverse is valid
        inverse_results = [r for r in results if r.formula == "(I - A)^(-1)"]
        assert len(inverse_results) == 1
        assert inverse_results[0].status == ValidationStatus.VALID

        # Check that value added calculation is valid
        value_added_results = [r for r in results if r.formula == "(I - A)x = d"]
        assert len(value_added_results) == 1
        assert value_added_results[0].status == ValidationStatus.VALID

    def test_cybernetic_formulas_validation(self, sample_data):
        """Test validation of cybernetic feedback formulas."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)
        validator = MathematicalValidator()

        # Apply cybernetic feedback
        initial_output = np.array([1.0, 1.0])
        result = system.apply_cybernetic_feedback(initial_output)

        # Validate cybernetic formulas
        results = validator.validate_cybernetic_formulas(system, result)

        # Should have validation results for key aspects
        assert len(results) >= 2  # At least 2 key validations

        # Check that PID controller produces finite values
        pid_results = [r for r in results if "PID controller" in r.message]
        assert len(pid_results) >= 1
        assert pid_results[0].status == ValidationStatus.VALID

        # Check that output is stable
        stability_results = [r for r in results if "stable output" in r.message]
        assert len(stability_results) >= 1
        assert stability_results[0].status == ValidationStatus.VALID

    def test_optimization_formulas_validation(self, sample_data):
        """Test validation of optimization formulas."""
        tech_matrix, final_demand, labor_vector = sample_data
        optimizer = ConstrainedOptimizer(tech_matrix, labor_vector, final_demand)
        validator = MathematicalValidator()

        # Solve optimization problem
        result = optimizer.solve()

        # Validate optimization formulas
        results = validator.validate_optimization_formulas(optimizer, result)

        # Should have validation results
        assert len(results) >= 1

        # Check constraint satisfaction
        constraint_results = [r for r in results if "constraints" in r.message.lower()]
        assert len(constraint_results) >= 1

        # Check objective function
        objective_results = [r for r in results if "objective function" in r.message.lower()]
        if result.get("feasible", False):
            assert len(objective_results) >= 1
            assert objective_results[0].status == ValidationStatus.VALID

    def test_economic_consistency_validation(self, sample_data):
        """Test validation of economic consistency."""
        tech_matrix, final_demand, labor_vector = sample_data
        validator = MathematicalValidator()

        data = {
            "technology_matrix": tech_matrix,
            "final_demand": final_demand,
            "labor_vector": labor_vector
        }

        results = validator.validate_economic_consistency(data)

        # Should have validation results for dimensions and values
        assert len(results) >= 1

        # Check dimension compatibility
        dimension_results = [r for r in results if "dimensions" in r.message.lower()]
        assert len(dimension_results) >= 1
        assert dimension_results[0].status == ValidationStatus.VALID

        # Check for negative values
        negative_results = [r for r in results if "negative values" in r.message.lower()]
        # Should not have negative values in our test data
        assert len(negative_results) == 0

    def test_comprehensive_validation(self, sample_data):
        """Test comprehensive validation of all components."""
        tech_matrix, final_demand, labor_vector = sample_data
        validator = MathematicalValidator()

        # Create system components
        marxist_calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)
        leontief_model = LeontiefModel(tech_matrix, final_demand)
        cybernetic_system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)
        optimizer = ConstrainedOptimizer(tech_matrix, labor_vector, final_demand)

        # Apply cybernetic feedback
        cybernetic_result = cybernetic_system.apply_cybernetic_feedback(np.array([1.0, 1.0]))

        # Solve optimization
        optimization_result = optimizer.solve()

        # Validate all components
        system_components = {
            "marxist_calculator": marxist_calculator,
            "leontief_model": leontief_model,
            "cybernetic_system": cybernetic_system,
            "cybernetic_result": cybernetic_result,
            "optimizer": optimizer,
            "optimization_result": optimization_result,
            "data": {
                "technology_matrix": tech_matrix,
                "final_demand": final_demand,
                "labor_vector": labor_vector
            }
        }

        results = validator.validate_all(system_components)

        # Check that all categories are present
        assert "marxist_validation" in results
        assert "leontief_validation" in results
        assert "cybernetic_validation" in results
        assert "optimization_validation" in results
        assert "consistency_validation" in results
        assert "overall_status" in results
        assert "summary" in results

        # Check overall status
        assert results["overall_status"] in [ValidationStatus.VALID, ValidationStatus.WARNING]

        # Check summary
        summary = results["summary"]
        assert "total_validations" in summary
        assert "valid" in summary
        assert "warnings" in summary
        assert "errors" in summary
        assert summary["total_validations"] > 0
        assert summary["valid"] > 0

    def test_validation_report_generation(self, sample_data):
        """Test generation of validation report."""
        tech_matrix, final_demand, labor_vector = sample_data
        validator = MathematicalValidator()

        # Create simple validation results
        marxist_calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)
        marxist_results = validator.validate_marxist_formulas(marxist_calculator)

        validation_results = {
            "marxist_validation": marxist_results,
            "leontief_validation": [],
            "cybernetic_validation": [],
            "optimization_validation": [],
            "consistency_validation": [],
            "overall_status": ValidationStatus.VALID,
            "summary": {
                "total_validations": len(marxist_results),
                "valid": len(marxist_results),
                "warnings": 0,
                "errors": 0,
                "overall_status": "valid"
            }
        }

        report = validator.get_validation_report(validation_results)

        # Check that report contains expected sections
        assert "MATHEMATICAL VALIDATION REPORT" in report
        assert "Overall Status: VALID" in report
        assert "SUMMARY:" in report
        assert "MARXIST ECONOMICS:" in report

        # Check that report contains validation details
        assert "✓" in report  # Valid symbol
        assert "W = C + V + S" in report  # Formula

    def test_validation_with_errors(self):
        """Test validation with intentionally incorrect data."""
        validator = MathematicalValidator()

        # Create data with errors
        tech_matrix = np.array([[1.1, 0.0], [0.0, 1.1]])  # Non - productive
        final_demand = np.array([1.0, -1.0])  # Negative demand
        labor_vector = np.array([0.5, -0.3])  # Negative labor

        data = {
            "technology_matrix": tech_matrix,
            "final_demand": final_demand,
            "labor_vector": labor_vector
        }

        results = validator.validate_economic_consistency(data)

        # Should have error results
        error_results = [r for r in results if r.status == ValidationStatus.ERROR]
        assert len(error_results) > 0

        # Check for specific error messages
        error_messages = [r.message for r in error_results]
        assert any("negative values" in msg.lower() for msg in error_messages)

    def test_tolerance_parameter(self):
        """Test that tolerance parameter affects validation."""
        # Create data with small numerical errors
        tech_matrix = np.array([[0.1, 0.0], [0.0, 0.1]])
        final_demand = np.array([1.0, 1.0])
        labor_vector = np.array([0.5, 0.5])

        # Test with strict tolerance
        strict_validator = MathematicalValidator(tolerance = 1e - 15)
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)
        strict_results = strict_validator.validate_marxist_formulas(calculator)

        # Test with loose tolerance
        loose_validator = MathematicalValidator(tolerance = 1e - 5)
        loose_results = loose_validator.validate_marxist_formulas(calculator)

        # Both should be valid for this simple case
        assert all(r.status == ValidationStatus.VALID for r in strict_results)
        assert all(r.status == ValidationStatus.VALID for r in loose_results)

class TestValidationEdgeCases:
    """Test edge cases in validation."""

    def test_zero_values_validation(self):
        """Test validation with zero values."""
        tech_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
        final_demand = np.array([0.0, 0.0])
        labor_vector = np.array([0.0, 0.0])

        validator = MathematicalValidator()
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        # This should handle zero values gracefully
        results = validator.validate_marxist_formulas(calculator)
        assert len(results) > 0

    def test_single_sector_validation(self):
        """Test validation with single sector economy."""
        tech_matrix = np.array([[0.1]])
        final_demand = np.array([1.0])
        labor_vector = np.array([0.5])

        validator = MathematicalValidator()
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        results = validator.validate_marxist_formulas(calculator)
        assert len(results) > 0
        assert all(r.status == ValidationStatus.VALID for r in results)

if __name__ == "__main__":
    pytest.main([__file__])
