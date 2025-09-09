"""
Comprehensive tests for Leontief Input - Output model.

Tests mathematical accuracy and economic validity of Leontief calculations.
"""

import pytest
from src.cybernetic_planning.core.leontief import LeontiefModel

class TestLeontiefModel:
    """Test the LeontiefModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample economic data for testing."""
        # Simple 2x2 economy
        technology_matrix = np.array([
            [0.2, 0.1],
            [0.1, 0.3]
        ])
        final_demand = np.array([1.0, 1.0])
        return technology_matrix, final_demand

    def test_initialization(self, sample_data):
        """Test model initialization."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        assert model.A.shape == (2, 2)
        assert model.d.shape == (2,)
        assert model._leontief_inverse is not None

    def test_validation_errors(self):
        """Test input validation."""
        # Test non - square matrix
        with pytest.raises(ValueError, match="Technology matrix must be square"):
            LeontiefModel(np.array([[1, 2]]), np.array([1, 2]))

        # Test dimension mismatch
        with pytest.raises(ValueError, match="Technology matrix and final demand must have compatible dimensions"):
            LeontiefModel(np.array([[1, 0], [0, 1]]), np.array([1, 2, 3]))

        # Test negative values in technology matrix
        with pytest.raises(ValueError, match="Technology matrix contains negative values"):
            LeontiefModel(np.array([[1, -1], [0, 1]]), np.array([1, 1]))

        # Test negative values in final demand
        with pytest.raises(ValueError, match="Final demand contains negative values"):
            LeontiefModel(np.array([[1, 0], [0, 1]]), np.array([1, -1]))

        # Test non - productive economy
        with pytest.raises(ValueError, match="Economy is not productive"):
            LeontiefModel(np.array([[1.1, 0], [0, 1.1]]), np.array([1, 1]))

    def test_leontief_inverse_calculation(self, sample_data):
        """Test Leontief inverse calculation."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        inverse = model.get_leontief_inverse()

        # Verify that (I - A) * (I - A)^(-1) = I
        I = np.eye(2)
        identity_check = (I - tech_matrix) @ inverse
        np.testing.assert_allclose(identity_check, I, atol = 1e - 10)

    def test_total_output_calculation(self, sample_data):
        """Test total output calculation."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        total_output = model.compute_total_output()

        # Verify that (I - A) * x = d
        I = np.eye(2)
        net_output = (I - tech_matrix) @ total_output
        np.testing.assert_allclose(net_output, final_demand, atol = 1e - 10)

        # Check that output is positive
        assert np.all(total_output > 0)

    def test_intermediate_demand_calculation(self, sample_data):
        """Test intermediate demand calculation."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        total_output = model.compute_total_output()
        intermediate_demand = model.compute_intermediate_demand(total_output)

        # Verify that intermediate demand = A * x
        expected_intermediate = tech_matrix @ total_output
        np.testing.assert_allclose(intermediate_demand, expected_intermediate, atol = 1e - 10)

    def test_value_added_calculation(self, sample_data):
        """Test value added calculation."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        total_output = model.compute_total_output()
        value_added = model.compute_value_added(total_output)

        # Verify that value added = x - Ax = (I - A) * x
        I = np.eye(2)
        expected_value_added = (I - tech_matrix) @ total_output
        np.testing.assert_allclose(value_added, expected_value_added, atol = 1e - 10)

        # Value added should equal final demand
        np.testing.assert_allclose(value_added, final_demand, atol = 1e - 10)

    def test_spectral_radius_calculation(self, sample_data):
        """Test spectral radius calculation."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        spectral_radius = model.get_spectral_radius()

        # Calculate manually
        eigenvals = np.linalg.eigvals(tech_matrix)
        expected_spectral_radius = np.max(np.abs(eigenvals))

        assert abs(spectral_radius - expected_spectral_radius) < 1e - 10
        assert spectral_radius < 1  # Economy should be productive

    def test_productivity_check(self, sample_data):
        """Test productivity check."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        assert model.is_productive()

        # Test with non - productive economy
        non_productive_matrix = np.array([[1.1, 0], [0, 1.1]])
        with pytest.raises(ValueError):
            LeontiefModel(non_productive_matrix, final_demand)

    def test_sensitivity_analysis(self, sample_data):
        """Test sensitivity analysis."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        # Test sensitivity to final demand changes
        demand_sensitivity = model.sensitivity_analysis("d")
        assert demand_sensitivity.shape == (2, 2)

        # Test sensitivity to technology matrix changes
        tech_sensitivity = model.sensitivity_analysis("A")
        assert tech_sensitivity.shape == (2, 2, 2)

        # Test invalid parameter
        with pytest.raises(ValueError, match="Parameter must be 'A' or 'd'"):
            model.sensitivity_analysis("invalid")

    def test_matrix_updates(self, sample_data):
        """Test technology matrix and final demand updates."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        # Update technology matrix
        new_tech_matrix = np.array([
            [0.3, 0.1],
            [0.1, 0.2]
        ])
        model.update_technology_matrix(new_tech_matrix)

        assert np.array_equal(model.A, new_tech_matrix)
        assert model._leontief_inverse is not None

        # Update final demand
        new_final_demand = np.array([2.0, 1.5])
        model.update_final_demand(new_final_demand)

        assert np.array_equal(model.d, new_final_demand)

    def test_mathematical_consistency(self, sample_data):
        """Test mathematical consistency of the model."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        # Test the fundamental Leontief equation: x = Ax + d
        total_output = model.compute_total_output()
        intermediate_demand = model.compute_intermediate_demand(total_output)

        # x = Ax + d should hold
        left_side = total_output
        right_side = intermediate_demand + final_demand

        np.testing.assert_allclose(left_side, right_side, atol = 1e - 10)

    def test_economic_interpretation(self, sample_data):
        """Test economic interpretation of results."""
        tech_matrix, final_demand = sample_data
        model = LeontiefModel(tech_matrix, final_demand)

        total_output = model.compute_total_output()
        intermediate_demand = model.compute_intermediate_demand(total_output)
        value_added = model.compute_value_added(total_output)

        # Total output should be positive
        assert np.all(total_output > 0)

        # Intermediate demand should be non - negative
        assert np.all(intermediate_demand >= 0)

        # Value added should equal final demand
        np.testing.assert_allclose(value_added, final_demand, atol = 1e - 10)

        # Total output should be greater than intermediate demand
        assert np.all(total_output > intermediate_demand)

    def test_numerical_stability(self):
        """Test numerical stability with different matrix properties."""
        # Test with nearly singular matrix
        tech_matrix = np.array([
            [0.99, 0.0],
            [0.0, 0.99]
        ])
        final_demand = np.array([1.0, 1.0])

        model = LeontiefModel(tech_matrix, final_demand)
        total_output = model.compute_total_output()

        # Should still produce reasonable results
        assert np.all(total_output > 0)
        assert np.all(np.isfinite(total_output))

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with diagonal matrix
        tech_matrix = np.array([
            [0.1, 0.0],
            [0.0, 0.2]
        ])
        final_demand = np.array([1.0, 1.0])

        model = LeontiefModel(tech_matrix, final_demand)
        total_output = model.compute_total_output()

        # For diagonal matrix, x_i = d_i / (1 - a_ii)
        expected_output = final_demand / (1 - np.diag(tech_matrix))
        np.testing.assert_allclose(total_output, expected_output, atol = 1e - 10)

        # Test with zero final demand
        zero_demand = np.array([0.0, 0.0])
        model_zero = LeontiefModel(tech_matrix, zero_demand)
        zero_output = model_zero.compute_total_output()

        # Should produce zero output
        np.testing.assert_allclose(zero_output, zero_demand, atol = 1e - 10)

class TestLeontiefTheoreticalAccuracy:
    """Test theoretical accuracy of Leontief calculations."""

    def test_leontief_equation(self):
        """Test that the fundamental Leontief equation is satisfied."""
        # Create a more complex economy
        tech_matrix = np.array([
            [0.1, 0.2, 0.0],
            [0.0, 0.1, 0.1],
            [0.1, 0.0, 0.2]
        ])
        final_demand = np.array([1.0, 2.0, 1.5])

        model = LeontiefModel(tech_matrix, final_demand)
        total_output = model.compute_total_output()

        # Verify x = Ax + d
        intermediate_demand = tech_matrix @ total_output
        leontief_equation = total_output - intermediate_demand - final_demand

        np.testing.assert_allclose(leontief_equation, 0, atol = 1e - 10)

    def test_leontief_inverse_properties(self):
        """Test properties of the Leontief inverse."""
        tech_matrix = np.array([
            [0.2, 0.1],
            [0.1, 0.3]
        ])
        final_demand = np.array([1.0, 1.0])

        model = LeontiefModel(tech_matrix, final_demand)
        leontief_inverse = model.get_leontief_inverse()

        # Leontief inverse should be non - negative (for productive economies)
        assert np.all(leontief_inverse >= 0)

        # Leontief inverse should be symmetric if A is symmetric
        if np.allclose(tech_matrix, tech_matrix.T):
            assert np.allclose(leontief_inverse, leontief_inverse.T, atol = 1e - 10)

    def test_multiplier_effects(self):
        """Test multiplier effects in the Leontief model."""
        tech_matrix = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ])
        final_demand = np.array([1.0, 1.0])

        model = LeontiefModel(tech_matrix, final_demand)
        leontief_inverse = model.get_leontief_inverse()

        # For diagonal matrix, multipliers should be 1/(1 - a_ii)
        expected_multipliers = 1 / (1 - np.diag(tech_matrix))
        actual_multipliers = np.diag(leontief_inverse)

        np.testing.assert_allclose(actual_multipliers, expected_multipliers, atol = 1e - 10)

if __name__ == "__main__":
    pytest.main([__file__])
