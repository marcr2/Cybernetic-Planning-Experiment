"""
Comprehensive tests for cybernetic feedback mechanisms.

Tests cybernetic feedback loops, stability, and adaptive control.
"""

import pytest
import numpy as np
from src.cybernetic_planning.core.cybernetic_feedback import CyberneticFeedbackSystem

class TestCyberneticFeedbackSystem:
    """Test the CyberneticFeedbackSystem class."""

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

    def test_initialization(self, sample_data):
        """Test system initialization."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        assert system.A.shape == (2, 2)
        assert system.d.shape == (2,)
        assert system.l.shape == (2,)
        assert system.feedback_strength == 0.1
        assert system.adaptation_rate == 0.05
        assert system.convergence_threshold == 1e - 6
        assert system.max_iterations == 100

    def test_cybernetic_feedback_application(self, sample_data):
        """Test application of cybernetic feedback."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        initial_output = np.array([1.0, 1.0])
        result = system.apply_cybernetic_feedback(initial_output)

        assert "final_output" in result
        assert "final_demand" in result
        assert "converged" in result
        assert "iterations" in result
        assert "cybernetic_metrics" in result

        assert len(result["final_output"]) == 2
        assert len(result["final_demand"]) == 2
        assert isinstance(result["converged"], bool)
        assert result["iterations"] > 0

    def test_feedback_signal_calculation(self, sample_data):
        """Test feedback signal calculation."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        # Test with demand satisfaction ratios
        demand_satisfaction = np.array([0.8, 1.2])
        feedback_signal = system._calculate_feedback_signal(demand_satisfaction)

        assert len(feedback_signal) == 2
        assert np.all(np.isfinite(feedback_signal))

    def test_cybernetic_constraints(self, sample_data):
        """Test application of cybernetic constraints."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        # Test with large feedback signal
        large_feedback = np.array([10.0, -5.0])
        constrained_feedback = system._apply_cybernetic_constraints(large_feedback)

        # Should be constrained to reasonable values
        assert np.all(np.abs(constrained_feedback) <= 0.5 * np.max(system.d))
        assert np.all(system.d + constrained_feedback >= 0)  # Non - negative total demand

    def test_cybernetic_metrics_calculation(self, sample_data):
        """Test cybernetic metrics calculation."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        # Add some mock feedback history
        system.feedback_history = [
            {'output_change': 0.1, 'demand_satisfaction': np.array([0.9, 1.1])},
            {'output_change': 0.05, 'demand_satisfaction': np.array([0.95, 1.05])},
            {'output_change': 0.01, 'demand_satisfaction': np.array([0.99, 1.01])}
        ]
        system.iteration = 2

        metrics = system._calculate_cybernetic_metrics()

        assert "stability" in metrics
        assert "convergence_rate" in metrics
        assert "efficiency" in metrics
        assert "adaptability" in metrics
        assert "cybernetic_health" in metrics

        assert 0 <= metrics["stability"] <= 1
        assert metrics["convergence_rate"] >= 0
        assert 0 <= metrics["efficiency"] <= 1
        assert 0 <= metrics["adaptability"] <= 1
        assert 0 <= metrics["cybernetic_health"] <= 1

    def test_system_diagnostics(self, sample_data):
        """Test system diagnostics."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        diagnostics = system.get_system_diagnostics()

        assert "cybernetic_parameters" in diagnostics
        assert "current_state" in diagnostics
        assert "system_metrics" in diagnostics
        assert "feedback_analysis" in diagnostics

        # Check parameters
        params = diagnostics["cybernetic_parameters"]
        assert "feedback_strength" in params
        assert "adaptation_rate" in params
        assert "convergence_threshold" in params
        assert "max_iterations" in params

    def test_parameter_updates(self, sample_data):
        """Test dynamic parameter updates."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        # Update parameters
        system.update_cybernetic_parameters(
            feedback_strength = 0.2,
            adaptation_rate = 0.1,
            convergence_threshold = 1e - 8
        )

        assert system.feedback_strength == 0.2
        assert system.adaptation_rate == 0.1
        assert system.convergence_threshold == 1e - 8

    def test_parameter_validation(self, sample_data):
        """Test parameter validation and clipping."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        # Test invalid parameters (should be clipped)
        system.update_cybernetic_parameters(
            feedback_strength = 1.5,  # Should be clipped to 1.0
            adaptation_rate=-0.1,   # Should be clipped to 0.0
            convergence_threshold = 0  # Should be set to minimum
        )

        assert system.feedback_strength == 1.0
        assert system.adaptation_rate == 0.0
        assert system.convergence_threshold == 1e - 10

    def test_feedback_pattern_analysis(self, sample_data):
        """Test feedback pattern analysis."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        # Add mock feedback history
        system.feedback_history = [
            {'output_change': 0.1, 'feedback_signal': np.array([0.1, -0.1])},
            {'output_change': 0.05, 'feedback_signal': np.array([0.05, -0.05])},
            {'output_change': 0.01, 'feedback_signal': np.array([0.01, -0.01])}
        ]

        analysis = system._analyze_feedback_patterns()

        assert "convergence_pattern" in analysis
        assert "dominant_feedback_sectors" in analysis
        assert "feedback_variance" in analysis
        assert "is_stable" in analysis
        assert "oscillation_detected" in analysis

    def test_system_reset(self, sample_data):
        """Test system state reset."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        # Set some state
        system.current_output = np.array([1.0, 1.0])
        system.iteration = 5
        system.converged = True
        system.feedback_history = [{'test': 'data'}]

        # Reset system
        system.reset_feedback_state()

        assert system.current_output is None
        assert system.iteration == 0
        assert system.converged is False
        assert len(system.feedback_history) == 0
        assert not hasattr(system, '_integral_term')

    def test_convergence_behavior(self, sample_data):
        """Test convergence behavior with different parameters."""
        tech_matrix, final_demand, labor_vector = sample_data

        # Test with high feedback strength (should converge faster)
        system_high = CyberneticFeedbackSystem(
            tech_matrix, final_demand, labor_vector,
            feedback_strength = 0.5,
            adaptation_rate = 0.2
        )

        initial_output = np.array([1.0, 1.0])
        result_high = system_high.apply_cybernetic_feedback(initial_output)

        # Test with low feedback strength (should converge slower)
        system_low = CyberneticFeedbackSystem(
            tech_matrix, final_demand, labor_vector,
            feedback_strength = 0.01,
            adaptation_rate = 0.01
        )

        result_low = system_low.apply_cybernetic_feedback(initial_output)

        # High feedback strength should converge in fewer iterations
        assert result_high["iterations"] <= result_low["iterations"]

    def test_stability_analysis(self, sample_data):
        """Test system stability analysis."""
        tech_matrix, final_demand, labor_vector = sample_data
        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)

        # Test with stable parameters
        initial_output = np.array([1.0, 1.0])
        result = system.apply_cybernetic_feedback(initial_output)

        # Check that output is finite and reasonable
        assert np.all(np.isfinite(result["final_output"]))
        assert np.all(result["final_output"] >= 0)

        # Check cybernetic health
        metrics = result["cybernetic_metrics"]
        assert metrics["cybernetic_health"] > 0

    def test_error_handling(self, sample_data):
        """Test error handling for invalid inputs."""
        tech_matrix, final_demand, labor_vector = sample_data

        # Test with non - productive economy (should raise error)
        non_productive_matrix = np.array([
            [1.1, 0.0],  # Spectral radius > 1
            [0.0, 1.1]
        ])

        with pytest.raises(ValueError, match="Cannot compute Leontief inverse"):
            system = CyberneticFeedbackSystem(non_productive_matrix, final_demand, labor_vector)
            system.apply_cybernetic_feedback(np.array([1.0, 1.0]))

class TestCyberneticPrinciples:
    """Test adherence to cybernetic principles."""

    def test_circular_causality(self):
        """Test that outputs influence inputs (circular causality)."""
        tech_matrix = np.array([[0.1, 0.0], [0.0, 0.1]])
        final_demand = np.array([1.0, 1.0])
        labor_vector = np.array([0.5, 0.5])

        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)
        initial_output = np.array([1.0, 1.0])

        result = system.apply_cybernetic_feedback(initial_output)

        # Check that demand was adjusted based on output
        assert not np.array_equal(result["final_demand"], final_demand)
        assert len(result["feedback_history"]) > 0

    def test_self_regulation(self):
        """Test that system maintains equilibrium (self - regulation)."""
        tech_matrix = np.array([[0.1, 0.0], [0.0, 0.1]])
        final_demand = np.array([1.0, 1.0])
        labor_vector = np.array([0.5, 0.5])

        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)
        initial_output = np.array([2.0, 0.5])  # Start far from equilibrium

        result = system.apply_cybernetic_feedback(initial_output)

        # System should converge to equilibrium
        if result["converged"]:
            assert result["iterations"] < system.max_iterations
            # Final output should be reasonable
            assert np.all(result["final_output"] > 0)

    def test_requisite_variety(self):
        """Test that control system has sufficient complexity."""
        tech_matrix = np.array([[0.1, 0.0], [0.0, 0.1]])
        final_demand = np.array([1.0, 1.0])
        labor_vector = np.array([0.5, 0.5])

        system = CyberneticFeedbackSystem(tech_matrix, final_demand, labor_vector)
        initial_output = np.array([1.0, 1.0])

        result = system.apply_cybernetic_feedback(initial_output)

        # Check that system can handle different demand patterns
        metrics = result["cybernetic_metrics"]
        assert metrics["adaptability"] > 0  # System should be adaptable

if __name__ == "__main__":
    pytest.main([__file__])
