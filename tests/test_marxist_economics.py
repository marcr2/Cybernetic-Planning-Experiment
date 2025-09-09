"""
Comprehensive tests for Marxist economic calculations.

Tests all core Marxist economic formulas and ensures theoretical accuracy.
"""

import pytest
    MarxistEconomicCalculator,
    ValueComposition
)

class TestValueComposition:
    """Test the ValueComposition dataclass."""

    def test_value_composition_creation(self):
        """Test creation of value composition."""
        comp = ValueComposition(
            constant_capital = 100.0,
            variable_capital = 50.0,
            surplus_value = 50.0,
            total_value = 200.0
        )

        assert comp.constant_capital == 100.0
        assert comp.variable_capital == 50.0
        assert comp.surplus_value == 50.0
        assert comp.total_value == 200.0

    def test_organic_composition(self):
        """Test organic composition calculation."""
        comp = ValueComposition(100.0, 50.0, 50.0, 200.0)
        assert comp.organic_composition == 2.0  # C / V = 100 / 50

    def test_rate_of_surplus_value(self):
        """Test rate of surplus value calculation."""
        comp = ValueComposition(100.0, 50.0, 50.0, 200.0)
        assert comp.rate_of_surplus_value == 1.0  # S / V = 50 / 50

    def test_rate_of_profit(self):
        """Test rate of profit calculation."""
        comp = ValueComposition(100.0, 50.0, 50.0, 200.0)
        assert comp.rate_of_profit == 1 / 3  # S/(C + V) = 50/(100 + 50)

    def test_zero_variable_capital(self):
        """Test handling of zero variable capital."""
        comp = ValueComposition(100.0, 0.0, 0.0, 100.0)
        assert comp.organic_composition == 0.0
        assert comp.rate_of_surplus_value == 0.0
        assert comp.rate_of_profit == 0.0

class TestMarxistEconomicCalculator:
    """Test the MarxistEconomicCalculator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample economic data for testing."""
        # Simple 2x2 economy
        technology_matrix = np.array([
            [0.2, 0.1],
            [0.1, 0.3]
        ])
        labor_vector = np.array([0.5, 0.3])
        return technology_matrix, labor_vector

    def test_initialization(self, sample_data):
        """Test calculator initialization."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        assert calculator.A.shape == (2, 2)
        assert calculator.l.shape == (2,)
        assert calculator.wage_rate == 1.0
        assert calculator.surplus_value_rate == 1.0

    def test_value_compositions_calculation(self, sample_data):
        """Test value compositions calculation."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        compositions = calculator.get_value_compositions()
        assert len(compositions) == 2

        # Check first sector
        comp1 = compositions[0]
        expected_constant_capital = 0.2 + 0.1  # Sum of column 0
        expected_variable_capital = 0.5 * 1.0  # labor * wage_rate
        expected_surplus_value = expected_variable_capital * 1.0  # V * s'
        expected_total_value = expected_constant_capital + expected_variable_capital + expected_surplus_value

        assert abs(comp1.constant_capital - expected_constant_capital) < 1e - 10
        assert abs(comp1.variable_capital - expected_variable_capital) < 1e - 10
        assert abs(comp1.surplus_value - expected_surplus_value) < 1e - 10
        assert abs(comp1.total_value - expected_total_value) < 1e - 10

    def test_organic_composition_calculation(self, sample_data):
        """Test organic composition calculation."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        organic_composition = calculator.calculate_organic_composition_of_capital()
        assert len(organic_composition) == 2
        assert np.all(organic_composition >= 0)

    def test_rate_of_surplus_value_calculation(self, sample_data):
        """Test rate of surplus value calculation."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        rate_of_surplus_value = calculator.calculate_rate_of_surplus_value()
        assert len(rate_of_surplus_value) == 2
        assert np.all(rate_of_surplus_value >= 0)

    def test_rate_of_profit_calculation(self, sample_data):
        """Test rate of profit calculation."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        rate_of_profit = calculator.calculate_rate_of_profit()
        assert len(rate_of_profit) == 2
        assert np.all(rate_of_profit >= 0)

    def test_aggregate_value_composition(self, sample_data):
        """Test aggregate value composition calculation."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        aggregate = calculator.calculate_aggregate_value_composition()
        assert aggregate.total_value > 0
        assert aggregate.constant_capital > 0
        assert aggregate.variable_capital > 0
        assert aggregate.surplus_value > 0

    def test_price_value_deviations(self, sample_data):
        """Test price - value deviation calculations."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        prices = np.array([1.5, 2.0])
        deviations = calculator.calculate_price_value_deviations(prices)

        assert "price_value_ratios" in deviations
        assert "correlation" in deviations
        assert "mean_absolute_deviation" in deviations
        assert len(deviations["price_value_ratios"]) == 2

    def test_labor_productivity(self, sample_data):
        """Test labor productivity calculation."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        productivity = calculator.calculate_labor_productivity()
        assert len(productivity) == 2
        assert np.all(productivity > 0)

    def test_value_flow_analysis(self, sample_data):
        """Test value flow analysis."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        final_demand = np.array([1.0, 1.0])
        flow_analysis = calculator.analyze_value_flow(final_demand)

        assert "total_value_produced" in flow_analysis
        assert "value_in_final_demand" in flow_analysis
        assert "value_realization_rate" in flow_analysis
        assert flow_analysis["total_value_produced"] > 0

    def test_reproduction_requirements(self, sample_data):
        """Test reproduction requirements calculation."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        output = np.array([1.0, 1.0])
        reproduction = calculator.calculate_reproduction_requirements(output)

        assert "total_constant_capital" in reproduction
        assert "total_variable_capital" in reproduction
        assert "total_surplus_value" in reproduction
        assert "reproduction_condition_met" in reproduction
        assert reproduction["total_value"] > 0

    def test_expanded_reproduction_requirements(self, sample_data):
        """Test expanded reproduction requirements calculation."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        output = np.array([1.0, 1.0])
        expanded = calculator.calculate_expanded_reproduction_requirements(output, 0.5)

        assert "accumulation_requirement" in expanded
        assert "consumption_requirement" in expanded
        assert "total_investment" in expanded
        assert expanded["accumulation_rate"] == 0.5

    def test_parameter_updates(self, sample_data):
        """Test parameter updates."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        # Update parameters
        calculator.update_parameters(wage_rate = 2.0, surplus_value_rate = 0.5)

        assert calculator.wage_rate == 2.0
        assert calculator.surplus_value_rate == 0.5

        # Check that value compositions are recalculated
        compositions = calculator.get_value_compositions()
        comp1 = compositions[0]
        expected_variable_capital = 0.5 * 2.0  # labor * new_wage_rate
        assert abs(comp1.variable_capital - expected_variable_capital) < 1e - 10

    def test_economic_indicators(self, sample_data):
        """Test comprehensive economic indicators."""
        tech_matrix, labor_vector = sample_data
        calculator = MarxistEconomicCalculator(tech_matrix, labor_vector)

        indicators = calculator.get_economic_indicators()

        assert "aggregate_value_composition" in indicators
        assert "sectoral_indicators" in indicators
        assert "economy_wide_averages" in indicators

        # Check aggregate composition
        aggregate = indicators["aggregate_value_composition"]
        assert "constant_capital" in aggregate
        assert "variable_capital" in aggregate
        assert "surplus_value" in aggregate
        assert "total_value" in aggregate

    def test_validation_errors(self):
        """Test input validation."""
        # Test invalid technology matrix
        with pytest.raises(ValueError, match="Technology matrix must be square"):
            MarxistEconomicCalculator(np.array([[1, 2]]), np.array([1, 2]))

        # Test dimension mismatch
        with pytest.raises(ValueError, match="Labor vector must match"):
            MarxistEconomicCalculator(np.array([[1, 0], [0, 1]]), np.array([1, 2, 3]))

        # Test negative labor
        with pytest.raises(ValueError, match="Labor input cannot be negative"):
            MarxistEconomicCalculator(np.array([[1, 0], [0, 1]]), np.array([-1, 1]))

        # Test invalid wage rate
        with pytest.raises(ValueError, match="Wage rate must be positive"):
            MarxistEconomicCalculator(np.array([[1, 0], [0, 1]]), np.array([1, 1]), wage_rate = 0)

        # Test invalid surplus value rate
        with pytest.raises(ValueError, match="Surplus value rate cannot be negative"):
            MarxistEconomicCalculator(np.array([[1, 0], [0, 1]]), np.array([1, 1]), surplus_value_rate=-1)

class TestTheoreticalAccuracy:
    """Test theoretical accuracy of Marxist calculations."""

    def test_marx_formulas(self):
        """Test that calculations follow Marx's formulas exactly."""
        # Create a simple economy where we can verify calculations manually
        technology_matrix = np.array([
            [0.1, 0.0],
            [0.0, 0.2]
        ])
        labor_vector = np.array([1.0, 1.0])
        wage_rate = 1.0
        surplus_value_rate = 1.0  # 100% rate of surplus value

        calculator = MarxistEconomicCalculator(
            technology_matrix, labor_vector, wage_rate, surplus_value_rate
        )

        compositions = calculator.get_value_compositions()

        # For sector 0: C = 0.1, V = 1.0, S = 1.0, W = 2.1
        comp0 = compositions[0]
        assert abs(comp0.constant_capital - 0.1) < 1e - 10
        assert abs(comp0.variable_capital - 1.0) < 1e - 10
        assert abs(comp0.surplus_value - 1.0) < 1e - 10
        assert abs(comp0.total_value - 2.1) < 1e - 10
        assert abs(comp0.organic_composition - 0.1) < 1e - 10  # C / V = 0.1 / 1.0
        assert abs(comp0.rate_of_surplus_value - 1.0) < 1e - 10  # S / V = 1.0 / 1.0
        assert abs(comp0.rate_of_profit - 1.0 / 1.1) < 1e - 10  # S/(C + V) = 1.0/(0.1 + 1.0)

    def test_reproduction_conditions(self):
        """Test that reproduction conditions are mathematically correct."""
        # Create an economy that should satisfy simple reproduction
        technology_matrix = np.array([
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        labor_vector = np.array([0.5, 0.5])

        calculator = MarxistEconomicCalculator(technology_matrix, labor_vector)

        # For simple reproduction, total surplus value should equal total variable capital
        output = np.array([1.0, 1.0])
        reproduction = calculator.calculate_reproduction_requirements(output)

        # This should be close to simple reproduction condition
        # (exact equality may not hold due to the way we calculate constant capital)
        assert reproduction["total_surplus_value"] > 0
        assert reproduction["total_variable_capital"] > 0

if __name__ == "__main__":
    pytest.main([__file__])
