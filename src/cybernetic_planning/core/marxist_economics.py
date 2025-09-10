"""
Enhanced Marxist Economic Calculations

Implements comprehensive Marxist economic theory including:
- C + V+S value composition formulas - Rate of surplus value and rate of profit calculations - Organic composition of capital - Simple and expanded reproduction schemas - Price - value transformation problem - Departmental balance analysis

Based on Marx's Capital Volumes 1, 2, and 3.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass

@dataclass
class ValueComposition:
    """Represents the value composition of a commodity (C + V+S)."""
    constant_capital: float  # C
    variable_capital: float  # V
    surplus_value: float     # S
    total_value: float       # C + V+S

    @property
    def organic_composition(self) -> float:
        """Organic composition of capital: C / V"""
        return self.constant_capital / self.variable_capital if self.variable_capital > 0 else 0

    @property
    def rate_of_surplus_value(self) -> float:
        """Rate of surplus value: S / V"""
        return self.surplus_value / self.variable_capital if self.variable_capital > 0 else 0

    @property
    def rate_of_profit(self) -> float:
        """Rate of profit: S/(C + V)"""
        return self.surplus_value / (self.constant_capital + self.variable_capital) if (self.constant_capital + self.variable_capital) > 0 else 0

class MarxistEconomicCalculator:
    """
    Comprehensive Marxist economic calculations based on Capital.

    Implements the core formulas from Marx's economic theory:
    - Value composition: W = C + V + S - Rate of surplus value: s' = S / V - Rate of profit: p' = S/(C + V)
    - Organic composition: OCC = C / V - Reproduction conditions
    """

    def __init__(
        self,
        technology_matrix: np.ndarray,
        labor_vector: np.ndarray,
        wage_rate: float = 1.0,
        surplus_value_rate: float = 1.0
    ):
        """
        Initialize the Marxist economic calculator.

        Args:
            technology_matrix: Technology matrix A (input coefficients)
            labor_vector: Direct labor input vector l
            wage_rate: Wage rate per unit labor (default: 1.0)
            surplus_value_rate: Rate of surplus value s' (default: 1.0 = 100%)
        """
        self.A = np.asarray(technology_matrix)
        self.l = np.asarray(labor_vector).flatten()
        self.wage_rate = wage_rate
        self.surplus_value_rate = surplus_value_rate

        # Validate inputs
        self._validate_inputs()

        # Calculate value compositions
        self._value_compositions = None
        self._calculate_value_compositions()

    def _validate_inputs(self) -> None:
        """Validate input matrices and parameters."""
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Technology matrix must be square")

        if self.l.shape[0] != self.A.shape[0]:
            raise ValueError("Labor vector must match technology matrix dimensions")

        if np.any(self.l < 0):
            raise ValueError("Labor input cannot be negative")

        if self.wage_rate <= 0:
            raise ValueError("Wage rate must be positive")

        if self.surplus_value_rate < 0:
            raise ValueError("Surplus value rate cannot be negative")

    def _calculate_value_compositions(self) -> None:
        """Calculate C + V+S composition for each sector."""
        n_sectors = self.A.shape[0]
        self._value_compositions = []

        for i in range(n_sectors):
            # Constant capital C: value of means of production used
            constant_capital = np.sum(self.A[:, i])

            # Variable capital V: wages paid to workers
            variable_capital = self.l[i] * self.wage_rate

            # Surplus value S: surplus value rate * variable capital
            surplus_value = variable_capital * self.surplus_value_rate

            # Total value W = C + V + S
            total_value = constant_capital + variable_capital + surplus_value

            composition = ValueComposition(
                constant_capital = constant_capital,
                variable_capital = variable_capital,
                surplus_value = surplus_value,
                total_value = total_value
            )

            self._value_compositions.append(composition)

    def get_value_compositions(self) -> List[ValueComposition]:
        """Get value compositions for all sectors."""
        return self._value_compositions.copy()

    def get_sector_value_composition(self, sector_index: int) -> ValueComposition:
        """Get value composition for a specific sector."""
        if sector_index < 0 or sector_index >= len(self._value_compositions):
            raise ValueError(f"Invalid sector index: {sector_index}")

        return self._value_compositions[sector_index]

    def calculate_aggregate_value_composition(self) -> ValueComposition:
        """Calculate aggregate value composition for the entire economy."""
        total_constant_capital = sum(comp.constant_capital for comp in self._value_compositions)
        total_variable_capital = sum(comp.variable_capital for comp in self._value_compositions)
        total_surplus_value = sum(comp.surplus_value for comp in self._value_compositions)
        total_value = sum(comp.total_value for comp in self._value_compositions)

        return ValueComposition(
            constant_capital = total_constant_capital,
            variable_capital = total_variable_capital,
            surplus_value = total_surplus_value,
            total_value = total_value
        )

    def calculate_organic_composition_of_capital(self) -> np.ndarray:
        """Calculate organic composition of capital for each sector."""
        return np.array([comp.organic_composition for comp in self._value_compositions])

    def calculate_rate_of_surplus_value(self) -> np.ndarray:
        """Calculate rate of surplus value for each sector."""
        return np.array([comp.rate_of_surplus_value for comp in self._value_compositions])

    def calculate_rate_of_profit(self) -> np.ndarray:
        """Calculate rate of profit for each sector."""
        return np.array([comp.rate_of_profit for comp in self._value_compositions])

    def calculate_average_rate_of_profit(self) -> float:
        """Calculate economy - wide average rate of profit."""
        aggregate = self.calculate_aggregate_value_composition()
        return aggregate.rate_of_profit

    def calculate_price_value_deviations(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        Calculate deviations between prices and values (transformation problem).

        Args:
            prices: Price vector for all commodities

        Returns:
            Dictionary with deviation analysis
        """
        values = np.array([comp.total_value for comp in self._value_compositions])

        # Calculate relative deviations
        price_value_ratios = prices / (values + 1e-10)  # Avoid division by zero
        mean_ratio = np.mean(price_value_ratios)
        normalized_ratios = price_value_ratios / mean_ratio

        # Calculate correlation
        correlation = np.corrcoef(prices, values)[0, 1]

        # Calculate mean absolute deviation
        mad = np.mean(np.abs(normalized_ratios - 1.0))

        return {
            "price_value_ratios": price_value_ratios,
            "normalized_ratios": normalized_ratios,
            "correlation": correlation,
            "mean_absolute_deviation": mad,
            "values": values,
            "prices": prices
        }

    def calculate_labor_productivity(self) -> np.ndarray:
        """Calculate labor productivity (output per unit labor) for each sector."""
        # Labor productivity = 1 / labor value per unit output
        labor_values = np.array([comp.total_value for comp in self._value_compositions])
        return 1.0 / (labor_values + 1e-10)  # Avoid division by zero

    def calculate_technical_composition_of_capital(self) -> np.ndarray:
        """Calculate technical composition of capital (means of production per worker)."""
        # TCC = constant capital / labor input
        constant_capitals = np.array([comp.constant_capital for comp in self._value_compositions])
        return constant_capitals / (self.l + 1e-10)  # Avoid division by zero

    def analyze_value_flow(self, final_demand: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the flow of value through the economy.

        Args:
            final_demand: Final demand vector

        Returns:
            Dictionary with value flow analysis
        """
        # Calculate total value produced
        total_value_produced = sum(comp.total_value for comp in self._value_compositions)

        # Calculate value embodied in final demand
        values = np.array([comp.total_value for comp in self._value_compositions])
        value_in_final_demand = np.dot(values, final_demand)

        # Calculate value in intermediate goods
        intermediate_demand = self.A @ final_demand
        value_in_intermediate = np.dot(values, intermediate_demand)

        # Calculate surplus value realization
        total_surplus_value = sum(comp.surplus_value for comp in self._value_compositions)
        surplus_value_realized = (value_in_final_demand / total_value_produced) * total_surplus_value

        return {
            "total_value_produced": total_value_produced,
            "value_in_final_demand": value_in_final_demand,
            "value_in_intermediate": value_in_intermediate,
            "surplus_value_realized": surplus_value_realized,
            "value_realization_rate": value_in_final_demand / total_value_produced,
            "surplus_realization_rate": surplus_value_realized / total_surplus_value
        }

    def calculate_reproduction_requirements(self, output: np.ndarray) -> Dict[str, Any]:
        """
        Calculate requirements for simple reproduction.

        Args:
            output: Output vector for each sector

        Returns:
            Dictionary with reproduction analysis
        """
        # Calculate total constant capital requirements
        total_constant_capital = np.sum(self.A @ output)

        # Calculate total variable capital (wages)
        total_variable_capital = np.sum(self.l * output * self.wage_rate)

        # Calculate total surplus value
        total_surplus_value = total_variable_capital * self.surplus_value_rate

        # Calculate total value
        total_value = total_constant_capital + total_variable_capital + total_surplus_value

        # Check reproduction conditions
        # For simple reproduction: total surplus value = total consumption
        reproduction_condition = abs(total_surplus_value - total_variable_capital) < 1e-6

        return {
            "total_constant_capital": total_constant_capital,
            "total_variable_capital": total_variable_capital,
            "total_surplus_value": total_surplus_value,
            "total_value": total_value,
            "reproduction_condition_met": reproduction_condition,
            "surplus_consumption_ratio": total_surplus_value / total_variable_capital if total_variable_capital > 0 else 0
        }

    def calculate_expanded_reproduction_requirements(
        self,
        output: np.ndarray,
        accumulation_rate: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate requirements for expanded reproduction.

        Args:
            output: Output vector for each sector
            accumulation_rate: Rate of surplus value accumulation (default: 50%)

        Returns:
            Dictionary with expanded reproduction analysis
        """
        # Get simple reproduction requirements
        simple_reproduction = self.calculate_reproduction_requirements(output)

        # Calculate accumulation requirements
        total_surplus_value = simple_reproduction["total_surplus_value"]
        accumulation_requirement = total_surplus_value * accumulation_rate
        consumption_requirement = total_surplus_value * (1 - accumulation_rate)

        # Calculate investment in constant capital
        # Assume 70% of accumulation goes to constant capital, 30% to variable capital
        constant_capital_investment = accumulation_requirement * 0.7
        variable_capital_investment = accumulation_requirement * 0.3

        # Calculate new output requirements
        # Additional constant capital requirements
        additional_constant_capital = constant_capital_investment / np.mean([comp.constant_capital for comp in self._value_compositions])

        # Additional variable capital requirements
        additional_variable_capital = variable_capital_investment / (self.wage_rate * np.mean(self.l))

        return {
            "simple_reproduction": simple_reproduction,
            "accumulation_rate": accumulation_rate,
            "accumulation_requirement": accumulation_requirement,
            "consumption_requirement": consumption_requirement,
            "constant_capital_investment": constant_capital_investment,
            "variable_capital_investment": variable_capital_investment,
            "additional_constant_capital": additional_constant_capital,
            "additional_variable_capital": additional_variable_capital,
            "total_investment": constant_capital_investment + variable_capital_investment
        }

    def update_parameters(
        self,
        wage_rate: Optional[float] = None,
        surplus_value_rate: Optional[float] = None
    ) -> None:
        """Update calculation parameters and recalculate."""
        if wage_rate is not None:
            self.wage_rate = wage_rate
        if surplus_value_rate is not None:
            self.surplus_value_rate = surplus_value_rate

        # Recalculate value compositions
        self._calculate_value_compositions()

    def get_economic_indicators(self) -> Dict[str, Any]:
        """Get comprehensive economic indicators."""
        aggregate = self.calculate_aggregate_value_composition()
        organic_composition = self.calculate_organic_composition_of_capital()
        rate_of_surplus_value = self.calculate_rate_of_surplus_value()
        rate_of_profit = self.calculate_rate_of_profit()

        return {
            "aggregate_value_composition": {
                "constant_capital": aggregate.constant_capital,
                "variable_capital": aggregate.variable_capital,
                "surplus_value": aggregate.surplus_value,
                "total_value": aggregate.total_value,
                "organic_composition": aggregate.organic_composition,
                "rate_of_surplus_value": aggregate.rate_of_surplus_value,
                "rate_of_profit": aggregate.rate_of_profit
            },
            "sectoral_indicators": {
                "organic_composition": organic_composition.tolist(),
                "rate_of_surplus_value": rate_of_surplus_value.tolist(),
                "rate_of_profit": rate_of_profit.tolist()
            },
            "economy_wide_averages": {
                "average_organic_composition": np.mean(organic_composition),
                "average_rate_of_surplus_value": np.mean(rate_of_surplus_value),
                "average_rate_of_profit": np.mean(rate_of_profit)
            }
        }
