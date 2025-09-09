"""
Dynamic Planning Implementation

Implements dynamic planning for years 2-5 with capital accumulation
and technological change.
"""

import numpy as np
from typing import Dict, List, Optional
import warnings


class DynamicPlanner:
    """
    Handles dynamic planning for multi-year economic plans.

    Implements capital accumulation and technological change over time:
    - x_t = A_t x_t + d_{c,t} + d_{i,t}
    - K_{t+1} = (I - δ)K_t + M d_{i,t}
    """

    def __init__(
        self,
        initial_technology_matrix: np.ndarray,
        initial_labor_vector: np.ndarray,
        initial_capital_stock: Optional[np.ndarray] = None,
        depreciation_rates: Optional[np.ndarray] = None,
        capital_formation_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initialize the dynamic planner.

        Args:
            initial_technology_matrix: Initial technology matrix A_0
            initial_labor_vector: Initial labor vector l_0
            initial_capital_stock: Initial capital stock K_0
            depreciation_rates: Depreciation rate matrix δ
            capital_formation_matrix: Capital formation matrix M
        """
        self.A_0 = np.asarray(initial_technology_matrix)
        self.l_0 = np.asarray(initial_labor_vector).flatten()
        self.n_sectors = self.A_0.shape[0]

        # Initialize capital stock
        if initial_capital_stock is None:
            self.K_0 = np.zeros(self.n_sectors)
        else:
            self.K_0 = np.asarray(initial_capital_stock).flatten()

        # Initialize depreciation rates
        if depreciation_rates is None:
            self.depreciation_rates = np.full(self.n_sectors, 0.1)  # 10% annual depreciation
        else:
            self.depreciation_rates = np.asarray(depreciation_rates).flatten()

        # Initialize capital formation matrix
        if capital_formation_matrix is None:
            self.M = np.eye(self.n_sectors)  # Identity matrix as default
        else:
            self.M = np.asarray(capital_formation_matrix)

        # Validate inputs
        self._validate_inputs()

        # Storage for dynamic variables
        self.technology_matrices = {0: self.A_0.copy()}
        self.labor_vectors = {0: self.l_0.copy()}
        self.capital_stocks = {0: self.K_0.copy()}
        self.plans = {}

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.A_0.ndim != 2 or self.A_0.shape[0] != self.A_0.shape[1]:
            raise ValueError("Technology matrix must be square")

        if self.l_0.shape[0] != self.n_sectors:
            raise ValueError("Labor vector must have same dimension as technology matrix")

        if self.K_0.shape[0] != self.n_sectors:
            raise ValueError("Capital stock must have same dimension as technology matrix")

        if self.depreciation_rates.shape[0] != self.n_sectors:
            raise ValueError("Depreciation rates must have same dimension as technology matrix")

        if self.M.shape != (self.n_sectors, self.n_sectors):
            raise ValueError("Capital formation matrix must be square with same dimension as technology matrix")

    def update_technology(
        self,
        year: int,
        new_technology_matrix: Optional[np.ndarray] = None,
        new_labor_vector: Optional[np.ndarray] = None,
        technological_change_rate: float = 0.02,
    ) -> None:
        """
        Update technology matrix and labor vector for a given year.

        Args:
            year: Year to update
            new_technology_matrix: New technology matrix (if None, applies technological change)
            new_labor_vector: New labor vector (if None, applies productivity growth)
            technological_change_rate: Rate of technological change (default 2% per year)
        """
        if year < 1:
            raise ValueError("Year must be >= 1")

        # Update technology matrix
        if new_technology_matrix is not None:
            self.technology_matrices[year] = np.asarray(new_technology_matrix)
        else:
            # Apply technological change (reduce input coefficients)
            prev_year = max([y for y in self.technology_matrices.keys() if y < year])
            prev_A = self.technology_matrices[prev_year]

            # Reduce input coefficients by technological change rate
            years_diff = year - prev_year
            change_factor = (1 - technological_change_rate) ** years_diff
            self.technology_matrices[year] = prev_A * change_factor

        # Update labor vector
        if new_labor_vector is not None:
            self.labor_vectors[year] = np.asarray(new_labor_vector).flatten()
        else:
            # Apply productivity growth (reduce labor requirements)
            prev_year = max([y for y in self.labor_vectors.keys() if y < year])
            prev_l = self.labor_vectors[prev_year]

            # Reduce labor requirements by productivity growth
            years_diff = year - prev_year
            change_factor = (1 - technological_change_rate) ** years_diff
            self.labor_vectors[year] = prev_l * change_factor

    def update_capital_stock(
        self, year: int, investment_vector: np.ndarray, new_capital_stock: Optional[np.ndarray] = None
    ) -> None:
        """
        Update capital stock based on investment.

        Args:
            year: Year to update
            investment_vector: Investment vector d_i
            new_capital_stock: New capital stock (if None, calculates from investment)
        """
        if year < 1:
            raise ValueError("Year must be >= 1")

        if new_capital_stock is not None:
            self.capital_stocks[year] = np.asarray(new_capital_stock).flatten()
        else:
            # Calculate new capital stock: K_t = (I - δ)K_{t-1} + M d_i
            prev_year = year - 1
            if prev_year not in self.capital_stocks:
                raise ValueError(f"Capital stock for year {prev_year} not available")

            prev_K = self.capital_stocks[prev_year]
            I = np.eye(self.n_sectors)
            delta = np.diag(self.depreciation_rates)

            new_K = (I - delta) @ prev_K + self.M @ investment_vector
            self.capital_stocks[year] = new_K

    def plan_year(
        self, year: int, consumption_demand: np.ndarray, investment_demand: np.ndarray, use_optimization: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Create economic plan for a specific year.

        Args:
            year: Year to plan
            consumption_demand: Consumption demand vector d_c
            investment_demand: Investment demand vector d_i
            use_optimization: Whether to use constrained optimization

        Returns:
            Dictionary with plan results
        """
        if year not in self.technology_matrices:
            self.update_technology(year)

        A_t = self.technology_matrices[year]
        l_t = self.labor_vectors[year]
        d_t = consumption_demand + investment_demand

        if use_optimization:
            # Use constrained optimization
            from .optimization import ConstrainedOptimizer

            optimizer = ConstrainedOptimizer(technology_matrix=A_t, direct_labor=l_t, final_demand=d_t)

            result = optimizer.solve()

            if result["feasible"]:
                total_output = result["solution"]
                total_labor_cost = result["total_labor_cost"]
            else:
                warnings.warn(f"Optimization failed for year {year}, using Leontief solution")
                from .leontief import LeontiefModel

                leontief = LeontiefModel(A_t, d_t)
                total_output = leontief.compute_total_output()
                total_labor_cost = np.dot(l_t, total_output)
        else:
            # Use simple Leontief model
            from .leontief import LeontiefModel

            leontief = LeontiefModel(A_t, d_t)
            total_output = leontief.compute_total_output()
            total_labor_cost = np.dot(l_t, total_output)

        # Update capital stock if investment is provided
        if np.any(investment_demand > 0):
            self.update_capital_stock(year, investment_demand)

        # Calculate labor values for this year
        from .labor_values import LaborValueCalculator
        labor_calc = LaborValueCalculator(A_t, l_t)
        labor_values = labor_calc.get_labor_values()

        # Store plan
        self.plans[year] = {
            "total_output": total_output,
            "consumption_demand": consumption_demand,
            "investment_demand": investment_demand,
            "total_demand": d_t,
            "total_labor_cost": total_labor_cost,
            "technology_matrix": A_t,
            "labor_vector": l_t,
            "labor_values": labor_values,
            "capital_stock": self.capital_stocks.get(year, np.zeros(self.n_sectors)),
        }

        return self.plans[year]

    def create_five_year_plan(
        self, consumption_demands: List[np.ndarray], investment_demands: List[np.ndarray], use_optimization: bool = True
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Create a complete 5-year economic plan.

        Args:
            consumption_demands: List of consumption demand vectors for years 1-5
            investment_demands: List of investment demand vectors for years 1-5
            use_optimization: Whether to use constrained optimization

        Returns:
            Dictionary with plans for each year
        """
        if len(consumption_demands) != 5 or len(investment_demands) != 5:
            raise ValueError("Must provide exactly 5 years of demand data")

        plans = {}

        for year in range(1, 6):
            plan = self.plan_year(
                year=year,
                consumption_demand=consumption_demands[year - 1],
                investment_demand=investment_demands[year - 1],
                use_optimization=use_optimization,
            )
            plans[year] = plan

        return plans

    def get_technology_matrix(self, year: int) -> np.ndarray:
        """Get technology matrix for a specific year."""
        if year not in self.technology_matrices:
            self.update_technology(year)
        return self.technology_matrices[year].copy()

    def get_labor_vector(self, year: int) -> np.ndarray:
        """Get labor vector for a specific year."""
        if year not in self.labor_vectors:
            self.update_technology(year)
        return self.labor_vectors[year].copy()

    def get_capital_stock(self, year: int) -> np.ndarray:
        """Get capital stock for a specific year."""
        return self.capital_stocks.get(year, np.zeros(self.n_sectors)).copy()

    def get_plan(self, year: int) -> Optional[Dict[str, np.ndarray]]:
        """Get plan for a specific year."""
        return self.plans.get(year)

    def calculate_growth_rates(self) -> Dict[str, np.ndarray]:
        """
        Calculate growth rates for key economic indicators.

        Returns:
            Dictionary with growth rates for output, labor, and capital
        """
        if len(self.plans) < 2:
            return {}

        years = sorted(self.plans.keys())
        growth_rates = {}

        for i in range(1, len(years)):
            year = years[i]
            prev_year = years[i - 1]

            # Output growth
            output_growth = self.plans[year]["total_output"] / self.plans[prev_year]["total_output"] - 1

            # Labor cost growth
            labor_growth = self.plans[year]["total_labor_cost"] / self.plans[prev_year]["total_labor_cost"] - 1

            # Capital stock growth
            capital_growth = self.plans[year]["capital_stock"] / self.plans[prev_year]["capital_stock"] - 1

            growth_rates[f"{prev_year}-{year}"] = {
                "output_growth": output_growth,
                "labor_growth": labor_growth,
                "capital_growth": capital_growth,
            }

        return growth_rates
