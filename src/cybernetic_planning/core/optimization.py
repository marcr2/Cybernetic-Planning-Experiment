"""
Constrained Optimization Implementation

Implements linear programming optimization to minimize labor time
subject to demand fulfillment and resource constraints.
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Dict, Any
from scipy.optimize import linprog
import warnings


class ConstrainedOptimizer:
    """
    Solves constrained optimization problems for economic planning.

    Minimizes total labor cost subject to:
    - Demand fulfillment: (I - A)x >= d
    - Resource constraints: Rx <= R_max
    - Non-negativity: x >= 0
    """

    def __init__(
        self,
        technology_matrix: np.ndarray,
        direct_labor: np.ndarray,
        final_demand: np.ndarray,
        resource_matrix: Optional[np.ndarray] = None,
        max_resources: Optional[np.ndarray] = None,
    ):
        """
        Initialize the constrained optimizer.

        Args:
            technology_matrix: Technology matrix A
            direct_labor: Direct labor input vector l
            final_demand: Final demand vector d
            resource_matrix: Resource constraint matrix R (optional)
            max_resources: Maximum resource availability R_max (optional)
        """
        self.A = np.asarray(technology_matrix)
        self.l = np.asarray(direct_labor).flatten()
        self.d = np.asarray(final_demand).flatten()
        self.R = resource_matrix
        self.R_max = max_resources

        # Validate inputs
        self._validate_inputs()

        # Problem variables
        self._problem = None
        self._solution = None
        self._status = None

    def _validate_inputs(self) -> None:
        """Validate input matrices and vectors."""
        n = self.A.shape[0]

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Technology matrix must be square")

        if self.l.shape[0] != n or self.d.shape[0] != n:
            raise ValueError("All vectors must have same dimension as technology matrix")

        if self.R is not None and self.R_max is not None:
            if self.R.shape[1] != n:
                raise ValueError("Resource matrix must have same number of columns as technology matrix")
            if self.R.shape[0] != self.R_max.shape[0]:
                raise ValueError("Resource matrix and max resources must have compatible dimensions")

    def _create_problem(self, use_cvxpy: bool = True) -> None:
        """
        Create the optimization problem.

        Args:
            use_cvxpy: Whether to use CVXPY (True) or scipy.optimize (False)
        """
        self.A.shape[0]

        if use_cvxpy:
            self._create_cvxpy_problem()
        else:
            self._create_scipy_problem()

    def _create_cvxpy_problem(self) -> None:
        """Create CVXPY optimization problem."""
        n = self.A.shape[0]

        # Decision variable: total output vector
        x = cp.Variable(n, nonneg=True)

        # Objective: minimize total labor cost
        objective = cp.Minimize(self.l @ x)

        # Constraints
        constraints = []

        # Demand fulfillment: (I - A)x = d (exact fulfillment as per Cockshott)
        I = np.eye(n)
        constraints.append((I - self.A) @ x == self.d)

        # Resource constraints: Rx <= R_max
        if self.R is not None and self.R_max is not None:
            constraints.append(self.R @ x <= self.R_max)

        # Create problem
        self._problem = cp.Problem(objective, constraints)

    def _create_scipy_problem(self) -> None:
        """Create scipy.optimize problem (for reference)."""
        n = self.A.shape[0]

        # Objective: minimize total labor cost
        c = self.l

        # Constraints: (I - A)x = d, so we need equality constraints
        I = np.eye(n)
        A_eq = (I - self.A)
        b_eq = self.d

        # Add resource constraints if provided
        A_ub = None
        b_ub = None
        if self.R is not None and self.R_max is not None:
            A_ub = self.R
            b_ub = self.R_max

        # Bounds: x >= 0
        bounds = [(0, None) for _ in range(n)]

        # Store problem parameters
        self._scipy_params = {
            "c": c,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "bounds": bounds,
            "method": "highs",  # Use HiGHS solver
        }

    def solve(self, use_cvxpy: bool = True, solver: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve the optimization problem.

        Args:
            use_cvxpy: Whether to use CVXPY (True) or scipy.optimize (False)
            solver: CVXPY solver to use (if use_cvxpy=True)

        Returns:
            Dictionary with solution information
        """
        self._create_problem(use_cvxpy)

        if use_cvxpy:
            return self._solve_cvxpy(solver)
        else:
            return self._solve_scipy()

    def _solve_cvxpy(self, solver: Optional[str] = None) -> Dict[str, Any]:
        """Solve using CVXPY."""
        if solver is None:
            solver = cp.ECOS  # Default solver

        try:
            self._problem.solve(solver=solver)
            self._status = self._problem.status

            if self._status == cp.OPTIMAL:
                self._solution = self._problem.variables()[0].value
                return {
                    "status": "optimal",
                    "solution": self._solution,
                    "objective_value": self._problem.value,
                    "total_labor_cost": self._problem.value,
                    "feasible": True,
                }
            elif self._status == cp.INFEASIBLE:
                return {
                    "status": "infeasible",
                    "solution": None,
                    "objective_value": None,
                    "total_labor_cost": None,
                    "feasible": False,
                }
            else:
                return {
                    "status": "unknown",
                    "solution": None,
                    "objective_value": None,
                    "total_labor_cost": None,
                    "feasible": False,
                }

        except Exception as e:
            warnings.warn(f"CVXPY solver failed: {e}")
            return {
                "status": "error",
                "solution": None,
                "objective_value": None,
                "total_labor_cost": None,
                "feasible": False,
                "error": str(e),
            }

    def _solve_scipy(self) -> Dict[str, Any]:
        """Solve using scipy.optimize."""
        try:
            result = linprog(**self._scipy_params)

            if result.success:
                self._solution = result.x
                return {
                    "status": "optimal",
                    "solution": self._solution,
                    "objective_value": result.fun,
                    "total_labor_cost": result.fun,
                    "feasible": True,
                }
            else:
                return {
                    "status": "failed",
                    "solution": None,
                    "objective_value": None,
                    "total_labor_cost": None,
                    "feasible": False,
                    "message": result.message,
                }

        except Exception as e:
            warnings.warn(f"SciPy solver failed: {e}")
            return {
                "status": "error",
                "solution": None,
                "objective_value": None,
                "total_labor_cost": None,
                "feasible": False,
                "error": str(e),
            }

    def get_solution(self) -> Optional[np.ndarray]:
        """Get the optimal solution vector."""
        return self._solution.copy() if self._solution is not None else None

    def get_objective_value(self) -> Optional[float]:
        """Get the optimal objective value (total labor cost)."""
        if self._problem is not None and self._status == cp.OPTIMAL:
            return self._problem.value
        return None

    def check_constraints(self, solution: np.ndarray) -> Dict[str, Any]:
        """
        Check constraint satisfaction for a given solution.

        Args:
            solution: Solution vector to check

        Returns:
            Dictionary with constraint satisfaction information
        """
        n = self.A.shape[0]
        I = np.eye(n)

        # Check demand fulfillment
        demand_satisfaction = (I - self.A) @ solution
        demand_violations = np.maximum(0, self.d - demand_satisfaction)

        # Check resource constraints
        resource_violations = None
        if self.R is not None and self.R_max is not None:
            resource_usage = self.R @ solution
            resource_violations = np.maximum(0, resource_usage - self.R_max)

        # Check non-negativity
        negativity_violations = np.maximum(0, -solution)

        return {
            "demand_satisfied": np.allclose(demand_violations, 0, atol=1e-6),
            "demand_violations": demand_violations,
            "resource_satisfied": resource_violations is None or np.allclose(resource_violations, 0, atol=1e-6),
            "resource_violations": resource_violations,
            "non_negative": np.allclose(negativity_violations, 0, atol=1e-6),
            "negativity_violations": negativity_violations,
            "all_constraints_satisfied": (
                np.allclose(demand_violations, 0, atol=1e-6)
                and (resource_violations is None or np.allclose(resource_violations, 0, atol=1e-6))
                and np.allclose(negativity_violations, 0, atol=1e-6)
            ),
        }

