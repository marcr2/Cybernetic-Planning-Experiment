"""
Constrained Optimization Implementation

Implements linear programming optimization to minimize labor time
subject to demand fulfillment and resource constraints.
"""

from typing import Optional, Dict, Any
import numpy as np
from scipy.optimize import linprog
import warnings
import cvxpy as cp

class ConstrainedOptimizer:
    """
    Solves constrained optimization problems for economic planning.

    Minimizes total labor cost subject to:
    - Demand fulfillment: (I - A)x >= d - Resource constraints: Rx <= R_max - Non - negativity: x >= 0
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
        
        # Check for invalid values
        if np.any(np.isnan(self.A)) or np.any(np.isinf(self.A)):
            raise ValueError("Technology matrix contains NaN or infinite values")
        
        if np.any(np.isnan(self.l)) or np.any(np.isinf(self.l)):
            raise ValueError("Labor vector contains NaN or infinite values")
            
        if np.any(np.isnan(self.d)) or np.any(np.isinf(self.d)):
            raise ValueError("Final demand vector contains NaN or infinite values")
        
        # Check for negative values where they shouldn't exist
        if np.any(self.l < 0):
            raise ValueError("Labor vector contains negative values")
            
        if np.any(self.d < 0):
            raise ValueError("Final demand contains negative values")
        
        # Check if economy is productive (spectral radius < 1)
        try:
            eigenvals = np.linalg.eigvals(self.A)
            spectral_radius = np.max(np.abs(eigenvals))
            if spectral_radius >= 1.0:
                warnings.warn(f"Economy may be non-productive (spectral radius = {spectral_radius:.4f})")
        except np.linalg.LinAlgError:
            warnings.warn("Could not compute spectral radius - matrix may be ill-conditioned")
        
        if self.R is not None and self.R_max is not None:
            if self.R.shape[1] != n:
                raise ValueError("Resource matrix must have same number of columns as technology matrix")
            if self.R.shape[0] != self.R_max.shape[0]:
                raise ValueError("Resource matrix and max resources must have compatible dimensions")
            
            # Check resource matrix for invalid values
            if np.any(np.isnan(self.R)) or np.any(np.isinf(self.R)):
                raise ValueError("Resource matrix contains NaN or infinite values")
            if np.any(np.isnan(self.R_max)) or np.any(np.isinf(self.R_max)):
                raise ValueError("Max resources contains NaN or infinite values")

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
        x = cp.Variable(n, nonneg = True)

        # Objective: minimize total labor cost
        objective = cp.Minimize(self.l @ x)

        # Constraints
        constraints = []

        # Demand fulfillment: (I - A)x >= d (meet or exceed demand as per cybernetic principles)
        # This allows for surplus production and proper feedback loops
        I = np.eye(n)
        constraints.append((I - self.A) @ x >= self.d)

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

        # Constraints: (I - A)x >= d, so we need inequality constraints
        I = np.eye(n)
        A_ub = -(I - self.A)  # Negative because scipy uses A_ub @ x <= b_ub
        b_ub = -self.d

        # Add resource constraints if provided
        if self.R is not None and self.R_max is not None:
            # Combine demand and resource constraints
            A_ub_combined = np.vstack([A_ub, self.R])
            b_ub_combined = np.hstack([b_ub, self.R_max])
        else:
            A_ub_combined = A_ub
            b_ub_combined = b_ub

        # Bounds: x >= 0
        bounds = [(0, None) for _ in range(n)]

        # Store problem parameters
        self._scipy_params = {
            "c": c,
            "A_ub": A_ub_combined,
            "b_ub": b_ub_combined,
            "bounds": bounds,
            "method": "highs",  # Use HiGHS solver
        }

    def solve(self, use_cvxpy: bool = True, solver: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve the optimization problem.

        Args:
            use_cvxpy: Whether to use CVXPY (True) or scipy.optimize (False)
            solver: CVXPY solver to use (if use_cvxpy = True)

        Returns:
            Dictionary with solution information
        """
        self._create_problem(use_cvxpy)

        if use_cvxpy:
            return self._solve_cvxpy(solver)
        else:
            return self._solve_scipy()

    def _solve_cvxpy(self, solver: Optional[str] = None) -> Dict[str, Any]:
        """Solve using CVXPY with fallback to multiple solvers."""
        # Try multiple solvers in order of preference
        solvers_to_try = []
        if solver is not None:
            solvers_to_try = [solver]
        else:
            # Try solvers in order of preference
            solvers_to_try = [cp.ECOS, cp.SCS, cp.OSQP, cp.CLARABEL]
        
        last_error = None
        for solver_name in solvers_to_try:
            try:
                self._problem.solve(solver=solver_name)
                self._status = self._problem.status

                if self._status == cp.OPTIMAL:
                    self._solution = self._problem.variables()[0].value
                    return {
                        "status": "optimal",
                        "solution": self._solution,
                        "objective_value": self._problem.value,
                        "total_labor_cost": np.dot(self.l, self._solution),
                        "feasible": True,
                        "solver_used": solver_name,
                    }
                elif self._status == cp.INFEASIBLE:
                    return {
                        "status": "infeasible",
                        "solution": None,
                        "objective_value": None,
                        "total_labor_cost": None,
                        "feasible": False,
                        "solver_used": solver_name,
                    }
                elif self._status == cp.UNBOUNDED:
                    return {
                        "status": "unbounded",
                        "solution": None,
                        "objective_value": None,
                        "total_labor_cost": None,
                        "feasible": False,
                        "solver_used": solver_name,
                    }
                else:
                    # Try next solver
                    continue

            except Exception as e:
                last_error = e
                warnings.warn(f"Solver {solver_name} failed: {str(e)}")
                # Try next solver
                continue
        
        # If all CVXPY solvers failed, try SciPy as fallback
        warnings.warn(f"All CVXPY solvers failed. Last error: {last_error}. Trying SciPy fallback...")
        try:
            return self._solve_scipy()
        except Exception as scipy_error:
            warnings.warn(f"SciPy fallback also failed: {scipy_error}")
            return {
                "status": "error",
                "solution": None,
                "objective_value": None,
                "total_labor_cost": None,
                "feasible": False,
                "error": f"All solvers failed. CVXPY: {last_error}, SciPy: {scipy_error}",
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

        # Check non - negativity
        negativity_violations = np.maximum(0, -solution)

        return {
            "demand_satisfied": np.allclose(demand_violations, 0, atol = 1e-6),
            "demand_violations": demand_violations,
            "resource_satisfied": resource_violations is None or np.allclose(resource_violations, 0, atol = 1e-6),
            "resource_violations": resource_violations,
            "non_negative": np.allclose(negativity_violations, 0, atol = 1e-6),
            "negativity_violations": negativity_violations,
            "all_constraints_satisfied": (
                np.allclose(demand_violations, 0, atol = 1e-6)
                and (resource_violations is None or np.allclose(resource_violations, 0, atol = 1e-6))
                and np.allclose(negativity_violations, 0, atol = 1e-6)
            ),
        }
