"""
Constrained Optimization Implementation

Implements linear programming optimization to minimize labor time
subject to demand fulfillment and resource constraints.
"""

from typing import Optional, Dict, Any, Union
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import issparse, csr_matrix, csc_matrix
import warnings

# Try to import cvxpy, but don't fail if it's not available
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

# Import GPU acceleration module
try:
    from .gpu_acceleration import (
        gpu_detector, solver_selector, performance_monitor, settings_manager,
        create_gpu_optimized_arrays, convert_gpu_to_cpu
    )
    GPU_ACCELERATION_AVAILABLE = True
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
    gpu_detector = None
    solver_selector = None
    performance_monitor = None
    settings_manager = None
    create_gpu_optimized_arrays = None
    convert_gpu_to_cpu = None

class ConstrainedOptimizer:
    """
    Solves constrained optimization problems for economic planning.

    Minimizes total labor cost subject to:
    - Demand fulfillment: (I - A)x >= d - Resource constraints: Rx <= R_max - Non - negativity: x >= 0
    """

    def __init__(
        self,
        technology_matrix: Union[np.ndarray, csr_matrix, csc_matrix],
        direct_labor: np.ndarray,
        final_demand: np.ndarray,
        resource_matrix: Optional[Union[np.ndarray, csr_matrix, csc_matrix]] = None,
        max_resources: Optional[np.ndarray] = None,
        use_sparse: bool = True,
        use_gpu: Optional[bool] = None,
    ):
        """
        Initialize the constrained optimizer.

        Args:
            technology_matrix: Technology matrix A (can be dense or sparse)
            direct_labor: Direct labor input vector l
            final_demand: Final demand vector d
            resource_matrix: Resource constraint matrix R (optional, can be dense or sparse)
            max_resources: Maximum resource availability R_max (optional)
            use_sparse: Whether to convert to sparse matrices if not already sparse
            use_gpu: Whether to use GPU acceleration (None = auto-detect from settings)
        """
        # Convert to sparse if requested and not already sparse
        if use_sparse and not issparse(technology_matrix):
            self.A = csr_matrix(technology_matrix)
        elif use_sparse and issparse(technology_matrix):
            # Ensure it's in CSR format for optimal performance
            if not isinstance(technology_matrix, csr_matrix):
                self.A = csr_matrix(technology_matrix)
            else:
                self.A = technology_matrix
        else:
            self.A = technology_matrix
            
        self.l = np.asarray(direct_labor).flatten()
        self.d = np.asarray(final_demand).flatten()
        
        # Handle resource matrix
        if resource_matrix is not None:
            if use_sparse and not issparse(resource_matrix):
                self.R = csr_matrix(resource_matrix)
            else:
                self.R = resource_matrix
        else:
            self.R = None
            
        self.R_max = max_resources

        # GPU acceleration setup
        self.use_gpu = use_gpu
        self.gpu_arrays = None
        self._setup_gpu_acceleration()

        # Validate inputs
        self._validate_inputs()

        # Problem variables
        self._problem = None
        self._solution = None
        self._status = None

    def _setup_gpu_acceleration(self) -> None:
        """Setup GPU acceleration if available and enabled."""
        if not GPU_ACCELERATION_AVAILABLE:
            self.use_gpu = False
            return
        
        # Auto-detect GPU usage from settings if not specified
        if self.use_gpu is None:
            self.use_gpu = settings_manager.is_gpu_enabled()
        
        # Check if GPU is actually available
        if self.use_gpu and not gpu_detector.is_gpu_available():
            warnings.warn("GPU acceleration requested but GPU not available, falling back to CPU")
            self.use_gpu = False
        
        # Convert arrays to GPU if enabled
        if self.use_gpu:
            try:
                arrays_to_convert = [self.l, self.d]
                if self.R is not None:
                    arrays_to_convert.append(self.R.toarray() if issparse(self.R) else self.R)
                if self.R_max is not None:
                    arrays_to_convert.append(self.R_max)
                
                self.gpu_arrays = create_gpu_optimized_arrays(arrays_to_convert, use_gpu=True)
            except Exception as e:
                warnings.warn(f"Failed to setup GPU acceleration: {e}, falling back to CPU")
                self.use_gpu = False
                self.gpu_arrays = None

    def _validate_inputs(self) -> None:
        """Validate input matrices and vectors."""
        n = self.A.shape[0]
        
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Technology matrix must be square")
        
        if self.l.shape[0] != n or self.d.shape[0] != n:
            raise ValueError("All vectors must have same dimension as technology matrix")
        
        # Check for invalid values
        if issparse(self.A):
            if np.any(np.isnan(self.A.data)) or np.any(np.isinf(self.A.data)):
                raise ValueError("Technology matrix contains NaN or infinite values")
        else:
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
            if issparse(self.A):
                eigenvals = np.linalg.eigvals(self.A.toarray())
            else:
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
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is not available. Please install it or use scipy optimization.")
        
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
        if issparse(self.A):
            # Convert sparse matrix to dense for CVXPY
            A_dense = self.A.toarray()
            constraints.append((I - A_dense) @ x >= self.d)
        else:
            constraints.append((I - self.A) @ x >= self.d)

        # Resource constraints: Rx <= R_max
        if self.R is not None and self.R_max is not None:
            if issparse(self.R):
                # Convert sparse matrix to dense for CVXPY
                R_dense = self.R.toarray()
                constraints.append(R_dense @ x <= self.R_max)
            else:
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
        if issparse(self.A):
            A_ub = -(I - self.A.toarray())  # Convert to dense for scipy
        else:
            A_ub = -(I - self.A)  # Negative because scipy uses A_ub @ x <= b_ub
        b_ub = -self.d

        # Add resource constraints if provided
        if self.R is not None and self.R_max is not None:
            # Convert resource matrix to dense if sparse
            if issparse(self.R):
                R_dense = self.R.toarray()
            else:
                R_dense = self.R
            # Combine demand and resource constraints
            A_ub_combined = np.vstack([A_ub, R_dense])
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
        # If CVXPY is not available, fall back to scipy
        if use_cvxpy and not CVXPY_AVAILABLE:
            warnings.warn("CVXPY not available, falling back to scipy.optimize")
            use_cvxpy = False
        
        # Select GPU solver if available and enabled
        if use_cvxpy and self.use_gpu and GPU_ACCELERATION_AVAILABLE:
            if solver is None:
                solver = solver_selector.select_gpu_solver(
                    use_gpu=True, 
                    solver_preference=settings_manager.get_solver_preference()
                )
        
        self._create_problem(use_cvxpy)

        # Start performance monitoring if enabled
        if GPU_ACCELERATION_AVAILABLE and performance_monitor:
            performance_monitor.start_monitoring()

        try:
            if use_cvxpy:
                result = self._solve_cvxpy(solver)
            else:
                result = self._solve_scipy()
            
            # Add GPU status to result
            if GPU_ACCELERATION_AVAILABLE:
                result["gpu_accelerated"] = self.use_gpu
                if self.use_gpu:
                    result["gpu_memory_usage"] = gpu_detector.get_gpu_memory_usage()
            
            return result
        finally:
            # Stop performance monitoring
            if GPU_ACCELERATION_AVAILABLE and performance_monitor:
                performance_monitor.stop_monitoring()

    def _solve_cvxpy(self, solver: Optional[str] = None) -> Dict[str, Any]:
        """Solve using CVXPY with fallback to multiple solvers."""
        # Try multiple solvers in order of preference
        solvers_to_try = []
        if solver is not None:
            solvers_to_try = [solver]
        else:
            # Try solvers in order of preference (removed CLARABEL as it requires juliacall)
            solvers_to_try = [cp.ECOS, cp.SCS, cp.OSQP]
        
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
            # Ensure scipy parameters are created
            if not hasattr(self, '_scipy_params') or self._scipy_params is None:
                self._create_scipy_problem()
            
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
        if issparse(self.A):
            demand_satisfaction = (I - self.A.toarray()) @ solution
        else:
            demand_satisfaction = (I - self.A) @ solution
        demand_violations = np.maximum(0, self.d - demand_satisfaction)

        # Check resource constraints
        resource_violations = None
        if self.R is not None and self.R_max is not None:
            if issparse(self.R):
                resource_usage = self.R.toarray() @ solution
            else:
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

    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get current GPU status and performance information.
        
        Returns:
            Dictionary with GPU status information
        """
        if not GPU_ACCELERATION_AVAILABLE:
            return {
                "gpu_available": False,
                "gpu_enabled": False,
                "error": "GPU acceleration module not available"
            }
        
        gpu_info = gpu_detector.gpu_info.copy()
        gpu_info.update({
            "gpu_available": gpu_detector.is_gpu_available(),
            "gpu_enabled": self.use_gpu,
            "gpu_arrays_created": self.gpu_arrays is not None,
            "memory_usage": gpu_detector.get_gpu_memory_usage()
        })
        
        return gpu_info

    def benchmark_gpu_vs_cpu(self) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance for this optimization problem.
        
        Returns:
            Dictionary with benchmark results
        """
        if not GPU_ACCELERATION_AVAILABLE or not performance_monitor:
            return {"error": "GPU benchmarking not available"}
        
        def gpu_solve():
            # Create a temporary GPU-enabled optimizer
            gpu_optimizer = ConstrainedOptimizer(
                self.A, self.l, self.d, self.R, self.R_max, 
                use_sparse=True, use_gpu=True
            )
            return gpu_optimizer.solve()
        
        def cpu_solve():
            # Create a temporary CPU-only optimizer
            cpu_optimizer = ConstrainedOptimizer(
                self.A, self.l, self.d, self.R, self.R_max, 
                use_sparse=True, use_gpu=False
            )
            return cpu_optimizer.solve()
        
        return performance_monitor.benchmark_operation(
            "optimization_solve", gpu_solve, cpu_solve
        )
