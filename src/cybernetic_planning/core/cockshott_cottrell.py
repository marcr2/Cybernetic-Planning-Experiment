"""
Cockshott & Cottrell Iterative Planning Implementation

Implements the iterative planning algorithm as described by Cockshott & Cottrell
for large-scale, disaggregated economic planning based on labor time accounting.

The core algorithm: output_plan_{t+1} = final_demand + A · output_plan_{t}
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, issparse
from scipy.sparse.linalg import spsolve
import warnings


class CockshottCottrellPlanner:
    """
    Implements the Cockshott & Cottrell iterative planning algorithm.
    
    This class handles large-scale economic planning using sparse matrices
    and iterative convergence to find optimal production plans.
    """

    def __init__(
        self,
        technology_matrix: np.ndarray,
        final_demand: np.ndarray,
        direct_labor: np.ndarray,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        use_sparse: bool = True
    ):
        """
        Initialize the Cockshott & Cottrell planner.

        Args:
            technology_matrix: Technology matrix A (n×n)
            final_demand: Final demand vector d (n×1)
            direct_labor: Direct labor input vector l (1×n)
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence tolerance
            use_sparse: Whether to use sparse matrix representation
        """
        # Convert to sparse matrices if requested
        if use_sparse and not issparse(technology_matrix):
            # Convert to CSR format for efficient row operations
            self.A = csr_matrix(technology_matrix)
        elif use_sparse and issparse(technology_matrix):
            # Ensure it's in CSR format for efficiency
            if not isinstance(technology_matrix, csr_matrix):
                self.A = csr_matrix(technology_matrix)
            else:
                self.A = technology_matrix
        else:
            self.A = np.asarray(technology_matrix)
        
        self.n_sectors = self.A.shape[0]
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_sparse = use_sparse

        self.d = np.asarray(final_demand).flatten()
        self.l = np.asarray(direct_labor).flatten()

        # Validate inputs
        self._validate_inputs()

        # Initialize planning state
        self.current_plan = None
        self.iteration_count = 0
        self.converged = False
        self.convergence_history = []

    def _validate_inputs(self) -> None:
        """Validate input matrices and vectors."""
        # Check if matrix has proper shape
        if len(self.A.shape) != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"Technology matrix must be square, got shape {self.A.shape}")

        if self.d.shape[0] != self.n_sectors:
            raise ValueError(f"Final demand must have same dimension as technology matrix, got {self.d.shape[0]} vs {self.n_sectors}")

        if self.l.shape[0] != self.n_sectors:
            raise ValueError(f"Labor vector must have same dimension as technology matrix, got {self.l.shape[0]} vs {self.n_sectors}")

        # Check for negative values
        if np.any(self.d < 0):
            raise ValueError("Final demand contains negative values")

        if np.any(self.l < 0):
            raise ValueError("Labor vector contains negative values")

        # Check if economy is productive (spectral radius < 1)
        if issparse(self.A):
            eigenvals = np.linalg.eigvals(self.A.toarray())
        else:
            eigenvals = np.linalg.eigvals(self.A)
        
        spectral_radius = np.max(np.abs(eigenvals))
        if spectral_radius >= 1.0:
            raise ValueError(f"Economy is not productive (spectral radius = {spectral_radius:.4f} >= 1)")

    def iterative_planning(self, initial_plan: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Execute the Cockshott & Cottrell iterative planning algorithm.

        Core algorithm: output_plan_{t+1} = final_demand + A · output_plan_{t}

        Args:
            initial_plan: Initial production plan (if None, uses final demand)

        Returns:
            Dictionary with planning results and convergence information
        """
        # Initialize production plan
        if initial_plan is None:
            self.current_plan = self.d.copy()
        else:
            self.current_plan = np.asarray(initial_plan).flatten()
            if self.current_plan.shape[0] != self.n_sectors:
                raise ValueError("Initial plan must have same dimension as technology matrix")

        self.iteration_count = 0
        self.converged = False
        self.convergence_history = []

        # Iterative planning loop
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            
            # Store previous plan for convergence check
            previous_plan = self.current_plan.copy()

            # Core Cockshott & Cottrell update step
            # output_plan_{t+1} = final_demand + A · output_plan_{t}
            if issparse(self.A):
                intermediate_demand = self.A @ self.current_plan
            else:
                intermediate_demand = self.A @ self.current_plan

            # Update production plan
            self.current_plan = self.d + intermediate_demand

            # Ensure non-negative production
            self.current_plan = np.maximum(self.current_plan, 0)

            # Check for convergence
            plan_change = np.linalg.norm(self.current_plan - previous_plan)
            relative_change = plan_change / (np.linalg.norm(self.current_plan) + 1e-10)
            
            self.convergence_history.append({
                'iteration': iteration,
                'plan_change': plan_change,
                'relative_change': relative_change,
                'total_output': np.sum(self.current_plan)
            })

            # Check convergence criteria
            if plan_change < self.convergence_threshold:
                self.converged = True
                break

            # Additional convergence check for relative change
            if relative_change < self.convergence_threshold:
                self.converged = True
                break

        # Calculate final metrics
        total_labor_cost = np.dot(self.l, self.current_plan)
        
        # Calculate intermediate demand for final plan
        if issparse(self.A):
            final_intermediate_demand = self.A @ self.current_plan
        else:
            final_intermediate_demand = self.A @ self.current_plan

        # Calculate net output (final demand fulfillment)
        net_output = self.current_plan - final_intermediate_demand
        demand_fulfillment = net_output / (self.d + 1e-10)  # Avoid division by zero

        # Get final convergence metrics
        final_plan_change = 0
        final_relative_change = 0
        if self.convergence_history:
            final_plan_change = self.convergence_history[-1]['plan_change']
            final_relative_change = self.convergence_history[-1]['relative_change']

        return {
            'production_plan': self.current_plan.copy(),
            'converged': self.converged,
            'iterations': self.iteration_count,
            'total_labor_cost': total_labor_cost,
            'intermediate_demand': final_intermediate_demand,
            'net_output': net_output,
            'demand_fulfillment': demand_fulfillment,
            'convergence_history': self.convergence_history,
            'final_plan_change': final_plan_change,
            'relative_change': final_relative_change
        }

    def calculate_labor_values(self) -> np.ndarray:
        """
        Calculate labor values using the Cockshott formula: v = (I - A^T)^{-1} l

        Returns:
            Labor value vector v
        """
        n = self.n_sectors
        I = np.eye(n)

        try:
            if issparse(self.A):
                # Convert identity to sparse format
                I_sparse = csr_matrix(I)
                labor_values = spsolve((I_sparse - self.A).T, self.l)
            else:
                labor_values = np.linalg.solve((I - self.A).T, self.l)

            # Ensure positive labor values
            if np.any(labor_values < 0):
                warnings.warn("Some labor values are negative - this violates the labor theory of value")
                labor_values = np.maximum(labor_values, 1e-10)

            return labor_values

        except np.linalg.LinAlgError as e:
            raise ValueError(f"Cannot compute labor values: {e}")

    def calculate_total_labor_cost(self, production_plan: Optional[np.ndarray] = None) -> float:
        """
        Calculate total labor cost for a given production plan.

        Args:
            production_plan: Production plan vector (if None, uses current plan)

        Returns:
            Total labor cost L = l · x
        """
        if production_plan is None:
            if self.current_plan is None:
                raise ValueError("No production plan available")
            production_plan = self.current_plan

        return np.dot(self.l, production_plan)

    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get summary of convergence behavior."""
        if not self.convergence_history:
            return {'error': 'No convergence history available'}

        final_change = self.convergence_history[-1]['plan_change']
        final_relative_change = self.convergence_history[-1]['relative_change']
        
        return {
            'converged': self.converged,
            'iterations': self.iteration_count,
            'final_plan_change': final_change,
            'final_relative_change': final_relative_change,
            'convergence_rate': self._calculate_convergence_rate(),
            'total_output_growth': self._calculate_output_growth()
        }

    def _calculate_convergence_rate(self) -> float:
        """Calculate the rate of convergence."""
        if len(self.convergence_history) < 2:
            return 0.0

        # Calculate average relative change over last 10 iterations
        recent_changes = [h['relative_change'] for h in self.convergence_history[-10:]]
        return np.mean(recent_changes)

    def _calculate_output_growth(self) -> float:
        """Calculate total output growth during planning."""
        if len(self.convergence_history) < 2:
            return 0.0

        initial_output = self.convergence_history[0]['total_output']
        final_output = self.convergence_history[-1]['total_output']
        
        if initial_output > 0:
            return (final_output - initial_output) / initial_output
        return 0.0

    def update_technology_matrix(self, new_A: np.ndarray) -> None:
        """Update technology matrix and reset planning state."""
        if new_A.shape != (self.n_sectors, self.n_sectors):
            raise ValueError("New technology matrix must have same dimensions")

        if self.use_sparse and not issparse(new_A):
            self.A = csr_matrix(new_A)
        else:
            self.A = np.asarray(new_A)

        # Reset planning state
        self.current_plan = None
        self.iteration_count = 0
        self.converged = False
        self.convergence_history = []

        # Re-validate
        self._validate_inputs()

    def update_final_demand(self, new_d: np.ndarray) -> None:
        """Update final demand vector."""
        if new_d.shape[0] != self.n_sectors:
            raise ValueError("New final demand must have same dimension as technology matrix")

        self.d = np.asarray(new_d).flatten()
        
        # Reset planning state
        self.current_plan = None
        self.iteration_count = 0
        self.converged = False
        self.convergence_history = []

    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive planning statistics."""
        if self.current_plan is None:
            return {'error': 'No planning results available'}

        total_output = np.sum(self.current_plan)
        total_labor_cost = self.calculate_total_labor_cost()
        
        # Calculate sector-wise statistics
        sector_stats = []
        for i in range(self.n_sectors):
            sector_stats.append({
                'sector_id': i,
                'production': self.current_plan[i],
                'labor_input': self.l[i] * self.current_plan[i],
                'labor_intensity': self.l[i],
                'demand_fulfillment': self.d[i] / (self.current_plan[i] + 1e-10)
            })

        return {
            'total_output': total_output,
            'total_labor_cost': total_labor_cost,
            'average_labor_intensity': np.mean(self.l),
            'sector_count': self.n_sectors,
            'convergence_info': self.get_convergence_summary(),
            'sector_statistics': sector_stats
        }
