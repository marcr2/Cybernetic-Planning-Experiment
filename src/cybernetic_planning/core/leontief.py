"""
Leontief Input - Output Model Implementation

Implements the core Leontief model for calculating total output requirements
given final demand and technology matrix.
"""

from typing import Optional, Dict, List
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
import warnings

class LeontiefModel:
    """
    Implements the Leontief Input - Output model for economic planning.

    The model solves the equation: x = Ax + d
    where:
    - x is the total output vector - A is the technology matrix (input coefficients)
    - d is the final demand vector
    """

    def __init__(self, technology_matrix: np.ndarray, final_demand: np.ndarray):
        """
        Initialize the Leontief model.

        Args:
            technology_matrix: A square matrix A of size n×n representing
                             input coefficients
            final_demand: A column vector d of size n×1 representing final demand
        """
        self.A = np.asarray(technology_matrix)
        self.d = np.asarray(final_demand).flatten()

        # Validate inputs
        self._validate_inputs()

        # Calculate Leontief inverse
        self._leontief_inverse = None
        self._compute_leontief_inverse()

    def _validate_inputs(self) -> None:
        """Validate input matrices and vectors."""
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Technology matrix must be square")

        if self.d.ndim != 1:
            raise ValueError("Final demand must be a vector")

        if self.A.shape[0] != self.d.shape[0]:
            raise ValueError("Technology matrix and final demand must have compatible dimensions")

        # Check for negative values
        if np.any(self.A < 0):
            raise ValueError("Technology matrix contains negative values - this is economically impossible")

        if np.any(self.d < 0):
            raise ValueError("Final demand contains negative values - this is economically impossible")

        # Check if economy is productive
        if not self.is_productive():
            raise ValueError(f"Economy is not productive (spectral radius = {self.get_spectral_radius():.4f} >= 1). "
                           "This means the economy cannot sustain itself.")

    def _compute_leontief_inverse(self) -> None:
        """Compute the Leontief inverse matrix (I - A)^(-1)."""
        n = self.A.shape[0]
        I = np.eye(n)

        try:
            # Check if matrix is sparse
            if issparse(self.A):
                self._leontief_inverse = spsolve(I - self.A, I)
            else:
                # Use LU decomposition for numerical stability
                self._leontief_inverse = np.linalg.solve(I - self.A, I)

        except np.linalg.LinAlgError as e:
            # Provide more detailed error information
            spectral_radius = self.get_spectral_radius()
            raise ValueError(
                f"Cannot compute Leontief inverse: {e}\n"
                f"Spectral radius: {spectral_radius:.6f}\n"
                f"Economy is {'productive' if spectral_radius < 1 else 'non - productive'}"
            )

    def get_leontief_inverse(self) -> np.ndarray:
        """Get the Leontief inverse matrix."""
        return self._leontief_inverse.copy()

    def compute_total_output(self) -> np.ndarray:
        """
        Compute total output vector using Leontief model.

        Returns:
            Total output vector x = (I - A)^(-1) * d
        """
        if self._leontief_inverse is None:
            self._compute_leontief_inverse()

        x = self._leontief_inverse @ self.d

        # Check for negative outputs
        if np.any(x < 0):
            warnings.warn("Model produces negative outputs, which may indicate infeasibility")

        return x

    def compute_intermediate_demand(self, total_output: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute intermediate demand (Ax).

        Args:
            total_output: Total output vector. If None, computes it first.

        Returns:
            Intermediate demand vector Ax
        """
        if total_output is None:
            total_output = self.compute_total_output()

        return self.A @ total_output

    def compute_value_added(self, total_output: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute value added (x - Ax).

        Args:
            total_output: Total output vector. If None, computes it first.

        Returns:
            Value added vector (I - A)x
        """
        if total_output is None:
            total_output = self.compute_total_output()

        return total_output - self.compute_intermediate_demand(total_output)

    def get_spectral_radius(self) -> float:
        """
        Get the spectral radius of the technology matrix.

        Returns:
            Spectral radius (largest eigenvalue magnitude)
        """
        eigenvals = np.linalg.eigvals(self.A)
        return np.max(np.abs(eigenvals))

    def is_productive(self, tolerance: float = 1e - 10) -> bool:
        """
        Check if the economy is productive (spectral radius < 1).

        Args:
            tolerance: Numerical tolerance for comparison

        Returns:
            True if economy is productive
        """
        return self.get_spectral_radius() < (1 - tolerance)

    def sensitivity_analysis(self, parameter: str, delta: float = 0.01) -> np.ndarray:
        """
        Perform sensitivity analysis on the Leontief model.

        Args:
            parameter: Parameter to analyze ('A' for technology matrix, 'd' for final demand)
            delta: Small change to apply

        Returns:
            Sensitivity matrix or vector
        """
        if parameter == "A":
            # Sensitivity of output to technology matrix changes
            # ∂x/∂A_ij = (I - A)^(-1) * (e_i * x_j^T)
            n = self.A.shape[0]
            sensitivity = np.zeros((n, n, n))
            x = self.compute_total_output()

            for i in range(n):
                for j in range(n):
                    e_i = np.zeros(n)
                    e_i[i] = 1
                    sensitivity[:, i, j] = self._leontief_inverse @ (e_i * x[j])

            return sensitivity

        elif parameter == "d":
            # Sensitivity of output to final demand changes
            # ∂x/∂d = (I - A)^(-1)
            return self._leontief_inverse

        else:
            raise ValueError("Parameter must be 'A' or 'd'")

    def update_technology_matrix(self, new_A: np.ndarray) -> None:
        """Update the technology matrix and recompute inverse."""
        self.A = np.asarray(new_A)
        self._validate_inputs()
        self._leontief_inverse = None
        self._compute_leontief_inverse()

    def update_final_demand(self, new_d: np.ndarray) -> None:
        """Update the final demand vector."""
        self.d = np.asarray(new_d).flatten()
        self._validate_inputs()

    def compute_multipliers(self) -> Dict[str, np.ndarray]:
        """
        Compute various economic multipliers.

        Returns:
            Dictionary containing different types of multipliers
        """
        leontief_inverse = self.get_leontief_inverse()

        # Output multipliers (column sums of Leontief inverse)
        output_multipliers = np.sum(leontief_inverse, axis = 0)

        # Income multipliers (if we had income coefficients)
        # For now, we'll use labor coefficients as proxy
        if hasattr(self, 'labor_coefficients'):
            income_multipliers = leontief_inverse @ self.labor_coefficients
        else:
            income_multipliers = None

        # Employment multipliers (if we had employment coefficients)
        if hasattr(self, 'employment_coefficients'):
            employment_multipliers = leontief_inverse @ self.employment_coefficients
        else:
            employment_multipliers = None

        # Value added multipliers (if we had value added coefficients)
        if hasattr(self, 'value_added_coefficients'):
            value_added_multipliers = leontief_inverse @ self.value_added_coefficients
        else:
            value_added_multipliers = None

        return {
            "output_multipliers": output_multipliers,
            "income_multipliers": income_multipliers,
            "employment_multipliers": employment_multipliers,
            "value_added_multipliers": value_added_multipliers
        }

    def compute_forward_linkages(self) -> np.ndarray:
        """
        Compute forward linkage indices (Rasmussen indices).

        Returns:
            Forward linkage indices for each sector
        """
        leontief_inverse = self.get_leontief_inverse()
        n = leontief_inverse.shape[0]

        # Forward linkages: row sums of Leontief inverse
        forward_linkages = np.sum(leontief_inverse, axis = 1)

        # Normalize by average
        average_forward = np.mean(forward_linkages)
        normalized_forward = forward_linkages / average_forward if average_forward > 0 else forward_linkages

        return normalized_forward

    def compute_backward_linkages(self) -> np.ndarray:
        """
        Compute backward linkage indices (Rasmussen indices).

        Returns:
            Backward linkage indices for each sector
        """
        leontief_inverse = self.get_leontief_inverse()
        n = leontief_inverse.shape[0]

        # Backward linkages: column sums of Leontief inverse
        backward_linkages = np.sum(leontief_inverse, axis = 0)

        # Normalize by average
        average_backward = np.mean(backward_linkages)
        normalized_backward = backward_linkages / average_backward if average_backward > 0 else backward_linkages

        return normalized_backward

    def compute_key_sectors(self, threshold: float = 1.0) -> Dict[str, List[int]]:
        """
        Identify key sectors based on linkage analysis.

        Args:
            threshold: Threshold for identifying key sectors

        Returns:
            Dictionary with different types of key sectors
        """
        forward_linkages = self.compute_forward_linkages()
        backward_linkages = self.compute_backward_linkages()

        # Key sectors: both forward and backward linkages above threshold
        key_sectors = np.where((forward_linkages >= threshold) & (backward_linkages >= threshold))[0].tolist()

        # Forward - oriented sectors: high forward, low backward
        forward_oriented = np.where((forward_linkages >= threshold) & (backward_linkages < threshold))[0].tolist()

        # Backward - oriented sectors: low forward, high backward
        backward_oriented = np.where((forward_linkages < threshold) & (backward_linkages >= threshold))[0].tolist()

        # Independent sectors: both low
        independent = np.where((forward_linkages < threshold) & (backward_linkages < threshold))[0].tolist()

        return {
            "key_sectors": key_sectors,
            "forward_oriented": forward_oriented,
            "backward_oriented": backward_oriented,
            "independent": independent,
            "forward_linkages": forward_linkages.tolist(),
            "backward_linkages": backward_linkages.tolist()
        }

    def compute_import_requirements(self, import_coefficients: np.ndarray) -> np.ndarray:
        """
        Compute import requirements for given final demand.

        Args:
            import_coefficients: Import coefficients vector

        Returns:
            Total import requirements
        """
        leontief_inverse = self.get_leontief_inverse()
        total_output = self.compute_total_output()

        # Import requirements = import coefficients * total output
        import_requirements = import_coefficients * total_output

        return import_requirements

    def compute_environmental_impact(self, environmental_coefficients: np.ndarray) -> np.ndarray:
        """
        Compute environmental impact for given final demand.

        Args:
            environmental_coefficients: Environmental impact coefficients

        Returns:
            Total environmental impact
        """
        leontief_inverse = self.get_leontief_inverse()
        total_output = self.compute_total_output()

        # Environmental impact = environmental coefficients * total output
        environmental_impact = environmental_coefficients * total_output

        return environmental_impact
