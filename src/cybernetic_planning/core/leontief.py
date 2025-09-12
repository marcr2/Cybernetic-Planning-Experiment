"""
Leontief Input - Output Model Implementation

Implements the core Leontief model for calculating total output requirements
given final demand and technology matrix.
"""

from typing import Optional, Dict, List, Union, Any
import numpy as np
from scipy.sparse import issparse, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import warnings
from .error_handling import MatrixValidator, ValidationResult, safe_matrix_operation

# Import GPU acceleration module
try:
    from .gpu_acceleration import (
        gpu_detector, create_gpu_optimized_arrays, convert_gpu_to_cpu, settings_manager
    )
    GPU_ACCELERATION_AVAILABLE = True
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
    gpu_detector = None
    create_gpu_optimized_arrays = None
    convert_gpu_to_cpu = None
    settings_manager = None

class LeontiefModel:
    """
    Implements the Leontief Input - Output model for economic planning.

    The model solves the equation: x = Ax + d
    where:
    - x is the total output vector - A is the technology matrix (input coefficients)
    - d is the final demand vector
    """

    def __init__(self, technology_matrix: Union[np.ndarray, csr_matrix, csc_matrix], final_demand: np.ndarray, use_sparse: bool = True, use_gpu: Optional[bool] = None):
        """
        Initialize the Leontief model.

        Args:
            technology_matrix: A square matrix A of size n×n representing
                             input coefficients (can be dense or sparse)
            final_demand: A column vector d of size n×1 representing final demand
            use_sparse: Whether to convert to sparse matrix if not already sparse
            use_gpu: Whether to use GPU acceleration (None = auto-detect from settings)
        """
        # Convert to sparse if requested and not already sparse
        if use_sparse and not issparse(technology_matrix):
            self.A = csr_matrix(technology_matrix)
        elif use_sparse and issparse(technology_matrix):
            # Ensure it's in CSR format for efficiency
            if not isinstance(technology_matrix, csr_matrix):
                self.A = csr_matrix(technology_matrix)
            else:
                self.A = technology_matrix
        else:
            self.A = technology_matrix
            
        self.d = np.asarray(final_demand).flatten()

        # GPU acceleration setup
        self.use_gpu = use_gpu
        self.gpu_arrays = None
        self._setup_gpu_acceleration()

        # Validate inputs with enhanced error handling
        self._validate_inputs()

        # Calculate Leontief inverse
        self._leontief_inverse = None
        self._compute_leontief_inverse()

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
                arrays_to_convert = [self.d]
                self.gpu_arrays = create_gpu_optimized_arrays(arrays_to_convert, use_gpu=True)
            except Exception as e:
                warnings.warn(f"Failed to setup GPU acceleration: {e}, falling back to CPU")
                self.use_gpu = False
                self.gpu_arrays = None

    def _validate_inputs(self) -> None:
        """Validate input matrices and vectors."""
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Technology matrix must be square")

        if self.d.ndim != 1:
            raise ValueError("Final demand must be a vector")

        if self.A.shape[0] != self.d.shape[0]:
            raise ValueError("Technology matrix and final demand must have compatible dimensions")

        # Check for negative values
        if issparse(self.A):
            if (self.A < 0).nnz > 0:  # Check if any negative values exist
                raise ValueError("Technology matrix contains negative values - this is economically impossible")
        else:
            if np.any(self.A < 0):
                raise ValueError("Technology matrix contains negative values - this is economically impossible")

        if np.any(self.d < 0):
            raise ValueError("Final demand contains negative values - this is economically impossible")

        # Check if economy is productive
        if not self.is_productive():
            raise ValueError(f"Economy is not productive (spectral radius = {self.get_spectral_radius():.4f} >= 1). "
                           "This means the economy cannot sustain itself.")

    def _compute_leontief_inverse(self) -> None:
        """Compute the Leontief inverse matrix (I - A)^(-1) with optimizations."""
        n = self.A.shape[0]
        
        try:
            # Check if matrix is sparse
            if issparse(self.A):
                # Ensure matrix is in CSR format for optimal performance
                if not isinstance(self.A, csr_matrix):
                    self.A = csr_matrix(self.A)
                
                # Create sparse identity matrix efficiently
                I_sparse = csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
                
                # Compute (I - A) efficiently using sparse operations
                I_minus_A = I_sparse - self.A
                
                # Use spsolve for sparse matrix solving
                self._leontief_inverse = spsolve(I_minus_A, I_sparse)
                
                # Convert result to dense for consistency with other operations
                if issparse(self._leontief_inverse):
                    self._leontief_inverse = self._leontief_inverse.toarray()
            else:
                # Use LU decomposition for numerical stability
                I = np.eye(n)
                self._leontief_inverse = np.linalg.solve(I - self.A, I)

        except np.linalg.LinAlgError as e:
            # Provide more detailed error information
            spectral_radius = self.get_spectral_radius()
            raise ValueError(
                f"Cannot compute Leontief inverse: {e}\n"
                f"Spectral radius: {spectral_radius:.6f}\n"
                f"Economy is {'productive' if spectral_radius < 1 else 'non-productive'}"
            )

    def get_leontief_inverse(self) -> np.ndarray:
        """Get the Leontief inverse matrix."""
        return self._leontief_inverse.copy()

    def compute_total_output(self) -> np.ndarray:
        """
        Compute total output vector using Leontief model with optimizations.

        Returns:
            Total output vector x = (I - A)^(-1) * d
        """
        if self._leontief_inverse is None:
            self._compute_leontief_inverse()

        # Use efficient matrix-vector multiplication
        if issparse(self._leontief_inverse):
            x = self._leontief_inverse.dot(self.d)
        else:
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
        if issparse(self.A):
            eigenvals = np.linalg.eigvals(self.A.toarray())
        else:
            eigenvals = np.linalg.eigvals(self.A)
        return np.max(np.abs(eigenvals))

    def is_productive(self, tolerance: float = 1e-10) -> bool:
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
            # Vectorized implementation
            n = self.A.shape[0]
            x = self.compute_total_output()
            
            # Initialize sensitivity tensor
            sensitivity = np.zeros((n, n, n))
            
            # For each (i, j) pair, compute sensitivity of all outputs
            for i in range(n):
                for j in range(n):
                    # Create unit vector e_i
                    e_i = np.zeros(n)
                    e_i[i] = 1
                    
                    # Compute sensitivity: (I - A)^(-1) @ (e_i * x_j)
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
