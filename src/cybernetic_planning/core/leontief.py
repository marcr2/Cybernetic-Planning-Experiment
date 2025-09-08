"""
Leontief Input-Output Model Implementation

Implements the core Leontief model for calculating total output requirements
given final demand and technology matrix.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import spsolve
import warnings


class LeontiefModel:
    """
    Implements the Leontief Input-Output model for economic planning.
    
    The model solves the equation: x = Ax + d
    where:
    - x is the total output vector
    - A is the technology matrix (input coefficients)
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
            warnings.warn("Technology matrix contains negative values")
        
        if np.any(self.d < 0):
            warnings.warn("Final demand contains negative values")
    
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
            raise ValueError(f"Cannot compute Leontief inverse: {e}")
    
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
        if parameter == 'A':
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
            
        elif parameter == 'd':
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
