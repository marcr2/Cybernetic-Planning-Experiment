"""
Labor Value Calculation Implementation

Implements Cockshott's labor-time accounting model for calculating
the total direct and indirect labor embodied in commodities.
"""

import numpy as np
from typing import Optional
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
import warnings


class LaborValueCalculator:
    """
    Calculates labor values using Cockshott's model.
    
    The labor value vector v represents the total direct and indirect
    labor embodied in one unit of each product:
    v = vA + l
    where:
    - v is the labor value vector
    - A is the technology matrix
    - l is the direct labor input vector
    """
    
    def __init__(self, technology_matrix: np.ndarray, direct_labor: np.ndarray):
        """
        Initialize the labor value calculator.
        
        Args:
            technology_matrix: Technology matrix A of size n×n
            direct_labor: Direct labor input vector l of size 1×n
        """
        self.A = np.asarray(technology_matrix)
        self.l = np.asarray(direct_labor).flatten()
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate labor values
        self._labor_values = None
        self._compute_labor_values()
    
    def _validate_inputs(self) -> None:
        """Validate input matrices and vectors."""
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Technology matrix must be square")
        
        if self.l.ndim != 1:
            raise ValueError("Direct labor input must be a vector")
        
        if self.A.shape[0] != self.l.shape[0]:
            raise ValueError("Technology matrix and labor vector must have compatible dimensions")
        
        # Check for negative values
        if np.any(self.l < 0):
            warnings.warn("Direct labor input contains negative values")
    
    def _compute_labor_values(self) -> None:
        """
        Compute labor values by solving v = vA + l.
        
        This is equivalent to solving v(I - A) = l, so v = l(I - A)^(-1)
        """
        n = self.A.shape[0]
        I = np.eye(n)
        
        try:
            # Check if matrix is sparse
            if issparse(self.A):
                self._labor_values = spsolve((I - self.A).T, self.l)
            else:
                # Use LU decomposition for numerical stability
                self._labor_values = np.linalg.solve((I - self.A).T, self.l)
                
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Cannot compute labor values: {e}")
    
    def get_labor_values(self) -> np.ndarray:
        """Get the labor value vector."""
        return self._labor_values.copy()
    
    def compute_total_labor_cost(self, final_demand: np.ndarray) -> float:
        """
        Compute total labor cost for a given final demand.
        
        Args:
            final_demand: Final demand vector d
            
        Returns:
            Total labor cost L = v · d
        """
        if self._labor_values is None:
            self._compute_labor_values()
        
        return np.dot(self._labor_values, final_demand)
    
    def compute_sectoral_labor_allocation(self, total_output: np.ndarray) -> np.ndarray:
        """
        Compute labor allocation for each sector.
        
        Args:
            total_output: Total output vector x
            
        Returns:
            Labor allocation vector l_j * x_j for each sector j
        """
        return self.l * total_output
    
    def compute_labor_intensity(self) -> np.ndarray:
        """
        Compute labor intensity (labor value per unit output).
        
        Returns:
            Labor intensity vector v
        """
        if self._labor_values is None:
            self._compute_labor_values()
        
        return self._labor_values.copy()
    
    def compute_labor_productivity(self) -> np.ndarray:
        """
        Compute labor productivity (output per unit labor).
        
        Returns:
            Labor productivity vector 1/v
        """
        labor_values = self.get_labor_values()
        
        # Avoid division by zero
        productivity = np.zeros_like(labor_values)
        mask = labor_values > 0
        productivity[mask] = 1.0 / labor_values[mask]
        
        return productivity
    
    def update_technology_matrix(self, new_A: np.ndarray) -> None:
        """Update the technology matrix and recompute labor values."""
        self.A = np.asarray(new_A)
        self._validate_inputs()
        self._labor_values = None
        self._compute_labor_values()
    
    def update_direct_labor(self, new_l: np.ndarray) -> None:
        """Update the direct labor input vector and recompute labor values."""
        self.l = np.asarray(new_l).flatten()
        self._validate_inputs()
        self._labor_values = None
        self._compute_labor_values()
    
    def get_labor_value_breakdown(self, sector_index: int) -> dict:
        """
        Get detailed breakdown of labor value for a specific sector.
        
        Args:
            sector_index: Index of the sector to analyze
            
        Returns:
            Dictionary with labor value components
        """
        if self._labor_values is None:
            self._compute_labor_values()
        
        # Direct labor component
        direct_labor = self.l[sector_index]
        
        # Indirect labor component (from inputs)
        indirect_labor = self._labor_values[sector_index] - direct_labor
        
        # Labor value from each input sector
        input_labor = np.zeros(self.A.shape[0])
        for j in range(self.A.shape[0]):
            if self.A[j, sector_index] > 0:
                input_labor[j] = self._labor_values[j] * self.A[j, sector_index]
        
        return {
            'total_labor_value': self._labor_values[sector_index],
            'direct_labor': direct_labor,
            'indirect_labor': indirect_labor,
            'input_labor_breakdown': input_labor,
            'labor_intensity': self._labor_values[sector_index],
            'labor_productivity': 1.0 / self._labor_values[sector_index] if self._labor_values[sector_index] > 0 else 0
        }
