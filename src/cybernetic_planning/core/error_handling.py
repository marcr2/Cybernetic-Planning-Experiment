"""
Enhanced Error Handling and Validation

Provides comprehensive error handling, validation, and debugging utilities
for the economic planning system.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from scipy.sparse import issparse, csr_matrix, csc_matrix
import warnings
import traceback
import logging
from datetime import datetime


class PlanningError(Exception):
    """Base exception for planning system errors."""
    pass


class MatrixValidationError(PlanningError):
    """Exception for matrix validation errors."""
    pass


class ConvergenceError(PlanningError):
    """Exception for convergence failures."""
    pass


class OptimizationError(PlanningError):
    """Exception for optimization failures."""
    pass


class DataIntegrityError(PlanningError):
    """Exception for data integrity issues."""
    pass


class ValidationResult:
    """Result of a validation check."""
    
    def __init__(self, is_valid: bool, message: str, details: Optional[Dict[str, Any]] = None):
        self.is_valid = is_valid
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()


class MatrixValidator:
    """Comprehensive matrix validation utilities."""
    
    @staticmethod
    def validate_technology_matrix(A: Union[np.ndarray, csr_matrix, csc_matrix], 
                                 name: str = "Technology matrix") -> ValidationResult:
        """
        Validate a technology matrix for economic planning.
        
        Args:
            A: Technology matrix to validate
            name: Name of the matrix for error messages
            
        Returns:
            ValidationResult with validation status and details
        """
        details = {}
        
        try:
            # Check if matrix is square
            if A.shape[0] != A.shape[1]:
                return ValidationResult(
                    False, 
                    f"{name} must be square, got shape {A.shape}",
                    {"shape": A.shape}
                )
            
            n = A.shape[0]
            details["size"] = n
            
            # Check for negative values
            if issparse(A):
                if (A < 0).nnz > 0:
                    return ValidationResult(
                        False,
                        f"{name} contains negative values - economically impossible",
                        {"negative_count": (A < 0).nnz}
                    )
                # Check for NaN or infinite values
                if np.any(np.isnan(A.data)) or np.any(np.isinf(A.data)):
                    return ValidationResult(
                        False,
                        f"{name} contains NaN or infinite values",
                        {"nan_count": np.sum(np.isnan(A.data)), "inf_count": np.sum(np.isinf(A.data))}
                    )
            else:
                if np.any(A < 0):
                    return ValidationResult(
                        False,
                        f"{name} contains negative values - economically impossible",
                        {"negative_count": np.sum(A < 0)}
                    )
                if np.any(np.isnan(A)) or np.any(np.isinf(A)):
                    return ValidationResult(
                        False,
                        f"{name} contains NaN or infinite values",
                        {"nan_count": np.sum(np.isnan(A)), "inf_count": np.sum(np.isinf(A))}
                    )
            
            # Check spectral radius (productivity)
            if issparse(A):
                eigenvals = np.linalg.eigvals(A.toarray())
            else:
                eigenvals = np.linalg.eigvals(A)
            
            spectral_radius = np.max(np.abs(eigenvals))
            details["spectral_radius"] = spectral_radius
            
            if spectral_radius >= 1.0:
                return ValidationResult(
                    False,
                    f"{name} represents non-productive economy (spectral radius = {spectral_radius:.4f} >= 1)",
                    {"spectral_radius": spectral_radius}
                )
            
            # Check sparsity for large matrices
            if n > 100:
                if issparse(A):
                    sparsity = 1.0 - A.nnz / (n * n)
                else:
                    sparsity = 1.0 - np.count_nonzero(A) / (n * n)
                details["sparsity"] = sparsity
                
                if not issparse(A) and sparsity > 0.5:
                    warnings.warn(f"{name} is {sparsity:.1%} sparse but stored as dense matrix. Consider using sparse format.")
            
            return ValidationResult(True, f"{name} is valid", details)
            
        except Exception as e:
            return ValidationResult(
                False,
                f"Error validating {name}: {str(e)}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
    @staticmethod
    def validate_demand_vector(d: np.ndarray, n_sectors: int, 
                             name: str = "Final demand") -> ValidationResult:
        """
        Validate a final demand vector.
        
        Args:
            d: Demand vector to validate
            n_sectors: Expected number of sectors
            name: Name of the vector for error messages
            
        Returns:
            ValidationResult with validation status and details
        """
        details = {}
        
        try:
            # Check dimensions
            if d.shape[0] != n_sectors:
                return ValidationResult(
                    False,
                    f"{name} must have {n_sectors} elements, got {d.shape[0]}",
                    {"expected_size": n_sectors, "actual_size": d.shape[0]}
                )
            
            # Check for negative values
            if np.any(d < 0):
                return ValidationResult(
                    False,
                    f"{name} contains negative values - economically impossible",
                    {"negative_count": np.sum(d < 0), "negative_indices": np.where(d < 0)[0].tolist()}
                )
            
            # Check for NaN or infinite values
            if np.any(np.isnan(d)) or np.any(np.isinf(d)):
                return ValidationResult(
                    False,
                    f"{name} contains NaN or infinite values",
                    {"nan_count": np.sum(np.isnan(d)), "inf_count": np.sum(np.isinf(d))}
                )
            
            # Check if all zeros
            if np.all(d == 0):
                warnings.warn(f"{name} is all zeros - no final demand")
            
            details["total_demand"] = np.sum(d)
            details["non_zero_count"] = np.count_nonzero(d)
            
            return ValidationResult(True, f"{name} is valid", details)
            
        except Exception as e:
            return ValidationResult(
                False,
                f"Error validating {name}: {str(e)}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
    @staticmethod
    def validate_labor_vector(l: np.ndarray, n_sectors: int,
                            name: str = "Labor vector") -> ValidationResult:
        """
        Validate a labor input vector.
        
        Args:
            l: Labor vector to validate
            n_sectors: Expected number of sectors
            name: Name of the vector for error messages
            
        Returns:
            ValidationResult with validation status and details
        """
        details = {}
        
        try:
            # Check dimensions
            if l.shape[0] != n_sectors:
                return ValidationResult(
                    False,
                    f"{name} must have {n_sectors} elements, got {l.shape[0]}",
                    {"expected_size": n_sectors, "actual_size": l.shape[0]}
                )
            
            # Check for negative values
            if np.any(l < 0):
                return ValidationResult(
                    False,
                    f"{name} contains negative values - economically impossible",
                    {"negative_count": np.sum(l < 0), "negative_indices": np.where(l < 0)[0].tolist()}
                )
            
            # Check for NaN or infinite values
            if np.any(np.isnan(l)) or np.any(np.isinf(l)):
                return ValidationResult(
                    False,
                    f"{name} contains NaN or infinite values",
                    {"nan_count": np.sum(np.isnan(l)), "inf_count": np.sum(np.isinf(l))}
                )
            
            # Check if all zeros
            if np.all(l == 0):
                warnings.warn(f"{name} is all zeros - no labor input")
            
            details["total_labor"] = np.sum(l)
            details["average_labor_intensity"] = np.mean(l)
            details["max_labor_intensity"] = np.max(l)
            
            return ValidationResult(True, f"{name} is valid", details)
            
        except Exception as e:
            return ValidationResult(
                False,
                f"Error validating {name}: {str(e)}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )


class ConvergenceMonitor:
    """Monitor convergence of iterative algorithms."""
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.history = []
        self.converged = False
        self.iteration_count = 0
    
    def check_convergence(self, current_value: np.ndarray, 
                         previous_value: np.ndarray) -> bool:
        """
        Check if convergence criteria are met.
        
        Args:
            current_value: Current iteration value
            previous_value: Previous iteration value
            
        Returns:
            True if converged, False otherwise
        """
        self.iteration_count += 1
        
        # Calculate change metrics
        absolute_change = np.linalg.norm(current_value - previous_value)
        relative_change = absolute_change / (np.linalg.norm(current_value) + 1e-10)
        
        # Store history
        self.history.append({
            "iteration": self.iteration_count,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
            "current_norm": np.linalg.norm(current_value)
        })
        
        # Check convergence
        if absolute_change < self.tolerance:
            self.converged = True
            return True
        
        if self.iteration_count >= self.max_iterations:
            raise ConvergenceError(
                f"Maximum iterations ({self.max_iterations}) reached without convergence. "
                f"Final change: {absolute_change:.2e} (tolerance: {self.tolerance:.2e})"
            )
        
        return False
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get summary of convergence behavior."""
        if not self.history:
            return {"error": "No convergence history available"}
        
        final_change = self.history[-1]["absolute_change"]
        final_relative_change = self.history[-1]["relative_change"]
        
        return {
            "converged": self.converged,
            "iterations": self.iteration_count,
            "final_absolute_change": final_change,
            "final_relative_change": final_relative_change,
            "convergence_rate": self._calculate_convergence_rate(),
            "history": self.history
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate the rate of convergence."""
        if len(self.history) < 2:
            return 0.0
        
        # Calculate average relative change over last 10 iterations
        recent_changes = [h["relative_change"] for h in self.history[-10:]]
        return np.mean(recent_changes)


class ErrorLogger:
    """Enhanced error logging and debugging utilities."""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger("cybernetic_planning")
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler if not exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_validation_error(self, result: ValidationResult, context: str = ""):
        """Log a validation error with context."""
        self.logger.error(f"Validation failed in {context}: {result.message}")
        if result.details:
            self.logger.error(f"Details: {result.details}")
    
    def log_convergence_failure(self, monitor: ConvergenceMonitor, context: str = ""):
        """Log convergence failure with details."""
        summary = monitor.get_convergence_summary()
        self.logger.error(f"Convergence failed in {context} after {summary['iterations']} iterations")
        self.logger.error(f"Final change: {summary['final_absolute_change']:.2e}")
    
    def log_optimization_failure(self, error: Exception, context: str = ""):
        """Log optimization failure with details."""
        self.logger.error(f"Optimization failed in {context}: {str(error)}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")


def safe_matrix_operation(operation_func, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
    """
    Safely execute a matrix operation with error handling.
    
    Args:
        operation_func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (result, error). Result is None if error occurred.
    """
    try:
        result = operation_func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e


def validate_economic_data(technology_matrix: Union[np.ndarray, csr_matrix, csc_matrix],
                         final_demand: np.ndarray,
                         labor_vector: np.ndarray) -> List[ValidationResult]:
    """
    Validate all economic data components.
    
    Args:
        technology_matrix: Technology matrix A
        final_demand: Final demand vector d
        labor_vector: Labor input vector l
        
    Returns:
        List of validation results
    """
    validator = MatrixValidator()
    results = []
    
    # Validate technology matrix
    results.append(validator.validate_technology_matrix(technology_matrix))
    
    # Validate demand vector
    n_sectors = technology_matrix.shape[0]
    results.append(validator.validate_demand_vector(final_demand, n_sectors))
    
    # Validate labor vector
    results.append(validator.validate_labor_vector(labor_vector, n_sectors))
    
    return results


def check_data_consistency(technology_matrix: Union[np.ndarray, csr_matrix, csc_matrix],
                          final_demand: np.ndarray,
                          labor_vector: np.ndarray) -> ValidationResult:
    """
    Check consistency between different data components.
    
    Args:
        technology_matrix: Technology matrix A
        final_demand: Final demand vector d
        labor_vector: Labor input vector l
        
    Returns:
        ValidationResult with consistency check results
    """
    details = {}
    
    try:
        n = technology_matrix.shape[0]
        
        # Check dimension consistency
        if final_demand.shape[0] != n or labor_vector.shape[0] != n:
            return ValidationResult(
                False,
                "Inconsistent dimensions between matrices and vectors",
                {
                    "matrix_size": n,
                    "demand_size": final_demand.shape[0],
                    "labor_size": labor_vector.shape[0]
                }
            )
        
        # Check for reasonable economic relationships
        if issparse(technology_matrix):
            A_dense = technology_matrix.toarray()
        else:
            A_dense = technology_matrix
        
        # Check if total intermediate demand is reasonable compared to final demand
        intermediate_demand = A_dense @ final_demand
        total_intermediate = np.sum(intermediate_demand)
        total_final = np.sum(final_demand)
        
        if total_final > 0:
            intermediate_ratio = total_intermediate / total_final
            details["intermediate_ratio"] = intermediate_ratio
            
            if intermediate_ratio > 10:  # Unrealistically high intermediate demand
                warnings.warn(f"Very high intermediate demand ratio: {intermediate_ratio:.2f}")
        
        # Check labor intensity distribution
        labor_std = np.std(labor_vector)
        labor_mean = np.mean(labor_vector)
        
        if labor_mean > 0:
            labor_cv = labor_std / labor_mean  # Coefficient of variation
            details["labor_cv"] = labor_cv
            
            if labor_cv > 5:  # Very high variation in labor intensity
                warnings.warn(f"Very high variation in labor intensity: CV = {labor_cv:.2f}")
        
        return ValidationResult(True, "Data consistency check passed", details)
        
    except Exception as e:
        return ValidationResult(
            False,
            f"Error in consistency check: {str(e)}",
            {"error": str(e), "traceback": traceback.format_exc()}
        )
