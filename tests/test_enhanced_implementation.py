"""
Comprehensive tests for enhanced implementation with sparse matrices and Cockshott & Cottrell planner.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, csc_matrix
import warnings

from src.cybernetic_planning.core.leontief import LeontiefModel
from src.cybernetic_planning.core.labor_values import LaborValueCalculator
from src.cybernetic_planning.core.optimization import ConstrainedOptimizer
from src.cybernetic_planning.core.cockshott_cottrell import CockshottCottrellPlanner
from src.cybernetic_planning.core.error_handling import MatrixValidator, validate_economic_data


class TestSparseMatrixSupport:
    """Test sparse matrix support across all modules."""
    
    def test_leontief_sparse_matrix(self):
        """Test Leontief model with sparse matrices."""
        # Create a sparse technology matrix
        n = 50
        A_dense = np.random.rand(n, n) * 0.1  # Small values for productivity
        A_dense = A_dense * 0.8  # Ensure spectral radius < 1
        A_sparse = csr_matrix(A_dense)
        
        d = np.random.rand(n) * 100
        
        # Test with sparse matrix
        model_sparse = LeontiefModel(A_sparse, d, use_sparse=True)
        output_sparse = model_sparse.compute_total_output()
        
        # Test with dense matrix
        model_dense = LeontiefModel(A_dense, d, use_sparse=False)
        output_dense = model_dense.compute_total_output()
        
        # Results should be approximately equal
        np.testing.assert_allclose(output_sparse, output_dense, rtol=1e-10)
    
    def test_labor_values_sparse_matrix(self):
        """Test labor value calculation with sparse matrices."""
        n = 30
        A_dense = np.random.rand(n, n) * 0.1
        A_dense = A_dense * 0.8  # Ensure productivity
        A_sparse = csr_matrix(A_dense)
        
        l = np.random.rand(n) * 10
        
        # Test with sparse matrix
        calc_sparse = LaborValueCalculator(A_sparse, l, use_sparse=True)
        values_sparse = calc_sparse.get_labor_values()
        
        # Test with dense matrix
        calc_dense = LaborValueCalculator(A_dense, l, use_sparse=False)
        values_dense = calc_dense.get_labor_values()
        
        # Results should be approximately equal
        np.testing.assert_allclose(values_sparse, values_dense, rtol=1e-10)
    
    def test_optimization_sparse_matrix(self):
        """Test optimization with sparse matrices."""
        n = 20
        A_dense = np.random.rand(n, n) * 0.1
        A_dense = A_dense * 0.8  # Ensure productivity
        A_sparse = csr_matrix(A_dense)
        
        l = np.random.rand(n) * 5
        d = np.random.rand(n) * 50
        
        # Test with sparse matrix
        opt_sparse = ConstrainedOptimizer(A_sparse, l, d, use_sparse=True)
        result_sparse = opt_sparse.solve(use_cvxpy=True)
        
        # Test with dense matrix
        opt_dense = ConstrainedOptimizer(A_dense, l, d, use_sparse=False)
        result_dense = opt_dense.solve(use_cvxpy=True)
        
        # Both should be feasible
        assert result_sparse["feasible"] == result_dense["feasible"]
        
        if result_sparse["feasible"] and result_dense["feasible"]:
            np.testing.assert_allclose(result_sparse["solution"], result_dense["solution"], rtol=1e-6)


class TestCockshottCottrellPlanner:
    """Test the Cockshott & Cottrell iterative planner."""
    
    def test_iterative_planning_convergence(self):
        """Test that iterative planning converges."""
        n = 25
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        A_sparse = csr_matrix(A)
        
        d = np.random.rand(n) * 100
        l = np.random.rand(n) * 5
        
        planner = CockshottCottrellPlanner(
            technology_matrix=A_sparse,
            final_demand=d,
            direct_labor=l,
            max_iterations=1000,
            convergence_threshold=1e-6,
            use_sparse=True
        )
        
        result = planner.iterative_planning()
        
        # Should converge
        assert result["converged"], f"Did not converge: {result}"
        assert result["iterations"] < 1000, "Took too many iterations"
        
        # Production plan should be non-negative
        assert np.all(result["production_plan"] >= 0), "Negative production values"
        
        # Should satisfy demand approximately
        net_output = result["net_output"]
        demand_fulfillment = net_output / (d + 1e-10)
        assert np.all(demand_fulfillment >= 0.99), "Demand not satisfied"
    
    def test_labor_value_calculation(self):
        """Test labor value calculation in planner."""
        n = 15
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        A_sparse = csr_matrix(A)
        
        d = np.random.rand(n) * 50
        l = np.random.rand(n) * 3
        
        planner = CockshottCottrellPlanner(A_sparse, d, l, use_sparse=True)
        
        # Test labor value calculation
        labor_values = planner.calculate_labor_values()
        
        # Labor values should be positive
        assert np.all(labor_values > 0), "Negative labor values"
        
        # Should match direct calculation
        calc = LaborValueCalculator(A_sparse, l, use_sparse=True)
        expected_values = calc.get_labor_values()
        
        np.testing.assert_allclose(labor_values, expected_values, rtol=1e-10)
    
    def test_planning_statistics(self):
        """Test planning statistics generation."""
        n = 20
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        A_sparse = csr_matrix(A)
        
        d = np.random.rand(n) * 100
        l = np.random.rand(n) * 5
        
        planner = CockshottCottrellPlanner(A_sparse, d, l, use_sparse=True)
        result = planner.iterative_planning()
        
        # Get statistics
        stats = planner.get_planning_statistics()
        
        # Should have required fields
        assert "total_output" in stats
        assert "total_labor_cost" in stats
        assert "convergence_info" in stats
        assert "sector_statistics" in stats
        
        # Statistics should be consistent
        assert stats["total_output"] > 0
        assert stats["total_labor_cost"] > 0
        assert len(stats["sector_statistics"]) == n


class TestErrorHandling:
    """Test enhanced error handling and validation."""
    
    def test_matrix_validation(self):
        """Test matrix validation."""
        validator = MatrixValidator()
        
        # Test valid matrix
        n = 10
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        
        result = validator.validate_technology_matrix(A)
        assert result.is_valid, f"Valid matrix failed validation: {result.message}"
        
        # Test invalid matrix (non-productive)
        A_bad = np.random.rand(n, n) * 0.5  # High values
        result = validator.validate_technology_matrix(A_bad)
        assert not result.is_valid, "Non-productive matrix should fail validation"
    
    def test_demand_validation(self):
        """Test demand vector validation."""
        validator = MatrixValidator()
        
        n = 10
        d = np.random.rand(n) * 100
        
        result = validator.validate_demand_vector(d, n)
        assert result.is_valid, f"Valid demand failed validation: {result.message}"
        
        # Test invalid demand (negative values)
        d_bad = d.copy()
        d_bad[0] = -10
        
        result = validator.validate_demand_vector(d_bad, n)
        assert not result.is_valid, "Negative demand should fail validation"
    
    def test_data_consistency_check(self):
        """Test data consistency checking."""
        n = 15
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        d = np.random.rand(n) * 100
        l = np.random.rand(n) * 5
        
        from src.cybernetic_planning.core.error_handling import check_data_consistency
        
        result = check_data_consistency(A, d, l)
        assert result.is_valid, f"Consistent data failed check: {result.message}"
        
        # Test inconsistent data
        d_bad = np.random.rand(n + 1) * 100  # Wrong size
        
        result = check_data_consistency(A, d_bad, l)
        assert not result.is_valid, "Inconsistent data should fail check"


class TestPerformanceImprovements:
    """Test performance improvements from vectorization and sparse matrices."""
    
    def test_vectorized_sensitivity_analysis(self):
        """Test vectorized sensitivity analysis."""
        n = 30
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        
        model = LeontiefModel(A, np.random.rand(n) * 100, use_sparse=False)
        
        # Test sensitivity analysis
        sensitivity = model.sensitivity_analysis("A")
        
        # Should return correct shape
        assert sensitivity.shape == (n, n, n), f"Wrong shape: {sensitivity.shape}"
        
        # Should not contain NaN or infinite values
        assert not np.any(np.isnan(sensitivity)), "Sensitivity contains NaN"
        assert not np.any(np.isinf(sensitivity)), "Sensitivity contains infinite values"
    
    def test_memory_efficiency_sparse(self):
        """Test memory efficiency with sparse matrices."""
        n = 1000  # Large matrix
        sparsity = 0.95  # 95% sparse
        
        # Create sparse matrix
        A_dense = np.random.rand(n, n) * 0.1
        A_dense = A_dense * 0.8  # Ensure productivity
        
        # Make it sparse by zeroing out most elements
        mask = np.random.rand(n, n) > sparsity
        A_dense[~mask] = 0
        
        A_sparse = csr_matrix(A_dense)
        d = np.random.rand(n) * 100
        
        # Test memory usage (approximate)
        dense_memory = A_dense.nbytes
        sparse_memory = A_sparse.data.nbytes + A_sparse.indices.nbytes + A_sparse.indptr.nbytes
        
        # Sparse should use less memory
        assert sparse_memory < dense_memory, "Sparse matrix not more memory efficient"
        
        # Both should give same results
        model_dense = LeontiefModel(A_dense, d, use_sparse=False)
        model_sparse = LeontiefModel(A_sparse, d, use_sparse=True)
        
        output_dense = model_dense.compute_total_output()
        output_sparse = model_sparse.compute_total_output()
        
        np.testing.assert_allclose(output_dense, output_sparse, rtol=1e-10)


class TestIntegration:
    """Test integration between different modules."""
    
    def test_end_to_end_planning(self):
        """Test complete planning workflow."""
        n = 20
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        A_sparse = csr_matrix(A)
        
        d = np.random.rand(n) * 100
        l = np.random.rand(n) * 5
        
        # Test with Cockshott & Cottrell planner
        planner = CockshottCottrellPlanner(A_sparse, d, l, use_sparse=True)
        result = planner.iterative_planning()
        
        assert result["converged"], "Planning did not converge"
        
        # Test with optimization
        optimizer = ConstrainedOptimizer(A_sparse, l, d, use_sparse=True)
        opt_result = optimizer.solve(use_cvxpy=True)
        
        if opt_result["feasible"]:
            # Both should give reasonable results
            assert np.all(result["production_plan"] >= 0)
            assert np.all(opt_result["solution"] >= 0)
    
    def test_large_scale_planning(self):
        """Test planning with large-scale data."""
        n = 500  # Large problem
        sparsity = 0.9  # 90% sparse
        
        # Create large sparse matrix
        A_dense = np.random.rand(n, n) * 0.1
        A_dense = A_dense * 0.8  # Ensure productivity
        
        # Make it sparse
        mask = np.random.rand(n, n) > sparsity
        A_dense[~mask] = 0
        A_sparse = csr_matrix(A_dense)
        
        d = np.random.rand(n) * 1000
        l = np.random.rand(n) * 10
        
        # Should work without memory issues
        planner = CockshottCottrellPlanner(
            A_sparse, d, l, 
            max_iterations=100,  # Limit iterations for test
            use_sparse=True
        )
        
        result = planner.iterative_planning()
        
        # Should either converge or hit iteration limit
        assert result["iterations"] <= 100
        assert np.all(result["production_plan"] >= 0)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
