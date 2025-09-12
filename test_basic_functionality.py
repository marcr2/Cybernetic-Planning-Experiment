"""
Basic functionality test for the enhanced implementation.
Tests core functionality without external dependencies.
"""

import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cockshott_cottrell_planner():
    """Test the Cockshott & Cottrell planner."""
    print("Testing Cockshott & Cottrell planner...")
    
    try:
        # Import without triggering cvxpy dependency
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from cybernetic_planning.core.cockshott_cottrell import CockshottCottrellPlanner
        
        # Create test data
        n = 10
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        A_sparse = csr_matrix(A)
        
        d = np.random.rand(n) * 100
        l = np.random.rand(n) * 5
        
        # Create planner
        planner = CockshottCottrellPlanner(
            technology_matrix=A_sparse,
            final_demand=d,
            direct_labor=l,
            max_iterations=100,
            convergence_threshold=1e-6,
            use_sparse=True
        )
        
        # Test iterative planning
        try:
            result = planner.iterative_planning()
            
            print(f"  - Converged: {result['converged']}")
            print(f"  - Iterations: {result['iterations']}")
            print(f"  - Total output: {np.sum(result['production_plan']):.2f}")
            print(f"  - Total labor cost: {result['total_labor_cost']:.2f}")
        except Exception as e:
            print(f"  - Error in iterative planning: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test labor value calculation
        labor_values = planner.calculate_labor_values()
        print(f"  - Labor values calculated: {len(labor_values)} values")
        print(f"  - All positive: {np.all(labor_values > 0)}")
        
        print("✓ Cockshott & Cottrell planner test passed")
        return True
        
    except Exception as e:
        print(f"✗ Cockshott & Cottrell planner test failed: {e}")
        return False

def test_sparse_matrix_support():
    """Test sparse matrix support."""
    print("Testing sparse matrix support...")
    
    try:
        from cybernetic_planning.core.leontief import LeontiefModel
        from cybernetic_planning.core.labor_values import LaborValueCalculator
        
        # Create test data
        n = 20
        A_dense = np.random.rand(n, n) * 0.1
        A_dense = A_dense * 0.8  # Ensure productivity
        A_sparse = csr_matrix(A_dense)
        
        d = np.random.rand(n) * 100
        l = np.random.rand(n) * 5
        
        # Test Leontief model with sparse matrix
        model_sparse = LeontiefModel(A_sparse, d, use_sparse=True)
        output_sparse = model_sparse.compute_total_output()
        
        # Test Leontief model with dense matrix
        model_dense = LeontiefModel(A_dense, d, use_sparse=False)
        output_dense = model_dense.compute_total_output()
        
        # Results should be approximately equal
        diff = np.max(np.abs(output_sparse - output_dense))
        print(f"  - Max difference between sparse and dense: {diff:.2e}")
        
        if diff < 1e-10:
            print("✓ Sparse matrix support test passed")
            return True
        else:
            print("✗ Sparse matrix support test failed: Results differ")
            return False
            
    except Exception as e:
        print(f"✗ Sparse matrix support test failed: {e}")
        return False

def test_error_handling():
    """Test enhanced error handling."""
    print("Testing error handling...")
    
    try:
        from cybernetic_planning.core.error_handling import MatrixValidator, validate_economic_data
        
        # Create test data
        n = 15
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        d = np.random.rand(n) * 100
        l = np.random.rand(n) * 5
        
        # Test validation
        validator = MatrixValidator()
        result = validator.validate_technology_matrix(A)
        
        print(f"  - Matrix validation: {result.is_valid}")
        
        # Test data validation
        results = validate_economic_data(A, d, l)
        all_valid = all(r.is_valid for r in results)
        
        print(f"  - All data valid: {all_valid}")
        
        if all_valid:
            print("✓ Error handling test passed")
            return True
        else:
            print("✗ Error handling test failed: Validation failed")
            return False
            
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_vectorized_operations():
    """Test vectorized operations."""
    print("Testing vectorized operations...")
    
    try:
        from cybernetic_planning.core.leontief import LeontiefModel
        
        # Create test data
        n = 25
        A = np.random.rand(n, n) * 0.05  # Smaller values
        A = A * 0.7  # Ensure productivity (spectral radius < 1)
        d = np.random.rand(n) * 100
        
        model = LeontiefModel(A, d, use_sparse=False)
        
        # Test sensitivity analysis (vectorized)
        sensitivity = model.sensitivity_analysis("A")
        
        print(f"  - Sensitivity shape: {sensitivity.shape}")
        print(f"  - No NaN values: {not np.any(np.isnan(sensitivity))}")
        print(f"  - No infinite values: {not np.any(np.isinf(sensitivity))}")
        
        if sensitivity.shape == (n, n, n) and not np.any(np.isnan(sensitivity)) and not np.any(np.isinf(sensitivity)):
            print("✓ Vectorized operations test passed")
            return True
        else:
            print("✗ Vectorized operations test failed")
            return False
            
    except Exception as e:
        print(f"✗ Vectorized operations test failed: {e}")
        return False

def test_optimization_without_cvxpy():
    """Test optimization without cvxpy dependency."""
    print("Testing optimization without cvxpy...")
    
    try:
        from cybernetic_planning.core.optimization import ConstrainedOptimizer
        
        # Create test data
        n = 15
        A = np.random.rand(n, n) * 0.1
        A = A * 0.8  # Ensure productivity
        l = np.random.rand(n) * 5
        d = np.random.rand(n) * 50
        
        # Test with scipy only
        optimizer = ConstrainedOptimizer(A, l, d, use_sparse=False)
        result = optimizer.solve(use_cvxpy=False)
        
        print(f"  - Optimization feasible: {result['feasible']}")
        if result['feasible']:
            print(f"  - Solution found: {np.all(result['solution'] >= 0)}")
            print(f"  - Total labor cost: {result['total_labor_cost']:.2f}")
        
        print("✓ Optimization without cvxpy test passed")
        return True
        
    except Exception as e:
        print(f"✗ Optimization without cvxpy test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running basic functionality tests...\n")
    
    tests = [
        test_cockshott_cottrell_planner,
        test_sparse_matrix_support,
        test_error_handling,
        test_vectorized_operations,
        test_optimization_without_cvxpy
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()
