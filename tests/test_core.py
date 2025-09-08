"""
Unit tests for core mathematical algorithms.
"""

import pytest
import numpy as np
from src.cybernetic_planning.core import LeontiefModel, LaborValueCalculator, ConstrainedOptimizer, DynamicPlanner


class TestLeontiefModel:
    """Test cases for LeontiefModel class."""
    
    def test_initialization(self):
        """Test LeontiefModel initialization."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        d = np.array([100, 200])
        
        model = LeontiefModel(A, d)
        assert model.A.shape == (2, 2)
        assert model.d.shape == (2,)
    
    def test_compute_total_output(self):
        """Test total output computation."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        d = np.array([100, 200])
        
        model = LeontiefModel(A, d)
        x = model.compute_total_output()
        
        # Check that x is positive
        assert np.all(x > 0)
        
        # Check that (I - A)x = d
        I = np.eye(2)
        assert np.allclose((I - A) @ x, d)
    
    def test_spectral_radius(self):
        """Test spectral radius calculation."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        d = np.array([100, 200])
        
        model = LeontiefModel(A, d)
        spectral_radius = model.get_spectral_radius()
        
        assert 0 <= spectral_radius < 1
        assert isinstance(spectral_radius, float)
    
    def test_is_productive(self):
        """Test productivity check."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        d = np.array([100, 200])
        
        model = LeontiefModel(A, d)
        assert model.is_productive()
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        d = np.array([100, 200])
        
        model = LeontiefModel(A, d)
        
        # Test technology sensitivity
        sens_A = model.sensitivity_analysis('A')
        assert sens_A.shape == (2, 2, 2)
        
        # Test demand sensitivity
        sens_d = model.sensitivity_analysis('d')
        assert sens_d.shape == (2, 2)


class TestLaborValueCalculator:
    """Test cases for LaborValueCalculator class."""
    
    def test_initialization(self):
        """Test LaborValueCalculator initialization."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        l = np.array([0.5, 0.8])
        
        calc = LaborValueCalculator(A, l)
        assert calc.A.shape == (2, 2)
        assert calc.l.shape == (2,)
    
    def test_compute_labor_values(self):
        """Test labor values computation."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        l = np.array([0.5, 0.8])
        
        calc = LaborValueCalculator(A, l)
        v = calc.get_labor_values()
        
        # Check that labor values are positive
        assert np.all(v > 0)
        assert len(v) == 2
    
    def test_compute_total_labor_cost(self):
        """Test total labor cost computation."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        l = np.array([0.5, 0.8])
        d = np.array([100, 200])
        
        calc = LaborValueCalculator(A, l)
        total_cost = calc.compute_total_labor_cost(d)
        
        assert total_cost > 0
        assert isinstance(total_cost, float)
    
    def test_labor_productivity(self):
        """Test labor productivity calculation."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        l = np.array([0.5, 0.8])
        
        calc = LaborValueCalculator(A, l)
        productivity = calc.compute_labor_productivity()
        
        assert len(productivity) == 2
        assert np.all(productivity >= 0)


class TestConstrainedOptimizer:
    """Test cases for ConstrainedOptimizer class."""
    
    def test_initialization(self):
        """Test ConstrainedOptimizer initialization."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        l = np.array([0.5, 0.8])
        d = np.array([100, 200])
        
        optimizer = ConstrainedOptimizer(A, l, d)
        assert optimizer.A.shape == (2, 2)
        assert optimizer.l.shape == (2,)
        assert optimizer.d.shape == (2,)
    
    def test_solve_optimization(self):
        """Test optimization solving."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        l = np.array([0.5, 0.8])
        d = np.array([100, 200])
        
        optimizer = ConstrainedOptimizer(A, l, d)
        result = optimizer.solve()
        
        assert 'status' in result
        assert 'solution' in result
        assert 'feasible' in result
    
    def test_constraint_checking(self):
        """Test constraint satisfaction checking."""
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        l = np.array([0.5, 0.8])
        d = np.array([100, 200])
        
        optimizer = ConstrainedOptimizer(A, l, d)
        result = optimizer.solve()
        
        if result['feasible']:
            solution = result['solution']
            constraints = optimizer.check_constraints(solution)
            
            assert 'demand_satisfied' in constraints
            assert 'non_negative' in constraints


class TestDynamicPlanner:
    """Test cases for DynamicPlanner class."""
    
    def test_initialization(self):
        """Test DynamicPlanner initialization."""
        A_0 = np.array([[0.1, 0.2], [0.3, 0.1]])
        l_0 = np.array([0.5, 0.8])
        
        planner = DynamicPlanner(A_0, l_0)
        assert planner.A_0.shape == (2, 2)
        assert planner.l_0.shape == (2,)
    
    def test_plan_year(self):
        """Test single year planning."""
        A_0 = np.array([[0.1, 0.2], [0.3, 0.1]])
        l_0 = np.array([0.5, 0.8])
        
        planner = DynamicPlanner(A_0, l_0)
        
        consumption_demand = np.array([100, 200])
        investment_demand = np.array([20, 40])
        
        plan = planner.plan_year(1, consumption_demand, investment_demand)
        
        assert 'total_output' in plan
        assert 'total_labor_cost' in plan
        assert len(plan['total_output']) == 2
    
    def test_five_year_plan(self):
        """Test five-year planning."""
        A_0 = np.array([[0.1, 0.2], [0.3, 0.1]])
        l_0 = np.array([0.5, 0.8])
        
        planner = DynamicPlanner(A_0, l_0)
        
        consumption_demands = [np.array([100, 200]) for _ in range(5)]
        investment_demands = [np.array([20, 40]) for _ in range(5)]
        
        plans = planner.create_five_year_plan(consumption_demands, investment_demands)
        
        assert len(plans) == 5
        for year in range(1, 6):
            assert year in plans
            assert 'total_output' in plans[year]
