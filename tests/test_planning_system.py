"""
Unit tests for the main planning system.
"""

import pytest
import numpy as np
from src.cybernetic_planning.planning_system import CyberneticPlanningSystem


class TestCyberneticPlanningSystem:
    """Test cases for CyberneticPlanningSystem class."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = CyberneticPlanningSystem()
        assert system.parser is not None
        assert system.matrix_builder is not None
        assert system.validator is not None
        assert system.manager_agent is not None
    
    def test_create_synthetic_data(self):
        """Test synthetic data creation."""
        system = CyberneticPlanningSystem()
        
        data = system.create_synthetic_data(n_sectors=5, technology_density=0.3)
        
        assert 'technology_matrix' in data
        assert 'final_demand' in data
        assert 'labor_input' in data
        assert data['technology_matrix'].shape == (5, 5)
        assert len(data['final_demand']) == 5
        assert len(data['labor_input']) == 5
    
    def test_load_data_from_dict(self):
        """Test loading data from dictionary."""
        system = CyberneticPlanningSystem()
        
        data = {
            'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
            'final_demand': np.array([100, 200]),
            'labor_input': np.array([0.5, 0.8])
        }
        
        result = system.load_data_from_dict(data)
        assert result == data
        assert system.current_data == data
    
    def test_create_plan(self):
        """Test plan creation."""
        system = CyberneticPlanningSystem()
        
        # Create synthetic data first
        system.create_synthetic_data(n_sectors=3, technology_density=0.3)
        
        # Create plan
        plan = system.create_plan()
        
        assert 'total_output' in plan
        assert 'total_labor_cost' in plan
        assert 'final_demand' in plan
        assert 'labor_values' in plan
        assert len(plan['total_output']) == 3
    
    def test_create_plan_with_policy_goals(self):
        """Test plan creation with policy goals."""
        system = CyberneticPlanningSystem()
        
        # Create synthetic data first
        system.create_synthetic_data(n_sectors=3, technology_density=0.3)
        
        policy_goals = ["Increase healthcare capacity by 15%", "Reduce carbon emissions by 20%"]
        
        # Create plan with policy goals
        plan = system.create_plan(policy_goals=policy_goals)
        
        assert 'total_output' in plan
        assert 'total_labor_cost' in plan
        assert len(plan['total_output']) == 3
    
    def test_create_five_year_plan(self):
        """Test five-year plan creation."""
        system = CyberneticPlanningSystem()
        
        # Create synthetic data first
        system.create_synthetic_data(n_sectors=3, technology_density=0.3)
        
        # Create five-year plan
        five_year_plan = system.create_five_year_plan()
        
        assert len(five_year_plan) == 5
        for year in range(1, 6):
            assert year in five_year_plan
            assert 'total_output' in five_year_plan[year]
            assert 'total_labor_cost' in five_year_plan[year]
    
    def test_generate_report(self):
        """Test report generation."""
        system = CyberneticPlanningSystem()
        
        # Create synthetic data and plan
        system.create_synthetic_data(n_sectors=3, technology_density=0.3)
        plan = system.create_plan()
        
        # Generate report
        report = system.generate_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Executive Summary" in report
        assert "Sector-by-Sector Analysis" in report
    
    def test_get_plan_summary(self):
        """Test plan summary retrieval."""
        system = CyberneticPlanningSystem()
        
        # Create synthetic data and plan
        system.create_synthetic_data(n_sectors=3, technology_density=0.3)
        plan = system.create_plan()
        
        # Get summary
        summary = system.get_plan_summary()
        
        assert 'total_economic_output' in summary
        assert 'total_labor_cost' in summary
        assert 'labor_efficiency' in summary
        assert 'sector_count' in summary
    
    def test_get_system_status(self):
        """Test system status retrieval."""
        system = CyberneticPlanningSystem()
        
        status = system.get_system_status()
        
        assert 'data_loaded' in status
        assert 'current_plan_available' in status
        assert 'plan_history_count' in status
        assert 'agent_status' in status
    
    def test_reset_system(self):
        """Test system reset."""
        system = CyberneticPlanningSystem()
        
        # Create some data and plans
        system.create_synthetic_data(n_sectors=3, technology_density=0.3)
        system.create_plan()
        
        # Reset system
        system.reset_system()
        
        assert not system.current_data
        assert system.current_plan is None
        assert len(system.plan_history) == 0
    
    def test_run_demo(self):
        """Test demo execution."""
        system = CyberneticPlanningSystem()
        
        result = system.run_demo()
        
        assert 'data' in result
        assert 'plan' in result
        assert 'report' in result
        assert 'summary' in result
        assert result['data']['technology_matrix'].shape == (8, 8)
        assert len(result['plan']['total_output']) == 8
