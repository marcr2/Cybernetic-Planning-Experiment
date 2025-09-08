"""
Unit tests for multi-agent system.
"""

import pytest
import numpy as np
from src.cybernetic_planning.agents import (
    ManagerAgent, EconomicsAgent, ResourceAgent, PolicyAgent, WriterAgent
)


class TestManagerAgent:
    """Test cases for ManagerAgent class."""
    
    def test_initialization(self):
        """Test ManagerAgent initialization."""
        agent = ManagerAgent()
        assert agent.agent_id == "manager"
        assert agent.name == "Central Planning Manager"
        assert agent.active
    
    def test_create_plan(self):
        """Test plan creation."""
        agent = ManagerAgent()
        
        task = {
            'type': 'create_plan',
            'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
            'final_demand': np.array([100, 200]),
            'labor_vector': np.array([0.5, 0.8])
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'plan' in result
    
    def test_get_capabilities(self):
        """Test capabilities retrieval."""
        agent = ManagerAgent()
        capabilities = agent.get_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert 'plan_coordination' in capabilities


class TestEconomicsAgent:
    """Test cases for EconomicsAgent class."""
    
    def test_initialization(self):
        """Test EconomicsAgent initialization."""
        agent = EconomicsAgent()
        assert agent.agent_id == "economics"
        assert agent.name == "Economics Analysis Agent"
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        agent = EconomicsAgent()
        
        task = {
            'type': 'sensitivity_analysis',
            'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
            'final_demand': np.array([100, 200])
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'technology_sensitivity' in result
        assert 'critical_sectors' in result
    
    def test_technology_forecast(self):
        """Test technology forecasting."""
        agent = EconomicsAgent()
        
        task = {
            'type': 'technology_forecast',
            'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
            'time_horizon': 5
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'forecast_matrices' in result


class TestResourceAgent:
    """Test cases for ResourceAgent class."""
    
    def test_initialization(self):
        """Test ResourceAgent initialization."""
        agent = ResourceAgent()
        assert agent.agent_id == "resource"
        assert agent.name == "Resource & Environmental Agent"
    
    def test_resource_optimization(self):
        """Test resource optimization."""
        agent = ResourceAgent()
        
        task = {
            'type': 'resource_optimization',
            'current_plan': {
                'total_output': np.array([100, 200]),
                'resource_usage': np.array([50, 75, 100]),
                'max_resources': np.array([100, 100, 100])
            }
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'current_usage' in result
        assert 'resource_utilization' in result
    
    def test_environmental_assessment(self):
        """Test environmental impact assessment."""
        agent = ResourceAgent()
        
        task = {
            'type': 'environmental_assessment',
            'current_plan': {
                'total_output': np.array([100, 200])
            },
            'environmental_matrix': np.array([[0.1, 0.2], [0.3, 0.1], [0.05, 0.15]])
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'environmental_impacts' in result


class TestPolicyAgent:
    """Test cases for PolicyAgent class."""
    
    def test_initialization(self):
        """Test PolicyAgent initialization."""
        agent = PolicyAgent()
        assert agent.agent_id == "policy"
        assert agent.name == "Social Policy Agent"
    
    def test_goal_translation(self):
        """Test goal translation."""
        agent = PolicyAgent()
        
        task = {
            'type': 'goal_translation',
            'goals': ["Increase healthcare capacity by 15%", "Reduce carbon emissions by 20%"]
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'translated_goals' in result
    
    def test_policy_adjustment(self):
        """Test policy adjustment."""
        agent = PolicyAgent()
        
        task = {
            'type': 'policy_adjustment',
            'current_plan': {
                'total_output': np.array([100, 200]),
                'final_demand': np.array([80, 150]),
                'labor_vector': np.array([0.5, 0.8])
            },
            'goals': ["Increase healthcare capacity by 15%"]
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'adjusted_plan' in result


class TestWriterAgent:
    """Test cases for WriterAgent class."""
    
    def test_initialization(self):
        """Test WriterAgent initialization."""
        agent = WriterAgent()
        assert agent.agent_id == "writer"
        assert agent.name == "Report Generation Agent"
    
    def test_generate_report(self):
        """Test report generation."""
        agent = WriterAgent()
        
        plan_data = {
            'total_output': np.array([100, 200]),
            'final_demand': np.array([80, 150]),
            'labor_values': np.array([0.5, 0.8]),
            'total_labor_cost': 150.0,
            'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]])
        }
        
        task = {
            'type': 'generate_report',
            'plan_data': plan_data
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'report' in result
        assert len(result['report']) > 0
    
    def test_generate_summary(self):
        """Test executive summary generation."""
        agent = WriterAgent()
        
        plan_data = {
            'total_output': np.array([100, 200]),
            'final_demand': np.array([80, 150]),
            'total_labor_cost': 150.0
        }
        
        task = {
            'type': 'generate_summary',
            'plan_data': plan_data
        }
        
        result = agent.process_task(task)
        assert result['status'] == 'success'
        assert 'summary' in result
        assert 'metrics' in result
