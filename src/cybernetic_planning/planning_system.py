"""
Main Planning System

Orchestrates the complete cybernetic planning process, integrating
all components to generate comprehensive economic plans.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from .core import LeontiefModel, LaborValueCalculator, ConstrainedOptimizer, DynamicPlanner
from .agents import ManagerAgent, EconomicsAgent, ResourceAgent, PolicyAgent, WriterAgent
from .data import IOParser, MatrixBuilder, DataValidator


class CyberneticPlanningSystem:
    """
    Main system for cybernetic central planning.
    
    Integrates all components to generate comprehensive 5-year economic plans
    using Input-Output analysis and labor-time accounting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cybernetic planning system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.parser = IOParser()
        self.matrix_builder = MatrixBuilder()
        self.validator = DataValidator()
        
        # Initialize agents
        self.manager_agent = ManagerAgent()
        self.economics_agent = EconomicsAgent()
        self.resource_agent = ResourceAgent()
        self.policy_agent = PolicyAgent()
        self.writer_agent = WriterAgent()
        
        # System state
        self.current_data = {}
        self.current_plan = None
        self.plan_history = []
        
    def load_data_from_file(self, file_path: Union[str, Path], 
                           format_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Load economic data from a file.
        
        Args:
            file_path: Path to the data file
            format_type: File format (auto-detected if None)
            
        Returns:
            Loaded data dictionary
        """
        try:
            data = self.parser.parse_file(file_path, format_type)
            
            # Validate loaded data
            validation_results = self.validator.validate_all(data)
            
            if not validation_results['overall_valid']:
                print("Warning: Data validation failed")
                print(f"Errors: {validation_results['summary']['total_errors']}")
                print(f"Warnings: {validation_results['summary']['total_warnings']}")
            
            self.current_data = data
            return data
            
        except Exception as e:
            raise ValueError(f"Error loading data from file: {e}")
    
    def load_data_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load economic data from a dictionary.
        
        Args:
            data: Data dictionary containing economic matrices and vectors
            
        Returns:
            Validated data dictionary
        """
        # Validate data
        validation_results = self.validator.validate_all(data)
        
        if not validation_results['overall_valid']:
            print("Warning: Data validation failed")
            print(f"Errors: {validation_results['summary']['total_errors']}")
            print(f"Warnings: {validation_results['summary']['total_warnings']}")
        
        self.current_data = data
        return data
    
    def create_synthetic_data(self, n_sectors: int = 10, 
                            technology_density: float = 0.3,
                            resource_count: int = 5) -> Dict[str, Any]:
        """
        Create synthetic economic data for testing and demonstration.
        
        Args:
            n_sectors: Number of economic sectors
            technology_density: Density of technology matrix
            resource_count: Number of resource types
            
        Returns:
            Synthetic data dictionary
        """
        data = self.matrix_builder.create_synthetic_data(
            n_sectors=n_sectors,
            technology_density=technology_density,
            resource_count=resource_count
        )
        
        self.current_data = data
        return data
    
    def create_plan(self, policy_goals: Optional[List[str]] = None,
                   use_optimization: bool = True,
                   max_iterations: int = 10) -> Dict[str, Any]:
        """
        Create a comprehensive economic plan.
        
        Args:
            policy_goals: List of policy goals in natural language
            use_optimization: Whether to use constrained optimization
            max_iterations: Maximum number of planning iterations
            
        Returns:
            Generated economic plan
        """
        if not self.current_data:
            raise ValueError("No economic data loaded. Please load data first.")
        
        # Set policy goals
        if policy_goals:
            self.manager_agent.update_state('policy_goals', policy_goals)
        
        # Create initial plan
        plan_task = {
            'type': 'create_plan',
            'technology_matrix': self.current_data['technology_matrix'],
            'final_demand': self.current_data['final_demand'],
            'labor_vector': self.current_data['labor_input'],
            'resource_matrix': self.current_data.get('resource_matrix'),
            'max_resources': self.current_data.get('max_resources')
        }
        
        result = self.manager_agent.process_task(plan_task)
        
        if result['status'] != 'success':
            raise ValueError(f"Failed to create plan: {result.get('error', 'Unknown error')}")
        
        self.current_plan = result['plan']
        
        # Refine plan through iterations
        for iteration in range(max_iterations):
            refine_task = {
                'type': 'refine_plan',
                'current_plan': self.current_plan,
                'policy_goals': policy_goals
            }
            
            refine_result = self.manager_agent.process_task(refine_task)
            
            if refine_result['status'] in ['converged', 'max_iterations_reached']:
                break
            
            self.current_plan = refine_result['plan']
        
        # Evaluate final plan
        eval_task = {
            'type': 'evaluate_plan',
            'current_plan': self.current_plan
        }
        
        eval_result = self.manager_agent.process_task(eval_task)
        self.current_plan.update(eval_result)
        
        # Store in history
        self.plan_history.append(self.current_plan.copy())
        
        return self.current_plan
    
    def create_five_year_plan(self, policy_goals: Optional[List[str]] = None,
                            consumption_growth_rate: float = 0.02,
                            investment_ratio: float = 0.2) -> Dict[int, Dict[str, Any]]:
        """
        Create a comprehensive 5-year economic plan.
        
        Args:
            policy_goals: List of policy goals in natural language
            consumption_growth_rate: Annual consumption growth rate
            investment_ratio: Ratio of investment to total output
            
        Returns:
            Dictionary with plans for each year
        """
        if not self.current_data:
            raise ValueError("No economic data loaded. Please load data first.")
        
        # Initialize dynamic planner
        dynamic_planner = DynamicPlanner(
            initial_technology_matrix=self.current_data['technology_matrix'],
            initial_labor_vector=self.current_data['labor_input']
        )
        
        # Create consumption and investment demands for each year
        base_final_demand = self.current_data['final_demand']
        consumption_demands = []
        investment_demands = []
        
        for year in range(1, 6):
            # Calculate consumption demand with growth
            consumption_demand = base_final_demand * ((1 + consumption_growth_rate) ** (year - 1))
            consumption_demands.append(consumption_demand)
            
            # Calculate investment demand
            investment_demand = consumption_demand * investment_ratio
            investment_demands.append(investment_demand)
        
        # Create 5-year plan
        five_year_plan = dynamic_planner.create_five_year_plan(
            consumption_demands=consumption_demands,
            investment_demands=investment_demands,
            use_optimization=True
        )
        
        # Apply policy goals if provided
        if policy_goals:
            for year, plan in five_year_plan.items():
                policy_task = {
                    'type': 'policy_adjustment',
                    'current_plan': plan,
                    'goals': policy_goals
                }
                
                policy_result = self.policy_agent.process_task(policy_task)
                if policy_result['status'] == 'success':
                    plan.update(policy_result['adjusted_plan'])
        
        # Store in history
        self.plan_history.append(five_year_plan)
        
        return five_year_plan
    
    def generate_report(self, plan_data: Optional[Dict[str, Any]] = None,
                       report_options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive markdown report.
        
        Args:
            plan_data: Plan data to include in report (uses current plan if None)
            report_options: Options for report generation
            
        Returns:
            Generated markdown report
        """
        if plan_data is None:
            if self.current_plan is None:
                raise ValueError("No plan data available. Please create a plan first.")
            plan_data = self.current_plan
        
        report_task = {
            'type': 'generate_report',
            'plan_data': plan_data,
            'options': report_options or {}
        }
        
        result = self.writer_agent.process_task(report_task)
        
        if result['status'] != 'success':
            raise ValueError(f"Failed to generate report: {result.get('error', 'Unknown error')}")
        
        return result['report']
    
    def save_plan(self, file_path: Union[str, Path], 
                 format_type: str = 'json') -> None:
        """
        Save the current plan to a file.
        
        Args:
            file_path: Path to save the plan
            format_type: File format ('json', 'csv', 'excel')
        """
        if self.current_plan is None:
            raise ValueError("No current plan to save")
        
        file_path = Path(file_path)
        
        if format_type == 'json':
            self._save_plan_json(file_path)
        elif format_type == 'csv':
            self._save_plan_csv(file_path)
        elif format_type == 'excel':
            self._save_plan_excel(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _save_plan_json(self, file_path: Path) -> None:
        """Save plan as JSON file."""
        def convert_numpy(obj, visited=None):
            """Recursively convert numpy arrays to lists, handling circular references."""
            if visited is None:
                visited = set()
            
            # Handle circular references
            obj_id = id(obj)
            if obj_id in visited:
                return "<circular_reference>"
            
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                visited.add(obj_id)
                result = {key: convert_numpy(value, visited) for key, value in obj.items()}
                visited.remove(obj_id)
                return result
            elif isinstance(obj, list):
                visited.add(obj_id)
                result = [convert_numpy(item, visited) for item in obj]
                visited.remove(obj_id)
                return result
            else:
                return obj
        
        # Convert all numpy arrays to lists for JSON serialization
        json_data = convert_numpy(self.current_plan)
        
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _save_plan_csv(self, file_path: Path) -> None:
        """Save plan as CSV file."""
        # Create DataFrame with plan data
        import pandas as pd
        
        data = {
            'sector': range(len(self.current_plan['total_output'])),
            'total_output': self.current_plan['total_output'],
            'final_demand': self.current_plan['final_demand'],
            'labor_values': self.current_plan['labor_values']
        }
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def _save_plan_excel(self, file_path: Path) -> None:
        """Save plan as Excel file."""
        import pandas as pd
        
        with pd.ExcelWriter(file_path) as writer:
            # Main plan data
            plan_data = {
                'sector': range(len(self.current_plan['total_output'])),
                'total_output': self.current_plan['total_output'],
                'final_demand': self.current_plan['final_demand'],
                'labor_values': self.current_plan['labor_values']
            }
            
            df = pd.DataFrame(plan_data)
            df.to_excel(writer, sheet_name='Plan_Data', index=False)
            
            # Technology matrix
            tech_df = pd.DataFrame(
                self.current_plan['technology_matrix'],
                index=range(len(self.current_plan['total_output'])),
                columns=range(len(self.current_plan['total_output']))
            )
            tech_df.to_excel(writer, sheet_name='Technology_Matrix')
    
    def load_plan(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a plan from a file.
        
        Args:
            file_path: Path to the plan file
            
        Returns:
            Loaded plan dictionary
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            for key, value in data.items():
                if isinstance(value, list) and key in ['total_output', 'final_demand', 'labor_values', 'technology_matrix']:
                    data[key] = np.array(value)
            
            self.current_plan = data
            return data
        else:
            raise ValueError(f"Unsupported plan format: {file_path.suffix}")
    
    def get_plan_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current plan.
        
        Returns:
            Plan summary dictionary
        """
        if self.current_plan is None:
            return {'error': 'No current plan available'}
        
        return {
            'total_economic_output': np.sum(self.current_plan['total_output']),
            'total_labor_cost': self.current_plan['total_labor_cost'],
            'labor_efficiency': np.sum(self.current_plan['total_output']) / self.current_plan['total_labor_cost'],
            'sector_count': len(self.current_plan['total_output']),
            'plan_quality_score': self.current_plan.get('plan_quality_score', 0),
            'constraint_violations': self.current_plan.get('constraint_violations', {})
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status dictionary
        """
        return {
            'data_loaded': bool(self.current_data),
            'current_plan_available': bool(self.current_plan),
            'plan_history_count': len(self.plan_history),
            'agent_status': {
                'manager': self.manager_agent.get_status(),
                'economics': self.economics_agent.get_status(),
                'resource': self.resource_agent.get_status(),
                'policy': self.policy_agent.get_status(),
                'writer': self.writer_agent.get_status()
            }
        }
    
    def reset_system(self) -> None:
        """Reset the system to initial state."""
        self.current_data = {}
        self.current_plan = None
        self.plan_history = []
        
        # Reset agents
        self.manager_agent.reset_planning()
        self.economics_agent.clear_cache()
        self.resource_agent.clear_cache()
    
    def export_data(self, output_path: Union[str, Path], 
                   format_type: str = 'json') -> None:
        """
        Export current data to a file.
        
        Args:
            output_path: Output file path
            format_type: Export format ('json', 'csv', 'excel')
        """
        if not self.current_data:
            raise ValueError("No data to export")
        
        self.parser.export_data(self.current_data, output_path, format_type)
    
    def run_demo(self) -> Dict[str, Any]:
        """
        Run a demonstration of the planning system.
        
        Returns:
            Demo results dictionary
        """
        print("Running Cybernetic Planning System Demo...")
        
        # Create synthetic data
        print("1. Creating synthetic economic data...")
        data = self.create_synthetic_data(n_sectors=8, technology_density=0.4, resource_count=3)
        print(f"   Created data with {data['sectors']} sectors and {len(data['resources'])} resource types")
        
        # Create plan
        print("2. Creating economic plan...")
        policy_goals = [
            "Increase healthcare capacity by 15%",
            "Reduce carbon emissions by 20%",
            "Improve education infrastructure"
        ]
        
        plan = self.create_plan(policy_goals=policy_goals)
        print(f"   Plan created with {len(plan['total_output'])} sectors")
        print(f"   Total economic output: {np.sum(plan['total_output']):.2f}")
        print(f"   Total labor cost: {plan['total_labor_cost']:.2f}")
        
        # Generate report
        print("3. Generating comprehensive report...")
        report = self.generate_report()
        print(f"   Report generated with {len(report.split())} words")
        
        # Save results
        print("4. Saving results...")
        self.save_plan("demo_plan.json")
        with open("demo_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("   Results saved to demo_plan.json and demo_report.md")
        
        return {
            'data': data,
            'plan': plan,
            'report': report,
            'summary': self.get_plan_summary()
        }
