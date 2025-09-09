"""
Main Planning System

Orchestrates the complete cybernetic planning process, integrating
all components to generate comprehensive economic plans.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .core import DynamicPlanner
from .core.validation import EconomicPlanValidator
from .core.marxist_economics import MarxistEconomicCalculator
from .core.cybernetic_feedback import CyberneticFeedbackSystem
from .core.mathematical_validation import MathematicalValidator
from .agents import ManagerAgent, EconomicsAgent, ResourceAgent, PolicyAgent, WriterAgent
from .data import IOParser, MatrixBuilder, DataValidator, EnhancedDataLoader

class CyberneticPlanningSystem:
    """
    Main system for cybernetic central planning.

    Integrates all components to generate comprehensive 5 - year economic plans
    using Input - Output analysis and labor - time accounting.
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
        self.enhanced_loader = EnhancedDataLoader()

        # Initialize agents
        self.manager_agent = ManagerAgent()
        self.economics_agent = EconomicsAgent()
        self.resource_agent = ResourceAgent()
        self.policy_agent = PolicyAgent()
        self.writer_agent = WriterAgent()

        # Initialize validator
        self.validator = EconomicPlanValidator()

        # Initialize new modules
        self.marxist_calculator = None
        self.cybernetic_feedback = None
        self.mathematical_validator = MathematicalValidator()

        # System state
        self.current_data = {}
        self.current_plan = None
        self.plan_history = []

    def load_data_from_file(self, file_path: Union[str, Path], format_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Load economic data from a file.

        Args:
            file_path: Path to the data file
            format_type: File format (auto - detected if None)

        Returns:
            Loaded data dictionary
        """
        try:
            data = self.parser.parse_file(file_path, format_type)

            # Validate loaded data
            validation_results = self.validator.validate_all(data)

            if not validation_results["overall_valid"]:
                # Log validation warnings instead of printing
                pass

            self.current_data = data

            # Initialize new modules with loaded data
            self._initialize_new_modules()

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

        if not validation_results["overall_valid"]:
            # Log validation warnings instead of printing
            pass

        self.current_data = data

        # Initialize new modules with loaded data
        self._initialize_new_modules()

        return data

    def create_synthetic_data(
        self, n_sectors: int = 10, technology_density: float = 0.3, resource_count: int = 5
    ) -> Dict[str, Any]:
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
            n_sectors = n_sectors, technology_density = technology_density, resource_count = resource_count
        )

        self.current_data = data

        # Initialize new modules with loaded data
        self._initialize_new_modules()

        return data

    def load_comprehensive_data(
        self, year: int = 2024, use_real_data: bool = True, eia_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load comprehensive economic planning data with real resource constraints.

        Args:
            year: Year for data collection
            use_real_data: Whether to collect real resource data
            eia_api_key: EIA API key for enhanced data access

        Returns:
            Comprehensive data dictionary
        """
        # Update enhanced loader with API key if provided
        if eia_api_key:
            self.enhanced_loader.data_collector.scrapers["eia"].api_key = eia_api_key

        # Load comprehensive data
        data = self.enhanced_loader.load_comprehensive_data(year = year, use_real_data = use_real_data)

        # Extract BEA data for compatibility
        bea_data = data.get("bea_data", {})

        # Safely extract resource matrices with null checks
        resource_matrices = data.get("resource_matrices")
        if resource_matrices is None:
            resource_matrices = {}

        # Convert lists to numpy arrays for mathematical operations
        technology_matrix = bea_data.get("technology_matrix")
        if isinstance(technology_matrix, list):
            technology_matrix = np.array(technology_matrix)

        final_demand = bea_data.get("final_demand")
        if isinstance(final_demand, list):
            final_demand = np.array(final_demand)

        labor_input = bea_data.get("labor_input")
        if isinstance(labor_input, list):
            labor_input = np.array(labor_input)

        resource_matrix = resource_matrices.get("combined_resource_matrix")
        if isinstance(resource_matrix, list):
            resource_matrix = np.array(resource_matrix)

        max_resources = resource_matrices.get("resource_constraints")
        if isinstance(max_resources, list):
            max_resources = np.array(max_resources)

        self.current_data = {
            "technology_matrix": technology_matrix,
            "final_demand": final_demand,
            "labor_input": labor_input,
            "sectors": bea_data.get("sectors", []),
            "sector_count": bea_data.get("sector_count", 175),
            "resource_matrix": resource_matrix,
            "max_resources": max_resources,
            "comprehensive_data": data,  # Store full data
        }

        # Initialize new modules with loaded data
        self._initialize_new_modules()

        return data

    def create_plan(
        self, policy_goals: Optional[List[str]] = None, use_optimization: bool = True, max_iterations: int = 10
    ) -> Dict[str, Any]:
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
            self.manager_agent.update_state("policy_goals", policy_goals)

        # Get sector mapping for policy goals
        sector_mapping = {}
        if "sectors" in self.current_data:
            sectors = self.current_data["sectors"]
            for i, sector_name in enumerate(sectors):
                sector_mapping[sector_name] = i
                # Also map by department
                if "Dept_I" in sector_name:
                    if "department_i" not in sector_mapping:
                        sector_mapping["department_i"] = []
                    sector_mapping["department_i"].append(i)
                elif "Dept_II" in sector_name:
                    if "department_ii" not in sector_mapping:
                        sector_mapping["department_ii"] = []
                    sector_mapping["department_ii"].append(i)
                elif "Dept_III" in sector_name:
                    if "department_iii" not in sector_mapping:
                        sector_mapping["department_iii"] = []
                    sector_mapping["department_iii"].append(i)

        # Create initial plan
        plan_task = {
            "type": "create_plan",
            "technology_matrix": self.current_data["technology_matrix"],
            "final_demand": self.current_data["final_demand"],
            "labor_vector": self.current_data["labor_input"],
            "resource_matrix": self.current_data.get("resource_matrix"),
            "max_resources": self.current_data.get("max_resources"),
            "sector_mapping": sector_mapping,  # Pass sector mapping
        }

        result = self.manager_agent.process_task(plan_task)

        if result["status"] != "success":
            raise ValueError(f"Failed to create plan: {result.get('error', 'Unknown error')}")

        self.current_plan = result["plan"]

        # Refine plan through iterations
        for iteration in range(max_iterations):
            refine_task = {
                "type": "refine_plan",
                "current_plan": self.current_plan,
                "policy_goals": policy_goals,
                "sector_mapping": sector_mapping  # Pass sector mapping
            }

            refine_result = self.manager_agent.process_task(refine_task)

            if refine_result["status"] in ["converged", "max_iterations_reached"]:
                break

            self.current_plan = refine_result["plan"]

        # Validate the plan
        validation_result = self.validator.validate_plan(self.current_plan)
        self.current_plan["validation"] = validation_result

        # Evaluate final plan
        eval_task = {"type": "evaluate_plan", "current_plan": self.current_plan}

        eval_result = self.manager_agent.process_task(eval_task)
        self.current_plan.update(eval_result)

        # Store in history
        self.plan_history.append(self.current_plan.copy())

        return self.current_plan

    def create_five_year_plan(
        self,
        policy_goals: Optional[List[str]] = None,
        consumption_growth_rate: float = 0.02,
        investment_ratio: float = 0.2,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Create a comprehensive 5 - year economic plan.

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
            initial_technology_matrix = self.current_data["technology_matrix"],
            initial_labor_vector = self.current_data["labor_input"],
        )

        # Create consumption and investment demands for each year
        base_final_demand = self.current_data["final_demand"]
        consumption_demands = []
        investment_demands = []

        for year in range(1, 6):
            # Calculate consumption demand with growth
            consumption_demand = base_final_demand * ((1 + consumption_growth_rate) ** (year - 1))
            consumption_demands.append(consumption_demand)

            # Calculate investment demand
            investment_demand = consumption_demand * investment_ratio
            investment_demands.append(investment_demand)

        # Create 5 - year plan
        five_year_plan = dynamic_planner.create_five_year_plan(
            consumption_demands = consumption_demands, investment_demands = investment_demands, use_optimization = True
        )

        # Apply policy goals if provided
        if policy_goals:
            for year, plan in five_year_plan.items():
                policy_task = {"type": "policy_adjustment", "current_plan": plan, "goals": policy_goals}

                policy_result = self.policy_agent.process_task(policy_task)
                if policy_result["status"] == "success":
                    plan.update(policy_result["adjusted_plan"])

        # Store in history
        self.plan_history.append(five_year_plan)

        return five_year_plan

    def generate_report(
        self, plan_data: Optional[Dict[str, Any]] = None, report_options: Optional[Dict[str, Any]] = None
    ) -> str:
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

        report_task = {"type": "generate_report", "plan_data": plan_data, "options": report_options or {}}

        result = self.writer_agent.process_task(report_task)

        if result["status"] != "success":
            raise ValueError(f"Failed to generate report: {result.get('error', 'Unknown error')}")

        return result["report"]

    def save_plan(self, file_path: Union[str, Path], format_type: str = "json") -> None:
        """
        Save the current plan to a file.

        Args:
            file_path: Path to save the plan
            format_type: File format ('json', 'csv', 'excel')
        """
        if self.current_plan is None:
            raise ValueError("No current plan to save")

        file_path = Path(file_path)

        if format_type == "json":
            self._save_plan_json(file_path)
        elif format_type == "csv":
            self._save_plan_csv(file_path)
        elif format_type == "excel":
            self._save_plan_excel(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _save_plan_json(self, file_path: Path) -> None:
        """Save plan as JSON file."""

        def convert_numpy(obj, visited = None):
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

        with open(file_path, "w") as f:
            json.dump(json_data, f, indent = 2)

    def _save_plan_csv(self, file_path: Path) -> None:
        """Save plan as CSV file."""
        # Create DataFrame with plan data

        data = {
            "sector": range(len(self.current_plan["total_output"])),
            "total_output": self.current_plan["total_output"],
            "final_demand": self.current_plan["final_demand"],
            "labor_values": self.current_plan["labor_values"],
        }

        df = pd.DataFrame(data)
        df.to_csv(file_path, index = False)

    def _save_plan_excel(self, file_path: Path) -> None:
        """Save plan as Excel file."""

        with pd.ExcelWriter(file_path) as writer:
            # Main plan data
            plan_data = {
                "sector": range(len(self.current_plan["total_output"])),
                "total_output": self.current_plan["total_output"],
                "final_demand": self.current_plan["final_demand"],
                "labor_values": self.current_plan["labor_values"],
            }

            df = pd.DataFrame(plan_data)
            df.to_excel(writer, sheet_name="Plan_Data", index = False)

            # Technology matrix
            tech_df = pd.DataFrame(
                self.current_plan["technology_matrix"],
                index = range(len(self.current_plan["total_output"])),
                columns = range(len(self.current_plan["total_output"])),
            )
            tech_df.to_excel(writer, sheet_name="Technology_Matrix")

    def load_plan(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a plan from a file.

        Args:
            file_path: Path to the plan file

        Returns:
            Loaded plan dictionary
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert lists back to numpy arrays
            for key, value in data.items():
                if isinstance(value, list) and key in [
                    "total_output",
                    "final_demand",
                    "labor_values",
                    "technology_matrix",
                ]:
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
            return {"error": "No current plan available"}

        return {
            "total_economic_output": np.sum(self.current_plan["total_output"]),
            "total_labor_cost": self.current_plan["total_labor_cost"],
            "labor_efficiency": np.sum(self.current_plan["total_output"]) / self.current_plan["total_labor_cost"],
            "sector_count": len(self.current_plan["total_output"]),
            "plan_quality_score": self.current_plan.get("plan_quality_score", 0),
            "constraint_violations": self.current_plan.get("constraint_violations", {}),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            System status dictionary
        """
        return {
            "data_loaded": bool(self.current_data),
            "current_plan_available": bool(self.current_plan),
            "plan_history_count": len(self.plan_history),
            "agent_status": {
                "manager": self.manager_agent.get_status(),
                "economics": self.economics_agent.get_status(),
                "resource": self.resource_agent.get_status(),
                "policy": self.policy_agent.get_status(),
                "writer": self.writer_agent.get_status(),
            },
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

    def export_data(self, output_path: Union[str, Path], format_type: str = "json") -> None:
        """
        Export current data to a file.

        Args:
            output_path: Output file path
            format_type: Export format ('json', 'csv', 'excel')
        """
        if not self.current_data:
            raise ValueError("No data to export")

        self.parser.export_data(self.current_data, output_path, format_type)

    def _initialize_new_modules(self):
        """Initialize new modules with current data."""
        if not self.current_data:
            return

        try:
            # Initialize Marxist economic calculator
            if all(key in self.current_data for key in ["technology_matrix", "labor_input"]):
                self.marxist_calculator = MarxistEconomicCalculator(
                    technology_matrix = self.current_data["technology_matrix"],
                    labor_vector = self.current_data["labor_input"]
                )

            # Initialize cybernetic feedback system
            if all(key in self.current_data for key in ["technology_matrix", "final_demand", "labor_input"]):
                self.cybernetic_feedback = CyberneticFeedbackSystem(
                    technology_matrix = self.current_data["technology_matrix"],
                    final_demand = self.current_data["final_demand"],
                    labor_vector = self.current_data["labor_input"]
                )
        except Exception as e:
            print(f"Warning: Could not initialize new modules: {e}")

    def get_marxist_analysis(self) -> Dict[str, Any]:
        """Get comprehensive Marxist economic analysis."""
        if not self.marxist_calculator:
            return {"error": "Marxist calculator not initialized. Please load data first."}

        return self.marxist_calculator.get_economic_indicators()

    def get_cybernetic_analysis(self, initial_output: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get cybernetic feedback analysis."""
        if not self.cybernetic_feedback:
            return {"error": "Cybernetic feedback system not initialized. Please load data first."}

        if initial_output is None:
            # Use Leontief solution as initial output
            try:
                I = np.eye(self.cybernetic_feedback.n_sectors)
                leontief_inverse = np.linalg.inv(I - self.cybernetic_feedback.A)
                initial_output = leontief_inverse @ self.cybernetic_feedback.d
            except np.linalg.LinAlgError:
                initial_output = np.ones(self.cybernetic_feedback.n_sectors)

        return self.cybernetic_feedback.apply_cybernetic_feedback(initial_output)

    def get_mathematical_validation(self) -> Dict[str, Any]:
        """Get mathematical validation results."""
        if not self.current_data:
            return {"error": "No data loaded for validation."}

        return self.mathematical_validator.validate_all_formulas(self.current_data)
