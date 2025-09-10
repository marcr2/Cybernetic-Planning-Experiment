"""
Main Planning System

Orchestrates the complete cybernetic planning process, integrating
all components to generate comprehensive economic plans.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json

from .core import DynamicPlanner
from .core.validation import EconomicPlanValidator
from .core.marxist_economics import MarxistEconomicCalculator
from .core.cybernetic_feedback import CyberneticFeedbackSystem
from .core.mathematical_validation import MathematicalValidator
from .core.leontief import LeontiefModel
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
        self.matrix_builder = MatrixBuilder(use_technology_tree = True)
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
            # Use enhanced data loader for better data processing
            if str(file_path).endswith('.json'):
                # Load JSON file and process with enhanced loader
                import json
                with open(file_path, 'r') as f:
                    raw_data = json.load(f)

                # Process with enhanced loader
                processed_data = self.enhanced_loader._convert_bea_data_format(raw_data)

                # Convert to numpy arrays
                technology_matrix = processed_data.get("technology_matrix")
                if isinstance(technology_matrix, list):
                    technology_matrix = np.array(technology_matrix)

                final_demand = processed_data.get("final_demand")
                if isinstance(final_demand, list):
                    final_demand = np.array(final_demand)

                labor_input = processed_data.get("labor_input")
                if isinstance(labor_input, list):
                    labor_input = np.array(labor_input)

                data = {
                    "technology_matrix": technology_matrix,
                    "final_demand": final_demand,
                    "labor_input": labor_input,
                    "sectors": processed_data.get("sectors", []),
                    "sector_count": processed_data.get("sector_count", 0),
                    "year": processed_data.get("year", 2024),
                    "data_source": processed_data.get("data_source", "file"),
                    "metadata": processed_data.get("metadata", {})
                }
            else:
                # Use basic parser for other formats
                data = self.parser.parse_file(file_path, format_type)

            # Validate loaded data
            validation_results = self.validator.validate_plan(data)

            if not validation_results["is_valid"]:
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
        validation_results = self.validator.validate_plan(data)

        if not validation_results["is_valid"]:
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

        # Debug: Check synthetic data generation
        print(f"DEBUG SYNTHETIC: Generated data keys: {list(data.keys())}")
        if "final_demand" in data:
            final_demand = data["final_demand"]
            print(f"DEBUG SYNTHETIC: Generated final_demand type: {type(final_demand)}")
            print(f"DEBUG SYNTHETIC: Generated final_demand sum: {np.sum(final_demand)}")
            print(f"DEBUG SYNTHETIC: Generated final_demand first 5 values: {final_demand[:5]}")
        else:
            print("DEBUG SYNTHETIC: WARNING - No final_demand in generated data!")

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
        self,
        policy_goals: Optional[List[str]] = None,
        use_optimization: bool = True,
        max_iterations: int = 10,
        production_multipliers: Optional[Dict[str, float]] = None,
        apply_reproduction: bool = True
    ) -> Dict[str, Any]:
        """
        Create a comprehensive economic plan.

        Args:
            policy_goals: List of policy goals in natural language
            use_optimization: Whether to use constrained optimization
            max_iterations: Maximum number of planning iterations
            production_multipliers: Dictionary of production multipliers for different departments
            apply_reproduction: Whether to apply Marxist reproduction adjustments

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

        # Get original final demand (production multipliers will be applied by reproduction system)
        final_demand = self.current_data["final_demand"].copy()

        # Debug: Check the original final_demand from data
        print(f"DEBUG PLANNING: Original final_demand from data type: {type(final_demand)}")
        print(f"DEBUG PLANNING: Original final_demand from data shape: {final_demand.shape if hasattr(final_demand, 'shape') else len(final_demand)}")
        print(f"DEBUG PLANNING: Original final_demand from data sum: {np.sum(final_demand)}")
        print(f"DEBUG PLANNING: Original final_demand from data first 5 values: {final_demand[:5]}")
        print(f"DEBUG PLANNING: Number of sectors: {len(self.current_data.get('sectors', []))}")

        # Create initial plan
        plan_task = {
            "type": "create_plan",
            "technology_matrix": self.current_data["technology_matrix"],
            "final_demand": final_demand,
            "labor_vector": self.current_data["labor_input"],
            "resource_matrix": self.current_data.get("resource_matrix"),
            "max_resources": self.current_data.get("max_resources"),
            "sector_mapping": sector_mapping,  # Pass sector mapping
            "apply_reproduction": apply_reproduction,  # Pass reproduction setting
            "production_multipliers": production_multipliers,  # Pass production multipliers
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

    def _apply_production_multipliers(self, final_demand: np.ndarray, production_multipliers: Dict[str, float]) -> np.ndarray:
        """
        Apply production multipliers to final demand.

        Args:
            final_demand: Original final demand vector
            production_multipliers: Dictionary with multipliers for 'overall', 'dept_I', 'dept_II', 'dept_III'

        Returns:
            Adjusted final demand vector
        """
        adjusted_demand = final_demand.copy()
        n_sectors = len(final_demand)

        # Apply overall multiplier first
        overall_multiplier = production_multipliers.get("overall", 1.0)
        adjusted_demand *= overall_multiplier

        # Apply department - specific multipliers
        # Assuming first 50 sectors are Dept I, next 50 are Dept II, rest are Dept III
        n_dept_I = min(50, n_sectors)
        n_dept_II = min(50, max(0, n_sectors - 50))
        n_dept_III = max(0, n_sectors - 100)

        # Department I multiplier
        dept_I_multiplier = production_multipliers.get("dept_I", 1.0)
        if n_dept_I > 0:
            adjusted_demand[:n_dept_I] *= dept_I_multiplier

        # Department II multiplier
        dept_II_multiplier = production_multipliers.get("dept_II", 1.0)
        if n_dept_II > 0:
            adjusted_demand[n_dept_I:n_dept_I + n_dept_II] *= dept_II_multiplier

        # Department III multiplier
        dept_III_multiplier = production_multipliers.get("dept_III", 1.0)
        if n_dept_III > 0:
            adjusted_demand[n_dept_I + n_dept_II:] *= dept_III_multiplier

        return adjusted_demand

    def create_five_year_plan(
        self,
        policy_goals: Optional[List[str]] = None,
        consumption_growth_rate: float = 0.05,  # Increased from 2% to 5% to account for population growth and rising living standards
        investment_ratio: float = 0.2,
        production_multipliers: Optional[Dict[str, float]] = None,
        apply_reproduction: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Create a comprehensive 5 - year economic plan.

        Args:
            policy_goals: List of policy goals in natural language
            consumption_growth_rate: Annual consumption growth rate
            investment_ratio: Ratio of investment to total output
            production_multipliers: Dictionary of production multipliers for different departments
            apply_reproduction: Whether to apply Marxist reproduction adjustments

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
        base_final_demand = self.current_data["final_demand"].copy()
        # Note: Production multipliers are handled by the reproduction system in individual year plans
        consumption_demands = []
        investment_demands = []

        # For the first year, use base growth rates
        for year in range(1, 6):
            if year == 1:
                # Year 1: Use base growth rates
                population_growth_rate = 0.01  # 1% annual population growth
                living_standards_growth_rate = consumption_growth_rate - population_growth_rate
                total_growth_rate = population_growth_rate + living_standards_growth_rate
                consumption_demand = base_final_demand * ((1 + total_growth_rate) ** (year - 1))
                investment_growth_rate = total_growth_rate * 1.2
                investment_demand = base_final_demand * investment_ratio * ((1 + investment_growth_rate) ** (year - 1))
            else:
                # Years 2 - 5: Use feedback - driven growth (will be calculated during planning)
                # Start with base growth and let feedback system adjust
                population_growth_rate = 0.01
                living_standards_growth_rate = consumption_growth_rate - population_growth_rate
                total_growth_rate = population_growth_rate + living_standards_growth_rate
                consumption_demand = base_final_demand * ((1 + total_growth_rate) ** (year - 1))
                investment_growth_rate = total_growth_rate * 1.2
                investment_demand = base_final_demand * investment_ratio * ((1 + investment_growth_rate) ** (year - 1))

            consumption_demands.append(consumption_demand)
            investment_demands.append(investment_demand)

        # Create 5 - year plan with feedback growth
        five_year_plan = dynamic_planner.create_five_year_plan(
            consumption_demands = consumption_demands,
            investment_demands = investment_demands,
            use_optimization = True,
            use_feedback_growth = True  # Re - enable feedback growth
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

        # Store performance feedback
        self.performance_feedback = dynamic_planner.get_performance_feedback()

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

        # Include automatic analysis results in the plan data
        if hasattr(self, 'automatic_analyses') and self.automatic_analyses:
            plan_data["automatic_analyses"] = self.automatic_analyses

        report_task = {"type": "generate_report", "plan_data": plan_data, "options": report_options or {}}

        result = self.writer_agent.process_task(report_task)

        if result["status"] != "success":
            raise ValueError(f"Failed to generate report: {result.get('error', 'Unknown error')}")

        return result["report"]

    def get_performance_feedback(self) -> Dict[str, Any]:
        """Get performance feedback from the latest plan."""
        if hasattr(self, 'performance_feedback'):
            return self.performance_feedback
        return {"message": "No performance feedback available"}

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

        def convert_for_json(obj, visited = None):
            """Recursively convert objects to JSON - serializable format, handling circular references."""
            if visited is None:
                visited = set()

            # Handle circular references
            obj_id = id(obj)
            if obj_id in visited:
                return "<circular_reference>"

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'value'):
                # Handle enum objects like ValidationStatus FIRST
                return obj.value
            elif hasattr(obj, '__dict__'):
                # Handle dataclass objects like ValidationResult
                visited.add(obj_id)
                result = {key: convert_for_json(value, visited) for key, value in obj.__dict__.items()}
                visited.remove(obj_id)
                return result
            elif isinstance(obj, (str, int, float, bool, type(None))):
                # Handle basic JSON - serializable types
                return obj
            elif isinstance(obj, dict):
                visited.add(obj_id)
                result = {key: convert_for_json(value, visited) for key, value in obj.items()}
                visited.remove(obj_id)
                return result
            elif isinstance(obj, (list, tuple)):
                visited.add(obj_id)
                result = [convert_for_json(item, visited) for item in obj]
                visited.remove(obj_id)
                return result
            else:
                return obj

        # Convert all objects to JSON - serializable format
        json_data = convert_for_json(self.current_plan)

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

    def export_plan_for_simulation(self, file_path: Union[str, Path]) -> None:
        """
        Export current plan in simulation - compatible format.

        Args:
            file_path: Path to save the simulation plan
        """
        if self.current_plan is None:
            raise ValueError("No current plan to export")

        file_path = Path(file_path)

        # Convert plan to simulation format
        simulation_plan = self._convert_plan_to_simulation_format(self.current_plan)

        # Save as JSON
        with open(file_path, "w") as f:
            json.dump(simulation_plan, f, indent = 2)

    def _convert_plan_to_simulation_format(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a planning system plan to simulation format."""
        # Extract data from planning plan
        total_output = np.array(plan.get('total_output', []))
        labor_vector = np.array(plan.get('labor_vector', []))
        technology_matrix = np.array(plan.get('technology_matrix', []))
        final_demand = np.array(plan.get('final_demand', []))

        # Get sector count
        n_sectors = len(total_output)

        # Create sector names if not available
        sectors = plan.get('sectors', [f"Sector_{i + 1}" for i in range(n_sectors)])

        # Convert to simulation format
        simulation_plan = {
            'sectors': sectors,
            'production_targets': total_output.tolist(),
            'labor_requirements': labor_vector.tolist(),
            'resource_allocations': {
                'technology_matrix': technology_matrix.tolist(),
                'final_demand': final_demand.tolist(),
                'total_labor_cost': plan.get('total_labor_cost', 0),
                'plan_quality_score': plan.get('plan_quality_score', 0)
            },
            'plan_metadata': {
                'year': plan.get('year', 1),
                'iteration': plan.get('iteration', 1),
                'status': plan.get('status', 'unknown'),
                'validation': plan.get('validation', {}),
                'constraint_violations': plan.get('constraint_violations', {}),
                'cybernetic_feedback': plan.get('cybernetic_feedback', {})
            }
        }

        return simulation_plan

    def _initialize_new_modules(self):
        """Initialize new modules with current data and run automatic analyses."""
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

            # Run automatic analyses after initialization
            self._run_automatic_analyses()

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

        # Create system components for comprehensive validation
        system_components = {
            "data": self.current_data,
            "marxist_calculator": self.marxist_calculator,
            "cybernetic_system": self.cybernetic_feedback,
        }

        # Add Leontief model if data is available
        if all(key in self.current_data for key in ["technology_matrix", "final_demand"]):
            try:
                leontief_model = LeontiefModel(
                    technology_matrix = self.current_data["technology_matrix"],
                    final_demand = self.current_data["final_demand"]
                )
                system_components["leontief_model"] = leontief_model
            except Exception as e:
                print(f"Warning: Could not create Leontief model for validation: {e}")

        # Add cybernetic result if available
        if self.cybernetic_feedback:
            try:
                cybernetic_result = self.get_cybernetic_analysis()
                system_components["cybernetic_result"] = cybernetic_result
            except Exception as e:
                print(f"Warning: Could not get cybernetic result for validation: {e}")

        # Run comprehensive validation
        return self.mathematical_validator.validate_all(system_components)

    def _run_automatic_analyses(self):
        """Run all three analyses automatically when data is loaded."""
        try:
            # Store analysis results in the system for easy access
            self.automatic_analyses = {}

            # Run Marxist analysis
            if self.marxist_calculator:
                try:
                    marxist_results = self.get_marxist_analysis()
                    self.automatic_analyses['marxist'] = marxist_results
                    print("✓ Marxist analysis completed automatically")
                except Exception as e:
                    print(f"Warning: Marxist analysis failed: {e}")
                    self.automatic_analyses['marxist'] = {"error": str(e)}

            # Run cybernetic feedback analysis
            if self.cybernetic_feedback:
                try:
                    cybernetic_results = self.get_cybernetic_analysis()
                    self.automatic_analyses['cybernetic'] = cybernetic_results
                    print("✓ Cybernetic feedback analysis completed automatically")
                except Exception as e:
                    print(f"Warning: Cybernetic analysis failed: {e}")
                    self.automatic_analyses['cybernetic'] = {"error": str(e)}

            # Run mathematical validation
            try:
                validation_results = self.get_mathematical_validation()
                self.automatic_analyses['mathematical'] = validation_results
                print("✓ Mathematical validation completed automatically")
            except Exception as e:
                print(f"Warning: Mathematical validation failed: {e}")
                self.automatic_analyses['mathematical'] = {"error": str(e)}

        except Exception as e:
            print(f"Warning: Automatic analyses failed: {e}")
            self.automatic_analyses = {"error": str(e)}

    def get_automatic_analyses(self) -> Dict[str, Any]:
        """Get results from automatic analyses run on data load."""
        return getattr(self, 'automatic_analyses', {"error": "No automatic analyses available"})
