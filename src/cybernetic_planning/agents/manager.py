"""
Manager Agent

Central coordinator for the multi - agent planning system.
Orchestrates the iterative refinement of economic plans.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .base import BaseAgent
from .policy import PolicyAgent

class ManagerAgent(BaseAgent):
    """
    Central coordinator agent that orchestrates the planning process.

    Manages the iterative refinement of economic plans by coordinating
    specialized agents and evaluating plan quality.
    """

    def __init__(self):
        """Initialize the manager agent."""
        super().__init__("manager", "Central Planning Manager")
        self.planning_iteration = 0
        self.max_iterations = 10
        self.convergence_threshold = 0.01
        self.current_plan = None
        self.plan_history = []
        self.agent_status = {}
        self.policy_agent = PolicyAgent()

    def get_capabilities(self) -> List[str]:
        """Get manager agent capabilities."""
        return [
            "plan_coordination",
            "iteration_management",
            "convergence_checking",
            "agent_orchestration",
            "plan_evaluation",
        ]

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a planning task.

        Args:
            task: Task description with planning parameters

        Returns:
            Task results including final plan
        """
        task_type = task.get("type", "unknown")

        if task_type == "create_plan":
            return self._create_economic_plan(task)
        elif task_type == "refine_plan":
            return self._refine_plan(task)
        elif task_type == "evaluate_plan":
            return self._evaluate_plan(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}

    def _create_economic_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new economic plan.

        Args:
            task: Task parameters including input data

        Returns:
            Initial economic plan
        """
        # Extract input data
        technology_matrix = task.get("technology_matrix")
        final_demand = task.get("final_demand")
        labor_vector = task.get("labor_vector")
        resource_matrix = task.get("resource_matrix")
        max_resources = task.get("max_resources")
        apply_reproduction = task.get("apply_reproduction", True)  # Default to True for backward compatibility
        production_multipliers = task.get("production_multipliers", {})  # Get production multipliers
        sector_mapping = task.get("sector_mapping", {})  # Get sector mapping for policy goals

        if any(x is None for x in [technology_matrix, final_demand, labor_vector]):
            return {"error": "Missing required input data"}

        # Initialize planning iteration
        self.planning_iteration = 0
        self.plan_history = []
        
        # Store sector mapping for policy goals
        self.update_state("sector_mapping", sector_mapping)

        # Create initial plan using core algorithms with cybernetic feedback
        from ..core.leontief import LeontiefModel
        from ..core.labor_values import LaborValueCalculator
        from ..core.optimization import ConstrainedOptimizer

        # Step 1: Calculate initial output using Leontief model
        leontief = LeontiefModel(technology_matrix, final_demand)
        initial_output = leontief.compute_total_output()

        # Step 2: Apply Marxist reproduction system for proper expanded reproduction (if enabled)
        if apply_reproduction:
            from ..core.marxist_reproduction import MarxistReproductionSystem

            # Debug: Check input final_demand
            print(f"DEBUG MANAGER: Original final_demand type: {type(final_demand)}")
            print(f"DEBUG MANAGER: Original final_demand shape: {final_demand.shape if hasattr(final_demand, 'shape') else len(final_demand)}")
            print(f"DEBUG MANAGER: Original final_demand sum: {np.sum(final_demand)}")
            print(f"DEBUG MANAGER: Original final_demand first 5 values: {final_demand[:5]}")

            # Calculate department sizes based on actual number of sectors
            n_sectors = len(final_demand)
            n_dept_I = min(50, n_sectors // 3)  # Department I: means of production
            n_dept_II = min(50, (n_sectors - n_dept_I) // 2)  # Department II: consumer goods
            n_dept_III = n_sectors - n_dept_I - n_dept_II  # Department III: services
            
            print(f"DEBUG MANAGER: Adjusted department sizes - Dept I: {n_dept_I}, Dept II: {n_dept_II}, Dept III: {n_dept_III}")
            
            reproduction_system = MarxistReproductionSystem(
                technology_matrix = technology_matrix,
                final_demand = final_demand,
                labor_vector = labor_vector,
                n_dept_I = n_dept_I,  # Department I: means of production
                n_dept_II = n_dept_II,  # Department II: consumer goods
                n_dept_III = n_dept_III  # Department III: services
            )

            # Adjust output for reproduction balance
            cybernetic_output = reproduction_system.adjust_for_reproduction_balance(initial_output)

            # Calculate expanded reproduction demands
            cybernetic_demand = reproduction_system.calculate_expanded_reproduction_demands(final_demand, production_multipliers=production_multipliers)

            # Debug: Check cybernetic_demand
            print(f"DEBUG MANAGER: Cybernetic_demand type: {type(cybernetic_demand)}")
            print(f"DEBUG MANAGER: Cybernetic_demand shape: {cybernetic_demand.shape if hasattr(cybernetic_demand, 'shape') else len(cybernetic_demand)}")
            print(f"DEBUG MANAGER: Cybernetic_demand sum: {np.sum(cybernetic_demand)}")
            print(f"DEBUG MANAGER: Cybernetic_demand first 5 values: {cybernetic_demand[:5]}")

            # Add accumulation requirements
            accumulation_demand = reproduction_system.calculate_accumulation_requirements(cybernetic_output)
            cybernetic_demand += accumulation_demand
            
            # Debug: Check final cybernetic_demand after accumulation
            print(f"DEBUG MANAGER: Final cybernetic_demand sum: {np.sum(cybernetic_demand)}")
            print(f"DEBUG MANAGER: Final cybernetic_demand first 5 values: {cybernetic_demand[:5]}")
            
            # Fallback: If cybernetic_demand is zero or very small, use original final_demand
            if np.sum(cybernetic_demand) < 1e-10:
                print("DEBUG MANAGER: WARNING - cybernetic_demand is zero, using original final_demand")
                cybernetic_demand = final_demand.copy()
                print(f"DEBUG MANAGER: Fallback cybernetic_demand sum: {np.sum(cybernetic_demand)}")
        else:
            # Use original values without reproduction adjustments
            cybernetic_output = initial_output
            cybernetic_demand = final_demand.copy()
            
            # Debug: Check final_demand when reproduction is disabled
            print(f"DEBUG MANAGER: Reproduction disabled - using original final_demand")
            print(f"DEBUG MANAGER: Original final_demand sum: {np.sum(final_demand)}")
            print(f"DEBUG MANAGER: Original final_demand first 5 values: {final_demand[:5]}")

        # Step 3: Calculate labor values with cybernetic output
        labor_calc = LaborValueCalculator(technology_matrix, labor_vector)
        labor_values = labor_calc.get_labor_values()
        total_labor_cost = labor_calc.compute_total_labor_cost(cybernetic_demand)

        # Step 4: Optimize with constraints using cybernetic demand
        optimizer = ConstrainedOptimizer(
            technology_matrix = technology_matrix,
            direct_labor = labor_vector,
            final_demand = cybernetic_demand,  # Use cybernetic - adjusted demand
            resource_matrix = resource_matrix,
            max_resources = max_resources,
        )

        optimization_result = optimizer.solve()

        if optimization_result["feasible"]:
            optimized_output = optimization_result["solution"]
            optimized_labor_cost = optimization_result["total_labor_cost"]
        else:
            # Fall back to Leontief solution
            optimized_output = initial_output
            optimized_labor_cost = total_labor_cost

        # Create initial plan with cybernetic information
        self.current_plan = {
            "year": 1,
            "iteration": self.planning_iteration,
            "technology_matrix": technology_matrix,
            "labor_vector": labor_vector,
            "final_demand": cybernetic_demand,  # Use cybernetic - adjusted demand
            "original_final_demand": final_demand,  # Keep original for reference
            "total_output": optimized_output,
            "labor_values": labor_values,
            "total_labor_cost": optimized_labor_cost,
            "resource_usage": None,
            "constraint_violations": None,
            "plan_quality_score": 0.0,
            "reproduction_applied": apply_reproduction,  # Store reproduction setting
            "cybernetic_feedback": {
                "applied": apply_reproduction,
                "feedback_strength": 0.1,
                "adaptation_rate": 0.05,
                "converged": True,  # Assume converged for initial plan
                "iterations": 1,
                "cybernetic_metrics": {
                    "demand_adjustment_factor": np.mean(cybernetic_demand / final_demand) if np.sum(final_demand) > 0 else 1.0,
                    "reproduction_balance": apply_reproduction
                },
                "demand_adjustment": (cybernetic_demand - final_demand).tolist(),
                "feedback_diagnostics": {
                    "system_stable": True,
                    "reproduction_balanced": apply_reproduction
                }
            }
        }

        # Store in history
        self.plan_history.append(self.current_plan.copy())

        return {
            "status": "success",
            "plan": self.current_plan,
            "iteration": self.planning_iteration,
            "message": "Initial plan created",
        }

    def _refine_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the current plan through agent collaboration.

        Args:
            task: Refinement parameters

        Returns:
            Refined plan
        """
        if self.current_plan is None:
            return {"error": "No current plan to refine"}

        self.planning_iteration += 1

        if self.planning_iteration > self.max_iterations:
            return {
                "status": "max_iterations_reached",
                "plan": self.current_plan,
                "message": "Maximum iterations reached",
            }

        # Check for convergence
        if self._check_convergence():
            return {"status": "converged", "plan": self.current_plan, "message": "Plan has converged"}

        # Create refinement tasks for specialized agents
        refinement_tasks = self._create_refinement_tasks()

        # Process refinements (in a real implementation, this would be done by actual agents)
        refined_plan = self._apply_refinements(refinement_tasks)

        # Debug: Check what's in refined_plan
        print(f"DEBUG REFINEMENT: Refined plan keys: {list(refined_plan.keys())}")
        if "final_demand" in refined_plan:
            refined_demand = refined_plan["final_demand"]
            print(f"DEBUG REFINEMENT: Refined final_demand type: {type(refined_demand)}")
            print(f"DEBUG REFINEMENT: Refined final_demand sum: {np.sum(refined_demand)}")
            print(f"DEBUG REFINEMENT: Refined final_demand first 5 values: {refined_demand[:5]}")
        
        # Update current plan
        self.current_plan.update(refined_plan)
        self.current_plan["iteration"] = self.planning_iteration
        
        # Debug: Check final plan after update
        if "final_demand" in self.current_plan:
            final_plan_demand = self.current_plan["final_demand"]
            print(f"DEBUG REFINEMENT: Final plan final_demand sum: {np.sum(final_plan_demand)}")
            print(f"DEBUG REFINEMENT: Final plan final_demand first 5 values: {final_plan_demand[:5]}")

        # Store in history
        self.plan_history.append(self.current_plan.copy())

        return {
            "status": "refined",
            "plan": self.current_plan,
            "iteration": self.planning_iteration,
            "message": "Plan refined",
        }

    def _evaluate_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of the current plan.

        Args:
            task: Evaluation parameters

        Returns:
            Plan evaluation results
        """
        if self.current_plan is None:
            return {"error": "No current plan to evaluate"}

        # Calculate plan quality score
        quality_score = self._calculate_plan_quality()

        # Check constraint violations
        constraint_violations = self._check_constraint_violations()

        # Update plan with evaluation results
        self.current_plan["plan_quality_score"] = quality_score
        self.current_plan["constraint_violations"] = constraint_violations

        return {
            "status": "evaluated",
            "quality_score": quality_score,
            "constraint_violations": constraint_violations,
            "plan": self.current_plan,
        }

    def _create_refinement_tasks(self) -> List[Dict[str, Any]]:
        """Create refinement tasks for specialized agents."""
        tasks = []

        # Economics agent task
        tasks.append(
            {
                "agent": "economics",
                "type": "sensitivity_analysis",
                "parameters": {
                    "technology_matrix": self.current_plan["technology_matrix"],
                    "final_demand": self.current_plan["final_demand"],
                },
            }
        )

        # Resource agent task
        tasks.append(
            {"agent": "resource", "type": "resource_optimization", "parameters": {"current_plan": self.current_plan}}
        )

        # Policy agent task
        tasks.append(
            {
                "agent": "policy",
                "type": "policy_adjustment",
                "parameters": {"current_plan": self.current_plan, "goals": self.get_state("policy_goals", [])},
            }
        )

        return tasks

    def _apply_refinements(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply refinements from specialized agents.

        Args:
            tasks: List of refinement tasks

        Returns:
            Refined plan components
        """
        refinements = {}

        # In a real implementation, this would coordinate with actual agents
        # For now, we'll simulate some basic refinements

        for task in tasks:
            agent_type = task["agent"]

            if agent_type == "economics":
                # Simulate economics agent refinement
                refinements.update(self._simulate_economics_refinement(task))
            elif agent_type == "resource":
                # Simulate resource agent refinement
                refinements.update(self._simulate_resource_refinement(task))
            elif agent_type == "policy":
                # Simulate policy agent refinement
                refinements.update(self._simulate_policy_refinement(task))

        return refinements

    def _simulate_economics_refinement(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate economics agent refinement."""
        # Add small random adjustments to simulate agent input
        current_output = self.current_plan["total_output"]
        adjustment = np.random.normal(0, 0.01, current_output.shape)
        refined_output = current_output * (1 + adjustment)

        return {"total_output": refined_output}

    def _simulate_resource_refinement(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate resource agent refinement."""
        # Simulate resource usage calculation
        current_output = self.current_plan["total_output"]
        resource_matrix = self.get_state("resource_matrix")

        if resource_matrix is not None:
            resource_usage = resource_matrix @ current_output
            return {"resource_usage": resource_usage}

        return {}

    def _simulate_policy_refinement(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply policy agent refinement with proper sector mapping."""
        policy_goals = task["parameters"].get("goals", [])
        
        if not policy_goals:
            return {}
        
        # Get sector mapping from state
        sector_mapping = self.get_state("sector_mapping", {})
        
        # Create policy task with sector mapping
        policy_task = {
            "type": "policy_adjustment",
            "current_plan": self.current_plan,
            "goals": policy_goals,
            "sector_mapping": sector_mapping
        }
        
        # Call the actual PolicyAgent
        policy_result = self.policy_agent.process_task(policy_task)
        
        if policy_result.get("status") == "success":
            # Extract the adjusted plan components
            adjusted_plan = policy_result.get("adjusted_plan", {})
            refinements = {}
            
            # Update final demand if it was modified
            if "final_demand" in adjusted_plan:
                refinements["final_demand"] = adjusted_plan["final_demand"]
            
            # Update labor vector if it was modified
            if "labor_vector" in adjusted_plan:
                refinements["labor_vector"] = adjusted_plan["labor_vector"]
            
            # Store constraint adjustments if any
            if "constraint_adjustments" in adjusted_plan:
                refinements["constraint_adjustments"] = adjusted_plan["constraint_adjustments"]
            
            return refinements
        else:
            # Fallback to simple simulation if PolicyAgent fails
            current_demand = self.current_plan["final_demand"]
            adjusted_demand = current_demand.copy()
            for goal in policy_goals:
                if "increase" in goal.lower():
                    adjustment = np.random.uniform(0.01, 0.05, current_demand.shape)
                    adjusted_demand *= 1 + adjustment
            return {"final_demand": adjusted_demand}

    def _check_convergence(self) -> bool:
        """Check if the plan has converged."""
        if len(self.plan_history) < 2:
            return False

        current_plan = self.plan_history[-1]
        previous_plan = self.plan_history[-2]

        # Check convergence of total output
        current_output = current_plan["total_output"]
        previous_output = previous_plan["total_output"]

        relative_change = np.linalg.norm(current_output - previous_output) / np.linalg.norm(previous_output)

        return relative_change < self.convergence_threshold

    def _calculate_plan_quality(self) -> float:
        """Calculate plan quality score."""
        if self.current_plan is None:
            return 0.0

        # Simple quality score based on labor efficiency
        total_labor_cost = self.current_plan["total_labor_cost"]
        total_output = self.current_plan["total_output"]

        if np.sum(total_output) > 0:
            labor_efficiency = 1.0 / (total_labor_cost / np.sum(total_output))
            return min(labor_efficiency, 1.0)

        return 0.0

    def _check_constraint_violations(self) -> Dict[str, Any]:
        """Check for constraint violations in the current plan."""
        if self.current_plan is None:
            return {}

        violations = {"demand_violations": [], "resource_violations": [], "non_negativity_violations": []}

        # Check non - negativity
        total_output = self.current_plan["total_output"]
        negative_outputs = total_output[total_output < 0]
        if len(negative_outputs) > 0:
            violations["non_negativity_violations"] = negative_outputs.tolist()

        return violations

    def get_plan_history(self) -> List[Dict[str, Any]]:
        """Get the complete plan history."""
        return self.plan_history.copy()

    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get the current plan."""
        return self.current_plan.copy() if self.current_plan else None

    def reset_planning(self) -> None:
        """Reset the planning process."""
        self.planning_iteration = 0
        self.current_plan = None
        self.plan_history = []
        self.agent_status = {}
