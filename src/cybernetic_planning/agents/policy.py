"""
Policy Agent

Specialized agent for social policy analysis and goal translation.
Converts natural language goals into quantitative plan adjustments.
"""

from typing import Dict, Any, List, Optional, Union
import re
import numpy as np
from .base import BaseAgent

class PolicyAgent(BaseAgent):
    """
    Social policy specialist agent.

    Translates qualitative social and political goals into quantitative
    adjustments to economic plans and constraints.
    """

    def __init__(self):
        """Initialize the policy agent."""
        super().__init__("policy", "Social Policy Agent")
        self.goal_templates = {}
        self.policy_database = {}
        self._initialize_goal_templates()

    def _initialize_goal_templates(self) -> None:
        """Initialize templates for common policy goals."""
        self.goal_templates = {
            "carbon_reduction": {
                "keywords": ["carbon", "emissions", "co2", "greenhouse", "climate"],
                "adjustment_type": "constraint",
                "target_sectors": ["energy", "transport", "manufacturing"],
                "adjustment_factor": -0.2,  # 20% reduction
            },
            "healthcare_increase": {
                "keywords": ["healthcare", "health", "medical", "hospital", "medicine"],
                "adjustment_type": "demand",
                "target_sectors": ["healthcare", "pharmaceuticals", "medical_equipment"],
                "adjustment_factor": 0.15,  # 15% increase
            },
            "education_improvement": {
                "keywords": ["education", "school", "university", "learning", "training"],
                "adjustment_type": "demand",
                "target_sectors": ["education", "publishing", "technology"],
                "adjustment_factor": 0.10,  # 10% increase
            },
            "food_security": {
                "keywords": ["food", "agriculture", "nutrition", "hunger", "calories"],
                "adjustment_type": "constraint",
                "target_sectors": ["agriculture", "food_processing"],
                "adjustment_factor": 0.05,  # 5% increase in minimum
            },
            "employment_increase": {
                "keywords": ["employment", "jobs", "unemployment", "work", "labor"],
                "adjustment_type": "labor",
                "target_sectors": ["all"],
                "adjustment_factor": 0.08,  # 8% increase in labor
            },
            "infrastructure_development": {
                "keywords": ["infrastructure", "construction", "roads", "bridges", "buildings"],
                "adjustment_type": "demand",
                "target_sectors": ["construction", "steel", "cement", "machinery"],
                "adjustment_factor": 0.12,  # 12% increase
            },
        }

    def get_capabilities(self) -> List[str]:
        """Get policy agent capabilities."""
        return [
            "goal_translation",
            "policy_analysis",
            "constraint_adjustment",
            "demand_modification",
            "social_impact_assessment",
        ]

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a policy analysis task.

        Args:
            task: Task description and parameters

        Returns:
            Policy analysis results
        """
        task_type = task.get("type", "unknown")

        if task_type == "policy_adjustment":
            return self._adjust_policy_goals(task)
        elif task_type == "goal_translation":
            return self._translate_goals(task)
        elif task_type == "social_impact_assessment":
            return self._assess_social_impact(task)
        elif task_type == "constraint_modification":
            return self._modify_constraints(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}

    def _adjust_policy_goals(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust economic plan based on policy goals.

        Args:
            task: Task parameters including current plan and policy goals

        Returns:
            Policy adjustment results
        """
        current_plan = task.get("current_plan")
        policy_goals = task.get("goals", [])
        sector_mapping = task.get("sector_mapping", {})

        if current_plan is None:
            return {"error": "Missing current plan"}

        if not policy_goals:
            return {"status": "success", "message": "No policy goals provided", "adjusted_plan": current_plan}

        # Translate goals into quantitative adjustments using sector mapping
        goal_adjustments = self._translate_goals_to_adjustments(policy_goals, sector_mapping)

        # Apply adjustments to the plan
        adjusted_plan = self._apply_goal_adjustments(current_plan, goal_adjustments)

        return {
            "status": "success",
            "goal_adjustments": goal_adjustments,
            "adjusted_plan": adjusted_plan,
            "analysis_type": "policy_adjustment",
        }

    def _translate_goals(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate natural language goals into quantitative adjustments.

        Args:
            task: Task parameters including goals and sector mapping

        Returns:
            Goal translation results
        """
        goals = task.get("goals", [])
        sector_mapping = task.get("sector_mapping", {})

        if not goals:
            return {"error": "No goals provided"}

        translated_goals = []

        for goal in goals:
            if isinstance(goal, str):
                translated_goal = self._translate_single_goal(goal, sector_mapping)
                if translated_goal:
                    translated_goals.append(translated_goal)
            elif isinstance(goal, dict):
                # Goal already in structured format
                translated_goals.append(goal)

        return {"status": "success", "translated_goals": translated_goals, "analysis_type": "goal_translation"}

    def _translate_single_goal(self, goal_text: str, sector_mapping: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """
        Translate a single natural language goal into quantitative adjustments.

        Args:
            goal_text: Natural language goal description
            sector_mapping: Mapping from sector names to indices

        Returns:
            Translated goal or None if not recognized
        """
        goal_text_lower = goal_text.lower()

        # Find matching goal template
        for template_name, template in self.goal_templates.items():
            if any(keyword in goal_text_lower for keyword in template["keywords"]):
                # Extract numerical values from goal text
                numbers = re.findall(r"\d+(?:\.\d+)?", goal_text)
                if numbers:
                    target_value = float(numbers[0])
                else:
                    target_value = template["adjustment_factor"]

                # Determine target sectors using improved mapping
                target_sectors = []
                if template["target_sectors"] == ["all"]:
                    # Get all sector indices from mapping
                    target_sectors = [idx for idx in sector_mapping.values() if isinstance(idx, int)]
                else:
                    for sector_name in template["target_sectors"]:
                        # Try exact name match first
                        if sector_name in sector_mapping:
                            if isinstance(sector_mapping[sector_name], int):
                                target_sectors.append(sector_mapping[sector_name])
                            elif isinstance(sector_mapping[sector_name], list):
                                target_sectors.extend(sector_mapping[sector_name])
                        else:
                            # Try partial name matching
                            for mapped_name, mapped_idx in sector_mapping.items():
                                if isinstance(mapped_name, str) and sector_name.lower() in mapped_name.lower():
                                    if isinstance(mapped_idx, int):
                                        target_sectors.append(mapped_idx)
                                    elif isinstance(mapped_idx, list):
                                        target_sectors.extend(mapped_idx)

                # Remove duplicates and ensure valid indices
                target_sectors = list(set([idx for idx in target_sectors if isinstance(idx, int) and idx >= 0]))

                return {
                    "template_name": template_name,
                    "original_text": goal_text,
                    "adjustment_type": template["adjustment_type"],
                    "target_sectors": target_sectors,
                    "target_value": target_value,
                    "adjustment_factor": template["adjustment_factor"],
                }

        return None

    def _translate_goals_to_adjustments(self, goals: List[Union[str, Dict[str, Any]]], sector_mapping: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Translate structured goals into plan adjustments.

        Args:
            goals: List of goals (strings or structured dictionaries)
            sector_mapping: Mapping from sector names to indices

        Returns:
            List of plan adjustments
        """
        if sector_mapping is None:
            sector_mapping = {}

        adjustments = []

        for goal in goals:
            if isinstance(goal, str):
                # Translate string goal to structured format first
                translated_goal = self._translate_single_goal(goal, sector_mapping)
                if translated_goal:
                    adjustment = {
                        "type": translated_goal["adjustment_type"],
                        "target_sectors": translated_goal["target_sectors"],
                        "value": translated_goal["target_value"],
                        "factor": translated_goal["adjustment_factor"],
                    }
                    adjustments.append(adjustment)
            elif isinstance(goal, dict):
                # Goal already in structured format
                adjustment = {
                    "type": goal["adjustment_type"],
                    "target_sectors": goal["target_sectors"],
                    "value": goal["target_value"],
                    "factor": goal["adjustment_factor"],
                }
                adjustments.append(adjustment)

        return adjustments

    def _apply_goal_adjustments(
        self, current_plan: Dict[str, Any], adjustments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply goal adjustments to the current plan.

        Args:
            current_plan: Current economic plan
            adjustments: List of adjustments to apply

        Returns:
            Adjusted plan
        """
        adjusted_plan = current_plan.copy()

        for adjustment in adjustments:
            if adjustment["type"] == "demand":
                adjusted_plan = self._apply_demand_adjustment(adjusted_plan, adjustment)
            elif adjustment["type"] == "constraint":
                adjusted_plan = self._apply_constraint_adjustment(adjusted_plan, adjustment)
            elif adjustment["type"] == "labor":
                adjusted_plan = self._apply_labor_adjustment(adjusted_plan, adjustment)

        return adjusted_plan

    def _apply_demand_adjustment(self, plan: Dict[str, Any], adjustment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply demand adjustment to the plan.

        Args:
            plan: Current plan
            adjustment: Demand adjustment parameters

        Returns:
            Plan with demand adjustment applied
        """
        if "final_demand" not in plan:
            return plan

        final_demand = plan["final_demand"].copy()
        target_sectors = adjustment["target_sectors"]
        factor = adjustment["factor"]

        for sector_idx in target_sectors:
            if 0 <= sector_idx < len(final_demand):
                final_demand[sector_idx] *= 1 + factor

        plan["final_demand"] = final_demand
        return plan

    def _apply_constraint_adjustment(self, plan: Dict[str, Any], adjustment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply constraint adjustment to the plan.

        Args:
            plan: Current plan
            adjustment: Constraint adjustment parameters

        Returns:
            Plan with constraint adjustment applied
        """
        # This would modify resource constraints or add new constraints
        # For now, we'll just store the adjustment for later use
        if "constraint_adjustments" not in plan:
            plan["constraint_adjustments"] = []

        plan["constraint_adjustments"].append(adjustment)
        return plan

    def _apply_labor_adjustment(self, plan: Dict[str, Any], adjustment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply labor adjustment to the plan.

        Args:
            plan: Current plan
            adjustment: Labor adjustment parameters

        Returns:
            Plan with labor adjustment applied
        """
        if "labor_vector" not in plan:
            return plan

        labor_vector = plan["labor_vector"].copy()
        target_sectors = adjustment["target_sectors"]
        factor = adjustment["factor"]

        for sector_idx in target_sectors:
            if 0 <= sector_idx < len(labor_vector):
                labor_vector[sector_idx] *= 1 + factor

        plan["labor_vector"] = labor_vector
        return plan

    def _assess_social_impact(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess social impact of the economic plan.

        Args:
            task: Task parameters including plan and social indicators

        Returns:
            Social impact assessment results
        """
        current_plan = task.get("current_plan")
        social_indicators = task.get("social_indicators", {})

        if current_plan is None:
            return {"error": "Missing current plan"}

        # Calculate social impact metrics
        social_metrics = self._calculate_social_metrics(current_plan, social_indicators)

        # Assess social goals achievement
        goal_achievement = self._assess_goal_achievement(current_plan, social_indicators)

        # Calculate social welfare score
        welfare_score = self._calculate_social_welfare_score(social_metrics)

        return {
            "status": "success",
            "social_metrics": social_metrics,
            "goal_achievement": goal_achievement,
            "welfare_score": welfare_score,
            "analysis_type": "social_impact_assessment",
        }

    def _calculate_social_metrics(self, plan: Dict[str, Any], social_indicators: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate social impact metrics.

        Args:
            plan: Current economic plan
            social_indicators: Social indicator data

        Returns:
            Dictionary of social metrics
        """
        total_output = plan.get("total_output", np.array([]))
        total_labor_cost = plan.get("total_labor_cost", 0)

        metrics = {
            "total_economic_output": np.sum(total_output),
            "average_sector_output": np.mean(total_output),
            "output_inequality": np.std(total_output) / (np.mean(total_output) + 1e-10),
            "total_labor_cost": total_labor_cost,
            "labor_efficiency": np.sum(total_output) / (total_labor_cost + 1e-10),
        }

        # Add sector - specific social metrics
        if "sector_social_weights" in social_indicators:
            social_weights = social_indicators["sector_social_weights"]
            if len(social_weights) == len(total_output):
                metrics["social_weighted_output"] = np.sum(social_weights * total_output)
                metrics["social_efficiency"] = metrics["social_weighted_output"] / (total_labor_cost + 1e-10)

        return metrics

    def _assess_goal_achievement(self, plan: Dict[str, Any], social_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess achievement of social goals.

        Args:
            plan: Current economic plan
            social_indicators: Social indicator data

        Returns:
            Goal achievement assessment
        """
        goal_achievement = {}

        # Check employment goals
        if "employment_target" in social_indicators:
            total_labor_cost = plan.get("total_labor_cost", 0)
            employment_target = social_indicators["employment_target"]
            goal_achievement["employment"] = {
                "target": employment_target,
                "actual": total_labor_cost,
                "achieved": total_labor_cost >= employment_target,
                "achievement_rate": total_labor_cost / (employment_target + 1e-10),
            }

        # Check output distribution goals
        if "output_distribution_target" in social_indicators:
            total_output = plan.get("total_output", np.array([]))
            distribution_target = social_indicators["output_distribution_target"]
            actual_distribution = total_output / (np.sum(total_output) + 1e-10)

            goal_achievement["output_distribution"] = {
                "target": distribution_target,
                "actual": actual_distribution,
                "achieved": np.allclose(actual_distribution, distribution_target, atol = 0.1),
                "achievement_rate": 1.0 - np.mean(np.abs(actual_distribution - distribution_target)),
            }

        return goal_achievement

    def _calculate_social_welfare_score(self, social_metrics: Dict[str, float]) -> float:
        """
        Calculate overall social welfare score.

        Args:
            social_metrics: Social impact metrics

        Returns:
            Social welfare score (0 - 1, higher is better)
        """
        # Weighted combination of social metrics
        weights = {
            "total_economic_output": 0.3,
            "labor_efficiency": 0.3,
            "output_inequality": -0.2,  # Negative weight for inequality
            "social_efficiency": 0.2,
        }

        welfare_score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in social_metrics:
                # Normalize metric to 0 - 1 range
                normalized_metric = min(max(social_metrics[metric] / 1000, 0), 1)  # Simple normalization
                welfare_score += weight * normalized_metric
                total_weight += abs(weight)

        if total_weight > 0:
            welfare_score /= total_weight

        return max(0, min(welfare_score, 1))

    def _modify_constraints(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify constraints based on policy requirements.

        Args:
            task: Task parameters including constraints and policy requirements

        Returns:
            Constraint modification results
        """
        current_constraints = task.get("constraints", {})
        policy_requirements = task.get("policy_requirements", [])

        modified_constraints = current_constraints.copy()

        for requirement in policy_requirements:
            if requirement["type"] == "minimum_output":
                modified_constraints = self._add_minimum_output_constraint(modified_constraints, requirement)
            elif requirement["type"] == "maximum_output":
                modified_constraints = self._add_maximum_output_constraint(modified_constraints, requirement)
            elif requirement["type"] == "resource_constraint":
                modified_constraints = self._add_resource_constraint(modified_constraints, requirement)

        return {
            "status": "success",
            "modified_constraints": modified_constraints,
            "analysis_type": "constraint_modification",
        }

    def _add_minimum_output_constraint(
        self, constraints: Dict[str, Any], requirement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add minimum output constraint."""
        if "minimum_outputs" not in constraints:
            constraints["minimum_outputs"] = {}

        constraints["minimum_outputs"][requirement["sector"]] = requirement["value"]
        return constraints

    def _add_maximum_output_constraint(
        self, constraints: Dict[str, Any], requirement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add maximum output constraint."""
        if "maximum_outputs" not in constraints:
            constraints["maximum_outputs"] = {}

        constraints["maximum_outputs"][requirement["sector"]] = requirement["value"]
        return constraints

    def _add_resource_constraint(self, constraints: Dict[str, Any], requirement: Dict[str, Any]) -> Dict[str, Any]:
        """Add resource constraint."""
        if "resource_constraints" not in constraints:
            constraints["resource_constraints"] = []

        constraints["resource_constraints"].append(requirement)
        return constraints

    def get_goal_templates(self) -> Dict[str, Any]:
        """Get available goal templates."""
        return self.goal_templates.copy()

    def add_goal_template(self, name: str, template: Dict[str, Any]) -> None:
        """Add a new goal template."""
        self.goal_templates[name] = template

    def update_goal_template(self, name: str, template: Dict[str, Any]) -> None:
        """Update an existing goal template."""
        if name in self.goal_templates:
            self.goal_templates[name].update(template)
        else:
            self.goal_templates[name] = template
