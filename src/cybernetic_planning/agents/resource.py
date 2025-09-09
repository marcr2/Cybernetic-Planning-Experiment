"""
Resource Agent

Specialized agent for resource and environmental analysis.
Manages resource constraints and environmental impact assessment.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .base import BaseAgent


class ResourceAgent(BaseAgent):
    """
    Resource and environmental specialist agent.

    Manages resource constraints, environmental impact assessment,
    and resource optimization strategies.
    """

    def __init__(self):
        """Initialize the resource agent."""
        super().__init__("resource", "Resource & Environmental Agent")
        self.resource_database = {}
        self.environmental_models = {}

    def get_capabilities(self) -> List[str]:
        """Get resource agent capabilities."""
        return [
            "resource_constraint_analysis",
            "environmental_impact_assessment",
            "resource_optimization",
            "sustainability_analysis",
            "resource_substitution",
        ]

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a resource analysis task.

        Args:
            task: Task description and parameters

        Returns:
            Analysis results
        """
        task_type = task.get("type", "unknown")

        if task_type == "resource_optimization":
            return self._optimize_resource_usage(task)
        elif task_type == "environmental_assessment":
            return self._assess_environmental_impact(task)
        elif task_type == "resource_constraint_analysis":
            return self._analyze_resource_constraints(task)
        elif task_type == "sustainability_analysis":
            return self._analyze_sustainability(task)
        elif task_type == "resource_substitution":
            return self._analyze_resource_substitution(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}

    def _optimize_resource_usage(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize resource usage for a given economic plan.

        Args:
            task: Task parameters including current plan and resource constraints

        Returns:
            Resource optimization results
        """
        current_plan = task.get("current_plan")
        resource_matrix = task.get("resource_matrix")
        max_resources = task.get("max_resources")

        if current_plan is None:
            return {"error": "Missing current plan"}

        total_output = current_plan.get("total_output")
        if total_output is None:
            return {"error": "Missing total output in plan"}

        # Calculate current resource usage
        if resource_matrix is not None and max_resources is not None:
            current_usage = resource_matrix @ total_output
            resource_utilization = current_usage / max_resources

            # Identify resource bottlenecks
            bottlenecks = self._identify_resource_bottlenecks(resource_utilization, max_resources)

            # Suggest optimization strategies
            optimization_strategies = self._suggest_optimization_strategies(bottlenecks, resource_matrix, total_output)

            return {
                "status": "success",
                "current_usage": current_usage,
                "resource_utilization": resource_utilization,
                "bottlenecks": bottlenecks,
                "optimization_strategies": optimization_strategies,
                "analysis_type": "resource_optimization",
            }
        else:
            # Calculate current usage from the plan data if available
            current_usage = current_plan.get("resource_usage")
            resource_utilization = None

            if current_usage is not None and max_resources is not None:
                resource_utilization = current_usage / max_resources

            return {
                "status": "success",
                "current_usage": current_usage,
                "resource_utilization": resource_utilization,
                "message": "No resource constraints provided",
                "analysis_type": "resource_optimization",
            }

    def _identify_resource_bottlenecks(
        self, utilization: np.ndarray, max_resources: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Identify resource bottlenecks based on utilization rates.

        Args:
            utilization: Resource utilization rates
            max_resources: Maximum resource availability

        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        threshold = 0.8  # 80% utilization threshold

        for i, (util, max_res) in enumerate(zip(utilization, max_resources)):
            if util > threshold:
                bottlenecks.append(
                    {
                        "resource_index": i,
                        "utilization_rate": util,
                        "max_availability": max_res,
                        "current_usage": util * max_res,
                        "severity": "high" if util > 0.95 else "medium",
                    }
                )

        return bottlenecks

    def _suggest_optimization_strategies(
        self, bottlenecks: List[Dict[str, Any]], resource_matrix: np.ndarray, total_output: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Suggest optimization strategies for resource bottlenecks.

        Args:
            bottlenecks: List of identified bottlenecks
            resource_matrix: Resource constraint matrix
            total_output: Current total output

        Returns:
            List of optimization strategies
        """
        strategies = []

        for bottleneck in bottlenecks:
            resource_idx = bottleneck["resource_index"]

            # Find sectors with highest resource consumption
            resource_consumption = resource_matrix[resource_idx, :] * total_output
            top_consumers = np.argsort(resource_consumption)[-3:]  # Top 3 consumers

            strategy = {
                "bottleneck_resource": resource_idx,
                "strategy_type": "reduce_consumption",
                "target_sectors": top_consumers.tolist(),
                "potential_savings": resource_consumption[top_consumers].sum(),
                "description": f"Reduce consumption in sectors {top_consumers.tolist()}",
            }
            strategies.append(strategy)

            # Suggest substitution strategies
            substitution_strategy = self._suggest_resource_substitution(resource_idx, resource_matrix, total_output)
            if substitution_strategy:
                strategies.append(substitution_strategy)

        return strategies

    def _suggest_resource_substitution(
        self, resource_idx: int, resource_matrix: np.ndarray, total_output: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest resource substitution strategies.

        Args:
            resource_idx: Index of the constrained resource
            resource_matrix: Resource constraint matrix
            total_output: Current total output

        Returns:
            Substitution strategy or None
        """
        # Find alternative resources with similar properties
        # This is a simplified implementation
        n_resources = resource_matrix.shape[0]

        if n_resources <= 1:
            return None

        # Calculate resource similarity
        similarity_scores = []
        for i in range(n_resources):
            if i != resource_idx:
                similarity = np.corrcoef(resource_matrix[resource_idx, :], resource_matrix[i, :])[0, 1]
                if not np.isnan(similarity):
                    similarity_scores.append((i, similarity))

        # Find best substitution candidate
        if similarity_scores:
            best_substitute = max(similarity_scores, key=lambda x: x[1])
            if best_substitute[1] > 0.7:  # High similarity threshold
                return {
                    "bottleneck_resource": resource_idx,
                    "strategy_type": "resource_substitution",
                    "substitute_resource": best_substitute[0],
                    "similarity_score": best_substitute[1],
                    "description": f"Substitute resource {resource_idx} with resource {best_substitute[0]}",
                }

        return None

    def _assess_environmental_impact(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess environmental impact of the economic plan.

        Args:
            task: Task parameters including current plan and environmental data

        Returns:
            Environmental impact assessment results
        """
        current_plan = task.get("current_plan")
        environmental_matrix = task.get("environmental_matrix")

        if current_plan is None:
            return {"error": "Missing current plan"}

        total_output = current_plan.get("total_output")
        if total_output is None:
            return {"error": "Missing total output in plan"}

        if environmental_matrix is not None:
            # Calculate environmental impacts
            environmental_impacts = environmental_matrix @ total_output

            # Calculate impact categories
            impact_categories = self._categorize_environmental_impacts(environmental_impacts)

            # Assess sustainability
            sustainability_score = self._calculate_sustainability_score(environmental_impacts)

            return {
                "status": "success",
                "environmental_impacts": environmental_impacts,
                "impact_categories": impact_categories,
                "sustainability_score": sustainability_score,
                "analysis_type": "environmental_assessment",
            }
        else:
            return {
                "status": "success",
                "message": "No environmental data provided",
                "analysis_type": "environmental_assessment",
            }

    def _categorize_environmental_impacts(self, impacts: np.ndarray) -> Dict[str, Any]:
        """
        Categorize environmental impacts by type.

        Args:
            impacts: Environmental impact vector

        Returns:
            Categorized impacts
        """
        # This is a simplified categorization
        # In practice, this would use actual environmental impact categories
        categories = {
            "carbon_emissions": impacts[0] if len(impacts) > 0 else 0,
            "water_usage": impacts[1] if len(impacts) > 1 else 0,
            "land_use": impacts[2] if len(impacts) > 2 else 0,
            "waste_generation": impacts[3] if len(impacts) > 3 else 0,
            "other_impacts": impacts[4:].sum() if len(impacts) > 4 else 0,
        }

        return categories

    def _calculate_sustainability_score(self, impacts: np.ndarray) -> float:
        """
        Calculate sustainability score based on environmental impacts.

        Args:
            impacts: Environmental impact vector

        Returns:
            Sustainability score (0-1, higher is better)
        """
        # Normalize impacts (simplified approach)
        normalized_impacts = impacts / (np.sum(impacts) + 1e-10)

        # Calculate sustainability score (inverse of total impact)
        sustainability_score = 1.0 / (1.0 + np.sum(normalized_impacts))

        return min(sustainability_score, 1.0)

    def _analyze_resource_constraints(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze resource constraints and their implications.

        Args:
            task: Task parameters including resource data

        Returns:
            Resource constraint analysis results
        """
        resource_matrix = task.get("resource_matrix")
        max_resources = task.get("max_resources")
        total_output = task.get("total_output")

        if resource_matrix is None or max_resources is None:
            return {"error": "Missing resource constraint data"}

        if total_output is None:
            return {"error": "Missing total output data"}

        # Calculate resource usage
        resource_usage = resource_matrix @ total_output

        # Calculate constraint violations
        violations = np.maximum(0, resource_usage - max_resources)

        # Calculate constraint tightness
        tightness = resource_usage / max_resources

        # Identify critical constraints
        critical_constraints = []
        for i, (usage, max_res, tight) in enumerate(zip(resource_usage, max_resources, tightness)):
            if tight > 0.9:  # 90% utilization threshold
                critical_constraints.append(
                    {
                        "resource_index": i,
                        "usage": usage,
                        "max_availability": max_res,
                        "tightness": tight,
                        "violation": violations[i],
                    }
                )

        return {
            "status": "success",
            "resource_usage": resource_usage,
            "constraint_tightness": tightness,
            "violations": violations,
            "critical_constraints": critical_constraints,
            "analysis_type": "resource_constraint_analysis",
        }

    def _analyze_sustainability(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overall sustainability of the economic plan.

        Args:
            task: Task parameters including plan and environmental data

        Returns:
            Sustainability analysis results
        """
        current_plan = task.get("current_plan")
        environmental_matrix = task.get("environmental_matrix")

        if current_plan is None:
            return {"error": "Missing current plan"}

        total_output = current_plan.get("total_output")
        if total_output is None:
            return {"error": "Missing total output in plan"}

        if environmental_matrix is not None:
            # Calculate environmental impacts
            environmental_impacts = environmental_matrix @ total_output

            # Calculate sustainability metrics
            sustainability_metrics = self._calculate_sustainability_metrics(environmental_impacts)

            # Assess long-term sustainability
            long_term_assessment = self._assess_long_term_sustainability(environmental_impacts)

            return {
                "status": "success",
                "environmental_impacts": environmental_impacts,
                "sustainability_metrics": sustainability_metrics,
                "long_term_assessment": long_term_assessment,
                "analysis_type": "sustainability_analysis",
            }
        else:
            return {
                "status": "success",
                "message": "No environmental data provided",
                "analysis_type": "sustainability_analysis",
            }

    def _calculate_sustainability_metrics(self, impacts: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive sustainability metrics.

        Args:
            impacts: Environmental impact vector

        Returns:
            Dictionary of sustainability metrics
        """
        metrics = {
            "total_environmental_impact": np.sum(impacts),
            "average_impact_per_unit": np.mean(impacts),
            "impact_variance": np.var(impacts),
            "sustainability_score": self._calculate_sustainability_score(impacts),
            "carbon_intensity": impacts[0] if len(impacts) > 0 else 0,
            "water_efficiency": 1.0 / (impacts[1] + 1e-10) if len(impacts) > 1 else 0,
        }

        return metrics

    def _assess_long_term_sustainability(self, impacts: np.ndarray) -> Dict[str, Any]:
        """
        Assess long-term sustainability implications.

        Args:
            impacts: Environmental impact vector

        Returns:
            Long-term sustainability assessment
        """
        # Project impacts over time (simplified model)
        time_horizon = 20  # years
        projected_impacts = impacts * (1.02**time_horizon)  # 2% annual growth

        # Assess sustainability thresholds
        thresholds = {
            "carbon_emissions": 1000,  # Example threshold
            "water_usage": 5000,  # Example threshold
            "land_use": 10000,  # Example threshold
        }

        threshold_violations = []
        for i, (impact, threshold) in enumerate(zip(projected_impacts, list(thresholds.values()))):
            if impact > threshold:
                threshold_violations.append(
                    {
                        "impact_type": list(thresholds.keys())[i],
                        "projected_impact": impact,
                        "threshold": threshold,
                        "violation_magnitude": impact - threshold,
                    }
                )

        return {
            "projected_impacts": projected_impacts,
            "threshold_violations": threshold_violations,
            "sustainable": len(threshold_violations) == 0,
            "time_horizon": time_horizon,
        }

    def _analyze_resource_substitution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze potential resource substitution opportunities.

        Args:
            task: Task parameters including resource data

        Returns:
            Resource substitution analysis results
        """
        resource_matrix = task.get("resource_matrix")
        total_output = task.get("total_output")

        if resource_matrix is None or total_output is None:
            return {"error": "Missing resource data"}

        n_resources, n_sectors = resource_matrix.shape

        # Calculate resource similarity matrix
        similarity_matrix = np.zeros((n_resources, n_resources))
        for i in range(n_resources):
            for j in range(n_resources):
                if i != j:
                    similarity = np.corrcoef(resource_matrix[i, :], resource_matrix[j, :])[0, 1]
                    if not np.isnan(similarity):
                        similarity_matrix[i, j] = similarity

        # Find substitution opportunities
        substitution_opportunities = []
        for i in range(n_resources):
            for j in range(n_resources):
                if i != j and similarity_matrix[i, j] > 0.7:
                    substitution_opportunities.append(
                        {
                            "from_resource": i,
                            "to_resource": j,
                            "similarity": similarity_matrix[i, j],
                            "substitution_potential": self._calculate_substitution_potential(
                                resource_matrix[i, :], resource_matrix[j, :], total_output
                            ),
                        }
                    )

        # Sort by substitution potential
        substitution_opportunities.sort(key=lambda x: x["substitution_potential"], reverse=True)

        return {
            "status": "success",
            "similarity_matrix": similarity_matrix,
            "substitution_opportunities": substitution_opportunities,
            "analysis_type": "resource_substitution",
        }

    def _calculate_substitution_potential(
        self, resource_i: np.ndarray, resource_j: np.ndarray, total_output: np.ndarray
    ) -> float:
        """
        Calculate substitution potential between two resources.

        Args:
            resource_i: First resource vector
            resource_j: Second resource vector
            total_output: Total output vector

        Returns:
            Substitution potential score
        """
        # Calculate current usage
        usage_i = np.sum(resource_i * total_output)
        usage_j = np.sum(resource_j * total_output)

        # Calculate substitution potential based on usage similarity
        if usage_i > 0 and usage_j > 0:
            usage_ratio = min(usage_i, usage_j) / max(usage_i, usage_j)
            return usage_ratio
        else:
            return 0.0

    def clear_cache(self) -> None:
        """Clear agent cache and reset state."""
        self.resource_database = {}
        self.environmental_models = {}
