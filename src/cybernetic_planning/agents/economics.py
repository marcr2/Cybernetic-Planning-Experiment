"""
Economics Agent

Specialized agent for economic analysis and sensitivity analysis.
Performs econometric modeling and forecasts technology matrix changes.
"""

from typing import Dict, Any, List
from .base import BaseAgent

class EconomicsAgent(BaseAgent):
    """
    Economics specialist agent for economic analysis.

    Performs sensitivity analysis on the Leontief inverse, forecasts
    technology matrix changes, and analyzes economic relationships.
    """

    def __init__(self):
        """Initialize the economics agent."""
        super().__init__("economics", "Economics Analysis Agent")
        self.sensitivity_cache = {}
        self.forecast_models = {}

    def get_capabilities(self) -> List[str]:
        """Get economics agent capabilities."""
        return [
            "sensitivity_analysis",
            "technology_forecasting",
            "economic_modeling",
            "supply_chain_analysis",
            "substitution_effects",
        ]

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an economics analysis task.

        Args:
            task: Task description and parameters

        Returns:
            Analysis results
        """
        task_type = task.get("type", "unknown")

        if task_type == "sensitivity_analysis":
            return self._perform_sensitivity_analysis(task)
        elif task_type == "technology_forecast":
            return self._forecast_technology_changes(task)
        elif task_type == "supply_chain_analysis":
            return self._analyze_supply_chains(task)
        elif task_type == "substitution_effects":
            return self._analyze_substitution_effects(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}

    def _perform_sensitivity_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on the Leontief model.

        Args:
            task: Task parameters including technology matrix and final demand

        Returns:
            Sensitivity analysis results
        """
        technology_matrix = task.get("technology_matrix")
        final_demand = task.get("final_demand")

        if technology_matrix is None or final_demand is None:
            return {"error": "Missing required parameters"}

        # Calculate sensitivity matrices
        sensitivity_A = self._calculate_technology_sensitivity(technology_matrix, final_demand)
        sensitivity_d = self._calculate_demand_sensitivity(technology_matrix, final_demand)

        # Identify critical sectors
        critical_sectors = self._identify_critical_sectors(sensitivity_A, sensitivity_d)

        # Calculate elasticity measures
        elasticities = self._calculate_elasticities(technology_matrix, final_demand)

        return {
            "status": "success",
            "technology_sensitivity": sensitivity_A,
            "demand_sensitivity": sensitivity_d,
            "critical_sectors": critical_sectors,
            "elasticities": elasticities,
            "analysis_type": "sensitivity_analysis",
        }

    def _calculate_technology_sensitivity(self, A: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Calculate sensitivity of output to technology matrix changes.

        Args:
            A: Technology matrix
            d: Final demand vector

        Returns:
            Sensitivity matrix ∂x/∂A_ij
        """
        n = A.shape[0]
        I = np.eye(n)

        try:
            # Calculate Leontief inverse
            leontief_inverse = np.linalg.inv(I - A)

            # Calculate total output
            x = leontief_inverse @ d

            # Calculate sensitivity matrix
            sensitivity = np.zeros((n, n, n))
            for i in range(n):
                for j in range(n):
                    e_i = np.zeros(n)
                    e_i[i] = 1
                    sensitivity[:, i, j] = leontief_inverse @ (e_i * x[j])

            return sensitivity

        except np.linalg.LinAlgError:
            return np.zeros((n, n, n))

    def _calculate_demand_sensitivity(self, A: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Calculate sensitivity of output to final demand changes.

        Args:
            A: Technology matrix
            d: Final demand vector

        Returns:
            Sensitivity matrix ∂x/∂d
        """
        n = A.shape[0]
        I = np.eye(n)

        try:
            # Calculate Leontief inverse
            leontief_inverse = np.linalg.inv(I - A)
            return leontief_inverse

        except np.linalg.LinAlgError:
            return np.zeros((n, n))

    def _identify_critical_sectors(self, sensitivity_A: np.ndarray, sensitivity_d: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify critical sectors based on sensitivity analysis.

        Args:
            sensitivity_A: Technology sensitivity matrix
            sensitivity_d: Demand sensitivity matrix

        Returns:
            List of critical sector information
        """
        n = sensitivity_A.shape[0]
        critical_sectors = []

        # Calculate sector importance scores
        for i in range(n):
            # Technology sensitivity score
            tech_sensitivity = np.sum(np.abs(sensitivity_A[i, :, :]))

            # Demand sensitivity score
            demand_sensitivity = np.sum(np.abs(sensitivity_d[i, :]))

            # Combined importance score
            importance_score = tech_sensitivity + demand_sensitivity

            if importance_score > np.percentile(
                [np.sum(np.abs(sensitivity_A[j, :, :])) + np.sum(np.abs(sensitivity_d[j, :])) for j in range(n)], 75
            ):
                critical_sectors.append(
                    {
                        "sector_index": i,
                        "tech_sensitivity": tech_sensitivity,
                        "demand_sensitivity": demand_sensitivity,
                        "importance_score": importance_score,
                    }
                )

        # Sort by importance score
        critical_sectors.sort(key = lambda x: x["importance_score"], reverse = True)

        return critical_sectors

    def _calculate_elasticities(self, A: np.ndarray, d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate elasticity measures for the economic system.

        Args:
            A: Technology matrix
            d: Final demand vector

        Returns:
            Dictionary with elasticity measures
        """
        n = A.shape[0]
        I = np.eye(n)

        try:
            # Calculate Leontief inverse
            leontief_inverse = np.linalg.inv(I - A)

            # Calculate total output
            x = leontief_inverse @ d

            # Calculate elasticities
            output_elasticities = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if x[j] != 0:
                        output_elasticities[i, j] = (leontief_inverse[i, j] * d[j]) / x[i]

            # Calculate sector elasticities
            sector_elasticities = np.zeros(n)
            for i in range(n):
                if x[i] != 0:
                    sector_elasticities[i] = np.sum(output_elasticities[i, :])

            return {"output_elasticities": output_elasticities, "sector_elasticities": sector_elasticities}

        except np.linalg.LinAlgError:
            return {"output_elasticities": np.zeros((n, n)), "sector_elasticities": np.zeros(n)}

    def _forecast_technology_changes(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forecast technology matrix changes over time.

        Args:
            task: Task parameters including current technology matrix and time horizon

        Returns:
            Technology forecast results
        """
        current_A = task.get("technology_matrix")
        time_horizon = task.get("time_horizon", 5)
        growth_rate = task.get("growth_rate", 0.02)

        if current_A is None:
            return {"error": "Missing technology matrix"}

        # Simple exponential decay model for technology improvement
        forecast_matrices = {}
        for t in range(1, time_horizon + 1):
            # Technology improves over time (reduces input requirements)
            improvement_factor = (1 - growth_rate) ** t
            forecast_matrices[t] = current_A * improvement_factor

        # Calculate forecasted changes
        changes = {}
        for t in range(1, time_horizon + 1):
            changes[t] = forecast_matrices[t] - current_A

        return {
            "status": "success",
            "forecast_matrices": forecast_matrices,
            "changes": changes,
            "analysis_type": "technology_forecast",
        }

    def _analyze_supply_chains(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze supply chain relationships and dependencies.

        Args:
            task: Task parameters including technology matrix

        Returns:
            Supply chain analysis results
        """
        A = task.get("technology_matrix")
        if A is None:
            return {"error": "Missing technology matrix"}

        n = A.shape[0]

        # Calculate supply chain strength matrix
        supply_chain_strength = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if A[i, j] > 0:
                    # Calculate indirect dependencies
                    indirect_deps = self._calculate_indirect_dependencies(A, i, j)
                    supply_chain_strength[i, j] = A[i, j] + indirect_deps

        # Identify key supply chains
        key_chains = self._identify_key_supply_chains(supply_chain_strength)

        # Calculate supply chain vulnerability
        vulnerability = self._calculate_supply_chain_vulnerability(A)

        return {
            "status": "success",
            "supply_chain_strength": supply_chain_strength,
            "key_chains": key_chains,
            "vulnerability": vulnerability,
            "analysis_type": "supply_chain_analysis",
        }

    def _calculate_indirect_dependencies(self, A: np.ndarray, i: int, j: int, max_depth: int = 3) -> float:
        """
        Calculate indirect dependencies between sectors.

        Args:
            A: Technology matrix
            i: Source sector
            j: Target sector
            max_depth: Maximum depth to search

        Returns:
            Indirect dependency strength
        """
        A.shape[0]
        total_indirect = 0.0

        for depth in range(1, max_depth + 1):
            A_power = np.linalg.matrix_power(A, depth)
            if A_power[i, j] > 0:
                total_indirect += A_power[i, j] * (0.5**depth)  # Decay with depth

        return total_indirect

    def _identify_key_supply_chains(self, strength_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify key supply chains based on strength matrix.

        Args:
            strength_matrix: Supply chain strength matrix

        Returns:
            List of key supply chains
        """
        n = strength_matrix.shape[0]
        key_chains = []

        # Find strongest connections
        for i in range(n):
            for j in range(n):
                if strength_matrix[i, j] > np.percentile(strength_matrix.flatten(), 90):
                    key_chains.append({"source_sector": i, "target_sector": j, "strength": strength_matrix[i, j]})

        # Sort by strength
        key_chains.sort(key = lambda x: x["strength"], reverse = True)

        return key_chains[:10]  # Return top 10

    def _calculate_supply_chain_vulnerability(self, A: np.ndarray) -> np.ndarray:
        """
        Calculate supply chain vulnerability for each sector.

        Args:
            A: Technology matrix

        Returns:
            Vulnerability vector
        """
        n = A.shape[0]
        vulnerability = np.zeros(n)

        for i in range(n):
            # Vulnerability based on input dependencies
            input_deps = np.sum(A[:, i])

            # Vulnerability based on output dependencies
            output_deps = np.sum(A[i, :])

            # Combined vulnerability score
            vulnerability[i] = input_deps + output_deps

        return vulnerability

    def _analyze_substitution_effects(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze substitution effects in the economy.

        Args:
            task: Task parameters including technology matrix

        Returns:
            Substitution analysis results
        """
        A = task.get("technology_matrix")
        if A is None:
            return {"error": "Missing technology matrix"}

        n = A.shape[0]

        # Calculate substitution possibilities
        substitution_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate substitution potential based on input similarity
                    input_similarity = np.corrcoef(A[:, i], A[:, j])[0, 1]
                    if not np.isnan(input_similarity):
                        substitution_matrix[i, j] = input_similarity

        # Identify substitution opportunities
        substitution_opportunities = []
        for i in range(n):
            for j in range(n):
                if i != j and substitution_matrix[i, j] > 0.7:
                    substitution_opportunities.append(
                        {"sector_i": i, "sector_j": j, "substitution_potential": substitution_matrix[i, j]}
                    )

        return {
            "status": "success",
            "substitution_matrix": substitution_matrix,
            "substitution_opportunities": substitution_opportunities,
            "analysis_type": "substitution_effects",
        }

    def get_sensitivity_cache(self) -> Dict[str, Any]:
        """Get the sensitivity analysis cache."""
        return self.sensitivity_cache.copy()

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.sensitivity_cache = {}
        self.forecast_models = {}
