"""
Matrix Builder

Constructs and manipulates matrices and vectors for the cybernetic planning system.
Provides utilities for matrix operations, validation, and transformation.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd

from .synthetic_sector_generator import SyntheticSectorGenerator
from .technology_tree_mapper import TechnologyTreeMapper

class MatrixBuilder:
    """
    Builder for constructing and manipulating economic matrices.

    Provides utilities for creating, validating, and transforming
    technology matrices, demand vectors, and constraint matrices.
    """

    def __init__(self, max_sectors: int = 1000, use_technology_tree: bool = True):
        """Initialize the matrix builder."""
        self.matrices = {}
        self.vectors = {}
        self.max_sectors = max_sectors
        self.use_technology_tree = use_technology_tree

        if use_technology_tree:
            self.sector_mapper = TechnologyTreeMapper()
        else:
            self.sector_mapper = SyntheticSectorGenerator(max_sectors=max_sectors, min_sectors=6)

    def create_technology_matrix(
        self,
        data: Union[np.ndarray, List[List[float]], pd.DataFrame],
        sectors: Optional[List[str]] = None,
        name: str = "technology_matrix",
    ) -> np.ndarray:
        """
        Create a technology matrix from input data.

        Args:
            data: Input data (array, list, or DataFrame)
            sectors: List of sector names
            name: Name for the matrix

        Returns:
            Technology matrix as numpy array
        """
        if isinstance(data, pd.DataFrame):
            matrix = data.values
            if sectors is None:
                sectors = data.index.tolist()
        elif isinstance(data, list):
            matrix = np.array(data)
        else:
            matrix = np.asarray(data)

        # Validate matrix
        if matrix.ndim != 2:
            raise ValueError("Technology matrix must be 2 - dimensional")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Technology matrix must be square")

        # Store matrix
        self.matrices[name] = {"data": matrix, "sectors": sectors, "shape": matrix.shape, "type": "technology"}

        return matrix

    def create_final_demand_vector(
        self,
        data: Union[np.ndarray, List[float], pd.Series],
        sectors: Optional[List[str]] = None,
        name: str = "final_demand",
    ) -> np.ndarray:
        """
        Create a final demand vector from input data.

        Args:
            data: Input data (array, list, or Series)
            sectors: List of sector names
            name: Name for the vector

        Returns:
            Final demand vector as numpy array
        """
        if isinstance(data, pd.Series):
            vector = data.values
            if sectors is None:
                sectors = data.index.tolist()
        elif isinstance(data, list):
            vector = np.array(data)
        else:
            vector = np.asarray(data).flatten()

        # Validate vector
        if vector.ndim != 1:
            raise ValueError("Final demand vector must be 1 - dimensional")

        # Store vector
        self.vectors[name] = {"data": vector, "sectors": sectors, "length": len(vector), "type": "final_demand"}

        return vector

    def create_labor_vector(
        self,
        data: Union[np.ndarray, List[float], pd.Series],
        sectors: Optional[List[str]] = None,
        name: str = "labor_input",
    ) -> np.ndarray:
        """
        Create a labor input vector from input data.

        Args:
            data: Input data (array, list, or Series)
            sectors: List of sector names
            name: Name for the vector

        Returns:
            Labor input vector as numpy array
        """
        if isinstance(data, pd.Series):
            vector = data.values
            if sectors is None:
                sectors = data.index.tolist()
        elif isinstance(data, list):
            vector = np.array(data)
        else:
            vector = np.asarray(data).flatten()

        # Validate vector
        if vector.ndim != 1:
            raise ValueError("Labor input vector must be 1 - dimensional")

        # Store vector
        self.vectors[name] = {"data": vector, "sectors": sectors, "length": len(vector), "type": "labor_input"}

        return vector

    def create_resource_matrix(
        self,
        data: Union[np.ndarray, List[List[float]], pd.DataFrame],
        resources: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None,
        name: str = "resource_matrix",
    ) -> np.ndarray:
        """
        Create a resource constraint matrix from input data.

        Args:
            data: Input data (array, list, or DataFrame)
            resources: List of resource names
            sectors: List of sector names
            name: Name for the matrix

        Returns:
            Resource matrix as numpy array
        """
        if isinstance(data, pd.DataFrame):
            matrix = data.values
            if resources is None:
                resources = data.index.tolist()
            if sectors is None:
                sectors = data.columns.tolist()
        elif isinstance(data, list):
            matrix = np.array(data)
        else:
            matrix = np.asarray(data)

        # Validate matrix
        if matrix.ndim != 2:
            raise ValueError("Resource matrix must be 2 - dimensional")

        # Store matrix
        self.matrices[name] = {
            "data": matrix,
            "resources": resources,
            "sectors": sectors,
            "shape": matrix.shape,
            "type": "resource",
        }

        return matrix

    def create_max_resources_vector(
        self,
        data: Union[np.ndarray, List[float], pd.Series],
        resources: Optional[List[str]] = None,
        name: str = "max_resources",
    ) -> np.ndarray:
        """
        Create a maximum resources vector from input data.

        Args:
            data: Input data (array, list, or Series)
            resources: List of resource names
            name: Name for the vector

        Returns:
            Maximum resources vector as numpy array
        """
        if isinstance(data, pd.Series):
            vector = data.values
            if resources is None:
                resources = data.index.tolist()
        elif isinstance(data, list):
            vector = np.array(data)
        else:
            vector = np.asarray(data).flatten()

        # Validate vector
        if vector.ndim != 1:
            raise ValueError("Maximum resources vector must be 1 - dimensional")

        # Store vector
        self.vectors[name] = {"data": vector, "resources": resources, "length": len(vector), "type": "max_resources"}

        return vector

    def create_synthetic_data(
        self, n_sectors: int, technology_density: float = 0.3, resource_count: int = 5, name_prefix: str = "synthetic"
    ) -> Dict[str, Any]:
        """
        Create synthetic I - O data for testing and demonstration.

        Args:
            n_sectors: Number of economic sectors
            technology_density: Density of technology matrix (0 - 1)
            resource_count: Number of resource types
            name_prefix: Prefix for generated data names

        Returns:
            Dictionary containing synthetic data
        """
        # DEBUG: Check sector counts
        print(f"DEBUG MATRIX_BUILDER: Requested sectors: {n_sectors}")
        max_available = self.sector_mapper.get_sector_count()
        print(f"DEBUG MATRIX_BUILDER: Available sectors from mapper: {max_available}")

        # BUG FIX: Don't cap the sector count if we're using technology tree
        # The technology tree mapper should handle generating sectors dynamically
        if self.use_technology_tree:
            # Don't limit n_sectors for technology tree - it can generate sectors on demand
            effective_n_sectors = n_sectors
            print(f"DEBUG MATRIX_BUILDER: Using technology tree, not capping sector count")
        else:
            # Only limit for hierarchical mapper if it can't expand
            if n_sectors > max_available:
                # Try to expand the hierarchical mapper first
                print(f"DEBUG MATRIX_BUILDER: Expanding hierarchical mapper from {max_available} to {n_sectors}")
                self.sector_mapper = SyntheticSectorGenerator(max_sectors=n_sectors, min_sectors=6)
                max_available = self.sector_mapper.get_sector_count()
            effective_n_sectors = min(n_sectors, max_available)

        print(f"DEBUG MATRIX_BUILDER: Effective sectors to generate: {effective_n_sectors}")
        n_sectors = effective_n_sectors

        # Get sector information based on mapper type
        if self.use_technology_tree:
            # Get available sectors from technology tree
            available_sector_ids = self.sector_mapper.get_available_sectors(n_sectors)
            sector_names = self.sector_mapper.get_sector_names_for_sectors(available_sector_ids)
        else:
            # Use hierarchical mapper
            if n_sectors > self.sector_mapper.max_sectors:
                # Reinitialize the sector mapper with the requested number of sectors
                self.sector_mapper = SyntheticSectorGenerator(max_sectors=n_sectors, min_sectors=6)
            sector_names = self.sector_mapper.get_sector_names()[:n_sectors]

        # Generate technology matrix with sector - aware interactions
        if self.use_technology_tree:
            # Use technology tree for more realistic sector interactions
            # Adjust n_sectors to match actual available sectors
            actual_n_sectors = len(available_sector_ids)
            n_sectors = actual_n_sectors  # Update n_sectors to match actual count
            tech_matrix = self._generate_technology_tree_matrix(n_sectors, technology_density, available_sector_ids)
        else:
            tech_matrix = self._generate_synthetic_technology_matrix(n_sectors, technology_density)

        self.create_technology_matrix(tech_matrix, sectors = sector_names, name = f"{name_prefix}_technology")

        # Generate final demand vector with sector importance weighting
        if self.use_technology_tree:
            final_demand = self._generate_technology_tree_final_demand(n_sectors, available_sector_ids)
        else:
            final_demand = self._generate_synthetic_final_demand(n_sectors)

        self.create_final_demand_vector(final_demand, sectors = sector_names, name = f"{name_prefix}_final_demand")

        # Generate labor input vector with sector - specific labor intensity
        if self.use_technology_tree:
            labor_input = self._generate_technology_tree_labor_input(n_sectors, available_sector_ids)
        else:
            labor_input = self._generate_synthetic_labor_input(n_sectors)

        self.create_labor_vector(labor_input, sectors = sector_names, name = f"{name_prefix}_labor")

        # Generate resource matrix
        resource_matrix = self._generate_synthetic_resource_matrix(resource_count, n_sectors)
        self.create_resource_matrix(resource_matrix, name = f"{name_prefix}_resource")

        # Generate max resources vector
        max_resources = self._generate_synthetic_max_resources(resource_count)
        self.create_max_resources_vector(max_resources, name = f"{name_prefix}_max_resources")

        # Calculate total labor cost
        total_labor_cost = np.dot(labor_input, final_demand)
        
        # Calculate plan quality score (based on economic efficiency)
        plan_quality_score = self._calculate_plan_quality_score(tech_matrix, final_demand, labor_input)
        
        # Create resource allocation data structure for simulation
        resource_allocations = {
            "technology_matrix": tech_matrix.tolist(),
            "final_demand": final_demand.tolist(),
            "total_labor_cost": float(total_labor_cost),
            "plan_quality_score": float(plan_quality_score),
            "resource_matrix": resource_matrix.tolist(),
            "max_resources": max_resources.tolist(),
            "resource_names": [f"Resource_{i}" for i in range(resource_count)]
        }

        # Create comprehensive data dictionary
        data = {
            "technology_matrix": tech_matrix,
            "final_demand": final_demand,
            "labor_input": labor_input,
            "resource_matrix": resource_matrix,
            "max_resources": max_resources,
            "sectors": sector_names,
            "resources": [f"Resource_{i}" for i in range(resource_count)],
            "sector_definitions": self.sector_mapper.sectors if hasattr(self.sector_mapper, 'sectors') else {},
            "resource_allocations": resource_allocations
        }

        # Add technology tree specific data
        if self.use_technology_tree:
            data["technology_tree"] = {
                "sector_ids": available_sector_ids,
                "development_stage": self.sector_mapper.current_development_stage.value,
                "unlockable_sectors": self.sector_mapper.get_unlockable_sectors(),
                "dependency_matrix": self.sector_mapper.get_technology_dependency_matrix(available_sector_ids).tolist()
            }

        return data

    def _calculate_plan_quality_score(self, tech_matrix: np.ndarray, final_demand: np.ndarray, labor_input: np.ndarray) -> float:
        """
        Calculate a plan quality score based on economic efficiency metrics.
        
        Args:
            tech_matrix: Technology matrix A
            final_demand: Final demand vector d
            labor_input: Labor input vector l
            
        Returns:
            Plan quality score between 0 and 1
        """
        try:
            # Calculate labor efficiency (output per unit labor)
            total_output = np.sum(final_demand)
            total_labor = np.sum(labor_input)
            labor_efficiency = total_output / (total_labor + 1e-10)
            
            # Calculate technology matrix efficiency (lower intermediate consumption is better)
            intermediate_consumption = np.sum(tech_matrix)
            technology_efficiency = 1.0 / (1.0 + intermediate_consumption)
            
            # Calculate demand diversity (more balanced demand is better)
            demand_std = np.std(final_demand)
            demand_mean = np.mean(final_demand)
            demand_diversity = 1.0 / (1.0 + demand_std / (demand_mean + 1e-10))
            
            # Calculate sector balance (more balanced sectors is better)
            sector_balance = 1.0 / (1.0 + np.std(labor_input) / (np.mean(labor_input) + 1e-10))
            
            # Weighted combination of metrics
            quality_score = (
                0.4 * min(1.0, labor_efficiency / 10.0) +  # Labor efficiency (normalized)
                0.3 * technology_efficiency +              # Technology efficiency
                0.2 * demand_diversity +                   # Demand diversity
                0.1 * sector_balance                       # Sector balance
            )
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            print(f"Warning: Could not calculate plan quality score: {e}")
            return 0.5  # Default moderate score

    def _generate_synthetic_technology_matrix(self, n_sectors: int, density: float) -> np.ndarray:
        """Generate synthetic technology matrix that is economically viable."""
        # Create sparse matrix with specified density
        matrix = np.zeros((n_sectors, n_sectors))

        # Get sector interaction matrix from hierarchical mapper
        if n_sectors <= self.sector_mapper.get_sector_count():
            interaction_matrix = self.sector_mapper.get_economic_impact_matrix()[:n_sectors, :n_sectors]
        else:
            interaction_matrix = np.ones((n_sectors, n_sectors))

        # Adjust coefficient ranges based on economy size for realism
        if n_sectors <= 10:
            # Small economy: higher coefficients (more interconnected)
            coeff_range = (0.05, 0.4)
            diagonal_range = (0.02, 0.15)
        elif n_sectors <= 50:
            # Medium economy: moderate coefficients
            coeff_range = (0.02, 0.25)
            diagonal_range = (0.01, 0.08)
        elif n_sectors <= 100:
            # Large economy: lower coefficients
            coeff_range = (0.01, 0.15)
            diagonal_range = (0.005, 0.05)
        else:
            # Very large economy: much lower coefficients
            coeff_range = (0.005, 0.08)
            diagonal_range = (0.002, 0.03)

        # Fill with random values ensuring economic viability
        n_elements = int(n_sectors * n_sectors * density)
        indices = np.random.choice(n_sectors * n_sectors, n_elements, replace = False)

        for idx in indices:
            i, j = divmod(idx, n_sectors)
            # Base coefficient
            base_coeff = np.random.uniform(coeff_range[0], coeff_range[1])

            # Apply sector interaction weighting
            interaction_weight = interaction_matrix[i, j] if i < interaction_matrix.shape[0] and j < interaction_matrix.shape[1] else 0.1

            # Apply sector - specific characteristics
            if i < len(self.sector_mapper.sectors):
                sector_i = self.sector_mapper.get_sector_by_id(i)
                if sector_i:
                    # High capital intensity sectors have lower input coefficients
                    if sector_i.capital_intensity in ["very_high", "high"]:
                        base_coeff *= 0.7
                    # High technology sectors have more complex interactions
                    if sector_i.technology_level in ["advanced", "high"]:
                        base_coeff *= 1.2

            matrix[i, j] = base_coeff * interaction_weight

        # Ensure diagonal elements are small (self - consumption)
        np.fill_diagonal(matrix, np.random.uniform(diagonal_range[0], diagonal_range[1], n_sectors))

        # Ensure the matrix is productive (spectral radius < 1)
        # This is crucial for economic viability
        max_iterations = 100
        for iteration in range(max_iterations):
            spectral_radius = np.max(np.abs(np.linalg.eigvals(matrix)))
            if spectral_radius < 0.95:  # Leave some margin
                break

            # Scale down the matrix to reduce spectral radius
            matrix *= 0.9

            # Ensure no negative values
            matrix = np.maximum(matrix, 0)

        return matrix

    def _generate_synthetic_final_demand(self, n_sectors: int) -> np.ndarray:
        """Generate synthetic final demand vector with realistic values."""
        # Scale final demand based on number of sectors to maintain realistic economy size
        # For larger economies, use smaller per - sector demand to avoid unrealistic totals

        if n_sectors <= 10:
            # Small economy: higher per - sector demand
            base_demand = np.random.uniform(50, 200, n_sectors)
            high_demand_multiplier = np.random.uniform(2, 4)
        elif n_sectors <= 50:
            # Medium economy: moderate per - sector demand
            base_demand = np.random.uniform(20, 80, n_sectors)
            high_demand_multiplier = np.random.uniform(2, 3)
        elif n_sectors <= 100:
            # Large economy: lower per - sector demand
            base_demand = np.random.uniform(10, 40, n_sectors)
            high_demand_multiplier = np.random.uniform(1.5, 2.5)
        else:
            # Very large economy: much lower per - sector demand
            base_demand = np.random.uniform(2, 15, n_sectors)
            high_demand_multiplier = np.random.uniform(1.2, 2.0)

        # Apply sector importance weighting
        if n_sectors <= self.sector_mapper.get_sector_count():
            importance_weights = self.sector_mapper.get_importance_weights()[:n_sectors]
            base_demand *= importance_weights

        # Add some sectors with higher demand (consumer goods, services)
        high_demand_sectors = np.random.choice(n_sectors, size = max(1, n_sectors // 4), replace = False)
        base_demand[high_demand_sectors] *= high_demand_multiplier

        # Ensure total demand is reasonable for the economy size
        total_demand = np.sum(base_demand)
        target_total = min(10000, n_sectors * 20)  # Reasonable total for any economy size

        if total_demand > target_total:
            # Scale down to maintain realistic total
            base_demand *= target_total / total_demand

        return base_demand

    def _generate_synthetic_labor_input(self, n_sectors: int) -> np.ndarray:
        """Generate synthetic labor input vector with realistic values."""
        # Labor input should be positive and represent person - hours per unit output
        # Some sectors are more labor - intensive than others
        base_labor = np.random.uniform(0.5, 3.0, n_sectors)

        # Apply sector - specific labor intensity
        if n_sectors <= self.sector_mapper.get_sector_count():
            for i in range(n_sectors):
                sector = self.sector_mapper.get_sector_by_id(i)
                if sector:
                    if sector.labor_intensity == "high":
                        base_labor[i] *= np.random.uniform(1.5, 2.5)
                    elif sector.labor_intensity == "medium":
                        base_labor[i] *= np.random.uniform(1.0, 1.5)
                    else:  # low
                        base_labor[i] *= np.random.uniform(0.5, 1.0)

        # Add some highly labor - intensive sectors (services, crafts)
        labor_intensive_sectors = np.random.choice(n_sectors, size = max(1, n_sectors // 5), replace = False)
        base_labor[labor_intensive_sectors] *= np.random.uniform(2, 4, len(labor_intensive_sectors))

        return base_labor

    def get_sector_mapper(self) -> SyntheticSectorGenerator:
        """Get the hierarchical sector mapper instance."""
        return self.sector_mapper

    def get_sector_info(self, sector_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific sector."""
        sector = self.sector_mapper.get_sector_by_name(sector_name)
        if sector:
            return {
                "id": sector.id,
                "name": sector.name,
                "category": sector.category,
                "parent_category": sector.parent_category,
                "description": sector.description,
                "importance_weight": sector.importance_weight,
                "economic_impact": sector.economic_impact,
                "labor_intensity": sector.labor_intensity,
                "capital_intensity": sector.capital_intensity,
                "technology_level": sector.technology_level,
                "environmental_impact": sector.environmental_impact
            }
        return None

    def _generate_synthetic_resource_matrix(self, n_resources: int, n_sectors: int) -> np.ndarray:
        """Generate synthetic resource matrix."""
        return np.random.uniform(0, 5, (n_resources, n_sectors))

    def _generate_synthetic_max_resources(self, n_resources: int) -> np.ndarray:
        """Generate synthetic maximum resources vector."""
        return np.random.uniform(100, 1000, n_resources)

    def validate_matrix_dimensions(self, matrix_name: str, vector_name: str) -> bool:
        """
        Validate that matrix and vector have compatible dimensions.

        Args:
            matrix_name: Name of the matrix
            vector_name: Name of the vector

        Returns:
            True if dimensions are compatible
        """
        if matrix_name not in self.matrices:
            raise ValueError(f"Matrix '{matrix_name}' not found")

        if vector_name not in self.vectors:
            raise ValueError(f"Vector '{vector_name}' not found")

        matrix = self.matrices[matrix_name]
        vector = self.vectors[vector_name]

        if matrix["type"] == "technology":
            return matrix["shape"][0] == vector["length"]
        elif matrix["type"] == "resource":
            return matrix["shape"][1] == vector["length"]
        else:
            return False

    def get_matrix(self, name: str) -> Optional[np.ndarray]:
        """Get matrix by name."""
        if name in self.matrices:
            return self.matrices[name]["data"].copy()
        return None

    def get_vector(self, name: str) -> Optional[np.ndarray]:
        """Get vector by name."""
        if name in self.vectors:
            return self.vectors[name]["data"].copy()
        return None

    def get_matrix_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get matrix information."""
        return self.matrices.get(name)

    def get_vector_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get vector information."""
        return self.vectors.get(name)

    def list_matrices(self) -> List[str]:
        """Get list of matrix names."""
        return list(self.matrices.keys())

    def list_vectors(self) -> List[str]:
        """Get list of vector names."""
        return list(self.vectors.keys())

    def clear_all(self) -> None:
        """Clear all matrices and vectors."""
        self.matrices = {}
        self.vectors = {}

    def export_to_dataframe(self, matrix_name: str, include_metadata: bool = True) -> pd.DataFrame:
        """
        Export matrix to pandas DataFrame.

        Args:
            matrix_name: Name of the matrix to export
            include_metadata: Whether to include metadata in the DataFrame

        Returns:
            DataFrame representation of the matrix
        """
        if matrix_name not in self.matrices:
            raise ValueError(f"Matrix '{matrix_name}' not found")

        matrix_info = self.matrices[matrix_name]
        matrix_data = matrix_info["data"]

        if matrix_info["type"] == "technology":
            sectors = matrix_info.get("sectors", [f"Sector_{i}" for i in range(matrix_data.shape[0])])
            df = pd.DataFrame(matrix_data, index = sectors, columns = sectors)
        elif matrix_info["type"] == "resource":
            resources = matrix_info.get("resources", [f"Resource_{i}" for i in range(matrix_data.shape[0])])
            sectors = matrix_info.get("sectors", [f"Sector_{i}" for i in range(matrix_data.shape[1])])
            df = pd.DataFrame(matrix_data, index = resources, columns = sectors)
        else:
            df = pd.DataFrame(matrix_data)

        if include_metadata:
            df.attrs["matrix_type"] = matrix_info["type"]
            df.attrs["shape"] = matrix_info["shape"]

        return df

    def export_vector_to_series(self, vector_name: str, include_metadata: bool = True) -> pd.Series:
        """
        Export vector to pandas Series.

        Args:
            vector_name: Name of the vector to export
            include_metadata: Whether to include metadata in the Series

        Returns:
            Series representation of the vector
        """
        if vector_name not in self.vectors:
            raise ValueError(f"Vector '{vector_name}' not found")

        vector_info = self.vectors[vector_name]
        vector_data = vector_info["data"]

        if vector_info["type"] in ["final_demand", "labor_input"]:
            sectors = vector_info.get("sectors", [f"Sector_{i}" for i in range(len(vector_data))])
            series = pd.Series(vector_data, index = sectors)
        elif vector_info["type"] == "max_resources":
            resources = vector_info.get("resources", [f"Resource_{i}" for i in range(len(vector_data))])
            series = pd.Series(vector_data, index = resources)
        else:
            series = pd.Series(vector_data)

        if include_metadata:
            series.attrs["vector_type"] = vector_info["type"]
            series.attrs["length"] = vector_info["length"]

        return series

    def _generate_technology_tree_matrix(self, n_sectors: int, density: float, sector_ids: List[int]) -> np.ndarray:
        """Generate technology matrix based on technology tree dependencies."""
        matrix = np.zeros((n_sectors, n_sectors))

        # Get dependency matrix from technology tree
        dependency_matrix = self.sector_mapper.get_technology_dependency_matrix(sector_ids)

        # Create technology matrix based on dependencies
        for i in range(n_sectors):
            for j in range(n_sectors):
                if i == j:
                    # Self - dependency (sector needs its own output)
                    matrix[i, j] = np.random.uniform(0.1, 0.3)
                elif dependency_matrix[i, j] > 0:
                    # Direct dependency (sector j is prerequisite for sector i)
                    matrix[i, j] = np.random.uniform(0.05, 0.2)
                elif np.random.random() < density:
                    # Random connections based on density
                    matrix[i, j] = np.random.uniform(0.01, 0.1)

        # Ensure economic viability
        row_sums = np.sum(matrix, axis = 1)
        for i in range(n_sectors):
            if row_sums[i] > 0.8:  # Prevent sectors from requiring more than 80% of total output
                matrix[i, :] *= 0.8 / row_sums[i]

        return matrix

    def _generate_technology_tree_final_demand(self, n_sectors: int, sector_ids: List[int]) -> np.ndarray:
        """Generate final demand vector based on technology tree sector importance."""
        # Get importance weights for the selected sectors
        importance_weights = self.sector_mapper.get_importance_weights_for_sectors(sector_ids)

        # Generate base demand scaled by importance
        base_demand = np.random.uniform(10, 100, n_sectors)
        final_demand = base_demand * importance_weights

        # Scale to maintain realistic economy size
        total_demand = np.sum(final_demand)
        target_total = min(10000, n_sectors * 20)

        if total_demand > target_total:
            final_demand *= target_total / total_demand

        return final_demand

    def _generate_technology_tree_labor_input(self, n_sectors: int, sector_ids: List[int]) -> np.ndarray:
        """Generate labor input vector based on technology tree sector characteristics."""
        labor_input = np.zeros(n_sectors)

        for i, sector_id in enumerate(sector_ids):
            if sector_id in self.sector_mapper.sectors:
                sector = self.sector_mapper.sectors[sector_id]

                # Base labor input based on technology level
                if sector.technology_level.value == "basic":
                    base_labor = np.random.uniform(2, 5)
                elif sector.technology_level.value == "intermediate":
                    base_labor = np.random.uniform(1, 3)
                elif sector.technology_level.value == "advanced":
                    base_labor = np.random.uniform(0.5, 2)
                elif sector.technology_level.value == "cutting_edge":
                    base_labor = np.random.uniform(0.2, 1)
                else:  # future
                    base_labor = np.random.uniform(0.1, 0.5)

                # Adjust based on labor intensity
                if sector.labor_intensity == "high":
                    base_labor *= 2.0
                elif sector.labor_intensity == "low":
                    base_labor *= 0.5

                labor_input[i] = base_labor

        return labor_input
