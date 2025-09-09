"""
Matrix Builder

Constructs and manipulates matrices and vectors for the cybernetic planning system.
Provides utilities for matrix operations, validation, and transformation.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd

class MatrixBuilder:
    """
    Builder for constructing and manipulating economic matrices.

    Provides utilities for creating, validating, and transforming
    technology matrices, demand vectors, and constraint matrices.
    """

    def __init__(self):
        """Initialize the matrix builder."""
        self.matrices = {}
        self.vectors = {}

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
        # Generate technology matrix
        tech_matrix = self._generate_synthetic_technology_matrix(n_sectors, technology_density)
        self.create_technology_matrix(tech_matrix, name = f"{name_prefix}_technology")

        # Generate final demand vector
        final_demand = self._generate_synthetic_final_demand(n_sectors)
        self.create_final_demand_vector(final_demand, name = f"{name_prefix}_final_demand")

        # Generate labor input vector
        labor_input = self._generate_synthetic_labor_input(n_sectors)
        self.create_labor_vector(labor_input, name = f"{name_prefix}_labor")

        # Generate resource matrix
        resource_matrix = self._generate_synthetic_resource_matrix(resource_count, n_sectors)
        self.create_resource_matrix(resource_matrix, name = f"{name_prefix}_resource")

        # Generate max resources vector
        max_resources = self._generate_synthetic_max_resources(resource_count)
        self.create_max_resources_vector(max_resources, name = f"{name_prefix}_max_resources")

        return {
            "technology_matrix": tech_matrix,
            "final_demand": final_demand,
            "labor_input": labor_input,
            "resource_matrix": resource_matrix,
            "max_resources": max_resources,
            "sectors": [f"Sector_{i}" for i in range(n_sectors)],
            "resources": [f"Resource_{i}" for i in range(resource_count)],
        }

    def _generate_synthetic_technology_matrix(self, n_sectors: int, density: float) -> np.ndarray:
        """Generate synthetic technology matrix that is economically viable."""
        # Create sparse matrix with specified density
        matrix = np.zeros((n_sectors, n_sectors))

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
            # Input coefficients should be positive and typically < 1
            matrix[i, j] = np.random.uniform(coeff_range[0], coeff_range[1])

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

        # Add some highly labor - intensive sectors (services, crafts)
        labor_intensive_sectors = np.random.choice(n_sectors, size = max(1, n_sectors // 5), replace = False)
        base_labor[labor_intensive_sectors] *= np.random.uniform(2, 4, len(labor_intensive_sectors))

        return base_labor

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
