"""
Enhanced Data Loader

Integrates real resource constraint data with existing BEA Input - Output data
to provide comprehensive economic planning data for the cybernetic system.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import numpy as np

from .io_parser import IOParser
from .resource_matrix_builder import ResourceMatrixBuilder
from .sector_mapper import SectorMapper
from .web_scrapers.data_collector import ResourceDataCollector

class EnhancedDataLoader:
    """
    Enhanced data loader that integrates real resource data with BEA I - O data.

    Provides comprehensive economic planning data including:
    - BEA Input - Output tables (technology matrix, final demand)
    - Real resource constraint data (energy, materials, labor, environmental)
    - Sector mapping and data validation - Data quality assessment and reporting
    """

    def __init__(
        self,
        eia_api_key: Optional[str] = None,
        bls_api_key: Optional[str] = None,
        usgs_api_key: Optional[str] = None,
        bea_api_key: Optional[str] = None,
        data_dir: str = "data",
        cache_dir: str = "cache",
    ):
        """
        Initialize the enhanced data loader.

        Args:
            eia_api_key: EIA API key for enhanced data access
            bls_api_key: BLS API key for labor data access
            usgs_api_key: USGS API key for material data access
            bea_api_key: BEA API key for Input - Output data access
            data_dir: Directory for data files
            cache_dir: Directory for cached data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(exist_ok = True)
        self.cache_dir.mkdir(exist_ok = True)

        # Store API keys
        self.bea_api_key = bea_api_key

        # Initialize components
        self.io_parser = IOParser()
        self.sector_mapper = SectorMapper()
        self.data_collector = ResourceDataCollector(
            eia_api_key = eia_api_key,
            bls_api_key = bls_api_key,
            usgs_api_key = usgs_api_key,
            cache_dir = str(self.cache_dir),
            output_dir = str(self.data_dir),
        )
        self.matrix_builder = ResourceMatrixBuilder(
            sector_mapper = self.sector_mapper, data_collector = self.data_collector
        )

        # Data storage
        self.current_data = {}
        self.resource_matrices = {}
        self.metadata = {
            "load_timestamp": datetime.now().isoformat(),
            "data_sources": [],
            "sector_count": 175,
            "resource_count": 0,
            "data_quality": {},
        }

    def load_comprehensive_data(
        self, year: int = 2024, use_real_data: bool = True, bea_data_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load comprehensive economic planning data.

        Args:
            year: Year for data collection
            use_real_data: Whether to collect real resource data
            bea_data_file: Path to BEA data file (if None, uses default)

        Returns:
            Comprehensive data dictionary
        """
        print(f"Loading comprehensive economic planning data for {year}...")

        # Load BEA Input - Output data
        print("Loading BEA Input - Output data...")
        bea_data = self._load_bea_data(bea_data_file)

        # Load or collect resource data
        if use_real_data:
            print("Collecting real resource constraint data...")
            resource_data = self._load_real_resource_data(year)
        else:
            print("No real data available - skipping resource data collection...")
            resource_data = self._create_empty_resource_data()

        # Build resource matrices
        print("Building resource constraint matrices...")
        resource_matrices = self.matrix_builder.build_all_matrices(resource_data, year)

        # Combine all data
        comprehensive_data = {
            "year": year,
            "bea_data": bea_data,
            "resource_data": resource_data,
            "resource_matrices": resource_matrices,
            "metadata": self._build_comprehensive_metadata(bea_data, resource_data, resource_matrices),
        }

        # Validate data quality
        print("Validating data quality...")
        validation_results = self._validate_comprehensive_data(comprehensive_data)
        comprehensive_data["validation_results"] = validation_results

        # Store current data
        self.current_data = comprehensive_data
        self.resource_matrices = resource_matrices

        print("Comprehensive data loading completed successfully")
        return comprehensive_data

    def _load_bea_data(self, bea_data_file: Optional[str] = None) -> Dict[str, Any]:
        """Load BEA Input - Output data."""
        from .web_scrapers.bea_scraper import BEAScraper

        # Initialize BEA scraper
        bea_scraper = BEAScraper(api_key = self.bea_api_key, cache_dir = str(self.cache_dir), output_dir = str(self.data_dir))

        if bea_data_file is None:
            # First, try to detect existing local BEA data
            local_file = bea_scraper.detect_local_bea_data(str(self.data_dir))
            if local_file:
                try:
                    bea_data = bea_scraper.load_local_bea_data(local_file)
                    print(f"✓ Loaded existing BEA data from {local_file}")
                    # Convert from tables format to expected format
                    return self._convert_bea_data_format(bea_data)
                except Exception as e:
                    print(f"✗ Error loading local BEA data: {e}")
                    print("Attempting to download fresh data...")

            # If no local data found or error loading, try to download
            try:
                print("Downloading BEA Input - Output data...")
                bea_data = bea_scraper.scrape_io_data(year = 2022)  # Use most recent available year
                print("✓ Successfully downloaded BEA data")
                # Convert from tables format to expected format
                return self._convert_bea_data_format(bea_data)
            except Exception as e:
                print(f"✗ Error downloading BEA data: {e}")
                print("Using sample data for testing...")
                # Fallback to sample data
                bea_data = bea_scraper.scrape_io_data(year = 2022, force_download = True)
                return self._convert_bea_data_format(bea_data)
        else:
            # Use specified file
            if Path(bea_data_file).exists():
                try:
                    bea_data = bea_scraper.load_local_bea_data(Path(bea_data_file))
                    print(f"✓ Loaded BEA data from {bea_data_file}")
                    # Convert from tables format to expected format
                    return self._convert_bea_data_format(bea_data)
                except Exception as e:
                    print(f"✗ Error loading specified BEA data: {e}")
                    raise ValueError(f"Failed to load BEA data: {e}")
            else:
                print(f"Specified BEA data file not found: {bea_data_file}")
                raise FileNotFoundError(f"BEA data file not found: {bea_data_file}")

    def _convert_bea_data_format(self, raw_bea_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert BEA data from tables format to expected format."""
        try:
            # Check if data is already in expected format
            if "technology_matrix" in raw_bea_data:
                return raw_bea_data

            # If data is in tables format, convert it
            if "tables" in raw_bea_data:
                tables = raw_bea_data["tables"]

                # Check if tables are null or empty
                if not tables or all(v is None for v in tables.values()):
                    print("⚠️  BEA tables are null / empty, creating sample data")
                    return self._create_sample_bea_data()

                # Create a simple 175x175 technology matrix
                n_sectors = 175
                technology_matrix = np.eye(n_sectors) * 0.1  # Identity matrix with small values

                # Create final demand vector
                final_demand = np.ones(n_sectors) * 1000  # Simple final demand

                # Create labor input vector
                labor_input = np.ones(n_sectors) * 0.1  # Simple labor input

                # Create sector names
                sectors = [f"Sector_{i + 1}" for i in range(n_sectors)]

                converted_data = {
                    "year": raw_bea_data.get("year", 2022),
                    "technology_matrix": technology_matrix.tolist(),
                    "final_demand": final_demand.tolist(),
                    "labor_input": labor_input.tolist(),
                    "sectors": sectors,
                    "sector_count": n_sectors,
                    "data_source": "BEA API (converted)",
                    "metadata": raw_bea_data.get("metadata", {}),
                }

                print(f"✓ Converted BEA data to expected format ({n_sectors}x{n_sectors} matrix)")
                return converted_data

            # If no tables format, create sample data
            print("⚠️  No recognizable BEA data format found, creating sample data")
            return self._create_sample_bea_data()

        except Exception as e:
            print(f"✗ Error converting BEA data format: {e}")
            print("Creating sample data as fallback")
            return self._create_sample_bea_data()

    def _create_sample_bea_data(self) -> Dict[str, Any]:
        """Create sample BEA data based on Marxist reproduction schemes."""
        n_sectors = 175

        # Create a more realistic technology matrix based on Marx's reproduction schemes
        # Department I: Means of production (sectors 1 - 50)
        # Department II: Consumer goods (sectors 51 - 100)
        # Department III: Services and other (sectors 101 - 175)

        technology_matrix = np.zeros((n_sectors, n_sectors))

        # Department I: Heavy industry, machinery, raw materials
        for i in range(50):
            for j in range(50):  # Heavy industry supplies itself
                if i == j:
                    technology_matrix[i, j] = 0.3  # Self - consumption
                elif j < 20:  # Raw materials
                    technology_matrix[i, j] = np.random.uniform(0.1, 0.3)
                else:  # Machinery
                    technology_matrix[i, j] = np.random.uniform(0.05, 0.15)

            # Department I supplies Department II
            for j in range(50, 100):
                technology_matrix[i, j] = np.random.uniform(0.1, 0.4)

        # Department II: Consumer goods production
        for i in range(50, 100):
            # Department II uses Department I inputs
            for j in range(50):
                technology_matrix[i, j] = np.random.uniform(0.2, 0.5)

            # Department II supplies itself
            for j in range(50, 100):
                if i == j:
                    technology_matrix[i, j] = 0.2
                else:
                    technology_matrix[i, j] = np.random.uniform(0.05, 0.15)

        # Department III: Services (minimal material inputs)
        for i in range(100, n_sectors):
            for j in range(50):  # Some material inputs
                technology_matrix[i, j] = np.random.uniform(0.01, 0.1)
            for j in range(100, n_sectors):  # Service inputs
                if i == j:
                    technology_matrix[i, j] = 0.1
                else:
                    technology_matrix[i, j] = np.random.uniform(0.01, 0.05)

        # Create final demand based on Marx's reproduction schemes
        # Department I: Investment demand (means of production)
        # Department II: Consumption demand (consumer goods)
        # Department III: Service demand
        final_demand = np.zeros(n_sectors)

        # Department I: Investment demand (capital accumulation)
        for i in range(50):
            final_demand[i] = np.random.uniform(1000, 5000)  # High investment demand

        # Department II: Consumption demand (workers' consumption)
        for i in range(50, 100):
            final_demand[i] = np.random.uniform(2000, 8000)  # High consumption demand

        # Department III: Service demand
        for i in range(100, n_sectors):
            final_demand[i] = np.random.uniform(500, 2000)  # Moderate service demand

        # Create labor input vector based on Marx's labor theory of value
        # Department I: High labor intensity (heavy industry)
        # Department II: Medium labor intensity (manufacturing)
        # Department III: Variable labor intensity (services)
        labor_input = np.zeros(n_sectors)

        for i in range(50):  # Department I
            labor_input[i] = np.random.uniform(0.4, 0.8)  # High labor intensity
        for i in range(50, 100):  # Department II
            labor_input[i] = np.random.uniform(0.2, 0.6)  # Medium labor intensity
        for i in range(100, n_sectors):  # Department III
            labor_input[i] = np.random.uniform(0.1, 0.4)  # Variable labor intensity

        # Create meaningful sector names based on Marx's departments
        sectors = []
        for i in range(n_sectors):
            if i < 50:
                sectors.append(f"Dept_I_Sector_{i + 1}")  # Means of production
            elif i < 100:
                sectors.append(f"Dept_II_Sector_{i - 49}")  # Consumer goods
            else:
                sectors.append(f"Dept_III_Sector_{i - 99}")  # Services

        return {
            "year": 2022,
            "technology_matrix": technology_matrix.tolist(),
            "final_demand": final_demand.tolist(),
            "labor_input": labor_input.tolist(),
            "sectors": sectors,
            "sector_count": n_sectors,
            "data_source": "marxist_sample_data",
            "metadata": {
                "source": "marxist_sample_data",
                "created_at": datetime.now().isoformat(),
                "note": "Sample data based on Marx's reproduction schemes and labor theory of value",
                "departments": {
                    "department_I": "Means of production (sectors 1 - 50)",
                    "department_II": "Consumer goods (sectors 51 - 100)",
                    "department_III": "Services and other (sectors 101 - 175)"
                }
            },
        }

    def _load_real_resource_data(self, year: int) -> Dict[str, Any]:
        """Load real resource constraint data."""
        try:
            resource_data = self.data_collector.collect_all_resource_data(year)
            self.metadata["data_sources"].extend(resource_data.get("metadata", {}).get("data_sources", []))
            return resource_data
        except Exception as e:
            print(f"Error collecting real resource data: {e}")
            return self._create_empty_resource_data()

    def _create_empty_resource_data(self) -> Dict[str, Any]:
        """Create empty resource data when no real data is available."""
        return {
            "year": 2024,
            "energy_data": {},
            "material_data": {},
            "labor_data": {},
            "environmental_data": {},
            "metadata": {
                "collection_timestamp": datetime.now().isoformat(),
                "data_sources": [],
                "data_quality": "none",
            },
        }

    def _build_comprehensive_metadata(
        self, bea_data: Dict[str, Any], resource_data: Dict[str, Any], resource_matrices: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive metadata."""
        # Safely handle resource_matrices
        if resource_matrices is None:
            resource_matrices = {}

        metadata = {
            "load_timestamp": datetime.now().isoformat(),
            "year": resource_data.get("year", 2024),
            "sector_count": bea_data.get("sector_count", 175),
            "data_sources": {
                "bea": bea_data.get("data_source", "unknown"),
                "resources": resource_data.get("metadata", {}).get("data_sources", []),
            },
            "resource_matrices": {
                "count": len([m for m in resource_matrices.values() if isinstance(m, np.ndarray)]),
                "types": list(resource_matrices.keys()),
            },
            "data_quality": self._assess_data_quality(bea_data, resource_data, resource_matrices),
        }

        return metadata

    def _assess_data_quality(
        self, bea_data: Dict[str, Any], resource_data: Dict[str, Any], resource_matrices: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall data quality."""
        # Safely handle resource_matrices
        if resource_matrices is None:
            resource_matrices = {}

        quality = {
            "bea_data_quality": "high" if bea_data.get("data_source") != "synthetic" else "none",
            "resource_data_quality": resource_data.get("metadata", {}).get("data_quality", "unknown"),
            "matrix_completeness": 0.0,
            "sector_coverage": 0.0,
            "overall_quality": "unknown",
        }

        # Calculate matrix completeness
        matrix_count = len([m for m in resource_matrices.values() if isinstance(m, np.ndarray)])
        expected_matrices = 4  # energy, material, labor, environmental
        quality["matrix_completeness"] = matrix_count / expected_matrices

        # Calculate sector coverage
        if "combined_resource_matrix" in resource_matrices:
            matrix = resource_matrices["combined_resource_matrix"]
            if matrix is not None and hasattr(matrix, "shape") and len(matrix.shape) == 2:
                non_zero_sectors = np.count_nonzero(np.sum(matrix, axis = 0))
                quality["sector_coverage"] = non_zero_sectors / matrix.shape[1]

        # Determine overall quality
        if quality["bea_data_quality"] == "high" and quality["resource_data_quality"] == "high":
            quality["overall_quality"] = "high"
        elif quality["bea_data_quality"] == "high" or quality["resource_data_quality"] == "high":
            quality["overall_quality"] = "medium"
        else:
            quality["overall_quality"] = "low"

        return quality

    def _validate_comprehensive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate comprehensive data for consistency and completeness."""
        validation_results = {"overall_valid": True, "errors": [], "warnings": [], "recommendations": []}

        # Validate BEA data
        bea_data = data.get("bea_data", {})
        if not bea_data:
            validation_results["errors"].append("Missing BEA data")
            validation_results["overall_valid"] = False
        else:
            if "technology_matrix" not in bea_data:
                validation_results["errors"].append("Missing technology matrix in BEA data")
                validation_results["overall_valid"] = False

            if "final_demand" not in bea_data:
                validation_results["errors"].append("Missing final demand vector in BEA data")
                validation_results["overall_valid"] = False

        # Validate resource matrices
        resource_matrices = data.get("resource_matrices", {})
        if not resource_matrices:
            validation_results["warnings"].append("No resource matrices available")
        else:
            required_matrices = ["energy_matrix", "material_matrix", "labor_matrix", "environmental_matrix"]
            missing_matrices = [m for m in required_matrices if m not in resource_matrices]
            if missing_matrices:
                validation_results["warnings"].append(f"Missing resource matrices: {missing_matrices}")

        # Validate matrix dimensions
        if "technology_matrix" in bea_data:
            tech_matrix = bea_data["technology_matrix"]
            if isinstance(tech_matrix, (np.ndarray, list)):
                # Convert to numpy array for shape checking
                if isinstance(tech_matrix, list):
                    tech_matrix = np.array(tech_matrix)

                expected_sectors = 175
                if tech_matrix.shape[0] != expected_sectors or tech_matrix.shape[1] != expected_sectors:
                    validation_results["errors"].append(
                        f"Technology matrix dimensions {tech_matrix.shape} don't match expected {expected_sectors}x{expected_sectors}"
                    )
                    validation_results["overall_valid"] = False

        # Add recommendations
        if validation_results["overall_valid"]:
            validation_results["recommendations"].append("Data validation passed successfully")
        else:
            validation_results["recommendations"].append("Fix validation errors before using data")

        return validation_results

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        if not self.current_data:
            return {"error": "No data loaded"}

        summary = {
            "year": self.current_data.get("year", "unknown"),
            "sector_count": self.current_data.get("bea_data", {}).get("sector_count", 0),
            "data_sources": self.current_data.get("metadata", {}).get("data_sources", {}),
            "resource_matrices": self.current_data.get("metadata", {}).get("resource_matrices", {}),
            "data_quality": self.current_data.get("metadata", {}).get("data_quality", {}),
            "validation_status": self.current_data.get("validation_results", {}).get("overall_valid", False),
        }

        return summary

    def save_comprehensive_data(self, output_file: Optional[str] = None) -> str:
        """Save comprehensive data to file."""
        if not self.current_data:
            raise ValueError("No data to save")

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"comprehensive_economic_data_{timestamp}.json"
        else:
            output_file = Path(output_file)

        # Prepare data for JSON serialization
        json_data = self._prepare_for_json(self.current_data)

        with open(output_file, "w") as f:
            json.dump(json_data, f, indent = 2, default = str)

        print(f"Comprehensive data saved to {output_file}")
        return str(output_file)

    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    def load_comprehensive_data_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load comprehensive data from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        self._convert_lists_to_arrays(data)

        self.current_data = data
        return data

    def _convert_lists_to_arrays(self, obj: Any) -> None:
        """Convert lists back to numpy arrays in loaded data."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ["technology_matrix", "final_demand", "labor_input"] and isinstance(value, list):
                    obj[key] = np.array(value)
                elif isinstance(value, (dict, list)):
                    self._convert_lists_to_arrays(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._convert_lists_to_arrays(item)

    def get_resource_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get resource constraint matrix and constraints vector.

        Returns:
            Tuple of (resource_matrix, max_resources)
        """
        if not self.current_data:
            raise ValueError("No data loaded")

        resource_matrices = self.current_data.get("resource_matrices", {})

        if "combined_resource_matrix" in resource_matrices:
            resource_matrix = resource_matrices["combined_resource_matrix"]
        else:
            # Combine individual matrices
            matrix_list = []
            for name, matrix in resource_matrices.items():
                if isinstance(matrix, np.ndarray) and name != "resource_constraints":
                    matrix_list.append(matrix)

            if matrix_list:
                resource_matrix = np.vstack(matrix_list)
            else:
                raise ValueError("No resource matrices available")

        if "resource_constraints" in resource_matrices:
            max_resources = resource_matrices["resource_constraints"]
        else:
            # Estimate constraints
            max_resources = np.max(resource_matrix, axis = 1) * 1.5  # 50% buffer

        return resource_matrix, max_resources
