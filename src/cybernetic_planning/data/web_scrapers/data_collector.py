"""
Resource Data Collector

Orchestrates the collection of resource constraint data from all sources
and integrates it with the existing BEA Input-Output data.
"""

import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import os
import sys

# Add project root to path for API key manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from .eia_scraper import EIAScraper
from .usgs_scraper import USGSScraper
from .bls_scraper import BLSScraper

try:
    from api_keys_config import APIKeyManager
except ImportError:
    # Fallback if API key manager is not available
    APIKeyManager = None


class ResourceDataCollector:
    """
    Main data collection orchestrator for resource constraint data.

    Coordinates data collection from all sources and integrates with
    existing BEA Input-Output data to create comprehensive resource
    constraint matrices for the cybernetic planning system.
    """

    def __init__(
        self,
        eia_api_key: Optional[str] = None,
        bls_api_key: Optional[str] = None,
        usgs_api_key: Optional[str] = None,
        cache_dir: str = "cache",
        output_dir: str = "data",
    ):
        """
        Initialize the resource data collector.

        Args:
            eia_api_key: EIA API key for enhanced data access (optional, will use environment variable if not provided)
            bls_api_key: BLS API key for labor data access (optional, will use environment variable if not provided)
            usgs_api_key: USGS API key for material data access (optional, will use environment variable if not provided)
            cache_dir: Directory for caching scraped data
            output_dir: Directory for output files
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize API key manager
        self.api_manager = APIKeyManager() if APIKeyManager else None

        # Get API keys from manager or use provided ones
        if self.api_manager:
            eia_key = eia_api_key or self.api_manager.get_api_key("EIA_API_KEY")
            bls_key = bls_api_key or self.api_manager.get_api_key("BLS_API_KEY")
            usgs_key = usgs_api_key or self.api_manager.get_api_key("USGS_API_KEY")
        else:
            eia_key = eia_api_key or os.getenv("EIA_API_KEY")
            bls_key = bls_api_key or os.getenv("BLS_API_KEY")
            usgs_key = usgs_api_key or os.getenv("USGS_API_KEY")

        # Initialize scrapers
        self.scrapers = {
            "eia": EIAScraper(api_key=eia_key, cache_dir=str(self.cache_dir)),
            "bls": BLSScraper(api_key=bls_key, cache_dir=str(self.cache_dir)),
            "usgs": USGSScraper(api_key=usgs_key, cache_dir=str(self.cache_dir)),
        }

        # Initialize data synchronizer
        from ..data_synchronizer import DataSynchronizer

        self.synchronizer = DataSynchronizer(cache_dir=str(self.cache_dir), output_dir=str(self.output_dir))

        # Data storage
        self.collected_data = {}
        self.resource_matrices = {}
        self.metadata = {
            "collection_timestamp": datetime.now().isoformat(),
            "data_sources": [],
            "sector_count": 175,
            "resource_types": [],
        }

    def check_api_key_status(self) -> Dict[str, Any]:
        """
        Check the status of all API keys.

        Returns:
            Dictionary with API key status information
        """
        if not self.api_manager:
            return {"api_manager_available": False, "message": "API key manager not available"}

        return {
            "api_manager_available": True,
            "required_keys": self.api_manager.get_required_api_keys(),
            "optional_keys": self.api_manager.get_optional_api_keys(),
            "validation_results": self.api_manager.validate_api_keys(),
            "capabilities": self.api_manager.get_data_collection_capabilities(),
        }

    def print_api_key_setup_instructions(self) -> None:
        """Print API key setup instructions."""
        if self.api_manager:
            self.api_manager.print_setup_instructions()
        else:
            # API key manager not available - use environment variables
            pass

    def collect_all_resource_data(self, year: int = 2024) -> Dict[str, Any]:
        """
        Collect resource data from all sources for a given year.

        Args:
            year: Year to collect data for

        Returns:
            Combined resource data dictionary
        """
        # Collecting resource data for year

        all_data = {
            "year": year,
            "energy_data": {},
            "material_data": {},
            "labor_data": {},
            "environmental_data": {},
            "resource_matrices": {},
            "metadata": self.metadata.copy(),
        }

        # Collect data from each source
        for source_name, scraper in self.scrapers.items():
            try:
                if source_name == "eia":
                    data = scraper.scrape_all_energy_data(year)
                    all_data["energy_data"] = data
                elif source_name == "bls":
                    data = scraper.scrape_all_labor_data(year)
                    all_data["labor_data"] = data
                elif source_name == "usgs":
                    data = scraper.scrape_all_material_data(year)
                    all_data["material_data"] = data

                self.metadata["data_sources"].append(source_name)

            except Exception as e:
                continue

        # Synchronize data from all sources
        # Synchronizing data from all sources
        synchronized_data = self.synchronizer.synchronize_resource_data(
            energy_data=all_data["energy_data"],
            material_data=all_data["material_data"],
            labor_data=all_data["labor_data"],
            environmental_data=all_data["environmental_data"],
            year=year,
        )

        # Update all_data with synchronized data
        all_data.update(synchronized_data)

        # Build resource constraint matrices
        # Building resource constraint matrices
        self.resource_matrices = self._build_resource_matrices(all_data)
        all_data["resource_matrices"] = self.resource_matrices

        # Save collected data
        self._save_collected_data(all_data)

        # Resource data collection completed
        return all_data

    def _build_resource_matrices(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build resource constraint matrices from collected data.

        Args:
            data: Collected resource data

        Returns:
            Resource matrices dictionary
        """
        matrices = {}

        # Energy resource matrix (n_energy_types × 175)
        energy_matrix = self._build_energy_matrix(data.get("energy_data", {}))
        if energy_matrix is not None:
            matrices["energy_matrix"] = energy_matrix

        # Material resource matrix (n_materials × 175)
        material_matrix = self._build_material_matrix(data.get("material_data", {}))
        if material_matrix is not None:
            matrices["material_matrix"] = material_matrix

        # Labor resource matrix (n_labor_categories × 175)
        labor_matrix = self._build_labor_matrix(data.get("labor_data", {}))
        if labor_matrix is not None:
            matrices["labor_matrix"] = labor_matrix

        # Environmental resource matrix (n_env_factors × 175)
        environmental_matrix = self._build_environmental_matrix(data.get("environmental_data", {}))
        if environmental_matrix is not None:
            matrices["environmental_matrix"] = environmental_matrix

        # Combined resource matrix
        combined_matrix = self._combine_resource_matrices(matrices)
        if combined_matrix is not None:
            matrices["combined_resource_matrix"] = combined_matrix

        return matrices

    def _build_energy_matrix(self, energy_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build energy resource constraint matrix."""
        try:
            # Energy types: coal, natural_gas, petroleum, nuclear, renewable
            energy_types = ["coal", "natural_gas", "petroleum", "nuclear", "renewable"]
            n_energy_types = len(energy_types)
            n_sectors = 175

            # Initialize matrix
            energy_matrix = np.zeros((n_energy_types, n_sectors))

            # Extract energy intensity data
            if "energy_intensity_by_sector" in energy_data:
                intensity_data = energy_data["energy_intensity_by_sector"].get("data", {})

                if "sector_intensities" in intensity_data:
                    for sector_id, intensities in intensity_data["sector_intensities"].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1  # Convert to 0-based index

                            for i, energy_type in enumerate(energy_types):
                                if energy_type in intensities:
                                    energy_matrix[i, sector_idx] = intensities[energy_type]

            # If no intensity data, use consumption data to estimate
            if np.all(energy_matrix == 0):
                return None

            return energy_matrix

        except Exception as e:
            return None

    def _build_material_matrix(self, material_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build material resource constraint matrix."""
        try:
            # Critical materials
            materials = ["lithium", "cobalt", "rare_earth", "copper", "aluminum", "steel"]
            n_materials = len(materials)
            n_sectors = 175

            # Initialize matrix
            material_matrix = np.zeros((n_materials, n_sectors))

            # Extract material intensity data
            if "material_intensity" in material_data:
                intensity_data = material_data["material_intensity"].get("data", {})

                if "sectors" in intensity_data:
                    for sector_id, intensities in intensity_data["sectors"].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1

                            for i, material in enumerate(materials):
                                if material in intensities:
                                    material_matrix[i, sector_idx] = intensities[material]

            # If no intensity data, return None
            if np.all(material_matrix == 0):
                return None

            return material_matrix

        except Exception as e:
            return None

    def _build_labor_matrix(self, labor_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build labor resource constraint matrix."""
        try:
            # Labor categories
            labor_categories = ["high_skilled", "medium_skilled", "low_skilled", "technical", "management"]
            n_categories = len(labor_categories)
            n_sectors = 175

            # Initialize matrix
            labor_matrix = np.zeros((n_categories, n_sectors))

            # Extract labor intensity data
            if "labor_intensity" in labor_data:
                intensity_data = labor_data["labor_intensity"].get("data", {})

                if "sectors" in intensity_data:
                    for sector_id, sector_data in intensity_data["sectors"].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1

                            # Use labor intensity and skill distribution
                            base_intensity = sector_data.get("labor_intensity", 0)
                            skill_requirements = sector_data.get("skill_requirements", {})

                            for i, category in enumerate(labor_categories):
                                if category in skill_requirements:
                                    labor_matrix[i, sector_idx] = base_intensity * skill_requirements[category]

            # If no intensity data, return None
            if np.all(labor_matrix == 0):
                return None

            return labor_matrix

        except Exception as e:
            return None

    def _build_environmental_matrix(self, environmental_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build environmental resource constraint matrix."""
        try:
            # Environmental factors
            env_factors = ["carbon_emissions", "water_usage", "waste_generation"]
            n_factors = len(env_factors)
            n_sectors = 175

            # Initialize matrix
            env_matrix = np.zeros((n_factors, n_sectors))

            # Extract environmental intensity data
            if "environmental_intensity" in environmental_data:
                intensity_data = environmental_data["environmental_intensity"].get("data", {})

                if "sectors" in intensity_data:
                    for sector_id, intensities in intensity_data["sectors"].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1

                            for i, factor in enumerate(env_factors):
                                if factor in intensities:
                                    env_matrix[i, sector_idx] = intensities[factor]

            # If no intensity data, return None
            if np.all(env_matrix == 0):
                return None

            return env_matrix

        except Exception as e:
            return None

    def _combine_resource_matrices(self, matrices: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Combine all resource matrices into a single matrix."""
        try:
            matrix_list = []
            resource_names = []

            for matrix_name, matrix in matrices.items():
                if matrix is not None and matrix.size > 0:
                    matrix_list.append(matrix)
                    resource_names.append(matrix_name)

            if not matrix_list:
                return None

            # Stack matrices vertically
            combined_matrix = np.vstack(matrix_list)

            # Store metadata
            self.metadata["resource_types"] = resource_names
            self.metadata["total_resources"] = combined_matrix.shape[0]

            return combined_matrix

        except Exception as e:
            return None

    def _save_collected_data(self, data: Dict[str, Any]) -> None:
        """Save collected data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save complete dataset
        complete_file = self.output_dir / f"resource_data_{timestamp}.json"
        with open(complete_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Save resource matrices separately
        if "resource_matrices" in data:
            matrices_file = self.output_dir / f"resource_matrices_{timestamp}.json"
            matrices_data = {}

            for name, matrix in data["resource_matrices"].items():
                if isinstance(matrix, np.ndarray):
                    matrices_data[name] = matrix.tolist()
                else:
                    matrices_data[name] = matrix

            with open(matrices_file, "w") as f:
                json.dump(matrices_data, f, indent=2)

        # Data saved successfully

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of data collection session."""
        return {
            "collection_timestamp": self.metadata["collection_timestamp"],
            "data_sources": self.metadata["data_sources"],
            "sector_count": self.metadata["sector_count"],
            "resource_types": self.metadata.get("resource_types", []),
            "total_resources": self.metadata.get("total_resources", 0),
            "scrapers_status": {name: scraper.get_collection_summary() for name, scraper in self.scrapers.items()},
        }
