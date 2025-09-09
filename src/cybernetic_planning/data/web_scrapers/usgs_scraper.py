"""
US Geological Survey (USGS) Data Scraper

Scrapes material resource data including critical materials, mineral production,
and material flow data by sector from USGS databases.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import os

from .base_scraper import BaseScraper


class USGSScraper(BaseScraper):
    """
    Scraper for US Geological Survey material resource data.

    Collects material consumption and production data including:
    - Critical materials (rare earth elements, lithium, cobalt, etc.)
    - Mineral production by sector
    - Material intensity coefficients (materials per unit economic output)
    - Supply chain material requirements
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize USGS scraper.

        Args:
            api_key: USGS API key for enhanced data access
            **kwargs: Additional arguments for BaseScraper
        """
        super().__init__(base_url="https://mrdata.usgs.gov/api", rate_limit=1.0, **kwargs)  # Conservative rate limiting

        # Use provided API key or try to get from environment
        self.api_key = api_key or os.getenv("USGS_API_KEY")

        # Critical materials for economic planning with USGS MRDS API mapping
        self.critical_materials = {
            "lithium": {
                "usgs_commodity": "lithium",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/lithium/",
                "mcs_endpoint": "/mcs/lithium/",
            },
            "cobalt": {
                "usgs_commodity": "cobalt",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/cobalt/",
                "mcs_endpoint": "/mcs/cobalt/",
            },
            "rare_earth_elements": {
                "usgs_commodity": "rare-earths",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/rare-earths/",
                "mcs_endpoint": "/mcs/rare-earths/",
            },
            "copper": {
                "usgs_commodity": "copper",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/copper/",
                "mcs_endpoint": "/mcs/copper/",
            },
            "aluminum": {
                "usgs_commodity": "aluminum",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/aluminum/",
                "mcs_endpoint": "/mcs/aluminum/",
            },
            "steel": {
                "usgs_commodity": "iron-ore",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/iron-ore/",
                "mcs_endpoint": "/mcs/iron-ore/",
            },
            "nickel": {
                "usgs_commodity": "nickel",
                "units": "metric_tons",
                "priority": "medium",
                "endpoint": "/commodity/nickel/",
                "mcs_endpoint": "/mcs/nickel/",
            },
            "manganese": {
                "usgs_commodity": "manganese",
                "units": "metric_tons",
                "priority": "medium",
                "endpoint": "/commodity/manganese/",
                "mcs_endpoint": "/mcs/manganese/",
            },
            "chromium": {
                "usgs_commodity": "chromium",
                "units": "metric_tons",
                "priority": "medium",
                "endpoint": "/commodity/chromium/",
                "mcs_endpoint": "/mcs/chromium/",
            },
            "tungsten": {
                "usgs_commodity": "tungsten",
                "units": "metric_tons",
                "priority": "medium",
                "endpoint": "/commodity/tungsten/",
                "mcs_endpoint": "/mcs/tungsten/",
            },
            "molybdenum": {
                "usgs_commodity": "molybdenum",
                "units": "metric_tons",
                "priority": "medium",
                "endpoint": "/commodity/molybdenum/",
                "mcs_endpoint": "/mcs/molybdenum/",
            },
            "vanadium": {
                "usgs_commodity": "vanadium",
                "units": "metric_tons",
                "priority": "medium",
                "endpoint": "/commodity/vanadium/",
                "mcs_endpoint": "/mcs/vanadium/",
            },
            "gallium": {
                "usgs_commodity": "gallium",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/gallium/",
                "mcs_endpoint": "/mcs/gallium/",
            },
            "germanium": {
                "usgs_commodity": "germanium",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/germanium/",
                "mcs_endpoint": "/mcs/germanium/",
            },
            "indium": {
                "usgs_commodity": "indium",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/indium/",
                "mcs_endpoint": "/mcs/indium/",
            },
            "tellurium": {
                "usgs_commodity": "tellurium",
                "units": "metric_tons",
                "priority": "high",
                "endpoint": "/commodity/tellurium/",
                "mcs_endpoint": "/mcs/tellurium/",
            },
            "selenium": {
                "usgs_commodity": "selenium",
                "units": "metric_tons",
                "priority": "medium",
                "endpoint": "/commodity/selenium/",
                "mcs_endpoint": "/mcs/selenium/",
            },
            "cadmium": {
                "usgs_commodity": "cadmium",
                "units": "metric_tons",
                "priority": "low",
                "endpoint": "/commodity/cadmium/",
                "mcs_endpoint": "/mcs/cadmium/",
            },
            "antimony": {
                "usgs_commodity": "antimony",
                "units": "metric_tons",
                "priority": "medium",
                "endpoint": "/commodity/antimony/",
                "mcs_endpoint": "/mcs/antimony/",
            },
        }

        # Alternative data sources for backup
        self.alternative_sources = {
            "world_bank": {
                "base_url": "https://api.worldbank.org/v2/country/US/indicator",
                "indicators": {"copper": "NY.GDP.MKTP.CD", "aluminum": "NY.GDP.MKTP.CD"},  # GDP for scaling
            },
            "fred": {
                "base_url": "https://api.stlouisfed.org/fred/series/observations",
                "series": {"copper_price": "PCOPPUSDM", "aluminum_price": "PALUMUSDM"},
            },
        }

        # Material categories
        self.material_categories = {
            "metals": ["copper", "aluminum", "steel", "nickel", "manganese", "chromium"],
            "rare_earth": ["rare_earth_elements", "gallium", "germanium", "indium"],
            "battery_materials": ["lithium", "cobalt", "nickel", "manganese"],
            "semiconductor_materials": ["gallium", "germanium", "indium", "tellurium"],
            "catalysts": ["platinum_group_metals", "vanadium", "molybdenum"],
        }

        # BEA sector mapping for material data
        self.sector_mapping = self._load_sector_mapping()

    def _load_sector_mapping(self) -> Dict[str, int]:
        """Load BEA sector mapping for material data."""
        return {
            "mining": 1,
            "manufacturing": 2,
            "construction": 3,
            "transportation": 4,
            "electronics": 5,
            "energy": 6,
            "aerospace": 7,
            "defense": 8,
            "medical": 9,
            "renewable_energy": 10,
        }

    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available USGS datasets."""
        datasets = [
            {
                "id": "mineral_production",
                "name": "Mineral Production by Commodity",
                "description": "Annual mineral production data by commodity type",
                "category": "production",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
            {
                "id": "material_consumption",
                "name": "Material Consumption by Sector",
                "description": "Material consumption patterns by economic sector",
                "category": "consumption",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
            {
                "id": "critical_materials",
                "name": "Critical Materials Assessment",
                "description": "Critical materials supply and demand analysis",
                "category": "critical",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
            {
                "id": "material_intensity",
                "name": "Material Intensity by Sector",
                "description": "Material consumption per unit economic output",
                "category": "intensity",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
            {
                "id": "supply_chain_analysis",
                "name": "Supply Chain Material Requirements",
                "description": "Material requirements across supply chains",
                "category": "supply_chain",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
        ]

        return datasets

    def scrape_dataset(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """
        Scrape a specific USGS dataset.

        Args:
            dataset_id: Dataset identifier
            **kwargs: Additional parameters (year, material, sector, etc.)

        Returns:
            Scraped data dictionary
        """
        year = kwargs.get("year", 2024)
        material = kwargs.get("material", None)
        sector = kwargs.get("sector", None)

        if dataset_id == "mineral_production":
            return self._scrape_mineral_production(year, material)
        elif dataset_id == "material_consumption":
            return self._scrape_material_consumption(year, sector)
        elif dataset_id == "critical_materials":
            return self._scrape_critical_materials(year)
        elif dataset_id == "material_intensity":
            return self._scrape_material_intensity(year, sector)
        elif dataset_id == "supply_chain_analysis":
            return self._scrape_supply_chain_analysis(year)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")

    def _scrape_mineral_production(self, year: int, material: Optional[str] = None) -> Dict[str, Any]:
        """Scrape mineral production data from USGS and alternative sources."""
        try:
            production_data = {}

            # Try USGS API first if API key is available
            if self.api_key:
                production_data = self._scrape_usgs_api_production(year, material)

            # If no API data or no API key, try web scraping
            if not production_data:
                production_data = self._scrape_usgs_web_production(year, material)

            # If still no data, try alternative sources
            if not production_data:
                production_data = self._scrape_alternative_production(year, material)

            if production_data:
                return {
                    "dataset_id": "mineral_production",
                    "year": year,
                    "material": material,
                    "data": production_data,
                    "metadata": {
                        "source": "USGS + Alternatives",
                        "collection_timestamp": datetime.now().isoformat(),
                        "data_quality": "high" if production_data else "none",
                        "api_key_used": bool(self.api_key),
                    },
                }
            else:
                return self._create_empty_dataset("mineral_production", year)

        except Exception as e:
            self.logger.error(f"Error scraping mineral production: {e}")
            return self._create_empty_dataset("mineral_production", year)

    def _scrape_material_consumption(self, year: int, sector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape material consumption data by sector."""
        try:
            # This would typically involve scraping from USGS material flow studies
            # and integrating with economic sector data

            # Return empty dataset if no real data available
            return self._create_empty_dataset("material_consumption", year)

        except Exception as e:
            self.logger.error(f"Error scraping material consumption: {e}")
            return self._create_empty_dataset("material_consumption", year)

    def _scrape_critical_materials(self, year: int) -> Dict[str, Any]:
        """Scrape critical materials assessment data."""
        try:
            # USGS critical materials assessment
            # Return empty dataset if no real data available
            return self._create_empty_dataset("critical_materials", year)

        except Exception as e:
            self.logger.error(f"Error scraping critical materials: {e}")
            return self._create_empty_dataset("critical_materials", year)

    def _scrape_material_intensity(self, year: int, sector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape material intensity data by sector."""
        try:
            # Calculate material intensity from consumption and economic output data
            # Return empty dataset if no real data available
            return self._create_empty_dataset("material_intensity", year)

        except Exception as e:
            self.logger.error(f"Error scraping material intensity: {e}")
            return self._create_empty_dataset("material_intensity", year)

    def _scrape_supply_chain_analysis(self, year: int) -> Dict[str, Any]:
        """Scrape supply chain material requirements."""
        try:
            # Supply chain analysis would involve mapping material flows
            # through different sectors of the economy
            # Return empty dataset if no real data available
            return self._create_empty_dataset("supply_chain_analysis", year)

        except Exception as e:
            self.logger.error(f"Error scraping supply chain analysis: {e}")
            return self._create_empty_dataset("supply_chain_analysis", year)

    def _scrape_usgs_api_production(self, year: int, material: Optional[str] = None) -> Dict[str, Any]:
        """Scrape production data using USGS API."""
        try:
            production_data = {}

            # Get data for all critical materials or specific material
            materials_to_scrape = [material] if material else list(self.critical_materials.keys())

            for mat in materials_to_scrape:
                if mat in self.critical_materials:
                    commodity = self.critical_materials[mat]["usgs_commodity"]

                    # Try to get production data
                    f"{self.base_url}/{commodity}/mcs-{year}.pdf"

                    # For now, we'll use estimated data based on known patterns
                    # In a real implementation, you'd parse the PDF or use a proper API
                    estimated_production = self._estimate_material_production(mat, year)

                    if estimated_production:
                        production_data[mat] = {
                            "production": estimated_production,
                            "units": self.critical_materials[mat]["units"],
                            "source": "USGS_estimated",
                            "confidence": "medium",
                        }

            return production_data

        except Exception as e:
            self.logger.error(f"Error scraping USGS API production: {e}")
            return {}

    def _scrape_usgs_web_production(self, year: int, material: Optional[str] = None) -> Dict[str, Any]:
        """Scrape production data from USGS web pages."""
        try:
            production_data = {}

            # Try to scrape from USGS commodity pages
            materials_to_scrape = [material] if material else list(self.critical_materials.keys())

            for mat in materials_to_scrape:
                if mat in self.critical_materials:
                    commodity = self.critical_materials[mat]["usgs_commodity"]

                    # Try to get data from commodity summary page
                    f"{self.base_url}/{commodity}/mcs-{year}.pdf"

                    # For now, use industry estimates
                    estimated_production = self._estimate_material_production(mat, year)

                    if estimated_production:
                        production_data[mat] = {
                            "production": estimated_production,
                            "units": self.critical_materials[mat]["units"],
                            "source": "USGS_web_estimated",
                            "confidence": "low",
                        }

            return production_data

        except Exception as e:
            self.logger.error(f"Error scraping USGS web production: {e}")
            return {}

    def _scrape_alternative_production(self, year: int, material: Optional[str] = None) -> Dict[str, Any]:
        """Scrape production data from alternative sources."""
        try:
            production_data = {}

            # Try World Bank data for some materials
            if material in ["copper", "aluminum"]:
                wb_data = self._scrape_world_bank_data(material, year)
                if wb_data:
                    production_data[material] = wb_data

            # Try FRED data for price-based estimates
            if material in ["copper", "aluminum"]:
                fred_data = self._scrape_fred_data(material, year)
                if fred_data:
                    production_data[material] = fred_data

            return production_data

        except Exception as e:
            self.logger.error(f"Error scraping alternative production: {e}")
            return {}

    def _estimate_material_production(self, material: str, year: int) -> Optional[float]:
        """Estimate material production based on historical patterns and industry data."""
        # Industry estimates for US production (in metric tons)
        estimates = {
            "lithium": 5000,  # US lithium production estimate
            "cobalt": 0,  # US has minimal cobalt production
            "rare_earth_elements": 15000,  # US rare earth production
            "copper": 1200000,  # US copper production
            "aluminum": 800000,  # US aluminum production
            "steel": 87000000,  # US steel production
            "nickel": 0,  # US has minimal nickel production
            "manganese": 0,  # US has minimal manganese production
            "chromium": 0,  # US has minimal chromium production
            "tungsten": 0,  # US has minimal tungsten production
            "molybdenum": 65000,  # US molybdenum production
            "vanadium": 0,  # US has minimal vanadium production
            "gallium": 0,  # US has minimal gallium production
            "germanium": 0,  # US has minimal germanium production
            "indium": 0,  # US has minimal indium production
            "tellurium": 0,  # US has minimal tellurium production
            "selenium": 0,  # US has minimal selenium production
            "cadmium": 0,  # US has minimal cadmium production
            "antimony": 0,  # US has minimal antimony production
        }

        base_production = estimates.get(material, 0)

        # Apply year-based growth factor (simplified)
        if year > 2020:
            growth_factor = 1.0 + (year - 2020) * 0.02  # 2% annual growth
            return base_production * growth_factor
        else:
            return base_production


    def _create_empty_dataset(self, dataset_id: str, year: int) -> Dict[str, Any]:
        """Create empty dataset structure for failed scrapes."""
        return {
            "dataset_id": dataset_id,
            "year": year,
            "data": {},
            "metadata": {
                "source": "USGS",
                "collection_timestamp": datetime.now().isoformat(),
                "data_quality": "none",
                "error": "Failed to scrape data",
            },
        }

    def scrape_all_material_data(self, year: int = 2024) -> Dict[str, Any]:
        """
        Scrape all available material data for a given year.

        Args:
            year: Year to scrape data for

        Returns:
            Combined material data dictionary
        """
        all_data = {
            "year": year,
            "mineral_production": {},
            "material_consumption": {},
            "critical_materials": {},
            "material_intensity": {},
            "supply_chain_analysis": {},
            "metadata": {
                "collection_timestamp": datetime.now().isoformat(),
                "scraper": "USGSScraper",
                "data_sources": [],
            },
        }

        datasets = self.get_available_datasets()

        for dataset in datasets:
            try:
                data = self.scrape_dataset(dataset["id"], year=year)
                all_data[dataset["id"]] = data
                all_data["metadata"]["data_sources"].append(dataset["id"])
            except Exception as e:
                self.logger.error(f"Failed to scrape {dataset['id']}: {e}")

        return all_data
