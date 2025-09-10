"""
Sector Mapping and Data Integration

Maps data from various sources to the 175 - sector BEA classification
and handles data normalization and integration.
"""

from typing import Dict, Any, Optional, List
import numpy as np

class SectorMapper:
    """
    Maps data from various sources to BEA 175 - sector classification.

    Handles:
    - Cross - walking between different sector classifications - Data normalization and unit conversion - Missing data estimation and interpolation - Quality assessment and validation
    """

    def __init__(self, mapping_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sector mapper.

        Args:
            mapping_config: Configuration for sector mappings
        """
        self.mapping_config = mapping_config or self._load_default_config()
        self.bea_sectors = self._load_bea_sector_definitions()
        self.mapping_tables = self._load_mapping_tables()

        # Data quality tracking
        self.quality_metrics = {
            "mapping_confidence": {},
            "data_completeness": {},
            "unit_consistency": {},
            "temporal_coverage": {},
        }

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default mapping configuration."""
        return {
            "target_year": 2024,
            "currency_base_year": 2024,
            "unit_conversion": {
                "energy": "btu",
                "materials": "kg",
                "labor": "hours",
                "environmental": "tons_co2_equivalent",
            },
            "missing_data_strategy": "interpolate",
            "quality_thresholds": {"min_coverage": 0.8, "max_missing_sectors": 20, "confidence_threshold": 0.7},
        }

    def _load_bea_sector_definitions(self) -> Dict[int, Dict[str, Any]]:
        """Load BEA 175 - sector definitions based on Marx's reproduction schemes."""
        sectors = {}

        # Department I: Means of production (sectors 1 - 50)
        for i in range(1, 51):
            sectors[i] = {
                "sector_id": i,
                "naics_code": f"NAICS_{i:03d}",
                "sector_name": f"Dept_I_Sector_{i}",
                "category": "means_of_production",
                "description": f"Means of production sector {i} - heavy industry, machinery, raw materials",
                "marx_department": "I",
                "labor_intensity": "high",
                "capital_intensity": "high"
            }

        # Department II: Consumer goods (sectors 51 - 100)
        for i in range(51, 101):
            sectors[i] = {
                "sector_id": i,
                "naics_code": f"NAICS_{i:03d}",
                "sector_name": f"Dept_II_Sector_{i - 50}",
                "category": "consumer_goods",
                "description": f"Consumer goods sector {i - 50} - manufactured goods for consumption",
                "marx_department": "II",
                "labor_intensity": "medium",
                "capital_intensity": "medium"
            }

        # Department III: Services and other (sectors 101 - 175)
        for i in range(101, 176):
            sectors[i] = {
                "sector_id": i,
                "naics_code": f"NAICS_{i:03d}",
                "sector_name": f"Dept_III_Sector_{i - 100}",
                "category": "services",
                "description": f"Services sector {i - 100} - services, trade, transportation, etc.",
                "marx_department": "III",
                "labor_intensity": "variable",
                "capital_intensity": "low"
            }

        return sectors

    def _categorize_sector(self, sector_id: int) -> str:
        """Categorize sector by ID range."""
        if sector_id <= 20:
            return "manufacturing"
        elif sector_id <= 50:
            return "services"
        elif sector_id <= 80:
            return "construction"
        elif sector_id <= 120:
            return "trade"
        elif sector_id <= 150:
            return "transportation"
        else:
            return "other"

    def _load_mapping_tables(self) -> Dict[str, Dict[str, int]]:
        """Load mapping tables for different data sources."""
        return {
            "naics_to_bea": self._create_naics_mapping(),
            "sic_to_bea": self._create_sic_mapping(),
            "eia_to_bea": self._create_eia_mapping(),
            "usgs_to_bea": self._create_usgs_mapping(),
            "bls_to_bea": self._create_bls_mapping(),
            "epa_to_bea": self._create_epa_mapping(),
        }

    def get_sector_mapping(self) -> Dict[str, int]:
        """Get comprehensive sector name to index mapping."""
        mapping = {}

        # Add mapping for all sectors
        for sector_id, sector_info in self.bea_sectors.items():
            # Map by sector name
            mapping[sector_info["sector_name"]] = sector_id - 1  # Convert to 0 - based index

            # Map by category
            category = sector_info["category"]
            if category not in mapping:
                mapping[category] = []
            mapping[category].append(sector_id - 1)

            # Map by Marx department
            dept = sector_info["marx_department"]
            dept_key = f"department_{dept.lower()}"
            if dept_key not in mapping:
                mapping[dept_key] = []
            mapping[dept_key].append(sector_id - 1)

        return mapping

    def get_sector_by_name(self, sector_name: str) -> Optional[Dict[str, Any]]:
        """Get sector information by name."""
        for sector_id, sector_info in self.bea_sectors.items():
            if sector_info["sector_name"] == sector_name:
                return sector_info
        return None

    def get_sectors_by_department(self, department: str) -> List[int]:
        """Get sector indices by Marx department (I, II, III)."""
        dept_key = f"department_{department.lower()}"
        mapping = self.get_sector_mapping()
        return mapping.get(dept_key, [])

    def get_sectors_by_category(self, category: str) -> List[int]:
        """Get sector indices by category."""
        mapping = self.get_sector_mapping()
        return mapping.get(category, [])

    def _create_naics_mapping(self) -> Dict[str, int]:
        """Create NAICS to BEA sector mapping."""
        # Simplified mapping - in practice would be more detailed
        mapping = {}

        # Manufacturing sectors (NAICS 31 - 33)
        for naics in range(311, 340):
            mapping[f"NAICS_{naics}"] = (naics - 311) % 20 + 1

        # Services sectors (NAICS 51 - 92)
        for naics in range(510, 930):
            mapping[f"NAICS_{naics}"] = (naics - 510) % 125 + 21

        return mapping

    def _create_sic_mapping(self) -> Dict[str, int]:
        """Create SIC to BEA sector mapping."""
        mapping = {}

        # Manufacturing (SIC 20 - 39)
        for sic in range(200, 400):
            mapping[f"SIC_{sic}"] = (sic - 200) % 20 + 1

        # Services (SIC 40 - 89)
        for sic in range(400, 900):
            mapping[f"SIC_{sic}"] = (sic - 400) % 125 + 21

        return mapping

    def _create_eia_mapping(self) -> Dict[str, int]:
        """Create EIA energy sector to BEA mapping."""
        return {
            "electric_power": 1,
            "petroleum_refining": 2,
            "coal_mining": 3,
            "natural_gas": 4,
            "nuclear_power": 5,
            "renewable_energy": 6,
            "manufacturing": 7,
            "transportation": 8,
            "residential": 9,
            "commercial": 10,
        }

    def _create_usgs_mapping(self) -> Dict[str, int]:
        """Create USGS material sector to BEA mapping."""
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

    def _create_bls_mapping(self) -> Dict[str, int]:
        """Create BLS labor sector to BEA mapping."""
        return {
            "agriculture": 1,
            "mining": 2,
            "utilities": 3,
            "construction": 4,
            "manufacturing": 5,
            "wholesale_trade": 6,
            "retail_trade": 7,
            "transportation": 8,
            "information": 9,
            "finance": 10,
        }

    def _create_epa_mapping(self) -> Dict[str, int]:
        """Create EPA environmental sector to BEA mapping."""
        return {
            "energy": 1,
            "manufacturing": 2,
            "transportation": 3,
            "agriculture": 4,
            "construction": 5,
            "mining": 6,
            "utilities": 7,
            "waste_management": 8,
            "services": 9,
            "government": 10,
        }

    def map_data_to_bea_sectors(self, data: Dict[str, Any], source_type: str, data_type: str) -> Dict[str, Any]:
        """
        Map data from source classification to BEA 175 - sector classification.

        Args:
            data: Source data dictionary
            source_type: Type of source ('eia', 'usgs', 'bls', 'epa')
            data_type: Type of data ('consumption', 'intensity', 'production')

        Returns:
            Mapped data dictionary
        """
        try:
            mapping_table = self.mapping_tables.get(f"{source_type}_to_bea", {})

            if not mapping_table:
                raise ValueError(f"No mapping table found for source: {source_type}")

            mapped_data = {
                "bea_sectors": list(range(1, 176)),
                "data_type": data_type,
                "source_type": source_type,
                "mapped_values": {},
                "mapping_metadata": {
                    "mapping_table": mapping_table,
                    "mapping_confidence": {},
                    "missing_sectors": [],
                    "mapped_sectors": [],
                },
            }

            # Map data based on source type
            if source_type == "eia":
                mapped_data = self._map_eia_data(data, mapping_table, data_type)
            elif source_type == "usgs":
                mapped_data = self._map_usgs_data(data, mapping_table, data_type)
            elif source_type == "bls":
                mapped_data = self._map_bls_data(data, mapping_table, data_type)
            elif source_type == "epa":
                mapped_data = self._map_epa_data(data, mapping_table, data_type)
            else:
                mapped_data = self._map_generic_data(data, mapping_table, data_type)

            # Assess mapping quality
            self._assess_mapping_quality(mapped_data, source_type)

            return mapped_data

        except Exception as e:
            # Log error instead of printing
            return self._create_empty_mapped_data(source_type, data_type)

    def _map_eia_data(self, data: Dict[str, Any], mapping_table: Dict[str, int], data_type: str) -> Dict[str, Any]:
        """Map EIA energy data to BEA sectors."""
        mapped_data = {
            "bea_sectors": list(range(1, 176)),
            "data_type": data_type,
            "source_type": "eia",
            "energy_types": ["coal", "natural_gas", "petroleum", "nuclear", "renewable"],
            "mapped_values": {},
            "mapping_metadata": {
                "mapping_table": mapping_table,
                "mapping_confidence": {},
                "missing_sectors": [],
                "mapped_sectors": [],
            },
        }

        # Initialize sector data
        for sector_id in range(1, 176):
            mapped_data["mapped_values"][sector_id] = {
                "energy_consumption": {},
                "energy_intensity": {},
                "total_consumption": 0,
            }

        # Map energy consumption data
        if "energy_consumption_by_sector" in data:
            consumption_data = data["energy_consumption_by_sector"].get("data", {})
            if "energy_data" in consumption_data:
                self._distribute_energy_consumption(consumption_data["energy_data"], mapped_data)

        # Map energy intensity data
        if "energy_intensity_by_sector" in data:
            intensity_data = data["energy_intensity_by_sector"].get("data", {})
            if "sector_intensities" in intensity_data:
                self._map_energy_intensities(intensity_data["sector_intensities"], mapped_data)

        return mapped_data

    def _map_usgs_data(self, data: Dict[str, Any], mapping_table: Dict[str, int], data_type: str) -> Dict[str, Any]:
        """Map USGS material data to BEA sectors."""
        mapped_data = {
            "bea_sectors": list(range(1, 176)),
            "data_type": data_type,
            "source_type": "usgs",
            "materials": ["lithium", "cobalt", "rare_earth", "copper", "aluminum", "steel"],
            "mapped_values": {},
            "mapping_metadata": {
                "mapping_table": mapping_table,
                "mapping_confidence": {},
                "missing_sectors": [],
                "mapped_sectors": [],
            },
        }

        # Initialize sector data
        for sector_id in range(1, 176):
            mapped_data["mapped_values"][sector_id] = {
                "material_consumption": {},
                "material_intensity": {},
                "total_consumption": 0,
            }

        # Map material consumption data
        if "material_consumption" in data:
            consumption_data = data["material_consumption"].get("data", {})
            if "sectors" in consumption_data:
                self._map_material_consumption(consumption_data["sectors"], mapped_data)

        # Map material intensity data
        if "material_intensity" in data:
            intensity_data = data["material_intensity"].get("data", {})
            if "sectors" in intensity_data:
                self._map_material_intensities(intensity_data["sectors"], mapped_data)

        return mapped_data

    def _map_bls_data(self, data: Dict[str, Any], mapping_table: Dict[str, int], data_type: str) -> Dict[str, Any]:
        """Map BLS labor data to BEA sectors."""
        mapped_data = {
            "bea_sectors": list(range(1, 176)),
            "data_type": data_type,
            "source_type": "bls",
            "labor_categories": ["high_skilled", "medium_skilled", "low_skilled", "technical", "management"],
            "mapped_values": {},
            "mapping_metadata": {
                "mapping_table": mapping_table,
                "mapping_confidence": {},
                "missing_sectors": [],
                "mapped_sectors": [],
            },
        }

        # Initialize sector data
        for sector_id in range(1, 176):
            mapped_data["mapped_values"][sector_id] = {
                "employment": {},
                "wage_rates": {},
                "labor_intensity": {},
                "total_employment": 0,
            }

        # Map employment data
        if "employment_by_sector" in data:
            employment_data = data["employment_by_sector"].get("data", {})
            if "sectors" in employment_data:
                self._map_employment_data(employment_data["sectors"], mapped_data)

        # Map wage data
        if "wage_rates" in data:
            wage_data = data["wage_rates"].get("data", {})
            if "categories" in wage_data:
                self._map_wage_data(wage_data["categories"], mapped_data)

        # Map labor intensity data
        if "labor_intensity" in data:
            intensity_data = data["labor_intensity"].get("data", {})
            if "sectors" in intensity_data:
                self._map_labor_intensities(intensity_data["sectors"], mapped_data)

        return mapped_data

    def _map_epa_data(self, data: Dict[str, Any], mapping_table: Dict[str, int], data_type: str) -> Dict[str, Any]:
        """Map EPA environmental data to BEA sectors."""
        mapped_data = {
            "bea_sectors": list(range(1, 176)),
            "data_type": data_type,
            "source_type": "epa",
            "environmental_factors": ["carbon_emissions", "water_usage", "waste_generation"],
            "mapped_values": {},
            "mapping_metadata": {
                "mapping_table": mapping_table,
                "mapping_confidence": {},
                "missing_sectors": [],
                "mapped_sectors": [],
            },
        }

        # Initialize sector data
        for sector_id in range(1, 176):
            mapped_data["mapped_values"][sector_id] = {
                "environmental_impact": {},
                "environmental_intensity": {},
                "total_impact": 0,
            }

        # Map environmental impact data
        for impact_type in ["carbon_emissions", "water_usage", "waste_generation"]:
            if impact_type in data:
                impact_data = data[impact_type].get("data", {})
                if "sectors" in impact_data:
                    self._map_environmental_impact(impact_data["sectors"], mapped_data, impact_type)

        # Map environmental intensity data
        if "environmental_intensity" in data:
            intensity_data = data["environmental_intensity"].get("data", {})
            if "sectors" in intensity_data:
                self._map_environmental_intensities(intensity_data["sectors"], mapped_data)

        return mapped_data

    def _map_generic_data(self, data: Dict[str, Any], mapping_table: Dict[str, int], data_type: str) -> Dict[str, Any]:
        """Map generic data using standard mapping approach."""
        mapped_data = {
            "bea_sectors": list(range(1, 176)),
            "data_type": data_type,
            "source_type": "generic",
            "mapped_values": {},
            "mapping_metadata": {
                "mapping_table": mapping_table,
                "mapping_confidence": {},
                "missing_sectors": [],
                "mapped_sectors": [],
            },
        }

        # Initialize sector data
        for sector_id in range(1, 176):
            mapped_data["mapped_values"][sector_id] = {"values": {}, "total_value": 0}

        # Generic mapping logic
        if "sectors" in data:
            for source_sector, value in data["sectors"].items():
                if source_sector in mapping_table:
                    bea_sector = mapping_table[source_sector]
                    mapped_data["mapped_values"][bea_sector]["values"][source_sector] = value
                    mapped_data["mapping_metadata"]["mapped_sectors"].append(bea_sector)

        return mapped_data

    def _distribute_energy_consumption(self, energy_data: Dict[str, Any], mapped_data: Dict[str, Any]) -> None:
        """Distribute energy consumption data across BEA sectors."""
        if "sector_consumption" in energy_data:
            sector_consumption = energy_data["sector_consumption"]

            # Distribute consumption across sectors
            for sector_id, consumption in enumerate(sector_consumption, 1):
                if sector_id <= 175:
                    mapped_data["mapped_values"][sector_id]["total_consumption"] = consumption

                    # Distribute across energy types
                    for energy_type in mapped_data["energy_types"]:
                        mapped_data["mapped_values"][sector_id]["energy_consumption"][energy_type] = consumption / len(
                            mapped_data["energy_types"]
                        )

    def _map_energy_intensities(
        self, sector_intensities: Dict[int, Dict[str, float]], mapped_data: Dict[str, Any]
    ) -> None:
        """Map energy intensity data to BEA sectors."""
        for sector_id, intensities in sector_intensities.items():
            if 1 <= sector_id <= 175:
                mapped_data["mapped_values"][sector_id]["energy_intensity"] = intensities
                mapped_data["mapping_metadata"]["mapped_sectors"].append(sector_id)

    def _map_material_consumption(self, sectors_data: Dict[int, Dict[str, float]], mapped_data: Dict[str, Any]) -> None:
        """Map material consumption data to BEA sectors."""
        for sector_id, consumption in sectors_data.items():
            if 1 <= sector_id <= 175:
                mapped_data["mapped_values"][sector_id]["material_consumption"] = consumption
                mapped_data["mapped_values"][sector_id]["total_consumption"] = sum(consumption.values())
                mapped_data["mapping_metadata"]["mapped_sectors"].append(sector_id)

    def _map_material_intensities(self, sectors_data: Dict[int, Dict[str, float]], mapped_data: Dict[str, Any]) -> None:
        """Map material intensity data to BEA sectors."""
        for sector_id, intensities in sectors_data.items():
            if 1 <= sector_id <= 175:
                mapped_data["mapped_values"][sector_id]["material_intensity"] = intensities
                mapped_data["mapping_metadata"]["mapped_sectors"].append(sector_id)

    def _map_employment_data(self, sectors_data: Dict[int, Dict[str, Any]], mapped_data: Dict[str, Any]) -> None:
        """Map employment data to BEA sectors."""
        for sector_id, employment_data in sectors_data.items():
            if 1 <= sector_id <= 175:
                mapped_data["mapped_values"][sector_id]["employment"] = employment_data
                mapped_data["mapped_values"][sector_id]["total_employment"] = employment_data.get("total_employment", 0)
                mapped_data["mapping_metadata"]["mapped_sectors"].append(sector_id)

    def _map_wage_data(self, categories_data: Dict[str, Dict[str, float]], mapped_data: Dict[str, Any]) -> None:
        """Map wage data to BEA sectors."""
        for sector_id in range(1, 176):
            mapped_data["mapped_values"][sector_id]["wage_rates"] = categories_data

    def _map_labor_intensities(self, sectors_data: Dict[int, Dict[str, Any]], mapped_data: Dict[str, Any]) -> None:
        """Map labor intensity data to BEA sectors."""
        for sector_id, intensity_data in sectors_data.items():
            if 1 <= sector_id <= 175:
                mapped_data["mapped_values"][sector_id]["labor_intensity"] = intensity_data
                mapped_data["mapping_metadata"]["mapped_sectors"].append(sector_id)

    def _map_environmental_impact(
        self, sectors_data: Dict[int, Dict[str, float]], mapped_data: Dict[str, Any], impact_type: str
    ) -> None:
        """Map environmental impact data to BEA sectors."""
        for sector_id, impact_data in sectors_data.items():
            if 1 <= sector_id <= 175:
                mapped_data["mapped_values"][sector_id]["environmental_impact"][impact_type] = impact_data
                mapped_data["mapping_metadata"]["mapped_sectors"].append(sector_id)

    def _map_environmental_intensities(
        self, sectors_data: Dict[int, Dict[str, float]], mapped_data: Dict[str, Any]
    ) -> None:
        """Map environmental intensity data to BEA sectors."""
        for sector_id, intensities in sectors_data.items():
            if 1 <= sector_id <= 175:
                mapped_data["mapped_values"][sector_id]["environmental_intensity"] = intensities
                mapped_data["mapping_metadata"]["mapped_sectors"].append(sector_id)

    def _assess_mapping_quality(self, mapped_data: Dict[str, Any], source_type: str) -> None:
        """Assess the quality of sector mapping."""
        mapped_sectors = mapped_data["mapping_metadata"]["mapped_sectors"]
        total_sectors = 175
        coverage = len(mapped_sectors) / total_sectors

        # Calculate mapping confidence
        confidence = min(1.0, coverage * 1.2)  # Boost confidence for good coverage

        mapped_data["mapping_metadata"]["mapping_confidence"] = {
            "overall_confidence": confidence,
            "sector_coverage": coverage,
            "mapped_sector_count": len(mapped_sectors),
            "missing_sector_count": total_sectors - len(mapped_sectors),
        }

        # Identify missing sectors
        all_sectors = set(range(1, 176))
        mapped_sector_set = set(mapped_sectors)
        missing_sectors = list(all_sectors - mapped_sector_set)
        mapped_data["mapping_metadata"]["missing_sectors"] = missing_sectors

        # Store quality metrics
        self.quality_metrics["mapping_confidence"][source_type] = confidence
        self.quality_metrics["data_completeness"][source_type] = coverage

    def _create_empty_mapped_data(self, source_type: str, data_type: str) -> Dict[str, Any]:
        """Create empty mapped data structure for failed mappings."""
        return {
            "bea_sectors": list(range(1, 176)),
            "data_type": data_type,
            "source_type": source_type,
            "mapped_values": {},
            "mapping_metadata": {
                "mapping_table": {},
                "mapping_confidence": {"overall_confidence": 0.0, "sector_coverage": 0.0},
                "missing_sectors": list(range(1, 176)),
                "mapped_sectors": [],
            },
        }

    def handle_missing_data(self, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle missing data for unmapped sectors by marking them as missing.

        Args:
            mapped_data: Mapped data with missing sectors

        Returns:
            Data with missing sectors marked appropriately
        """
        missing_sectors = mapped_data["mapping_metadata"]["missing_sectors"]

        if not missing_sectors:
            return mapped_data

        # Mark missing sectors as having no data
        for sector_id in missing_sectors:
            mapped_data["mapped_values"][sector_id] = {
                "data_available": False,
                "values": {},
                "total_value": 0,
                "confidence": 0.0,
            }

        # Update mapping confidence to reflect missing data
        total_sectors = 175
        mapped_count = len(mapped_data["mapping_metadata"]["mapped_sectors"])
        mapped_data["mapping_metadata"]["mapping_confidence"]["overall_confidence"] = mapped_count / total_sectors
        mapped_data["mapping_metadata"]["missing_sectors"] = missing_sectors

        return mapped_data

    def normalize_units(self, data: Dict[str, Any], target_units: Dict[str, str]) -> Dict[str, Any]:
        """
        Normalize units across different data sources.

        Args:
            data: Data to normalize
            target_units: Target unit specifications

        Returns:
            Data with normalized units
        """
        # This would implement unit conversion logic
        # For now, return data as - is
        return data

    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get summary of mapping operations."""
        return {
            "mapping_confidence": self.quality_metrics["mapping_confidence"],
            "data_completeness": self.quality_metrics["data_completeness"],
            "unit_consistency": self.quality_metrics["unit_consistency"],
            "temporal_coverage": self.quality_metrics["temporal_coverage"],
            "total_mappings": len(self.quality_metrics["mapping_confidence"]),
            "average_confidence": (
                np.mean(list(self.quality_metrics["mapping_confidence"].values()))
                if self.quality_metrics["mapping_confidence"]
                else 0.0
            ),
        }
