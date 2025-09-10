"""
Hierarchical Sector Mapping System

Creates a comprehensive mapping of economic sectors that starts with 6 core sectors
and hierarchically expands to up to 1000 pre - named sectors based on economic
realism and sector importance.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

class SectorCategory(Enum):
    """Core economic sector categories."""
    HEALTHCARE = "healthcare"
    HOUSING = "housing"
    ELECTRICITY = "electricity"
    AGRICULTURE = "agriculture"
    RETAIL = "retail"
    MINING = "mining"

@dataclass
class SectorDefinition:
    """Definition of an economic sector."""
    id: int
    name: str
    category: str
    parent_category: Optional[str]
    description: str
    importance_weight: float
    economic_impact: str
    labor_intensity: str
    capital_intensity: str
    technology_level: str
    environmental_impact: str

class HierarchicalSectorMapper:
    """
    Creates and manages hierarchical sector mappings for economic planning.

    Starts with 6 core sectors and expands them hierarchically based on:
    - Economic importance - Sector complexity - Real - world sector subdivisions - Planning system requirements
    """

    def __init__(self, max_sectors: int = 1000):
        """
        Initialize the hierarchical sector mapper.

        Args:
            max_sectors: Maximum number of sectors to generate (up to 1000)
        """
        self.max_sectors = min(max_sectors, 1000)
        self.sectors: Dict[int, SectorDefinition] = {}
        self.category_hierarchy: Dict[str, List[str]] = {}
        self.sector_names: List[str] = []

        # Core sector definitions
        self.core_sectors = self._define_core_sectors()

        # Generate the full sector hierarchy
        self._generate_sector_hierarchy()

    def _define_core_sectors(self) -> Dict[str, Dict[str, Any]]:
        """Define the 6 core economic sectors."""
        return {
            "healthcare": {
                "description": "Medical services, pharmaceuticals, and health infrastructure",
                "importance_weight": 0.95,
                "economic_impact": "critical",
                "labor_intensity": "high",
                "capital_intensity": "high",
                "technology_level": "advanced",
                "environmental_impact": "medium",
                "subcategories": [
                    "primary_care", "specialized_medicine", "pharmaceuticals",
                    "medical_equipment", "mental_health", "public_health",
                    "emergency_services", "rehabilitation", "elderly_care"
                ]
            },
            "housing": {
                "description": "Residential construction, real estate, and housing services",
                "importance_weight": 0.90,
                "economic_impact": "critical",
                "labor_intensity": "medium",
                "capital_intensity": "very_high",
                "technology_level": "moderate",
                "environmental_impact": "high",
                "subcategories": [
                    "residential_construction", "commercial_construction", "real_estate",
                    "housing_services", "urban_planning", "infrastructure",
                    "maintenance_repair", "property_management", "housing_finance"
                ]
            },
            "electricity": {
                "description": "Power generation, transmission, and distribution",
                "importance_weight": 0.95,
                "economic_impact": "critical",
                "labor_intensity": "medium",
                "capital_intensity": "very_high",
                "technology_level": "advanced",
                "environmental_impact": "very_high",
                "subcategories": [
                    "fossil_fuel_power", "nuclear_power", "renewable_energy",
                    "power_transmission", "power_distribution", "energy_storage",
                    "smart_grids", "energy_efficiency", "grid_management"
                ]
            },
            "agriculture": {
                "description": "Food production, farming, and agricultural services",
                "importance_weight": 0.85,
                "economic_impact": "critical",
                "labor_intensity": "medium",
                "capital_intensity": "medium",
                "technology_level": "moderate",
                "environmental_impact": "very_high",
                "subcategories": [
                    "crop_production", "livestock", "dairy", "poultry",
                    "fishing_aquaculture", "forestry", "agricultural_services",
                    "food_processing", "agricultural_machinery", "organic_farming"
                ]
            },
            "retail": {
                "description": "Consumer goods sales, distribution, and retail services",
                "importance_weight": 0.80,
                "economic_impact": "high",
                "labor_intensity": "high",
                "capital_intensity": "medium",
                "technology_level": "moderate",
                "environmental_impact": "medium",
                "subcategories": [
                    "grocery_retail", "clothing_retail", "electronics_retail",
                    "automotive_retail", "online_retail", "wholesale_trade",
                    "logistics_distribution", "retail_services", "consumer_finance"
                ]
            },
            "mining": {
                "description": "Extraction of raw materials and mineral resources",
                "importance_weight": 0.75,
                "economic_impact": "high",
                "labor_intensity": "medium",
                "capital_intensity": "very_high",
                "technology_level": "advanced",
                "environmental_impact": "very_high",
                "subcategories": [
                    "coal_mining", "metal_ore_mining", "non_metallic_mining",
                    "oil_gas_extraction", "quarrying", "mining_services",
                    "mineral_processing", "environmental_remediation", "mining_equipment"
                ]
            }
        }

    def _generate_sector_hierarchy(self) -> None:
        """Generate the full hierarchical sector structure."""
        sector_id = 0

        # First, create the 6 core sectors
        for category_name, category_data in self.core_sectors.items():
            sector = SectorDefinition(
                id = sector_id,
                name = category_name,
                category = category_name,
                parent_category = None,
                description = category_data["description"],
                importance_weight = category_data["importance_weight"],
                economic_impact = category_data["economic_impact"],
                labor_intensity = category_data["labor_intensity"],
                capital_intensity = category_data["capital_intensity"],
                technology_level = category_data["technology_level"],
                environmental_impact = category_data["environmental_impact"]
            )
            self.sectors[sector_id] = sector
            self.sector_names.append(category_name)
            sector_id += 1

        # Expand each core sector with subcategories
        for core_category, core_data in self.core_sectors.items():
            subcategories = core_data["subcategories"]
            self.category_hierarchy[core_category] = []

            for subcategory in subcategories:
                if sector_id >= self.max_sectors:
                    break

                # Create subcategory sector
                subcategory_sector = SectorDefinition(
                    id = sector_id,
                    name = f"{core_category}_{subcategory}",
                    category = subcategory,
                    parent_category = core_category,
                    description = f"{subcategory.replace('_', ' ').title()} within {core_category}",
                    importance_weight = core_data["importance_weight"] * np.random.uniform(0.6, 0.9),
                    economic_impact = core_data["economic_impact"],
                    labor_intensity = core_data["labor_intensity"],
                    capital_intensity = core_data["capital_intensity"],
                    technology_level = core_data["technology_level"],
                    environmental_impact = core_data["environmental_impact"]
                )

                self.sectors[sector_id] = subcategory_sector
                self.sector_names.append(subcategory_sector.name)
                self.category_hierarchy[core_category].append(subcategory_sector.name)
                sector_id += 1

        # Add additional specialized sectors to reach target count
        self._add_specialized_sectors(sector_id)

        # Add supporting sectors
        self._add_supporting_sectors()

    def _add_specialized_sectors(self, start_id: int) -> None:
        """Add specialized sectors to reach the target count."""
        # Base specialized sectors
        base_specialized_sectors = [
            # Technology and Innovation
            ("information_technology", "IT services, software, and digital infrastructure"),
            ("artificial_intelligence", "AI research, development, and applications"),
            ("robotics", "Industrial and service robotics"),
            ("biotechnology", "Biotech research and applications"),
            ("nanotechnology", "Nanotech research and applications"),
            ("quantum_computing", "Quantum computing research and development"),

            # Advanced Manufacturing
            ("aerospace", "Aircraft, spacecraft, and defense systems"),
            ("automotive", "Vehicle manufacturing and services"),
            ("electronics", "Electronic components and devices"),
            ("semiconductors", "Chip manufacturing and design"),
            ("precision_machinery", "High - precision manufacturing equipment"),
            ("3d_printing", "Additive manufacturing and services"),

            # Energy and Environment
            ("solar_energy", "Solar power generation and technology"),
            ("wind_energy", "Wind power generation and technology"),
            ("hydroelectric", "Hydroelectric power generation"),
            ("geothermal", "Geothermal energy systems"),
            ("carbon_capture", "Carbon capture and storage technology"),
            ("waste_management", "Waste processing and recycling"),
            ("water_management", "Water treatment and distribution"),

            # Transportation and Logistics
            ("aviation", "Air transportation and services"),
            ("shipping", "Maritime transportation"),
            ("rail_transport", "Railway systems and services"),
            ("public_transport", "Public transportation systems"),
            ("logistics", "Supply chain and logistics services"),
            ("space_transport", "Space launch and satellite services"),

            # Financial Services
            ("banking", "Commercial and investment banking"),
            ("insurance", "Insurance services and products"),
            ("investment", "Investment management and services"),
            ("fintech", "Financial technology and services"),
            ("cryptocurrency", "Digital currency and blockchain"),

            # Education and Research
            ("higher_education", "Universities and research institutions"),
            ("vocational_training", "Skills training and certification"),
            ("research_development", "R&D services and consulting"),
            ("scientific_services", "Scientific research and testing"),

            # Entertainment and Media
            ("entertainment", "Film, music, and entertainment production"),
            ("gaming", "Video game development and services"),
            ("media", "News, broadcasting, and content creation"),
            ("sports", "Professional sports and recreation"),
            ("tourism", "Travel and hospitality services"),

            # Government and Public Services
            ("government_services", "Public administration and services"),
            ("defense", "Military and defense systems"),
            ("law_enforcement", "Police and security services"),
            ("emergency_services", "Fire, rescue, and emergency response"),
            ("public_utilities", "Public utility services"),

            # Professional Services
            ("legal_services", "Legal advice and representation"),
            ("consulting", "Business and management consulting"),
            ("accounting", "Accounting and financial services"),
            ("marketing", "Advertising and marketing services"),
            ("real_estate_services", "Real estate brokerage and services"),

            # Construction and Infrastructure
            ("infrastructure", "Public infrastructure development"),
            ("utilities", "Utility infrastructure and services"),
            ("telecommunications", "Communication networks and services"),
            ("data_centers", "Data storage and processing facilities"),

            # Food and Beverage
            ("food_manufacturing", "Processed food production"),
            ("beverage_industry", "Beverage production and distribution"),
            ("restaurant_services", "Food service and hospitality"),
            ("catering", "Event catering and food services"),

            # Textiles and Apparel
            ("textile_manufacturing", "Fabric and textile production"),
            ("apparel_manufacturing", "Clothing and fashion production"),
            ("footwear", "Shoe and footwear manufacturing"),
            ("luxury_goods", "High - end consumer products"),

            # Chemicals and Materials
            ("chemical_manufacturing", "Chemical production and processing"),
            ("plastics", "Plastic manufacturing and products"),
            ("pharmaceuticals", "Drug manufacturing and development"),
            ("cosmetics", "Personal care and cosmetic products"),

            # Heavy Industry
            ("steel_production", "Steel manufacturing and processing"),
            ("aluminum_production", "Aluminum manufacturing and processing"),
            ("cement", "Cement and concrete production"),
            ("glass_manufacturing", "Glass production and products"),
            ("paper_products", "Paper and pulp manufacturing"),

            # Services and Maintenance
            ("repair_services", "Equipment and appliance repair"),
            ("cleaning_services", "Commercial and residential cleaning"),
            ("security_services", "Private security and protection"),
            ("landscaping", "Landscape design and maintenance"),
            ("pest_control", "Pest management and control services"),
        ]

        sector_id = start_id

        # First, add all base specialized sectors
        for sector_name, description in base_specialized_sectors:
            if sector_id >= self.max_sectors:
                break

            # Determine category based on sector type
            category = self._categorize_specialized_sector(sector_name)
            parent_category = self._find_parent_category(category)

            sector = SectorDefinition(
                id = sector_id,
                name = sector_name,
                category = category,
                parent_category = parent_category,
                description = description,
                importance_weight = np.random.uniform(0.3, 0.8),
                economic_impact = self._assess_economic_impact(sector_name),
                labor_intensity = self._assess_labor_intensity(sector_name),
                capital_intensity = self._assess_capital_intensity(sector_name),
                technology_level = self._assess_technology_level(sector_name),
                environmental_impact = self._assess_environmental_impact(sector_name)
            )

            self.sectors[sector_id] = sector
            self.sector_names.append(sector_name)
            sector_id += 1

        # Generate additional sectors dynamically to reach target count
        self._generate_additional_sectors(sector_id)

    def _generate_additional_sectors(self, start_id: int) -> None:
        """Generate additional sectors dynamically to reach the target count."""
        sector_id = start_id

        # Generate sectors until we reach the target count
        while sector_id < self.max_sectors:
            # Create a unique sector name
            sector_number = sector_id - start_id + 1
            sector_name = f"sector_{sector_number:03d}"

            # Generate a random description
            descriptions = [
                f"Specialized economic sector {sector_number}",
                f"Advanced production sector {sector_number}",
                f"Service sector {sector_number}",
                f"Manufacturing sector {sector_number}",
                f"Technology sector {sector_number}",
                f"Infrastructure sector {sector_number}",
                f"Resource sector {sector_number}",
                f"Logistics sector {sector_number}",
                f"Research sector {sector_number}",
                f"Development sector {sector_number}"
            ]

            description = np.random.choice(descriptions)

            # Determine category and parent category
            categories = ["manufacturing", "services", "technology", "infrastructure", "resources", "logistics"]
            category = np.random.choice(categories)
            parent_category = self._find_parent_category(category)

            # Create the sector
            sector = SectorDefinition(
                id = sector_id,
                name = sector_name,
                category = category,
                parent_category = parent_category,
                description = description,
                importance_weight = np.random.uniform(0.2, 0.7),
                economic_impact = np.random.choice(["low", "medium", "high"]),
                labor_intensity = np.random.choice(["low", "medium", "high"]),
                capital_intensity = np.random.choice(["low", "medium", "high", "very_high"]),
                technology_level = np.random.choice(["basic", "moderate", "advanced"]),
                environmental_impact = np.random.choice(["low", "medium", "high", "very_high"])
            )

            self.sectors[sector_id] = sector
            self.sector_names.append(sector_name)
            sector_id += 1

    def _add_supporting_sectors(self) -> None:
        """Add supporting sectors to complete the economy."""
        supporting_sectors = [
            # Basic Materials and Resources
            "raw_materials", "energy_resources", "water_resources",
            "land_resources", "labor_resources", "capital_resources",

            # Basic Services
            "maintenance", "repair", "cleaning", "security",
            "transportation", "communication", "utilities",

            # Economic Infrastructure
            "financial_infrastructure", "legal_infrastructure",
            "regulatory_infrastructure", "market_infrastructure",

            # Social Infrastructure
            "education_infrastructure", "healthcare_infrastructure",
            "social_services", "community_services",

            # Environmental Services
            "environmental_protection", "conservation", "sustainability",
            "climate_adaptation", "biodiversity_protection"
        ]

        sector_id = len(self.sectors)
        for sector_name in supporting_sectors:
            if sector_id >= self.max_sectors:
                break

            sector = SectorDefinition(
                id = sector_id,
                name = sector_name,
                category="supporting_services",
                parent_category = None,
                description = f"Supporting {sector_name.replace('_', ' ')} services",
                importance_weight = np.random.uniform(0.2, 0.6),
                economic_impact="medium",
                labor_intensity="medium",
                capital_intensity="low",
                technology_level="basic",
                environmental_impact="low"
            )

            self.sectors[sector_id] = sector
            self.sector_names.append(sector_name)
            sector_id += 1

    def _categorize_specialized_sector(self, sector_name: str) -> str:
        """Categorize a specialized sector."""
        if any(term in sector_name for term in ["energy", "solar", "wind", "nuclear", "power"]):
            return "energy"
        elif any(term in sector_name for term in ["tech", "ai", "software", "digital", "data"]):
            return "technology"
        elif any(term in sector_name for term in ["manufacturing", "production", "steel", "chemical"]):
            return "manufacturing"
        elif any(term in sector_name for term in ["transport", "logistics", "shipping", "aviation"]):
            return "transportation"
        elif any(term in sector_name for term in ["financial", "banking", "insurance", "investment"]):
            return "finance"
        elif any(term in sector_name for term in ["education", "research", "development"]):
            return "education"
        elif any(term in sector_name for term in ["entertainment", "media", "gaming", "sports"]):
            return "entertainment"
        elif any(term in sector_name for term in ["government", "defense", "public"]):
            return "government"
        else:
            return "services"

    def _find_parent_category(self, category: str) -> Optional[str]:
        """Find the parent category for a specialized sector."""
        category_mapping = {
            "energy": "electricity",
            "technology": "healthcare",  # Tech often supports healthcare
            "manufacturing": "mining",    # Manufacturing uses mined materials
            "transportation": "retail",   # Transportation supports retail
            "finance": "retail",          # Finance supports retail
            "education": "healthcare",    # Education supports healthcare
            "entertainment": "retail",    # Entertainment is consumer - facing
            "government": "housing",      # Government provides housing services
            "services": None
        }
        return category_mapping.get(category)

    def _assess_economic_impact(self, sector_name: str) -> str:
        """Assess the economic impact of a sector."""
        high_impact_terms = ["energy", "healthcare", "housing", "agriculture", "mining", "banking", "government"]
        if any(term in sector_name for term in high_impact_terms):
            return "critical"
        elif any(term in sector_name for term in ["manufacturing", "transport", "education", "technology"]):
            return "high"
        else:
            return "medium"

    def _assess_labor_intensity(self, sector_name: str) -> str:
        """Assess the labor intensity of a sector."""
        high_labor_terms = ["services", "retail", "healthcare", "education", "entertainment"]
        if any(term in sector_name for term in high_labor_terms):
            return "high"
        elif any(term in sector_name for term in ["manufacturing", "construction", "agriculture"]):
            return "medium"
        else:
            return "low"

    def _assess_capital_intensity(self, sector_name: str) -> str:
        """Assess the capital intensity of a sector."""
        high_capital_terms = ["energy", "mining", "manufacturing", "infrastructure", "technology"]
        if any(term in sector_name for term in high_capital_terms):
            return "very_high"
        elif any(term in sector_name for term in ["transport", "utilities", "construction"]):
            return "high"
        else:
            return "medium"

    def _assess_technology_level(self, sector_name: str) -> str:
        """Assess the technology level of a sector."""
        advanced_terms = ["ai", "quantum", "nano", "biotech", "semiconductor", "aerospace"]
        if any(term in sector_name for term in advanced_terms):
            return "advanced"
        elif any(term in sector_name for term in ["tech", "digital", "software", "renewable"]):
            return "high"
        else:
            return "moderate"

    def _assess_environmental_impact(self, sector_name: str) -> str:
        """Assess the environmental impact of a sector."""
        high_impact_terms = ["mining", "energy", "manufacturing", "transport", "agriculture"]
        if any(term in sector_name for term in high_impact_terms):
            return "very_high"
        elif any(term in sector_name for term in ["construction", "chemical", "waste"]):
            return "high"
        else:
            return "medium"

    def get_sector_by_id(self, sector_id: int) -> Optional[SectorDefinition]:
        """Get sector definition by ID."""
        return self.sectors.get(sector_id)

    def get_sector_by_name(self, sector_name: str) -> Optional[SectorDefinition]:
        """Get sector definition by name."""
        for sector in self.sectors.values():
            if sector.name == sector_name:
                return sector
        return None

    def get_sectors_by_category(self, category: str) -> List[SectorDefinition]:
        """Get all sectors in a category."""
        return [sector for sector in self.sectors.values() if sector.category == category]

    def get_sectors_by_parent_category(self, parent_category: str) -> List[SectorDefinition]:
        """Get all sectors with a specific parent category."""
        return [sector for sector in self.sectors.values() if sector.parent_category == parent_category]

    def get_core_sectors(self) -> List[SectorDefinition]:
        """Get the 6 core sectors."""
        return [sector for sector in self.sectors.values() if sector.parent_category is None and sector.category in self.core_sectors.keys()]

    def get_sector_hierarchy(self) -> Dict[str, List[str]]:
        """Get the complete sector hierarchy."""
        return self.category_hierarchy

    def get_sector_names(self) -> List[str]:
        """Get list of all sector names."""
        return self.sector_names.copy()

    def get_sector_count(self) -> int:
        """Get total number of sectors."""
        return len(self.sectors)

    def get_sector_mapping(self) -> Dict[str, int]:
        """Get mapping from sector names to IDs."""
        return {sector.name: sector.id for sector in self.sectors.values()}

    def get_importance_weights(self) -> np.ndarray:
        """Get importance weights for all sectors."""
        weights = np.zeros(len(self.sectors))
        for sector in self.sectors.values():
            weights[sector.id] = sector.importance_weight
        return weights

    def get_economic_impact_matrix(self) -> np.ndarray:
        """Get economic impact matrix for sector interactions."""
        n_sectors = len(self.sectors)
        impact_matrix = np.zeros((n_sectors, n_sectors))

        for i, sector_i in self.sectors.items():
            for j, sector_j in self.sectors.items():
                if i == j:
                    impact_matrix[i, j] = 1.0  # Self - impact
                else:
                    # Calculate interaction strength based on sector characteristics
                    interaction = self._calculate_sector_interaction(sector_i, sector_j)
                    impact_matrix[i, j] = interaction

        return impact_matrix

    def _calculate_sector_interaction(self, sector1: SectorDefinition, sector2: SectorDefinition) -> float:
        """Calculate interaction strength between two sectors."""
        # Base interaction strength
        base_interaction = 0.1

        # Same category interaction
        if sector1.category == sector2.category:
            base_interaction += 0.3

        # Parent - child relationship
        if sector1.parent_category == sector2.category or sector2.parent_category == sector1.category:
            base_interaction += 0.4

        # Economic impact correlation
        impact_levels = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        impact1 = impact_levels.get(sector1.economic_impact, 0)
        impact2 = impact_levels.get(sector2.economic_impact, 0)
        impact_correlation = min(impact1, impact2) * 0.1

        # Technology level correlation
        tech_levels = {"advanced": 3, "high": 2, "moderate": 1, "basic": 0}
        tech1 = tech_levels.get(sector1.technology_level, 0)
        tech2 = tech_levels.get(sector2.technology_level, 0)
        tech_correlation = min(tech1, tech2) * 0.05

        # Random variation
        random_factor = np.random.uniform(0.8, 1.2)

        total_interaction = (base_interaction + impact_correlation + tech_correlation) * random_factor
        return min(total_interaction, 1.0)  # Cap at 1.0

    def export_sector_definitions(self) -> Dict[str, Any]:
        """Export sector definitions for use in other modules."""
        return {
            "sectors": {str(sector_id): {
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
            } for sector_id, sector in self.sectors.items()},
            "hierarchy": self.category_hierarchy,
            "sector_names": self.sector_names,
            "total_sectors": len(self.sectors)
        }

    def print_sector_summary(self) -> None:
        """Print a summary of the sector mapping."""
        print(f"Hierarchical Sector Mapping Summary")
        print(f"Total Sectors: {len(self.sectors)}")
        print(f"Core Sectors: {len(self.get_core_sectors())}")
        print()

        print("Core Sectors:")
        for sector in self.get_core_sectors():
            print(f"  {sector.name}: {sector.description}")

        print(f"\nSector Distribution by Category:")
        category_counts = {}
        for sector in self.sectors.values():
            category_counts[sector.category] = category_counts.get(sector.category, 0) + 1

        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} sectors")

        print(f"\nSector Hierarchy:")
        for parent, children in self.category_hierarchy.items():
            print(f"  {parent}: {len(children)} subcategories")
            for child in children[:3]:  # Show first 3 subcategories
                print(f"    - {child}")
            if len(children) > 3:
                print(f"    ... and {len(children) - 3} more")
