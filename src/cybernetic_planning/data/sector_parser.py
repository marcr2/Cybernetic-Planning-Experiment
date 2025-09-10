"""
Sector Parser for 1000 Economic Sectors

Parses the sectors.md file to extract all 1000 sectors and their categories,
then creates a technology tree structure with dependencies.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json

class TechnologyLevel(Enum):
    """Technology development levels for sector unlocking."""
    BASIC = "basic"           # Available from start
    INTERMEDIATE = "intermediate"  # Requires basic sectors
    ADVANCED = "advanced"     # Requires intermediate sectors
    CUTTING_EDGE = "cutting_edge"  # Requires advanced sectors
    FUTURE = "future"         # Requires cutting - edge sectors

class SectorCategory(Enum):
    """Main sector categories."""
    HEALTHCARE = "healthcare"
    AGRICULTURE = "agriculture"
    ENERGY = "energy"
    CONSTRUCTION = "construction"
    EDUCATION = "education"
    TRANSPORTATION = "transportation"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"
    SERVICES = "services"
    RESEARCH = "research"
    INFRASTRUCTURE = "infrastructure"
    FINANCE = "finance"
    ENTERTAINMENT = "entertainment"
    GOVERNMENT = "government"
    ENVIRONMENTAL = "environmental"

@dataclass
class SectorDefinition:
    """Definition of an economic sector with technology tree properties."""
    id: int
    name: str
    category: SectorCategory
    subcategory: str
    technology_level: TechnologyLevel
    prerequisites: List[int]  # IDs of prerequisite sectors
    unlocks: List[int]        # IDs of sectors this unlocks
    description: str
    importance_weight: float
    economic_impact: str
    labor_intensity: str
    capital_intensity: str
    environmental_impact: str
    development_cost: float   # Cost to develop this sector
    research_requirements: List[str]  # Research areas needed

class SectorParser:
    """Parser for extracting sectors from markdown and building technology tree."""

    def __init__(self, sectors_file_path: str = "src / cybernetic_planning / data / sectors.md"):
        self.sectors_file_path = sectors_file_path
        self.sectors: Dict[int, SectorDefinition] = {}
        self.sector_names: List[str] = []
        self.technology_tree: Dict[int, List[int]] = {}  # sector_id -> unlocked_sectors
        self.prerequisites: Dict[int, List[int]] = {}    # sector_id -> required_sectors

    def parse_sectors(self) -> Dict[int, SectorDefinition]:
        """Parse all sectors from the markdown file."""
        with open(self.sectors_file_path, 'r', encoding='utf - 8') as f:
            content = f.read()

        # Extract sectors using regex
        sector_pattern = r'(\d+)\.\s+(.+?)(?=\n\d+\.|\n\n|$)'
        matches = re.findall(sector_pattern, content, re.MULTILINE | re.DOTALL)

        sector_id = 1
        current_category = SectorCategory.SERVICES  # Default

        for match in matches:
            number_str, name_and_desc = match
            sector_number = int(number_str)

            # Clean up the name
            name = name_and_desc.strip()
            if '\n' in name:
                name = name.split('\n')[0].strip()

            # Determine category based on sector number ranges
            category = self._determine_category(sector_number)

            # Determine technology level based on sector number and name
            tech_level = self._determine_technology_level(sector_number, name)

            # Create sector definition
            sector = SectorDefinition(
                id = sector_number - 1,  # Convert to 0 - based indexing
                name = name,
                category = category,
                subcategory = self._determine_subcategory(name, category),
                technology_level = tech_level,
                prerequisites=[],  # Will be filled later
                unlocks=[],        # Will be filled later
                description = f"Economic sector: {name}",
                importance_weight = self._calculate_importance_weight(sector_number, name),
                economic_impact = self._assess_economic_impact(name),
                labor_intensity = self._assess_labor_intensity(name),
                capital_intensity = self._assess_capital_intensity(name),
                environmental_impact = self._assess_environmental_impact(name),
                development_cost = self._calculate_development_cost(tech_level),
                research_requirements = self._determine_research_requirements(name, tech_level)
            )

            self.sectors[sector_number - 1] = sector
            self.sector_names.append(name)

        # Build technology tree dependencies
        self._build_technology_tree()

        return self.sectors

    def _determine_category(self, sector_number: int) -> SectorCategory:
        """Determine the main category based on sector number."""
        if 1 <= sector_number <= 6:
            return SectorCategory.HEALTHCARE  # Core sectors
        elif 7 <= sector_number <= 25:
            return SectorCategory.HEALTHCARE
        elif 26 <= sector_number <= 50:
            return SectorCategory.AGRICULTURE
        elif 51 <= sector_number <= 75:
            return SectorCategory.ENERGY
        elif 76 <= sector_number <= 100:
            return SectorCategory.CONSTRUCTION
        elif 101 <= sector_number <= 125:
            return SectorCategory.EDUCATION
        elif 126 <= sector_number <= 150:
            return SectorCategory.TRANSPORTATION
        elif 151 <= sector_number <= 200:
            return SectorCategory.MANUFACTURING
        elif 201 <= sector_number <= 300:
            return SectorCategory.TECHNOLOGY
        elif 301 <= sector_number <= 400:
            return SectorCategory.SERVICES
        elif 401 <= sector_number <= 500:
            return SectorCategory.RESEARCH
        elif 501 <= sector_number <= 600:
            return SectorCategory.INFRASTRUCTURE
        elif 601 <= sector_number <= 700:
            return SectorCategory.FINANCE
        elif 701 <= sector_number <= 800:
            return SectorCategory.ENTERTAINMENT
        elif 801 <= sector_number <= 900:
            return SectorCategory.GOVERNMENT
        elif 901 <= sector_number <= 1000:
            return SectorCategory.ENVIRONMENTAL
        else:
            return SectorCategory.SERVICES

    def _determine_subcategory(self, name: str, category: SectorCategory) -> str:
        """Determine subcategory based on sector name and main category."""
        name_lower = name.lower()

        if category == SectorCategory.HEALTHCARE:
            if any(term in name_lower for term in ['pharmaceutical', 'drug', 'medicine']):
                return 'pharmaceuticals'
            elif any(term in name_lower for term in ['equipment', 'device', 'medical']):
                return 'medical_equipment'
            elif any(term in name_lower for term in ['hospital', 'clinic', 'care']):
                return 'healthcare_services'
            elif any(term in name_lower for term in ['research', 'development']):
                return 'healthcare_research'
            else:
                return 'healthcare_services'

        elif category == SectorCategory.TECHNOLOGY:
            if any(term in name_lower for term in ['ai', 'artificial', 'machine learning']):
                return 'artificial_intelligence'
            elif any(term in name_lower for term in ['quantum', 'quantum computing']):
                return 'quantum_technology'
            elif any(term in name_lower for term in ['blockchain', 'cryptocurrency']):
                return 'blockchain_technology'
            elif any(term in name_lower for term in ['cloud', 'computing']):
                return 'cloud_computing'
            elif any(term in name_lower for term in ['cyber', 'security']):
                return 'cybersecurity'
            else:
                return 'general_technology'

        elif category == SectorCategory.MANUFACTURING:
            if any(term in name_lower for term in ['automotive', 'vehicle', 'car']):
                return 'automotive'
            elif any(term in name_lower for term in ['aerospace', 'aircraft', 'space']):
                return 'aerospace'
            elif any(term in name_lower for term in ['electronics', 'semiconductor']):
                return 'electronics'
            elif any(term in name_lower for term in ['steel', 'metal', 'materials']):
                return 'materials'
            else:
                return 'general_manufacturing'

        else:
            return 'general'

    def _determine_technology_level(self, sector_number: int, name: str) -> TechnologyLevel:
        """Determine technology level based on sector number and name."""
        name_lower = name.lower()

        # Core sectors (1 - 6) are basic
        if 1 <= sector_number <= 6:
            return TechnologyLevel.BASIC

        # Sectors 7 - 100 are generally intermediate
        elif 7 <= sector_number <= 100:
            return TechnologyLevel.INTERMEDIATE

        # Sectors 101 - 300 are advanced
        elif 101 <= sector_number <= 300:
            return TechnologyLevel.ADVANCED

        # Sectors 301 - 600 are cutting - edge
        elif 301 <= sector_number <= 600:
            return TechnologyLevel.CUTTING_EDGE

        # Sectors 601 - 1000 are future technology
        elif 601 <= sector_number <= 1000:
            return TechnologyLevel.FUTURE

        # Special cases based on keywords
        if any(term in name_lower for term in ['quantum', 'ai', 'artificial intelligence', 'blockchain', 'cryptocurrency']):
            return TechnologyLevel.CUTTING_EDGE
        elif any(term in name_lower for term in ['research', 'development', 'innovation']):
            return TechnologyLevel.ADVANCED
        else:
            return TechnologyLevel.INTERMEDIATE

    def _calculate_importance_weight(self, sector_number: int, name: str) -> float:
        """Calculate importance weight based on sector characteristics."""
        base_weight = 0.5

        # Core sectors are more important
        if 1 <= sector_number <= 6:
            base_weight = 0.9

        # Technology sectors are important
        if any(term in name.lower() for term in ['technology', 'ai', 'quantum', 'blockchain']):
            base_weight = max(base_weight, 0.8)

        # Research sectors are important
        if any(term in name.lower() for term in ['research', 'development', 'innovation']):
            base_weight = max(base_weight, 0.7)

        # Add some randomness
        import random
        return min(1.0, base_weight + random.uniform(-0.1, 0.1))

    def _assess_economic_impact(self, name: str) -> str:
        """Assess economic impact of a sector."""
        name_lower = name.lower()

        if any(term in name_lower for term in ['healthcare', 'energy', 'food', 'housing', 'education', 'transportation']):
            return 'critical'
        elif any(term in name_lower for term in ['technology', 'manufacturing', 'research']):
            return 'high'
        elif any(term in name_lower for term in ['entertainment', 'luxury', 'specialty']):
            return 'medium'
        else:
            return 'medium'

    def _assess_labor_intensity(self, name: str) -> str:
        """Assess labor intensity of a sector."""
        name_lower = name.lower()

        if any(term in name_lower for term in ['services', 'healthcare', 'education', 'retail']):
            return 'high'
        elif any(term in name_lower for term in ['manufacturing', 'construction', 'agriculture']):
            return 'medium'
        elif any(term in name_lower for term in ['technology', 'research', 'ai', 'quantum']):
            return 'low'
        else:
            return 'medium'

    def _assess_capital_intensity(self, name: str) -> str:
        """Assess capital intensity of a sector."""
        name_lower = name.lower()

        if any(term in name_lower for term in ['manufacturing', 'energy', 'infrastructure', 'technology']):
            return 'high'
        elif any(term in name_lower for term in ['research', 'development', 'quantum', 'ai']):
            return 'very_high'
        elif any(term in name_lower for term in ['services', 'retail', 'entertainment']):
            return 'low'
        else:
            return 'medium'

    def _assess_environmental_impact(self, name: str) -> str:
        """Assess environmental impact of a sector."""
        name_lower = name.lower()

        if any(term in name_lower for term in ['energy', 'manufacturing', 'mining', 'chemical']):
            return 'high'
        elif any(term in name_lower for term in ['agriculture', 'transportation', 'construction']):
            return 'medium'
        elif any(term in name_lower for term in ['technology', 'services', 'research']):
            return 'low'
        else:
            return 'medium'

    def _calculate_development_cost(self, tech_level: TechnologyLevel) -> float:
        """Calculate development cost based on technology level."""
        costs = {
            TechnologyLevel.BASIC: 1.0,
            TechnologyLevel.INTERMEDIATE: 2.0,
            TechnologyLevel.ADVANCED: 5.0,
            TechnologyLevel.CUTTING_EDGE: 10.0,
            TechnologyLevel.FUTURE: 20.0
        }
        return costs[tech_level]

    def _determine_research_requirements(self, name: str, tech_level: TechnologyLevel) -> List[str]:
        """Determine research requirements for a sector."""
        name_lower = name.lower()
        requirements = []

        if tech_level in [TechnologyLevel.ADVANCED, TechnologyLevel.CUTTING_EDGE, TechnologyLevel.FUTURE]:
            requirements.append('basic_research')

        if any(term in name_lower for term in ['ai', 'artificial', 'machine learning']):
            requirements.extend(['computer_science', 'mathematics', 'data_science'])

        if any(term in name_lower for term in ['quantum', 'quantum computing']):
            requirements.extend(['physics', 'quantum_mechanics', 'materials_science'])

        if any(term in name_lower for term in ['biotech', 'pharmaceutical', 'medical']):
            requirements.extend(['biology', 'chemistry', 'medicine'])

        if any(term in name_lower for term in ['energy', 'renewable', 'solar', 'wind']):
            requirements.extend(['physics', 'engineering', 'materials_science'])

        return requirements

    def _build_technology_tree(self):
        """Build the technology tree with dependencies."""
        # Initialize prerequisites and unlocks
        for sector_id in self.sectors:
            self.prerequisites[sector_id] = []
            self.technology_tree[sector_id] = []

        # Build dependencies based on technology levels and categories
        for sector_id, sector in self.sectors.items():
            # Find prerequisite sectors
            prerequisites = self._find_prerequisites(sector_id, sector)
            self.prerequisites[sector_id] = prerequisites
            sector.prerequisites = prerequisites

            # Update unlocks for prerequisite sectors
            for prereq_id in prerequisites:
                if prereq_id in self.technology_tree:
                    self.technology_tree[prereq_id].append(sector_id)
                    if prereq_id in self.sectors:
                        self.sectors[prereq_id].unlocks.append(sector_id)

    def _find_prerequisites(self, sector_id: int, sector: SectorDefinition) -> List[int]:
        """Find prerequisite sectors for a given sector."""
        prerequisites = []

        # Core sectors (1 - 6) have no prerequisites
        if sector_id < 6:
            return prerequisites

        # Technology level - based prerequisites
        if sector.technology_level == TechnologyLevel.INTERMEDIATE:
            # Need at least one basic sector from the same category
            basic_sectors = [s_id for s_id, s in self.sectors.items()
                           if s.technology_level == TechnologyLevel.BASIC and s.category == sector.category]
            if basic_sectors:
                prerequisites.append(basic_sectors[0])

        elif sector.technology_level == TechnologyLevel.ADVANCED:
            # Need intermediate sectors from the same category
            intermediate_sectors = [s_id for s_id, s in self.sectors.items()
                                 if s.technology_level == TechnologyLevel.INTERMEDIATE and s.category == sector.category]
            if intermediate_sectors:
                prerequisites.extend(intermediate_sectors[:2])  # Need 2 intermediate sectors

        elif sector.technology_level == TechnologyLevel.CUTTING_EDGE:
            # Need advanced sectors from the same category
            advanced_sectors = [s_id for s_id, s in self.sectors.items()
                             if s.technology_level == TechnologyLevel.ADVANCED and s.category == sector.category]
            if advanced_sectors:
                prerequisites.extend(advanced_sectors[:2])  # Need 2 advanced sectors

        elif sector.technology_level == TechnologyLevel.FUTURE:
            # Need cutting - edge sectors from the same category
            cutting_edge_sectors = [s_id for s_id, s in self.sectors.items()
                                 if s.technology_level == TechnologyLevel.CUTTING_EDGE and s.category == sector.category]
            if cutting_edge_sectors:
                prerequisites.extend(cutting_edge_sectors[:2])  # Need 2 cutting - edge sectors

        # Cross - category dependencies for technology sectors
        if sector.category == SectorCategory.TECHNOLOGY:
            # Technology sectors often depend on research and manufacturing
            research_sectors = [s_id for s_id, s in self.sectors.items()
                             if s.category == SectorCategory.RESEARCH and s.technology_level.value <= sector.technology_level.value]
            if research_sectors:
                prerequisites.extend(research_sectors[:1])

        # Limit prerequisites to avoid circular dependencies
        return prerequisites[:5]  # Maximum 5 prerequisites

    def get_sector_count(self) -> int:
        """Get total number of sectors."""
        return len(self.sectors)

    def get_sector_names(self) -> List[str]:
        """Get list of all sector names."""
        return self.sector_names.copy()

    def get_sectors_by_technology_level(self, level: TechnologyLevel) -> List[SectorDefinition]:
        """Get all sectors at a specific technology level."""
        return [sector for sector in self.sectors.values() if sector.technology_level == level]

    def get_unlocked_sectors(self, developed_sectors: Set[int]) -> List[int]:
        """Get sectors that can be unlocked given the currently developed sectors."""
        unlocked = []
        for sector_id, prerequisites in self.prerequisites.items():
            if sector_id not in developed_sectors:
                if all(prereq in developed_sectors for prereq in prerequisites):
                    unlocked.append(sector_id)
        return unlocked

    def get_technology_tree_path(self, target_sector_id: int) -> List[int]:
        """Get the path of sectors needed to unlock a target sector."""
        if target_sector_id not in self.sectors:
            return []

        path = []
        to_process = [target_sector_id]
        processed = set()

        while to_process:
            current = to_process.pop(0)
            if current in processed:
                continue

            processed.add(current)
            path.append(current)

            # Add prerequisites
            for prereq in self.prerequisites.get(current, []):
                if prereq not in processed:
                    to_process.append(prereq)

        return path

    def export_technology_tree(self, filepath: str):
        """Export the technology tree to a JSON file."""
        tree_data = {
            'sectors': {
                str(sector_id): {
                    'name': sector.name,
                    'category': sector.category.value,
                    'subcategory': sector.subcategory,
                    'technology_level': sector.technology_level.value,
                    'prerequisites': sector.prerequisites,
                    'unlocks': sector.unlocks,
                    'description': sector.description,
                    'importance_weight': sector.importance_weight,
                    'economic_impact': sector.economic_impact,
                    'labor_intensity': sector.labor_intensity,
                    'capital_intensity': sector.capital_intensity,
                    'environmental_impact': sector.environmental_impact,
                    'development_cost': sector.development_cost,
                    'research_requirements': sector.research_requirements
                }
                for sector_id, sector in self.sectors.items()
            },
            'technology_tree': {str(k): v for k, v in self.technology_tree.items()},
            'prerequisites': {str(k): v for k, v in self.prerequisites.items()}
        }

        with open(filepath, 'w', encoding='utf - 8') as f:
            json.dump(tree_data, f, indent = 2, ensure_ascii = False)

if __name__ == "__main__":
    # Test the parser
    parser = SectorParser()
    sectors = parser.parse_sectors()

    print(f"Parsed {len(sectors)} sectors")
    print(f"Basic sectors: {len(parser.get_sectors_by_technology_level(TechnologyLevel.BASIC))}")
    print(f"Intermediate sectors: {len(parser.get_sectors_by_technology_level(TechnologyLevel.INTERMEDIATE))}")
    print(f"Advanced sectors: {len(parser.get_sectors_by_technology_level(TechnologyLevel.ADVANCED))}")
    print(f"Cutting - edge sectors: {len(parser.get_sectors_by_technology_level(TechnologyLevel.CUTTING_EDGE))}")
    print(f"Future sectors: {len(parser.get_sectors_by_technology_level(TechnologyLevel.FUTURE))}")

    # Export the technology tree
    parser.export_technology_tree("data / technology_tree.json")
    print("Technology tree exported to data / technology_tree.json")
