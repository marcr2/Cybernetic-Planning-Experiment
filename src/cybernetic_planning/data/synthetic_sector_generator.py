"""
Synthetic Sector Generation System for Economic Simulation

This module implements a dynamic sector generation system that:
1. Starts with the mandatory 6 core sectors from sectors.md
2. Subdivides them if user limit allows
3. Dynamically generates new sectors based on technological progress
4. Ensures all sector names come from sectors.md
5. Integrates with employment and economic planning systems
"""

import re
import random
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
from datetime import datetime, timedelta

class TechnologyLevel(Enum):
    """Technology development levels for sector unlocking."""
    BASIC = "basic"           # Available from start
    INTERMEDIATE = "intermediate"  # Requires basic sectors
    ADVANCED = "advanced"     # Requires intermediate sectors
    CUTTING_EDGE = "cutting_edge"  # Requires advanced sectors
    FUTURE = "future"         # Requires cutting-edge sectors

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
    employment_capacity: float = 0.0  # Number of workers this sector can employ
    is_core_sector: bool = False  # Whether this is one of the 6 core sectors
    is_subdivision: bool = False  # Whether this is a subdivision of a core sector
    parent_sector_id: Optional[int] = None  # ID of parent sector if subdivision
    creation_year: Optional[int] = None  # Year this sector was created
    technological_breakthrough_required: bool = False  # Whether this requires a breakthrough

@dataclass
class TechnologicalBreakthrough:
    """Represents a technological breakthrough that unlocks new sectors."""
    name: str
    description: str
    required_technology_level: TechnologyLevel
    required_research_investment: float
    unlocks_sectors: List[int]
    breakthrough_year: Optional[int] = None
    is_achieved: bool = False

class SyntheticSectorGenerator:
    """
    Generates and manages synthetic economic sectors dynamically.
    
    Features:
    - Starts with 6 mandatory core sectors from sectors.md
    - Subdivides core sectors if user limit allows
    - Dynamically creates new sectors based on technological progress
    - All sector names sourced from sectors.md
    - Integrates with employment and economic planning
    """
    
    def __init__(self, 
                 sectors_file_path: Optional[str] = None,
                 max_sectors: int = 1000,
                 min_sectors: int = 6,
                 starting_technology_level: float = 0.0):
        """
        Initialize the synthetic sector generator.
        
        Args:
            sectors_file_path: Path to sectors.md file
            max_sectors: Maximum number of sectors (1-1000)
            min_sectors: Minimum number of sectors (6-1000)
            starting_technology_level: Starting technology level (0.0-1.0)
        """
        # Validate constraints
        if max_sectors < min_sectors:
            raise ValueError("max_sectors must be >= min_sectors")
        if min_sectors < 6:
            raise ValueError("min_sectors must be >= 6")
        if max_sectors > 1000:
            raise ValueError("max_sectors must be <= 1000")
        if not 0.0 <= starting_technology_level <= 1.0:
            raise ValueError("starting_technology_level must be between 0.0 and 1.0")
            
        self.max_sectors = max_sectors
        self.min_sectors = min_sectors
        self.starting_technology_level = starting_technology_level
        
        # Set up sectors file path
        if sectors_file_path is None:
            current_dir = Path(__file__).parent
            self.sectors_file_path = current_dir / "sectors.md"
        else:
            self.sectors_file_path = Path(sectors_file_path)
            
        # Initialize data structures
        self.sectors: Dict[int, SectorDefinition] = {}
        self.sector_names: List[str] = []
        self.available_sector_names: List[str] = []  # All names from sectors.md
        self.core_sector_names: List[str] = []  # Names of sectors 1-6
        self.subdivision_sector_names: Dict[str, List[str]] = {}  # Core sector -> subdivisions
        self.technology_tree: Dict[int, List[int]] = {}  # sector_id -> unlocked_sectors
        self.prerequisites: Dict[int, List[int]] = {}    # sector_id -> required_sectors
        
        # Simulation state
        self.current_year: int = 2024
        self.technological_level: float = self.starting_technology_level  # 0.0 to 1.0
        self.research_investment: float = 0.0
        self.total_research_spent: float = 0.0
        self.technological_breakthroughs: List[TechnologicalBreakthrough] = []
        self.achieved_breakthroughs: Set[str] = set()
        
        # Employment integration
        self.total_population: float = 1000000.0  # Default population
        self.employment_by_sector: Dict[int, float] = {}
        self.unemployment_rate: float = 0.05  # 5% unemployment
        
        # Load and parse sectors.md
        self._load_sector_names()
        
        # Generate initial sectors
        self._generate_initial_sectors()
        
        # Set up technological breakthroughs
        self._setup_technological_breakthroughs()
    
    def _load_sector_names(self):
        """Load all sector names from sectors.md file."""
        if not self.sectors_file_path.exists():
            raise FileNotFoundError(f"Sectors file not found: {self.sectors_file_path}")
            
        with open(self.sectors_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract sectors using regex
        sector_pattern = r'(\d+)\.\s+(.+?)(?=\n\d+\.|\n\n|$)'
        matches = re.findall(sector_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            number_str, name_and_desc = match
            sector_number = int(number_str)
            
            # Clean up the name
            name = name_and_desc.strip()
            if '\n' in name:
                name = name.split('\n')[0].strip()
                
            self.available_sector_names.append(name)
            
            # Identify core sectors (1-6)
            if 1 <= sector_number <= 6:
                self.core_sector_names.append(name)
                
            # Identify subdivisions by category
            self._categorize_sector_for_subdivision(sector_number, name)
    
    def _categorize_sector_for_subdivision(self, sector_number: int, name: str):
        """Categorize sectors for subdivision purposes."""
        # Map sector numbers to core sectors
        core_mapping = {
            1: "Healthcare",
            2: "Food and Agriculture", 
            3: "Energy",
            4: "Housing and Construction",
            5: "Education",
            6: "Transportation"
        }
        
        # Healthcare subdivisions (7-25)
        if 7 <= sector_number <= 25:
            core_sector = "Healthcare"
        # Agriculture subdivisions (26-50)
        elif 26 <= sector_number <= 50:
            core_sector = "Food and Agriculture"
        # Energy subdivisions (51-75)
        elif 51 <= sector_number <= 75:
            core_sector = "Energy"
        # Construction subdivisions (76-100)
        elif 76 <= sector_number <= 100:
            core_sector = "Housing and Construction"
        # Education subdivisions (101-125)
        elif 101 <= sector_number <= 125:
            core_sector = "Education"
        # Transportation subdivisions (126-150)
        elif 126 <= sector_number <= 150:
            core_sector = "Transportation"
        else:
            return
            
        if core_sector not in self.subdivision_sector_names:
            self.subdivision_sector_names[core_sector] = []
        self.subdivision_sector_names[core_sector].append(name)
    
    def _generate_initial_sectors(self):
        """Generate the initial set of sectors based on constraints."""
        sector_id = 0
        
        # Step 1: Create the 6 mandatory core sectors
        for i, core_name in enumerate(self.core_sector_names):
            sector = self._create_sector_definition(
                sector_id=sector_id,
                name=core_name,
                is_core_sector=True,
                technology_level=TechnologyLevel.BASIC
            )
            self.sectors[sector_id] = sector
            self.sector_names.append(core_name)
            sector_id += 1
        
        # Step 2: Subdivide core sectors if space allows
        remaining_slots = self.max_sectors - sector_id
        if remaining_slots > 0:
            sector_id = self._subdivide_core_sectors(sector_id, remaining_slots)
        
        # Step 3: Add additional sectors to reach max_sectors limit
        if len(self.sectors) < self.max_sectors:
            additional_needed = self.max_sectors - len(self.sectors)
            sector_id = self._add_additional_sectors(sector_id, additional_needed)
        
        # Build technology tree dependencies
        self._build_technology_tree()
        
        # Initialize employment
        self._initialize_employment()
    
    def _subdivide_core_sectors(self, start_sector_id: int, max_slots: int) -> int:
        """Subdivide core sectors if space allows."""
        sector_id = start_sector_id
        slots_used = 0
        
        # Calculate how many subdivisions we can add per core sector
        subdivisions_per_core = max_slots // len(self.core_sector_names)
        
        for core_sector_name in self.core_sector_names:
            if slots_used >= max_slots:
                break
                
            # Get subdivisions for this core sector
            subdivisions = self.subdivision_sector_names.get(core_sector_name, [])
            
            # Add up to subdivisions_per_core subdivisions
            subdivisions_to_add = min(subdivisions_per_core, len(subdivisions), max_slots - slots_used)
            
            for i in range(subdivisions_to_add):
                if slots_used >= max_slots:
                    break
                    
                subdivision_name = subdivisions[i]
                
                # Find parent sector ID
                parent_sector_id = None
                for sid, sector in self.sectors.items():
                    if sector.name == core_sector_name:
                        parent_sector_id = sid
                        break
                
                sector = self._create_sector_definition(
                    sector_id=sector_id,
                    name=subdivision_name,
                    is_core_sector=False,
                    is_subdivision=True,
                    parent_sector_id=parent_sector_id,
                    technology_level=TechnologyLevel.INTERMEDIATE
                )
                
                self.sectors[sector_id] = sector
                self.sector_names.append(subdivision_name)
                sector_id += 1
                slots_used += 1
        
        return sector_id
    
    def _add_additional_sectors(self, start_sector_id: int, count: int) -> int:
        """Add additional sectors to reach minimum count."""
        sector_id = start_sector_id
        
        # Get remaining sector names (excluding core and subdivisions already used)
        used_names = set(self.sector_names)
        available_names = [name for name in self.available_sector_names if name not in used_names]
        
        # Add sectors up to the count needed
        for i in range(min(count, len(available_names))):
            sector_name = available_names[i]
            
            sector = self._create_sector_definition(
                sector_id=sector_id,
                name=sector_name,
                is_core_sector=False,
                is_subdivision=False,
                technology_level=self._determine_technology_level_by_name(sector_name)
            )
            
            self.sectors[sector_id] = sector
            self.sector_names.append(sector_name)
            sector_id += 1
        
        return sector_id
    
    def _create_sector_definition(self, 
                                 sector_id: int, 
                                 name: str,
                                 is_core_sector: bool = False,
                                 is_subdivision: bool = False,
                                 parent_sector_id: Optional[int] = None,
                                 technology_level: Optional[TechnologyLevel] = None) -> SectorDefinition:
        """Create a sector definition with appropriate properties."""
        
        if technology_level is None:
            technology_level = self._determine_technology_level_by_name(name)
        
        category = self._determine_category_by_name(name)
        subcategory = self._determine_subcategory_by_name(name, category)
        
        return SectorDefinition(
            id=sector_id,
            name=name,
            category=category,
            subcategory=subcategory,
            technology_level=technology_level,
            prerequisites=[],  # Will be filled later
            unlocks=[],        # Will be filled later
            description=f"Economic sector: {name}",
            importance_weight=self._calculate_importance_weight(name, is_core_sector),
            economic_impact=self._assess_economic_impact(name),
            labor_intensity=self._assess_labor_intensity(name),
            capital_intensity=self._assess_capital_intensity(name),
            environmental_impact=self._assess_environmental_impact(name),
            development_cost=self._calculate_development_cost(technology_level),
            research_requirements=self._determine_research_requirements(name, technology_level),
            employment_capacity=self._calculate_employment_capacity(name, is_core_sector),
            is_core_sector=is_core_sector,
            is_subdivision=is_subdivision,
            parent_sector_id=parent_sector_id,
            creation_year=self.current_year if not is_core_sector else None,
            technological_breakthrough_required=self._requires_breakthrough(name, technology_level)
        )
    
    def _determine_technology_level_by_name(self, name: str) -> TechnologyLevel:
        """Determine technology level based on sector name."""
        name_lower = name.lower()
        
        # Check for advanced technology keywords
        if any(term in name_lower for term in ['quantum', 'ai', 'artificial intelligence', 'blockchain', 'cryptocurrency', 'nanotechnology']):
            return TechnologyLevel.CUTTING_EDGE
        elif any(term in name_lower for term in ['research', 'development', 'innovation', 'technology', 'digital']):
            return TechnologyLevel.ADVANCED
        elif any(term in name_lower for term in ['manufacturing', 'production', 'processing']):
            return TechnologyLevel.INTERMEDIATE
        else:
            return TechnologyLevel.BASIC
    
    def _determine_category_by_name(self, name: str) -> SectorCategory:
        """Determine category based on sector name."""
        name_lower = name.lower()
        
        if any(term in name_lower for term in ['health', 'medical', 'pharmaceutical', 'hospital', 'clinic']):
            return SectorCategory.HEALTHCARE
        elif any(term in name_lower for term in ['food', 'agriculture', 'farming', 'crop', 'livestock']):
            return SectorCategory.AGRICULTURE
        elif any(term in name_lower for term in ['energy', 'power', 'electric', 'solar', 'wind', 'nuclear']):
            return SectorCategory.ENERGY
        elif any(term in name_lower for term in ['housing', 'construction', 'building', 'real estate']):
            return SectorCategory.CONSTRUCTION
        elif any(term in name_lower for term in ['education', 'school', 'university', 'training']):
            return SectorCategory.EDUCATION
        elif any(term in name_lower for term in ['transportation', 'transport', 'shipping', 'aviation', 'rail']):
            return SectorCategory.TRANSPORTATION
        elif any(term in name_lower for term in ['manufacturing', 'production', 'factory']):
            return SectorCategory.MANUFACTURING
        elif any(term in name_lower for term in ['technology', 'software', 'digital', 'ai', 'tech']):
            return SectorCategory.TECHNOLOGY
        elif any(term in name_lower for term in ['service', 'retail', 'consulting']):
            return SectorCategory.SERVICES
        elif any(term in name_lower for term in ['research', 'development', 'innovation']):
            return SectorCategory.RESEARCH
        elif any(term in name_lower for term in ['infrastructure', 'utility', 'telecommunications']):
            return SectorCategory.INFRASTRUCTURE
        elif any(term in name_lower for term in ['financial', 'banking', 'insurance', 'investment']):
            return SectorCategory.FINANCE
        elif any(term in name_lower for term in ['entertainment', 'media', 'gaming', 'sports']):
            return SectorCategory.ENTERTAINMENT
        elif any(term in name_lower for term in ['government', 'public', 'administration']):
            return SectorCategory.GOVERNMENT
        elif any(term in name_lower for term in ['environmental', 'renewable', 'sustainability']):
            return SectorCategory.ENVIRONMENTAL
        else:
            return SectorCategory.SERVICES
    
    def _determine_subcategory_by_name(self, name: str, category: SectorCategory) -> str:
        """Determine subcategory based on sector name and category."""
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
        else:
            return 'general'
    
    def _calculate_importance_weight(self, name: str, is_core_sector: bool) -> float:
        """Calculate importance weight based on sector characteristics."""
        if is_core_sector:
            return 0.9
        
        base_weight = 0.5
        
        # Technology sectors are important
        if any(term in name.lower() for term in ['technology', 'ai', 'quantum', 'blockchain']):
            base_weight = max(base_weight, 0.8)
        
        # Research sectors are important
        if any(term in name.lower() for term in ['research', 'development', 'innovation']):
            base_weight = max(base_weight, 0.7)
        
        # Add some randomness
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
    
    def _calculate_employment_capacity(self, name: str, is_core_sector: bool) -> float:
        """Calculate employment capacity for a sector."""
        # Base capacity
        if is_core_sector:
            base_capacity = self.total_population * 0.15  # Core sectors employ 15% each
        else:
            base_capacity = self.total_population * 0.01  # Other sectors employ 1% each
        
        # Adjust based on labor intensity
        name_lower = name.lower()
        if any(term in name_lower for term in ['services', 'healthcare', 'education', 'retail']):
            base_capacity *= 1.5  # High labor intensity
        elif any(term in name_lower for term in ['technology', 'research', 'ai', 'quantum']):
            base_capacity *= 0.5  # Low labor intensity
        
        return max(100.0, base_capacity)  # Minimum 100 workers
    
    def _requires_breakthrough(self, name: str, tech_level: TechnologyLevel) -> bool:
        """Determine if a sector requires a technological breakthrough."""
        if tech_level in [TechnologyLevel.CUTTING_EDGE, TechnologyLevel.FUTURE]:
            return True
        
        name_lower = name.lower()
        breakthrough_keywords = ['quantum', 'ai', 'artificial intelligence', 'blockchain', 'nanotechnology', 'space']
        return any(term in name_lower for term in breakthrough_keywords)
    
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
        
        # Core sectors have no prerequisites
        if sector.is_core_sector:
            return prerequisites
        
        # Subdivisions depend on their parent
        if sector.is_subdivision and sector.parent_sector_id is not None:
            prerequisites.append(sector.parent_sector_id)
        
        # Technology level-based prerequisites
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
            # Need cutting-edge sectors from the same category
            cutting_edge_sectors = [s_id for s_id, s in self.sectors.items()
                                 if s.technology_level == TechnologyLevel.CUTTING_EDGE and s.category == sector.category]
            if cutting_edge_sectors:
                prerequisites.extend(cutting_edge_sectors[:2])  # Need 2 cutting-edge sectors
        
        # Limit prerequisites to avoid circular dependencies
        return prerequisites[:5]  # Maximum 5 prerequisites
    
    def _initialize_employment(self):
        """Initialize employment distribution across sectors."""
        total_employment_capacity = sum(sector.employment_capacity for sector in self.sectors.values())
        available_workers = self.total_population * (1 - self.unemployment_rate)
        
        # Scale employment capacity to match available workers
        if total_employment_capacity > 0:
            scale_factor = available_workers / total_employment_capacity
            for sector_id, sector in self.sectors.items():
                self.employment_by_sector[sector_id] = sector.employment_capacity * scale_factor
        else:
            # Distribute workers equally among sectors
            workers_per_sector = available_workers / len(self.sectors)
            for sector_id in self.sectors:
                self.employment_by_sector[sector_id] = workers_per_sector
    
    def _setup_technological_breakthroughs(self):
        """Set up technological breakthroughs that unlock new sectors."""
        self.technological_breakthroughs = [
            TechnologicalBreakthrough(
                name="Artificial Intelligence Revolution",
                description="Breakthrough in AI and machine learning technologies",
                required_technology_level=TechnologyLevel.ADVANCED,
                required_research_investment=1000000.0,
                unlocks_sectors=[]
            ),
            TechnologicalBreakthrough(
                name="Quantum Computing Breakthrough",
                description="Achievement of practical quantum computing",
                required_technology_level=TechnologyLevel.CUTTING_EDGE,
                required_research_investment=5000000.0,
                unlocks_sectors=[]
            ),
            TechnologicalBreakthrough(
                name="Space Technology Revolution",
                description="Advanced space exploration and colonization technologies",
                required_technology_level=TechnologyLevel.CUTTING_EDGE,
                required_research_investment=2000000.0,
                unlocks_sectors=[]
            ),
            TechnologicalBreakthrough(
                name="Biotechnology Revolution",
                description="Advanced biotechnology and genetic engineering",
                required_technology_level=TechnologyLevel.ADVANCED,
                required_research_investment=1500000.0,
                unlocks_sectors=[]
            ),
            TechnologicalBreakthrough(
                name="Renewable Energy Revolution",
                description="Breakthrough in renewable energy technologies",
                required_technology_level=TechnologyLevel.ADVANCED,
                required_research_investment=800000.0,
                unlocks_sectors=[]
            )
        ]
    
    # Public API methods
    
    def advance_simulation_year(self, research_investment: float = 0.0):
        """Advance the simulation by one year and check for technological breakthroughs."""
        self.current_year += 1
        self.research_investment = research_investment
        self.total_research_spent += research_investment
        
        # Update technological level based on research investment
        self.technological_level = min(1.0, self.technological_level + (research_investment / 10000000.0))
        
        # Check for technological breakthroughs
        self._check_technological_breakthroughs()
        
        # Generate new sectors if breakthroughs occurred
        self._generate_new_sectors_from_breakthroughs()
    
    def _check_technological_breakthroughs(self):
        """Check if any technological breakthroughs have been achieved."""
        for breakthrough in self.technological_breakthroughs:
            if breakthrough.is_achieved:
                continue
                
            # Check if conditions are met
            tech_level_met = self._get_tech_level_value() >= self._get_tech_level_value(breakthrough.required_technology_level)
            research_met = self.total_research_spent >= breakthrough.required_research_investment
            
            if tech_level_met and research_met:
                breakthrough.is_achieved = True
                breakthrough.breakthrough_year = self.current_year
                self.achieved_breakthroughs.add(breakthrough.name)
                print(f"Technological breakthrough achieved: {breakthrough.name} in {self.current_year}")
    
    def _get_tech_level_value(self, tech_level: Optional[TechnologyLevel] = None) -> float:
        """Get numeric value for technology level."""
        if tech_level is None:
            return self.technological_level
            
        values = {
            TechnologyLevel.BASIC: 0.0,
            TechnologyLevel.INTERMEDIATE: 0.25,
            TechnologyLevel.ADVANCED: 0.5,
            TechnologyLevel.CUTTING_EDGE: 0.75,
            TechnologyLevel.FUTURE: 1.0
        }
        return values[tech_level]
    
    def _generate_new_sectors_from_breakthroughs(self):
        """Generate new sectors based on achieved breakthroughs."""
        if len(self.sectors) >= self.max_sectors:
            return  # Can't add more sectors
            
        # Find sectors that require breakthroughs and can now be unlocked
        breakthrough_sectors = []
        for sector_id, sector in self.sectors.items():
            if (sector.technological_breakthrough_required and 
                not sector.is_core_sector and 
                sector.creation_year is None):
                breakthrough_sectors.append(sector_id)
        
        # Add new sectors based on breakthroughs
        for breakthrough_name in self.achieved_breakthroughs:
            if len(self.sectors) >= self.max_sectors:
                break
                
            # Find available sector names that match this breakthrough
            matching_names = self._find_sectors_for_breakthrough(breakthrough_name)
            
            for name in matching_names:
                if len(self.sectors) >= self.max_sectors:
                    break
                    
                # Check if this sector name is already used
                if name in self.sector_names:
                    continue
                
                # Create new sector
                new_sector_id = max(self.sectors.keys()) + 1
                sector = self._create_sector_definition(
                    sector_id=new_sector_id,
                    name=name,
                    is_core_sector=False,
                    is_subdivision=False,
                    technology_level=self._determine_technology_level_by_name(name)
                )
                
                self.sectors[new_sector_id] = sector
                self.sector_names.append(name)
                
                # Update employment
                self.employment_by_sector[new_sector_id] = sector.employment_capacity
                
                print(f"New sector created: {name} (ID: {new_sector_id}) in {self.current_year}")
    
    def _find_sectors_for_breakthrough(self, breakthrough_name: str) -> List[str]:
        """Find sector names that match a technological breakthrough."""
        breakthrough_keywords = {
            "Artificial Intelligence Revolution": ['ai', 'artificial', 'machine learning', 'neural', 'intelligent'],
            "Quantum Computing Breakthrough": ['quantum', 'computing', 'quantum computing'],
            "Space Technology Revolution": ['space', 'satellite', 'aerospace', 'spacecraft'],
            "Biotechnology Revolution": ['biotech', 'biotechnology', 'genetic', 'pharmaceutical'],
            "Renewable Energy Revolution": ['renewable', 'solar', 'wind', 'clean energy']
        }
        
        keywords = breakthrough_keywords.get(breakthrough_name, [])
        matching_names = []
        
        for name in self.available_sector_names:
            if name in self.sector_names:  # Already used
                continue
                
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in keywords):
                matching_names.append(name)
        
        return matching_names[:5]  # Limit to 5 new sectors per breakthrough
    
    def get_sector_count(self) -> int:
        """Get total number of sectors."""
        return len(self.sectors)
    
    def get_sector_names(self) -> List[str]:
        """Get list of all sector names."""
        return self.sector_names.copy()
    
    def get_core_sectors(self) -> List[SectorDefinition]:
        """Get the 6 core sectors."""
        return [sector for sector in self.sectors.values() if sector.is_core_sector]
    
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
    
    def get_available_sectors_by_tech_level(self) -> Dict[TechnologyLevel, List[int]]:
        """Get sectors available at current technology level."""
        available = {}
        
        for tech_level in TechnologyLevel:
            sectors_at_level = []
            for sector_id, sector in self.sectors.items():
                if sector.technology_level == tech_level:
                    # Check if technology level allows this sector
                    if self._is_sector_available_at_tech_level(sector, tech_level):
                        sectors_at_level.append(sector_id)
            available[tech_level] = sectors_at_level
        
        return available
    
    def _is_sector_available_at_tech_level(self, sector: SectorDefinition, tech_level: TechnologyLevel) -> bool:
        """Check if a sector is available at the current technology level."""
        tech_level_thresholds = {
            TechnologyLevel.BASIC: 0.0,
            TechnologyLevel.INTERMEDIATE: 0.2,
            TechnologyLevel.ADVANCED: 0.5,
            TechnologyLevel.CUTTING_EDGE: 0.8,
            TechnologyLevel.FUTURE: 0.95
        }
        
        threshold = tech_level_thresholds.get(tech_level, 1.0)
        return self.technological_level >= threshold
    
    def get_technology_tree_visualization(self) -> Dict[str, Any]:
        """Get technology tree visualization data."""
        available_by_level = self.get_available_sectors_by_tech_level()
        
        # Create nodes for each sector
        nodes = []
        for sector_id, sector in self.sectors.items():
            is_available = self._is_sector_available_at_tech_level(sector, sector.technology_level)
            is_core = sector.is_core_sector
            
            node = {
                "id": sector_id,
                "name": sector.name,
                "technology_level": sector.technology_level.value,
                "is_available": is_available,
                "is_core": is_core,
                "category": sector.category.value if hasattr(sector.category, 'value') else str(sector.category),
                "prerequisites": sector.prerequisites,
                "importance_weight": sector.importance_weight,
                "economic_impact": sector.economic_impact
            }
            nodes.append(node)
        
        # Create edges for prerequisites
        edges = []
        for sector_id, sector in self.sectors.items():
            for prereq_id in sector.prerequisites:
                edges.append({
                    "from": prereq_id,
                    "to": sector_id,
                    "type": "prerequisite"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "technology_level": self.technological_level,
            "available_by_level": {
                level.value: sectors for level, sectors in available_by_level.items()
            },
            "total_sectors": len(self.sectors),
            "available_sectors": sum(len(sectors) for sectors in available_by_level.values())
        }
    
    def get_employment_by_sector(self) -> Dict[int, float]:
        """Get employment distribution by sector."""
        return self.employment_by_sector.copy()
    
    def get_total_employment(self) -> float:
        """Get total employment across all sectors."""
        return sum(self.employment_by_sector.values())
    
    def get_unemployment_rate(self) -> float:
        """Get current unemployment rate."""
        total_employment = self.get_total_employment()
        if self.total_population > 0:
            return 1.0 - (total_employment / self.total_population)
        return 0.0
    
    def update_population(self, new_population: float):
        """Update total population and redistribute employment."""
        self.total_population = new_population
        
        # Redistribute employment proportionally
        total_current_employment = sum(self.employment_by_sector.values())
        if total_current_employment > 0:
            scale_factor = (new_population * (1 - self.unemployment_rate)) / total_current_employment
            for sector_id in self.employment_by_sector:
                self.employment_by_sector[sector_id] *= scale_factor
    
    def export_sector_data(self) -> Dict[str, Any]:
        """Export sector data for use in other modules."""
        return {
            "sectors": {
                str(sector_id): {
                    "id": sector.id,
                    "name": sector.name,
                    "category": sector.category.value,
                    "subcategory": sector.subcategory,
                    "technology_level": sector.technology_level.value,
                    "prerequisites": sector.prerequisites,
                    "unlocks": sector.unlocks,
                    "description": sector.description,
                    "importance_weight": sector.importance_weight,
                    "economic_impact": sector.economic_impact,
                    "labor_intensity": sector.labor_intensity,
                    "capital_intensity": sector.capital_intensity,
                    "environmental_impact": sector.environmental_impact,
                    "development_cost": sector.development_cost,
                    "research_requirements": sector.research_requirements,
                    "employment_capacity": sector.employment_capacity,
                    "is_core_sector": sector.is_core_sector,
                    "is_subdivision": sector.is_subdivision,
                    "parent_sector_id": sector.parent_sector_id,
                    "creation_year": sector.creation_year,
                    "technological_breakthrough_required": sector.technological_breakthrough_required
                }
                for sector_id, sector in self.sectors.items()
            },
            "technology_tree": {str(k): v for k, v in self.technology_tree.items()},
            "prerequisites": {str(k): v for k, v in self.prerequisites.items()},
            "employment_by_sector": {str(k): v for k, v in self.employment_by_sector.items()},
            "simulation_state": {
                "current_year": self.current_year,
                "technological_level": self.technological_level,
                "total_research_spent": self.total_research_spent,
                "total_population": self.total_population,
                "unemployment_rate": self.get_unemployment_rate(),
                "achieved_breakthroughs": list(self.achieved_breakthroughs)
            },
            "sector_names": self.sector_names,
            "total_sectors": len(self.sectors)
        }
    
    def print_sector_summary(self):
        """Print a summary of the sector generation."""
        print(f"Synthetic Sector Generation Summary")
        print(f"Total Sectors: {len(self.sectors)}")
        print(f"Core Sectors: {len(self.get_core_sectors())}")
        print(f"Current Year: {self.current_year}")
        print(f"Technological Level: {self.technological_level:.2f}")
        print(f"Total Research Spent: ${self.total_research_spent:,.0f}")
        print(f"Population: {self.total_population:,.0f}")
        print(f"Total Employment: {self.get_total_employment():,.0f}")
        print(f"Unemployment Rate: {self.get_unemployment_rate():.1%}")
        print()
        
        print("Core Sectors:")
        for sector in self.get_core_sectors():
            print(f"  {sector.name}: {sector.description}")
        
        print(f"\nSector Distribution by Technology Level:")
        tech_level_counts = {}
        for sector in self.sectors.values():
            tech_level_counts[sector.technology_level.value] = tech_level_counts.get(sector.technology_level.value, 0) + 1
        
        for tech_level, count in sorted(tech_level_counts.items()):
            print(f"  {tech_level}: {count} sectors")
        
        if self.achieved_breakthroughs:
            print(f"\nAchieved Technological Breakthroughs:")
            for breakthrough in self.achieved_breakthroughs:
                print(f"  - {breakthrough}")
        
        print(f"\nEmployment Distribution (Top 10 Sectors):")
        sorted_employment = sorted(self.employment_by_sector.items(), key=lambda x: x[1], reverse=True)
        for sector_id, employment in sorted_employment[:10]:
            sector_name = self.sectors[sector_id].name
            print(f"  {sector_name}: {employment:,.0f} workers")

if __name__ == "__main__":
    # Test the synthetic sector generator
    generator = SyntheticSectorGenerator(max_sectors=50, min_sectors=6)
    
    print("Initial Sector Generation:")
    generator.print_sector_summary()
    
    print("\n" + "="*50)
    print("Simulating 5 years with research investment...")
    
    # Simulate 5 years with increasing research investment
    for year in range(5):
        research_investment = 500000 * (year + 1)  # Increasing investment
        generator.advance_simulation_year(research_investment)
        print(f"\nYear {generator.current_year}:")
        print(f"Research Investment: ${research_investment:,.0f}")
        print(f"Technological Level: {generator.technological_level:.2f}")
        print(f"Total Sectors: {generator.get_sector_count()}")
    
    print("\n" + "="*50)
    print("Final Sector Generation:")
    generator.print_sector_summary()
