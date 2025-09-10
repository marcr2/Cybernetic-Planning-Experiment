"""
Technology Tree Sector Mapper

Creates a comprehensive mapping of economic sectors based on a technology tree
structure, allowing for progressive unlocking of sectors as technology develops.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .sector_parser import SectorParser, SectorDefinition, TechnologyLevel, SectorCategory


class DevelopmentStage(Enum):
    """Stages of economic development."""
    AGRARIAN = "agrarian"           # Basic agriculture and simple manufacturing
    INDUSTRIAL = "industrial"       # Heavy industry and basic technology
    POST_INDUSTRIAL = "post_industrial"  # Services and advanced technology
    KNOWLEDGE = "knowledge"         # High-tech and research-intensive
    FUTURE = "future"              # Cutting-edge and speculative technology


@dataclass
class TechnologyNode:
    """A node in the technology tree representing a sector."""
    sector: SectorDefinition
    is_unlocked: bool = False
    development_progress: float = 0.0  # 0.0 to 1.0
    research_investment: float = 0.0
    production_capacity: float = 0.0


class TechnologyTreeMapper:
    """
    Maps economic sectors using a technology tree structure.
    
    Features:
    - Progressive sector unlocking based on prerequisites
    - Technology development stages
    - Research investment requirements
    - Sector interdependencies
    - Development cost calculations
    """
    
    def __init__(self, sectors_file_path: str = "src/cybernetic_planning/data/sectors.md"):
        self.parser = SectorParser(sectors_file_path)
        self.sectors: Dict[int, SectorDefinition] = {}
        self.technology_nodes: Dict[int, TechnologyNode] = {}
        self.sector_names: List[str] = []
        self.current_development_stage = DevelopmentStage.AGRARIAN
        self.developed_sectors: Set[int] = set()
        self.research_budget: float = 1000.0
        self.total_research_investment: float = 0.0
        
        # Initialize the technology tree
        self._initialize_technology_tree()
    
    def _initialize_technology_tree(self):
        """Initialize the technology tree from the sector parser."""
        self.sectors = self.parser.parse_sectors()
        self.sector_names = self.parser.get_sector_names()
        
        # Create technology nodes
        for sector_id, sector in self.sectors.items():
            node = TechnologyNode(sector=sector)
            self.technology_nodes[sector_id] = node
        
        # Unlock basic sectors initially
        self._unlock_basic_sectors()
    
    def _unlock_basic_sectors(self):
        """Unlock all basic technology level sectors."""
        basic_sectors = self.parser.get_sectors_by_technology_level(TechnologyLevel.BASIC)
        for sector in basic_sectors:
            self._unlock_sector(sector.id)
    
    def _unlock_sector(self, sector_id: int):
        """Unlock a specific sector."""
        if sector_id in self.technology_nodes:
            self.technology_nodes[sector_id].is_unlocked = True
            self.technology_nodes[sector_id].development_progress = 1.0
            self.developed_sectors.add(sector_id)
    
    def get_available_sectors(self, max_sectors: int = 1000) -> List[int]:
        """Get available sectors up to the specified limit."""
        # Start with basic sectors
        available = []
        
        # Add basic sectors first (all of them, not just 6)
        basic_sectors = [s.id for s in self.parser.get_sectors_by_technology_level(TechnologyLevel.BASIC)]
        available.extend(basic_sectors[:max_sectors])
        
        # Add intermediate sectors
        if len(available) < max_sectors:
            intermediate_sectors = [s.id for s in self.parser.get_sectors_by_technology_level(TechnologyLevel.INTERMEDIATE)]
            available.extend(intermediate_sectors[:max_sectors - len(available)])
        
        # Add advanced sectors
        if len(available) < max_sectors:
            advanced_sectors = [s.id for s in self.parser.get_sectors_by_technology_level(TechnologyLevel.ADVANCED)]
            available.extend(advanced_sectors[:max_sectors - len(available)])
        
        # Add cutting-edge sectors
        if len(available) < max_sectors:
            cutting_edge_sectors = [s.id for s in self.parser.get_sectors_by_technology_level(TechnologyLevel.CUTTING_EDGE)]
            available.extend(cutting_edge_sectors[:max_sectors - len(available)])
        
        # Add future sectors
        if len(available) < max_sectors:
            future_sectors = [s.id for s in self.parser.get_sectors_by_technology_level(TechnologyLevel.FUTURE)]
            available.extend(future_sectors[:max_sectors - len(available)])
        
        return available[:max_sectors]
    
    def get_sector_count(self) -> int:
        """Get total number of sectors."""
        return len(self.sectors)
    
    def get_sector_names(self) -> List[str]:
        """Get list of all sector names."""
        return self.sector_names.copy()
    
    def get_sector_names_for_sectors(self, sector_ids: List[int]) -> List[str]:
        """Get sector names for specific sector IDs."""
        return [self.sectors[sector_id].name for sector_id in sector_ids if sector_id in self.sectors]
    
    def get_importance_weights(self) -> np.ndarray:
        """Get importance weights for all sectors."""
        weights = np.zeros(len(self.sectors))
        for sector_id, sector in self.sectors.items():
            weights[sector_id] = sector.importance_weight
        return weights
    
    def get_importance_weights_for_sectors(self, sector_ids: List[int]) -> np.ndarray:
        """Get importance weights for specific sector IDs."""
        weights = np.zeros(len(sector_ids))
        for i, sector_id in enumerate(sector_ids):
            if sector_id in self.sectors:
                weights[i] = self.sectors[sector_id].importance_weight
        return weights
    
    def get_economic_impact_matrix(self) -> np.ndarray:
        """Get economic impact matrix for sector interactions."""
        n_sectors = len(self.sectors)
        impact_matrix = np.zeros((n_sectors, n_sectors))
        
        for sector_id, sector in self.sectors.items():
            # Set self-impact
            impact_matrix[sector_id, sector_id] = 1.0
            
            # Set impact on prerequisite sectors
            for prereq_id in sector.prerequisites:
                if prereq_id < n_sectors:
                    impact_matrix[sector_id, prereq_id] = 0.3
            
            # Set impact on sectors this unlocks
            for unlock_id in sector.unlocks:
                if unlock_id < n_sectors:
                    impact_matrix[unlock_id, sector_id] = 0.5
        
        return impact_matrix
    
    def get_technology_dependency_matrix(self, sector_ids: List[int]) -> np.ndarray:
        """Get technology dependency matrix for specific sectors."""
        n_sectors = len(sector_ids)
        dependency_matrix = np.zeros((n_sectors, n_sectors))
        
        # Create mapping from sector IDs to indices
        id_to_index = {sector_id: i for i, sector_id in enumerate(sector_ids)}
        
        for i, sector_id in enumerate(sector_ids):
            if sector_id in self.sectors:
                sector = self.sectors[sector_id]
                
                # Set dependencies
                for prereq_id in sector.prerequisites:
                    if prereq_id in id_to_index:
                        prereq_index = id_to_index[prereq_id]
                        dependency_matrix[i, prereq_index] = 1.0
        
        return dependency_matrix
    
    def get_development_costs(self, sector_ids: List[int]) -> np.ndarray:
        """Get development costs for specific sectors."""
        costs = np.zeros(len(sector_ids))
        for i, sector_id in enumerate(sector_ids):
            if sector_id in self.sectors:
                costs[i] = self.sectors[sector_id].development_cost
        return costs
    
    def get_research_requirements(self, sector_ids: List[int]) -> Dict[int, List[str]]:
        """Get research requirements for specific sectors."""
        requirements = {}
        for sector_id in sector_ids:
            if sector_id in self.sectors:
                requirements[sector_id] = self.sectors[sector_id].research_requirements
        return requirements
    
    def can_unlock_sector(self, sector_id: int) -> bool:
        """Check if a sector can be unlocked given current development."""
        if sector_id not in self.sectors:
            return False
        
        sector = self.sectors[sector_id]
        return all(prereq in self.developed_sectors for prereq in sector.prerequisites)
    
    def get_unlockable_sectors(self) -> List[int]:
        """Get sectors that can be unlocked with current development."""
        return self.parser.get_unlocked_sectors(self.developed_sectors)
    
    def invest_in_research(self, sector_id: int, amount: float) -> bool:
        """Invest research funding in a sector."""
        if sector_id not in self.technology_nodes:
            return False
        
        if amount > self.research_budget:
            return False
        
        node = self.technology_nodes[sector_id]
        sector = node.sector
        
        # Calculate required research investment
        required_investment = sector.development_cost * 0.1  # 10% of development cost
        
        node.research_investment += amount
        self.research_budget -= amount
        self.total_research_investment += amount
        
        # Check if sector can be unlocked
        if (node.research_investment >= required_investment and 
            self.can_unlock_sector(sector_id)):
            self._unlock_sector(sector_id)
            return True
        
        return False
    
    def advance_development_stage(self):
        """Advance to the next development stage."""
        stages = list(DevelopmentStage)
        current_index = stages.index(self.current_development_stage)
        
        if current_index < len(stages) - 1:
            self.current_development_stage = stages[current_index + 1]
            
            # Unlock sectors appropriate for the new stage
            self._unlock_stage_appropriate_sectors()
    
    def _unlock_stage_appropriate_sectors(self):
        """Unlock sectors appropriate for the current development stage."""
        if self.current_development_stage == DevelopmentStage.INDUSTRIAL:
            # Unlock intermediate manufacturing sectors
            manufacturing_sectors = [s.id for s in self.sectors.values() 
                                   if s.category == SectorCategory.MANUFACTURING and 
                                   s.technology_level == TechnologyLevel.INTERMEDIATE]
            for sector_id in manufacturing_sectors[:5]:  # Unlock first 5
                if self.can_unlock_sector(sector_id):
                    self._unlock_sector(sector_id)
        
        elif self.current_development_stage == DevelopmentStage.POST_INDUSTRIAL:
            # Unlock advanced technology sectors
            tech_sectors = [s.id for s in self.sectors.values() 
                          if s.category == SectorCategory.TECHNOLOGY and 
                          s.technology_level == TechnologyLevel.ADVANCED]
            for sector_id in tech_sectors[:10]:  # Unlock first 10
                if self.can_unlock_sector(sector_id):
                    self._unlock_sector(sector_id)
        
        elif self.current_development_stage == DevelopmentStage.KNOWLEDGE:
            # Unlock cutting-edge research sectors
            research_sectors = [s.id for s in self.sectors.values() 
                              if s.category == SectorCategory.RESEARCH and 
                              s.technology_level == TechnologyLevel.CUTTING_EDGE]
            for sector_id in research_sectors[:15]:  # Unlock first 15
                if self.can_unlock_sector(sector_id):
                    self._unlock_sector(sector_id)
    
    def get_technology_tree_visualization(self, max_depth: int = 3) -> Dict[str, Any]:
        """Get a visualization of the technology tree."""
        tree = {
            'nodes': [],
            'edges': [],
            'stages': {}
        }
        
        # Add nodes
        for sector_id, sector in self.sectors.items():
            node = {
                'id': sector_id,
                'name': sector.name,
                'category': sector.category.value,
                'technology_level': sector.technology_level.value,
                'is_unlocked': sector_id in self.developed_sectors,
                'development_cost': sector.development_cost,
                'importance_weight': sector.importance_weight
            }
            tree['nodes'].append(node)
        
        # Add edges (dependencies)
        for sector_id, sector in self.sectors.items():
            for prereq_id in sector.prerequisites:
                edge = {
                    'from': prereq_id,
                    'to': sector_id,
                    'type': 'prerequisite'
                }
                tree['edges'].append(edge)
        
        # Group by development stages
        for stage in DevelopmentStage:
            stage_sectors = [s.id for s in self.sectors.values() 
                           if self._get_sector_development_stage(s.id) == stage]
            tree['stages'][stage.value] = stage_sectors
        
        return tree
    
    def _get_sector_development_stage(self, sector_id: int) -> DevelopmentStage:
        """Get the development stage appropriate for a sector."""
        if sector_id not in self.sectors:
            return DevelopmentStage.AGRARIAN
        
        sector = self.sectors[sector_id]
        
        if sector.technology_level == TechnologyLevel.BASIC:
            return DevelopmentStage.AGRARIAN
        elif sector.technology_level == TechnologyLevel.INTERMEDIATE:
            return DevelopmentStage.INDUSTRIAL
        elif sector.technology_level == TechnologyLevel.ADVANCED:
            return DevelopmentStage.POST_INDUSTRIAL
        elif sector.technology_level == TechnologyLevel.CUTTING_EDGE:
            return DevelopmentStage.KNOWLEDGE
        else:
            return DevelopmentStage.FUTURE
    
    def export_technology_tree(self, filepath: str):
        """Export the complete technology tree to a JSON file."""
        tree_data = self.get_technology_tree_visualization()
        tree_data['metadata'] = {
            'total_sectors': len(self.sectors),
            'developed_sectors': len(self.developed_sectors),
            'current_stage': self.current_development_stage.value,
            'research_budget': self.research_budget,
            'total_research_investment': self.total_research_investment
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
    
    def get_sector_by_id(self, sector_id: int) -> Optional[SectorDefinition]:
        """Get sector definition by ID."""
        return self.sectors.get(sector_id)
    
    def get_sector_by_name(self, sector_name: str) -> Optional[SectorDefinition]:
        """Get sector definition by name."""
        for sector in self.sectors.values():
            if sector.name == sector_name:
                return sector
        return None
    
    def get_sectors_by_category(self, category: SectorCategory) -> List[SectorDefinition]:
        """Get all sectors in a category."""
        return [sector for sector in self.sectors.values() if sector.category == category]
    
    def get_sectors_by_technology_level(self, level: TechnologyLevel) -> List[SectorDefinition]:
        """Get all sectors at a specific technology level."""
        return [sector for sector in self.sectors.values() if sector.technology_level == level]


if __name__ == "__main__":
    # Test the technology tree mapper
    mapper = TechnologyTreeMapper()
    
    print(f"Total sectors: {mapper.get_sector_count()}")
    print(f"Current development stage: {mapper.current_development_stage.value}")
    print(f"Developed sectors: {len(mapper.developed_sectors)}")
    
    # Test sector availability
    available = mapper.get_available_sectors(100)
    print(f"Available sectors (first 100): {len(available)}")
    
    # Test technology tree visualization
    tree = mapper.get_technology_tree_visualization()
    print(f"Technology tree nodes: {len(tree['nodes'])}")
    print(f"Technology tree edges: {len(tree['edges'])}")
    
    # Export the technology tree
    mapper.export_technology_tree("data/technology_tree_visualization.json")
    print("Technology tree exported to data/technology_tree_visualization.json")
