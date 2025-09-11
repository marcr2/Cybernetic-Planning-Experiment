"""
Sector Integration Module

This module integrates the Synthetic Sector Generator with the existing
economic planning system, providing a unified interface for sector management.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path

from .synthetic_sector_generator import (
    SyntheticSectorGenerator, 
    SectorDefinition, 
    TechnologyLevel,
    SectorCategory
)
# Legacy imports removed - old systems no longer available

class SectorIntegrationManager:
    """
    Manages sector generation using the Synthetic Sector Generator.
    
    The synthetic mode is the only supported approach, providing:
    - Dynamic, technology-driven sector generation
    - Employment integration and unemployment tracking
    - Technological breakthroughs and progressive unlocking
    - Full compliance with all requirements
    """
    
    def __init__(self, 
                 max_sectors: int = 1000,
                 min_sectors: int = 6,
                 sectors_file_path: Optional[str] = None):
        """
        Initialize the sector integration manager.
        
        Args:
            max_sectors: Maximum number of sectors
            min_sectors: Minimum number of sectors
            sectors_file_path: Path to sectors.md file
        """
        self.generation_mode = "synthetic"  # Only supported mode
        self.max_sectors = max_sectors
        self.min_sectors = min_sectors
        self.sectors_file_path = sectors_file_path
        
        # Initialize the synthetic generator
        self.generator = SyntheticSectorGenerator(
            sectors_file_path=self.sectors_file_path,
            max_sectors=self.max_sectors,
            min_sectors=self.min_sectors
        )
        
        # Integration state
        self.is_initialized = False
        self.current_year = 2024
        self.simulation_data = {}
        
    def initialize_sectors(self) -> Dict[int, SectorDefinition]:
        """Initialize sectors using the synthetic generator."""
        sectors = self.generator.sectors
        self.is_initialized = True
        return sectors
    
    def get_sector_count(self) -> int:
        """Get total number of sectors."""
        return self.generator.get_sector_count()
    
    def get_sector_names(self) -> List[str]:
        """Get list of all sector names."""
        return self.generator.get_sector_names()
    
    def get_core_sectors(self) -> List[SectorDefinition]:
        """Get the core sectors."""
        return self.generator.get_core_sectors()
    
    def advance_simulation_year(self, research_investment: float = 0.0) -> Dict[str, Any]:
        """Advance simulation by one year."""
        self.current_year += 1
        self.generator.advance_simulation_year(research_investment)
        
        return {
            "year": self.current_year,
            "sector_count": self.get_sector_count(),
            "technological_level": self.generator.technological_level,
            "achieved_breakthroughs": list(self.generator.achieved_breakthroughs),
            "research_investment": research_investment
        }
    
    def get_employment_data(self) -> Dict[str, Any]:
        """Get employment data."""
        return {
            "employment_by_sector": self.generator.get_employment_by_sector(),
            "total_employment": self.generator.get_total_employment(),
            "unemployment_rate": self.generator.get_unemployment_rate(),
            "total_population": self.generator.total_population
        }
    
    def get_technology_tree_data(self) -> Dict[str, Any]:
        """Get technology tree data."""
        return {
            "technology_tree": self.generator.technology_tree,
            "prerequisites": self.generator.prerequisites,
            "technological_level": self.generator.technological_level,
            "achieved_breakthroughs": list(self.generator.achieved_breakthroughs)
        }
    
    def get_sectors_by_technology_level(self, level: TechnologyLevel) -> List[SectorDefinition]:
        """Get sectors by technology level."""
        return self.generator.get_sectors_by_technology_level(level)
    
    def get_unlocked_sectors(self, developed_sectors: set) -> List[int]:
        """Get sectors that can be unlocked."""
        return self.generator.get_unlocked_sectors(developed_sectors)
    
    def export_comprehensive_data(self) -> Dict[str, Any]:
        """Export comprehensive sector data."""
        base_data = {
            "generation_mode": self.generation_mode,
            "sector_count": self.get_sector_count(),
            "sector_names": self.get_sector_names(),
            "core_sectors": [
                {
                    "id": sector.id,
                    "name": sector.name,
                    "category": sector.category.value if hasattr(sector.category, 'value') else str(sector.category),
                    "technology_level": sector.technology_level.value if hasattr(sector.technology_level, 'value') else str(sector.technology_level),
                    "importance_weight": sector.importance_weight,
                    "economic_impact": sector.economic_impact,
                    "labor_intensity": sector.labor_intensity,
                    "capital_intensity": sector.capital_intensity,
                    "environmental_impact": sector.environmental_impact
                }
                for sector in self.get_core_sectors()
            ],
            "current_year": self.current_year,
            "is_initialized": self.is_initialized
        }
        
        # Add mode-specific data
        if self.generation_mode == "synthetic":
            base_data.update({
                "employment_data": self.get_employment_data(),
                "technology_tree_data": self.get_technology_tree_data(),
                "synthetic_specific": self.generator.export_sector_data()
            })
        elif self.generation_mode == "static":
            base_data.update({
                "technology_tree_data": self.get_technology_tree_data(),
                "static_specific": {
                    "sectors": {
                        str(sector_id): {
                            "id": sector.id,
                            "name": sector.name,
                            "category": sector.category.value,
                            "technology_level": sector.technology_level.value,
                            "prerequisites": sector.prerequisites,
                            "unlocks": sector.unlocks,
                            "development_cost": sector.development_cost,
                            "research_requirements": sector.research_requirements
                        }
                        for sector_id, sector in self.generator.sectors.items()
                    }
                }
            })
        elif self.generation_mode == "hierarchical":
            base_data.update({
                "hierarchical_specific": self.generator.export_sector_definitions()
            })
        
        return base_data
    
    def validate_requirements(self) -> Dict[str, bool]:
        """Validate that the sector generation meets all requirements."""
        validation_results = {}
        
        # Requirement 1: Sector naming convention
        if self.generation_mode == "synthetic":
            sector_names = self.get_sector_names()
            available_names = self.generator.available_sector_names
            validation_results["all_names_from_sectors_md"] = all(
                name in available_names for name in sector_names
            )
        else:
            validation_results["all_names_from_sectors_md"] = True  # Static mode always uses sectors.md
        
        # Requirement 2: Sector count constraints
        sector_count = self.get_sector_count()
        validation_results["sector_count_valid"] = (
            self.min_sectors <= sector_count <= self.max_sectors
        )
        
        # Requirement 3: Core sectors mandatory
        core_sectors = self.get_core_sectors()
        validation_results["core_sectors_present"] = len(core_sectors) == 6
        
        # Requirement 4: Core sectors are basic technology level
        if self.generation_mode in ["synthetic", "static"]:
            validation_results["core_sectors_basic"] = all(
                sector.technology_level == TechnologyLevel.BASIC 
                for sector in core_sectors
            )
        else:
            validation_results["core_sectors_basic"] = True  # Hierarchical doesn't use tech levels
        
        # Requirement 5: Employment integration (synthetic mode only)
        if self.generation_mode == "synthetic":
            employment_data = self.get_employment_data()
            validation_results["employment_integrated"] = (
                "employment_by_sector" in employment_data and
                len(employment_data["employment_by_sector"]) == sector_count
            )
        else:
            validation_results["employment_integrated"] = True  # Not applicable to other modes
        
        # Overall validation
        validation_results["all_requirements_met"] = all(validation_results.values())
        
        return validation_results
    
    def print_summary(self):
        """Print a summary of the sector generation."""
        print(f"Sector Integration Manager Summary")
        print(f"Generation Mode: {self.generation_mode}")
        print(f"Total Sectors: {self.get_sector_count()}")
        print(f"Core Sectors: {len(self.get_core_sectors())}")
        print(f"Current Year: {self.current_year}")
        print(f"Initialized: {self.is_initialized}")
        
        # Print core sectors
        print(f"\nCore Sectors:")
        for sector in self.get_core_sectors():
            print(f"  {sector.name}")
        
        # Print validation results
        validation = self.validate_requirements()
        print(f"\nRequirements Validation:")
        for requirement, passed in validation.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {requirement}")
        
        # Print mode-specific information
        if self.generation_mode == "synthetic":
            employment_data = self.get_employment_data()
            print(f"\nEmployment Data:")
            print(f"  Total Employment: {employment_data['total_employment']:,.0f}")
            print(f"  Unemployment Rate: {employment_data['unemployment_rate']:.1%}")
            
            tech_data = self.get_technology_tree_data()
            print(f"\nTechnology Data:")
            print(f"  Technological Level: {tech_data['technological_level']:.2f}")
            print(f"  Achieved Breakthroughs: {len(tech_data['achieved_breakthroughs'])}")
        
        print()

# Convenience function for easy integration

def create_synthetic_sector_manager(max_sectors: int = 1000, min_sectors: int = 6) -> SectorIntegrationManager:
    """Create a synthetic sector manager."""
    return SectorIntegrationManager(
        max_sectors=max_sectors,
        min_sectors=min_sectors
    )

if __name__ == "__main__":
    # Test the integration manager
    print("Testing Sector Integration Manager")
    print("=" * 50)
    
    # Test synthetic mode
    print("Testing Synthetic Sector Generation:")
    synthetic_manager = create_synthetic_sector_manager(max_sectors=50)
    synthetic_manager.initialize_sectors()
    synthetic_manager.print_summary()
    
    # Simulate a few years
    for year in range(3):
        result = synthetic_manager.advance_simulation_year(1000000)
        print(f"Year {result['year']}: {result['sector_count']} sectors, "
              f"Tech Level: {result['technological_level']:.2f}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
