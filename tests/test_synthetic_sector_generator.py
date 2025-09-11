"""
Test suite for the Synthetic Sector Generator

Tests all requirements:
1. Sector naming convention (all names from sectors.md)
2. Sector count constraints (6-1000)
3. Initial sector setup (6 core sectors mandatory)
4. Subdivision logic
5. Dynamic sector evolution
6. Employment integration
"""

import unittest
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cybernetic_planning.data.synthetic_sector_generator import (
    SyntheticSectorGenerator, 
    TechnologyLevel, 
    SectorCategory,
    SectorDefinition
)

class TestSyntheticSectorGenerator(unittest.TestCase):
    """Test cases for the Synthetic Sector Generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SyntheticSectorGenerator(max_sectors=50, min_sectors=6)
    
    def test_sector_naming_convention(self):
        """Test that all sector names come from sectors.md."""
        # Get all sector names
        sector_names = self.generator.get_sector_names()
        
        # Check that all names are from the available sector names
        available_names = self.generator.available_sector_names
        
        for name in sector_names:
            self.assertIn(name, available_names, 
                         f"Sector name '{name}' not found in sectors.md")
    
    def test_sector_count_constraints(self):
        """Test sector count constraints (6-1000)."""
        # Test minimum constraint
        generator_min = SyntheticSectorGenerator(max_sectors=6, min_sectors=6)
        self.assertEqual(generator_min.get_sector_count(), 6)
        
        # Test maximum constraint
        generator_max = SyntheticSectorGenerator(max_sectors=1000, min_sectors=6)
        self.assertLessEqual(generator_max.get_sector_count(), 1000)
        
        # Test current generator
        self.assertGreaterEqual(self.generator.get_sector_count(), 6)
        self.assertLessEqual(self.generator.get_sector_count(), 50)
    
    def test_core_sectors_mandatory(self):
        """Test that the first 6 sectors are mandatory core sectors."""
        core_sectors = self.generator.get_core_sectors()
        
        # Should have exactly 6 core sectors
        self.assertEqual(len(core_sectors), 6)
        
        # Check that all core sectors are marked as core
        for sector in core_sectors:
            self.assertTrue(sector.is_core_sector)
            self.assertEqual(sector.technology_level, TechnologyLevel.BASIC)
        
        # Check that we have the expected core sector names
        core_names = [sector.name for sector in core_sectors]
        expected_core_names = [
            "Healthcare",
            "Food and Agriculture", 
            "Energy",
            "Housing and Construction",
            "Education",
            "Transportation"
        ]
        
        for expected_name in expected_core_names:
            self.assertIn(expected_name, core_names, 
                        f"Core sector '{expected_name}' not found")
    
    def test_subdivision_logic(self):
        """Test subdivision logic for core sectors."""
        # Get all sectors
        all_sectors = list(self.generator.sectors.values())
        
        # Count subdivisions
        subdivisions = [s for s in all_sectors if s.is_subdivision]
        
        # Should have subdivisions if we have more than 6 sectors
        if self.generator.get_sector_count() > 6:
            self.assertGreater(len(subdivisions), 0, 
                             "Should have subdivisions when sector count > 6")
            
            # Check that subdivisions have parent sectors
            for subdivision in subdivisions:
                self.assertIsNotNone(subdivision.parent_sector_id)
                parent_sector = self.generator.sectors[subdivision.parent_sector_id]
                self.assertTrue(parent_sector.is_core_sector)
    
    def test_dynamic_sector_evolution(self):
        """Test dynamic sector evolution based on technological progress."""
        initial_count = self.generator.get_sector_count()
        
        # Simulate several years with research investment
        for year in range(5):
            research_investment = 1000000 * (year + 1)
            self.generator.advance_simulation_year(research_investment)
        
        final_count = self.generator.get_sector_count()
        
        # Should have more sectors after technological progress
        self.assertGreaterEqual(final_count, initial_count)
        
        # Check that technological level increased
        self.assertGreater(self.generator.technological_level, 0.0)
        
        # Check that some breakthroughs were achieved
        self.assertGreater(len(self.generator.achieved_breakthroughs), 0)
    
    def test_employment_integration(self):
        """Test employment integration."""
        # Check that employment is distributed across sectors
        employment_by_sector = self.generator.get_employment_by_sector()
        
        # Should have employment data for all sectors
        self.assertEqual(len(employment_by_sector), self.generator.get_sector_count())
        
        # Check that total employment is reasonable
        total_employment = self.generator.get_total_employment()
        self.assertGreater(total_employment, 0)
        
        # Check unemployment rate
        unemployment_rate = self.generator.get_unemployment_rate()
        self.assertGreaterEqual(unemployment_rate, 0.0)
        self.assertLessEqual(unemployment_rate, 1.0)
        
        # Core sectors should have higher employment capacity
        core_sectors = self.generator.get_core_sectors()
        for core_sector in core_sectors:
            core_employment = employment_by_sector[core_sector.id]
            # Core sectors should have significant employment
            self.assertGreater(core_employment, 1000)
    
    def test_technology_levels(self):
        """Test technology level distribution."""
        # Get sectors by technology level
        basic_sectors = self.generator.get_sectors_by_technology_level(TechnologyLevel.BASIC)
        intermediate_sectors = self.generator.get_sectors_by_technology_level(TechnologyLevel.INTERMEDIATE)
        
        # Should have basic sectors (core sectors)
        self.assertGreater(len(basic_sectors), 0)
        
        # Should have intermediate sectors (subdivisions and others)
        self.assertGreater(len(intermediate_sectors), 0)
        
        # All core sectors should be basic
        core_sectors = self.generator.get_core_sectors()
        for core_sector in core_sectors:
            self.assertEqual(core_sector.technology_level, TechnologyLevel.BASIC)
    
    def test_sector_properties(self):
        """Test that sectors have proper properties."""
        for sector_id, sector in self.generator.sectors.items():
            # Check required properties
            self.assertIsInstance(sector.id, int)
            self.assertIsInstance(sector.name, str)
            self.assertIsInstance(sector.category, SectorCategory)
            self.assertIsInstance(sector.technology_level, TechnologyLevel)
            self.assertIsInstance(sector.importance_weight, float)
            self.assertIsInstance(sector.employment_capacity, float)
            
            # Check property ranges
            self.assertGreaterEqual(sector.importance_weight, 0.0)
            self.assertLessEqual(sector.importance_weight, 1.0)
            self.assertGreater(sector.employment_capacity, 0.0)
            
            # Check economic impact values
            valid_impacts = ['critical', 'high', 'medium', 'low']
            self.assertIn(sector.economic_impact, valid_impacts)
            
            # Check labor intensity values
            valid_labor = ['high', 'medium', 'low']
            self.assertIn(sector.labor_intensity, valid_labor)
            
            # Check capital intensity values
            valid_capital = ['very_high', 'high', 'medium', 'low']
            self.assertIn(sector.capital_intensity, valid_capital)
    
    def test_technological_breakthroughs(self):
        """Test technological breakthrough system."""
        # Initially no breakthroughs
        self.assertEqual(len(self.generator.achieved_breakthroughs), 0)
        
        # Simulate high research investment
        for year in range(10):
            self.generator.advance_simulation_year(2000000)  # High investment
        
        # Should have achieved some breakthroughs
        self.assertGreater(len(self.generator.achieved_breakthroughs), 0)
        
        # Check that breakthroughs have proper properties
        for breakthrough in self.generator.technological_breakthroughs:
            if breakthrough.is_achieved:
                self.assertIsNotNone(breakthrough.breakthrough_year)
                self.assertGreater(breakthrough.required_research_investment, 0)
    
    def test_export_functionality(self):
        """Test export functionality."""
        export_data = self.generator.export_sector_data()
        
        # Check required keys
        required_keys = ['sectors', 'technology_tree', 'prerequisites', 
                        'employment_by_sector', 'simulation_state', 
                        'sector_names', 'total_sectors']
        for key in required_keys:
            self.assertIn(key, export_data)
        
        # Check sector data structure
        sectors_data = export_data['sectors']
        self.assertEqual(len(sectors_data), self.generator.get_sector_count())
        
        # Check simulation state
        simulation_state = export_data['simulation_state']
        self.assertIn('current_year', simulation_state)
        self.assertIn('technological_level', simulation_state)
        self.assertIn('total_population', simulation_state)
    
    def test_population_updates(self):
        """Test population update functionality."""
        initial_population = self.generator.total_population
        initial_employment = self.generator.get_total_employment()
        
        # Update population
        new_population = initial_population * 1.5
        self.generator.update_population(new_population)
        
        # Check that population was updated
        self.assertEqual(self.generator.total_population, new_population)
        
        # Employment should be scaled proportionally
        new_employment = self.generator.get_total_employment()
        expected_employment = initial_employment * 1.5
        self.assertAlmostEqual(new_employment, expected_employment, delta=1000)
    
    def test_constraint_validation(self):
        """Test constraint validation."""
        # Test invalid min_sectors
        with self.assertRaises(ValueError):
            SyntheticSectorGenerator(min_sectors=5)  # Less than 6
        
        # Test invalid max_sectors
        with self.assertRaises(ValueError):
            SyntheticSectorGenerator(max_sectors=1001)  # More than 1000
        
        # Test min > max
        with self.assertRaises(ValueError):
            SyntheticSectorGenerator(min_sectors=10, max_sectors=8)
    
    def test_sector_unlocking(self):
        """Test sector unlocking based on prerequisites."""
        # Get sectors that can be unlocked
        developed_sectors = {sector.id for sector in self.generator.get_core_sectors()}
        unlocked_sectors = self.generator.get_unlocked_sectors(developed_sectors)
        
        # Should have some unlocked sectors
        self.assertGreater(len(unlocked_sectors), 0)
        
        # Check that unlocked sectors have their prerequisites met
        for sector_id in unlocked_sectors:
            sector = self.generator.sectors[sector_id]
            for prereq_id in sector.prerequisites:
                self.assertIn(prereq_id, developed_sectors)

class TestSectorGeneratorIntegration(unittest.TestCase):
    """Integration tests for the sector generator."""
    
    def test_large_scale_generation(self):
        """Test generation with maximum sectors."""
        generator = SyntheticSectorGenerator(max_sectors=1000, min_sectors=6)
        
        # Should generate close to maximum sectors
        self.assertGreaterEqual(generator.get_sector_count(), 6)
        self.assertLessEqual(generator.get_sector_count(), 1000)
        
        # Should have all core sectors
        self.assertEqual(len(generator.get_core_sectors()), 6)
    
    def test_minimal_generation(self):
        """Test generation with minimum sectors."""
        generator = SyntheticSectorGenerator(max_sectors=6, min_sectors=6)
        
        # Should generate exactly 6 sectors
        self.assertEqual(generator.get_sector_count(), 6)
        
        # Should have all core sectors
        self.assertEqual(len(generator.get_core_sectors()), 6)
        
        # Should have no subdivisions
        subdivisions = [s for s in generator.sectors.values() if s.is_subdivision]
        self.assertEqual(len(subdivisions), 0)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
