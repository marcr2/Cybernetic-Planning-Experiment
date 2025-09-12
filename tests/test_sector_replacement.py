#!/usr/bin/env python3
"""
Test Suite for Sector Generation System Replacement

This test suite verifies that the old sector generation systems have been
successfully replaced with the new synthetic sector generation system.
"""

import unittest
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cybernetic_planning.data import (
    SyntheticSectorGenerator,
    SectorIntegrationManager,
    create_synthetic_sector_manager,
    MatrixBuilder,
    TechnologyTreeMapper
)

class TestSectorReplacement(unittest.TestCase):
    """Test that the old sector generation has been replaced with the new system."""
    
    def test_matrix_builder_uses_synthetic_generator(self):
        """Test that MatrixBuilder now uses SyntheticSectorGenerator."""
        # Test with synthetic generator (use_technology_tree=False)
        mb = MatrixBuilder(max_sectors=50, use_technology_tree=False)
        sector_mapper = mb.get_sector_mapper()
        
        # Should be a SyntheticSectorGenerator
        self.assertIsInstance(sector_mapper, SyntheticSectorGenerator)
        
        # Should have sectors
        self.assertGreater(sector_mapper.get_sector_count(), 0)
        
        # Should have core sectors
        core_sectors = sector_mapper.get_core_sectors()
        self.assertEqual(len(core_sectors), 6)
        
        # Core sectors should have the right names
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
            self.assertIn(expected_name, core_names)
    
    def test_technology_tree_mapper_uses_synthetic_generator(self):
        """Test that TechnologyTreeMapper now uses SyntheticSectorGenerator."""
        ttm = TechnologyTreeMapper(max_sectors=50)
        
        # Should have sectors
        self.assertGreater(len(ttm.sectors), 0)
        
        # Should have sector names
        self.assertGreater(len(ttm.sector_names), 0)
        
        # Should have technology nodes
        self.assertGreater(len(ttm.technology_nodes), 0)
        
        # Should have basic sectors unlocked
        self.assertGreater(len(ttm.developed_sectors), 0)
    
    def test_sector_integration_manager_defaults_to_synthetic(self):
        """Test that SectorIntegrationManager defaults to synthetic mode."""
        # Test default behavior
        manager = SectorIntegrationManager()
        self.assertEqual(manager.generation_mode, "synthetic")
        
        # Test convenience function
        synthetic_manager = create_synthetic_sector_manager(max_sectors=50)
        self.assertEqual(synthetic_manager.generation_mode, "synthetic")
        
        # Should initialize successfully
        sectors = synthetic_manager.initialize_sectors()
        self.assertGreater(len(sectors), 0)
        
        # Should have core sectors
        core_sectors = synthetic_manager.get_core_sectors()
        self.assertEqual(len(core_sectors), 6)
    
    def test_synthetic_generator_meets_requirements(self):
        """Test that the synthetic generator meets all requirements."""
        generator = SyntheticSectorGenerator(max_sectors=100, min_sectors=6)
        
        # Requirement 1: Sector naming convention
        sector_names = generator.get_sector_names()
        available_names = generator.available_sector_names
        
        for name in sector_names:
            self.assertIn(name, available_names, 
                         f"Sector name '{name}' not found in sectors.md")
        
        # Requirement 2: Sector count constraints
        sector_count = generator.get_sector_count()
        self.assertGreaterEqual(sector_count, 6)
        self.assertLessEqual(sector_count, 100)
        
        # Requirement 3: Core sectors mandatory
        core_sectors = generator.get_core_sectors()
        self.assertEqual(len(core_sectors), 6)
        
        for sector in core_sectors:
            self.assertTrue(sector.is_core_sector)
        
        # Requirement 4: Employment integration
        employment_data = generator.get_employment_by_sector()
        self.assertEqual(len(employment_data), sector_count)
        
        total_employment = generator.get_total_employment()
        self.assertGreater(total_employment, 0)
    
    def test_dynamic_sector_evolution(self):
        """Test that sectors can evolve dynamically."""
        generator = SyntheticSectorGenerator(max_sectors=50, min_sectors=6)
        initial_count = generator.get_sector_count()
        
        # Simulate technological progress
        for year in range(5):
            generator.advance_simulation_year(1000000)
        
        final_count = generator.get_sector_count()
        
        # Should have more sectors or at least the same
        self.assertGreaterEqual(final_count, initial_count)
        
        # Should have achieved some breakthroughs
        self.assertGreater(len(generator.achieved_breakthroughs), 0)
        
        # Technological level should have increased
        self.assertGreater(generator.technological_level, 0.0)
    
    def test_synthetic_only_mode(self):
        """Test that only synthetic mode is supported."""
        # Only synthetic mode should be supported
        manager = SectorIntegrationManager(max_sectors=50)
        self.assertEqual(manager.generation_mode, "synthetic")
        
        # Should initialize successfully
        sectors = manager.initialize_sectors()
        self.assertGreater(len(sectors), 0)
    
    def test_import_structure(self):
        """Test that the new import structure works correctly."""
        # Test direct imports
        from cybernetic_planning.data.synthetic_sector_generator import (
            SyntheticSectorGenerator, SectorDefinition, TechnologyLevel, SectorCategory
        )
        
        # Test integration imports
        from cybernetic_planning.data.sector_integration import (
            SectorIntegrationManager, create_synthetic_sector_manager
        )
        
        # Test that classes are properly defined
        self.assertTrue(hasattr(SyntheticSectorGenerator, '__init__'))
        self.assertTrue(hasattr(SectorDefinition, '__init__'))
        self.assertTrue(hasattr(TechnologyLevel, 'BASIC'))
        self.assertTrue(hasattr(SectorCategory, 'HEALTHCARE'))
    
    def test_performance_comparison(self):
        """Test that the new system performs well."""
        import time
        
        # Test initialization speed
        start_time = time.time()
        generator = SyntheticSectorGenerator(max_sectors=100, min_sectors=6)
        init_time = time.time() - start_time
        
        # Should initialize quickly (less than 1 second)
        self.assertLess(init_time, 1.0)
        
        # Test year advancement speed
        start_time = time.time()
        for year in range(10):
            generator.advance_simulation_year(1000000)
        advance_time = time.time() - start_time
        
        # Should advance years quickly (less than 0.1 seconds per year)
        self.assertLess(advance_time, 1.0)
    
    def test_data_export_compatibility(self):
        """Test that exported data is compatible with existing systems."""
        generator = SyntheticSectorGenerator(max_sectors=50, min_sectors=6)
        
        # Export data
        export_data = generator.export_sector_data()
        
        # Should have required keys
        required_keys = [
            'sectors', 'technology_tree', 'prerequisites', 
            'employment_by_sector', 'simulation_state', 
            'sector_names', 'total_sectors'
        ]
        
        for key in required_keys:
            self.assertIn(key, export_data)
        
        # Should have proper data types
        self.assertIsInstance(export_data['sectors'], dict)
        self.assertIsInstance(export_data['sector_names'], list)
        self.assertIsInstance(export_data['total_sectors'], int)
        
        # Should have sector data
        sectors_data = export_data['sectors']
        self.assertGreater(len(sectors_data), 0)
        
        # Check sample sector data structure
        first_sector_id = list(sectors_data.keys())[0]
        first_sector = sectors_data[first_sector_id]
        
        required_sector_keys = [
            'id', 'name', 'category', 'technology_level', 
            'importance_weight', 'economic_impact', 'employment_capacity'
        ]
        
        for key in required_sector_keys:
            self.assertIn(key, first_sector)

class TestIntegrationCompatibility(unittest.TestCase):
    """Test integration with existing systems."""
    
    def test_matrix_builder_integration(self):
        """Test MatrixBuilder integration with synthetic generator."""
        mb = MatrixBuilder(max_sectors=50, use_technology_tree=False)
        
        # Should be able to create matrices
        import numpy as np
        test_data = np.random.rand(10, 10)
        
        try:
            matrix = mb.create_technology_matrix(test_data, name="test_matrix")
            self.assertIsNotNone(matrix)
        except Exception as e:
            self.fail(f"MatrixBuilder failed to create matrix: {e}")
    
    def test_technology_tree_integration(self):
        """Test TechnologyTreeMapper integration with synthetic generator."""
        ttm = TechnologyTreeMapper(max_sectors=50)
        
        # Should be able to get available sectors
        try:
            available = ttm.get_available_sectors(max_sectors=20)
            self.assertIsInstance(available, list)
            self.assertLessEqual(len(available), 20)
        except Exception as e:
            self.fail(f"TechnologyTreeMapper failed to get available sectors: {e}")
    
    def test_sector_integration_manager_compatibility(self):
        """Test SectorIntegrationManager compatibility."""
        manager = create_synthetic_sector_manager(max_sectors=50)
        
        # Should validate requirements
        validation = manager.validate_requirements()
        self.assertIn('all_requirements_met', validation)
        self.assertTrue(validation['all_requirements_met'])
        
        # Should export comprehensive data
        export_data = manager.export_comprehensive_data()
        self.assertIn('generation_mode', export_data)
        self.assertEqual(export_data['generation_mode'], 'synthetic')

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
