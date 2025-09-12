#!/usr/bin/env python3
"""
Test Suite for Resource Flow and Transportation Cost Model

This test suite validates the resource flow model implementation,
including resource definitions, consumption matrices, transportation
cost calculations, and integration with the economic planning system.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cybernetic_planning.data.resource_flow_model import (
    ResourceFlowModel, ResourceDefinition, ResourceType,
    ResourceInputMatrix, TransportationCostCalculator,
    create_example_resource_model
)
from cybernetic_planning.data.resource_flow_integration import (
    ResourcePlanningIntegration, create_integrated_resource_system
)
from cybernetic_planning.utils.transportation_system import (
    Location, TransportMode, VehicleType
)


class TestResourceDefinition(unittest.TestCase):
    """Test the ResourceDefinition data structure."""
    
    def test_resource_creation(self):
        """Test creating a resource definition."""
        resource = ResourceDefinition(
            resource_id="R001",
            resource_name="Iron Ore",
            resource_type=ResourceType.RAW_MATERIAL,
            producing_sector_id="S051",
            base_unit="ton",
            density=5000.0,
            value_per_unit=50.0
        )
        
        self.assertEqual(resource.resource_id, "R001")
        self.assertEqual(resource.resource_name, "Iron Ore")
        self.assertEqual(resource.resource_type, ResourceType.RAW_MATERIAL)
        self.assertEqual(resource.producing_sector_id, "S051")
        self.assertEqual(resource.base_unit, "ton")
        self.assertEqual(resource.density, 5000.0)
        self.assertEqual(resource.value_per_unit, 50.0)
    
    def test_resource_defaults(self):
        """Test resource definition with default values."""
        resource = ResourceDefinition(
            resource_id="R002",
            resource_name="Test Resource",
            resource_type=ResourceType.MANUFACTURED_GOOD,
            producing_sector_id="S001",
            base_unit="unit"
        )
        
        self.assertEqual(resource.density, 1.0)
        self.assertEqual(resource.value_per_unit, 0.0)
        self.assertEqual(resource.perishability, 0.0)
        self.assertEqual(resource.hazard_class, 0)
        self.assertEqual(resource.substitutability, 0.5)
        self.assertEqual(resource.criticality, 0.5)


class TestResourceInputMatrix(unittest.TestCase):
    """Test the ResourceInputMatrix class."""
    
    def setUp(self):
        """Set up test resources and sectors."""
        self.resources = [
            ResourceDefinition("R001", "Resource 1", ResourceType.RAW_MATERIAL, "S001", "unit"),
            ResourceDefinition("R002", "Resource 2", ResourceType.MANUFACTURED_GOOD, "S002", "unit"),
            ResourceDefinition("R003", "Resource 3", ResourceType.ENERGY, "S003", "unit")
        ]
        self.sectors = ["S001", "S002", "S003"]
        self.matrix = ResourceInputMatrix(self.resources, self.sectors)
    
    def test_matrix_initialization(self):
        """Test matrix initialization."""
        self.assertEqual(self.matrix.n_resources, 3)
        self.assertEqual(self.matrix.n_sectors, 3)
        self.assertEqual(self.matrix.matrix.shape, (3, 3))
        self.assertTrue(np.all(self.matrix.matrix == 0))
    
    def test_set_consumption(self):
        """Test setting resource consumption rates."""
        self.matrix.set_consumption("R001", "S001", 0.5, "direct")
        self.matrix.set_consumption("R002", "S002", 1.0, "direct")
        self.matrix.set_consumption("R003", "S003", 0.2, "direct")
        
        self.assertEqual(self.matrix.get_consumption("R001", "S001"), 0.5)
        self.assertEqual(self.matrix.get_consumption("R002", "S002"), 1.0)
        self.assertEqual(self.matrix.get_consumption("R003", "S003"), 0.2)
        self.assertEqual(self.matrix.get_consumption("R001", "S002"), 0.0)
    
    def test_invalid_resource_sector(self):
        """Test error handling for invalid resource/sector IDs."""
        with self.assertRaises(ValueError):
            self.matrix.set_consumption("R999", "S001", 0.5)
        
        with self.assertRaises(ValueError):
            self.matrix.set_consumption("R001", "S999", 0.5)
    
    def test_total_resource_demand(self):
        """Test total resource demand calculation."""
        # Set consumption rates
        self.matrix.set_consumption("R001", "S001", 0.5)
        self.matrix.set_consumption("R001", "S002", 0.3)
        self.matrix.set_consumption("R002", "S002", 1.0)
        
        # Test with sector outputs
        sector_outputs = np.array([100, 200, 150])
        
        # R001 demand: 0.5*100 + 0.3*200 + 0*150 = 50 + 60 + 0 = 110
        r001_demand = self.matrix.get_total_resource_demand("R001", sector_outputs)
        self.assertAlmostEqual(r001_demand, 110.0)
        
        # R002 demand: 0*100 + 1.0*200 + 0*150 = 200
        r002_demand = self.matrix.get_total_resource_demand("R002", sector_outputs)
        self.assertAlmostEqual(r002_demand, 200.0)
    
    def test_sector_resource_requirements(self):
        """Test getting resource requirements for a sector."""
        self.matrix.set_consumption("R001", "S001", 0.5)
        self.matrix.set_consumption("R002", "S001", 0.2)
        self.matrix.set_consumption("R003", "S001", 0.1)
        
        requirements = self.matrix.get_sector_resource_requirements("S001")
        
        self.assertEqual(len(requirements), 3)
        self.assertEqual(requirements["R001"], 0.5)
        self.assertEqual(requirements["R002"], 0.2)
        self.assertEqual(requirements["R003"], 0.1)


class TestTransportationCostCalculator(unittest.TestCase):
    """Test the TransportationCostCalculator class."""
    
    def setUp(self):
        """Set up test data."""
        self.calculator = TransportationCostCalculator()
        self.resource = ResourceDefinition(
            "R001", "Test Resource", ResourceType.RAW_MATERIAL,
            "S001", "ton", density=1000.0, value_per_unit=100.0
        )
        self.origin = Location("ORIGIN", "Origin", 40.0, -100.0)
        self.destination = Location("DEST", "Destination", 41.0, -99.0)
    
    def test_cost_calculation(self):
        """Test basic transportation cost calculation."""
        cost_details = self.calculator.calculate_transportation_cost(
            resource=self.resource,
            quantity=100.0,
            origin=self.origin,
            destination=self.destination,
            transport_mode=TransportMode.TRUCK
        )
        
        self.assertIn("total_cost", cost_details)
        self.assertIn("distance_km", cost_details)
        self.assertIn("emissions_kg_co2", cost_details)
        self.assertGreater(cost_details["total_cost"], 0)
        self.assertGreater(cost_details["distance_km"], 0)
    
    def test_different_transport_modes(self):
        """Test cost calculation for different transport modes."""
        modes = [TransportMode.TRUCK, TransportMode.RAIL, TransportMode.AIRCRAFT]
        costs = {}
        
        for mode in modes:
            cost_details = self.calculator.calculate_transportation_cost(
                resource=self.resource,
                quantity=100.0,
                origin=self.origin,
                destination=self.destination,
                transport_mode=mode
            )
            costs[mode] = cost_details["total_cost"]
        
        # Aircraft should be most expensive, rail should be cheapest
        self.assertGreater(costs[TransportMode.AIRCRAFT], costs[TransportMode.TRUCK])
        self.assertGreater(costs[TransportMode.TRUCK], costs[TransportMode.RAIL])
    
    def test_quantity_scaling(self):
        """Test that costs scale appropriately with quantity."""
        quantities = [100.0, 500.0, 1000.0]
        costs = []
        
        for quantity in quantities:
            cost_details = self.calculator.calculate_transportation_cost(
                resource=self.resource,
                quantity=quantity,
                origin=self.origin,
                destination=self.destination,
                transport_mode=TransportMode.TRUCK
            )
            costs.append(cost_details["total_cost"])
        
        # Costs should increase with quantity (though not necessarily linearly)
        for i in range(1, len(costs)):
            self.assertGreater(costs[i], costs[i-1])
    
    def test_resource_specific_factors(self):
        """Test resource-specific cost factors."""
        # High-value resource
        high_value_resource = ResourceDefinition(
            "R002", "High Value", ResourceType.MANUFACTURED_GOOD,
            "S001", "unit", density=1000.0, value_per_unit=10000.0
        )
        
        # Hazardous resource
        hazardous_resource = ResourceDefinition(
            "R003", "Hazardous", ResourceType.RAW_MATERIAL,
            "S001", "unit", density=1000.0, value_per_unit=100.0,
            hazard_class=3
        )
        
        base_cost = self.calculator.calculate_transportation_cost(
            resource=self.resource,
            quantity=100.0,
            origin=self.origin,
            destination=self.destination,
            transport_mode=TransportMode.TRUCK
        )["total_cost"]
        
        high_value_cost = self.calculator.calculate_transportation_cost(
            resource=high_value_resource,
            quantity=100.0,
            origin=self.origin,
            destination=self.destination,
            transport_mode=TransportMode.TRUCK
        )["total_cost"]
        
        hazardous_cost = self.calculator.calculate_transportation_cost(
            resource=hazardous_resource,
            quantity=100.0,
            origin=self.origin,
            destination=self.destination,
            transport_mode=TransportMode.TRUCK
        )["total_cost"]
        
        # High-value and hazardous resources should cost more
        self.assertGreater(high_value_cost, base_cost)
        self.assertGreater(hazardous_cost, base_cost)


class TestResourceFlowModel(unittest.TestCase):
    """Test the ResourceFlowModel class."""
    
    def setUp(self):
        """Set up test model."""
        self.model = ResourceFlowModel()
        self.resources = [
            ResourceDefinition("R001", "Resource 1", ResourceType.RAW_MATERIAL, "S001", "unit"),
            ResourceDefinition("R002", "Resource 2", ResourceType.MANUFACTURED_GOOD, "S002", "unit")
        ]
        self.sectors = ["S001", "S002"]
    
    def test_add_resources(self):
        """Test adding resources to the model."""
        self.model.add_resources(self.resources)
        
        self.assertEqual(len(self.model.resources), 2)
        self.assertIn("R001", self.model.resources)
        self.assertIn("R002", self.model.resources)
    
    def test_initialize_matrix(self):
        """Test initializing the resource matrix."""
        self.model.add_resources(self.resources)
        self.model.initialize_resource_matrix(self.sectors)
        
        self.assertIsNotNone(self.model.resource_matrix)
        self.assertEqual(self.model.resource_matrix.n_resources, 2)
        self.assertEqual(self.model.resource_matrix.n_sectors, 2)
    
    def test_set_consumption(self):
        """Test setting resource consumption."""
        self.model.add_resources(self.resources)
        self.model.initialize_resource_matrix(self.sectors)
        
        self.model.set_resource_consumption("R001", "S001", 0.5)
        self.model.set_resource_consumption("R002", "S002", 1.0)
        
        self.assertEqual(self.model.resource_matrix.get_consumption("R001", "S001"), 0.5)
        self.assertEqual(self.model.resource_matrix.get_consumption("R002", "S002"), 1.0)
    
    def test_calculate_total_demand(self):
        """Test calculating total resource demand."""
        self.model.add_resources(self.resources)
        self.model.initialize_resource_matrix(self.sectors)
        self.model.set_resource_consumption("R001", "S001", 0.5)
        self.model.set_resource_consumption("R002", "S002", 1.0)
        
        sector_outputs = np.array([100, 200])
        total_demand = self.model.calculate_total_resource_demand(sector_outputs)
        
        self.assertIn("R001", total_demand)
        self.assertIn("R002", total_demand)
        self.assertAlmostEqual(total_demand["R001"], 50.0)  # 0.5 * 100
        self.assertAlmostEqual(total_demand["R002"], 200.0)  # 1.0 * 200
    
    def test_save_load_model(self):
        """Test saving and loading the model."""
        # Set up model
        self.model.add_resources(self.resources)
        self.model.initialize_resource_matrix(self.sectors)
        self.model.set_resource_consumption("R001", "S001", 0.5)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.model.save_model(temp_path)
            
            # Load into new model
            new_model = ResourceFlowModel()
            new_model.load_model(temp_path)
            
            # Verify loaded model
            self.assertEqual(len(new_model.resources), 2)
            self.assertIn("R001", new_model.resources)
            self.assertEqual(new_model.resources["R001"].resource_name, "Resource 1")
            self.assertIsNotNone(new_model.resource_matrix)
            self.assertEqual(new_model.resource_matrix.get_consumption("R001", "S001"), 0.5)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestResourcePlanningIntegration(unittest.TestCase):
    """Test the ResourcePlanningIntegration class."""
    
    def setUp(self):
        """Set up test integration."""
        self.model = create_example_resource_model()
        self.integration = ResourcePlanningIntegration(self.model)
    
    def test_integration_creation(self):
        """Test creating integration layer."""
        self.assertIsNotNone(self.integration.resource_model)
        self.assertIsNotNone(self.integration.transportation_calculator)
        self.assertIn("created_at", self.integration.integration_metadata)
    
    def test_optimize_resource_allocation(self):
        """Test resource allocation optimization."""
        sector_outputs = np.array([1000, 2000, 1500, 500, 100, 5000, 3000, 2000])
        
        result = self.integration.optimize_resource_allocation(sector_outputs)
        
        self.assertIn("allocation_plan", result)
        self.assertIn("optimization_details", result)
        self.assertIn("total_cost", result["optimization_details"])
        self.assertIn("efficiency", result["optimization_details"])
    
    def test_calculate_transportation_network_costs(self):
        """Test transportation network cost calculation."""
        locations = {
            "ORIGIN": Location("ORIGIN", "Origin", 40.0, -100.0),
            "DEST": Location("DEST", "Destination", 41.0, -99.0)
        }
        
        resource_flows = {
            ("R001", "DEST"): 100.0,
            ("R002", "DEST"): 50.0
        }
        
        result = self.integration.calculate_transportation_network_costs(resource_flows, locations)
        
        self.assertIn("transport_costs", result)
        self.assertIn("cost_analysis", result)
        self.assertIn("recommendations", result)
        self.assertIn("total_transport_cost", result)
    
    def test_generate_resource_flow_report(self):
        """Test generating resource flow report."""
        sector_outputs = np.array([1000, 2000, 1500, 500, 100, 5000, 3000, 2000])
        
        report = self.integration.generate_resource_flow_report(sector_outputs)
        
        self.assertIn("report_metadata", report)
        self.assertIn("resource_summary", report)
        self.assertIn("allocation_analysis", report)
        self.assertIn("recommendations", report)
        
        # Check report structure
        self.assertIn("total_requirements", report["resource_summary"])
        self.assertIn("total_value", report["resource_summary"])
        self.assertIn("resource_types", report["resource_summary"])


class TestExampleModel(unittest.TestCase):
    """Test the example resource model creation."""
    
    def test_create_example_model(self):
        """Test creating the example resource model."""
        model = create_example_resource_model()
        
        self.assertIsInstance(model, ResourceFlowModel)
        self.assertGreater(len(model.resources), 0)
        self.assertIsNotNone(model.resource_matrix)
        
        # Check that example resources are present
        expected_resources = ["R001", "R002", "R003", "R004", "R005", "R006"]
        for resource_id in expected_resources:
            self.assertIn(resource_id, model.resources)
        
        # Check resource properties
        iron_ore = model.resources["R001"]
        self.assertEqual(iron_ore.resource_name, "Iron Ore")
        self.assertEqual(iron_ore.resource_type, ResourceType.RAW_MATERIAL)
        self.assertEqual(iron_ore.base_unit, "ton")
    
    def test_example_model_consumption_patterns(self):
        """Test that example model has consumption patterns set."""
        model = create_example_resource_model()
        
        # Check that some consumption patterns are set
        consumption_found = False
        for resource_id in model.resources:
            for sector_id in model.sectors:
                if model.resource_matrix.get_consumption(resource_id, sector_id) > 0:
                    consumption_found = True
                    break
            if consumption_found:
                break
        
        self.assertTrue(consumption_found, "Example model should have consumption patterns")


class TestIntegrationSystem(unittest.TestCase):
    """Test the integrated resource system."""
    
    def test_create_integrated_system(self):
        """Test creating the integrated resource system."""
        integration = create_integrated_resource_system()
        
        self.assertIsInstance(integration, ResourcePlanningIntegration)
        self.assertIsNotNone(integration.resource_model)
        self.assertIsNotNone(integration.sector_mapper)
        self.assertIsNotNone(integration.data_loader)
    
    def test_integrated_system_functionality(self):
        """Test that integrated system has full functionality."""
        integration = create_integrated_resource_system()
        
        # Test resource allocation
        sector_outputs = np.array([1000, 2000, 1500, 500, 100, 5000, 3000, 2000])
        allocation_result = integration.optimize_resource_allocation(sector_outputs)
        
        self.assertIn("allocation_plan", allocation_result)
        self.assertIn("optimization_details", allocation_result)
        
        # Test report generation
        report = integration.generate_resource_flow_report(sector_outputs)
        
        self.assertIn("resource_summary", report)
        self.assertIn("allocation_analysis", report)
        self.assertIn("recommendations", report)


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)
