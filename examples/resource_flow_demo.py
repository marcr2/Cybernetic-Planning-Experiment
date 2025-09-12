#!/usr/bin/env python3
"""
Resource Flow and Transportation Cost Model Demo

This script demonstrates the comprehensive resource flow and transportation cost model
designed for the cybernetic planning system. It shows how to:

1. Define resources and map them to production sectors
2. Create resource consumption matrices
3. Calculate transportation costs for different modes
4. Integrate with the economic planning system
5. Generate resource flow reports and optimization recommendations

Example Output Format:
- Sectors: S1: Healthcare, S2: Food and Agriculture, S3: Energy
- Resources: R1.1: Medicines, R1.2: PPE, R2: Farm Equipment, R3: Electric Components
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cybernetic_planning.data.resource_flow_model import (
    ResourceFlowModel, ResourceDefinition, ResourceType,
    create_example_resource_model, TransportationCostCalculator
)
from cybernetic_planning.data.resource_flow_integration import (
    ResourcePlanningIntegration, create_integrated_resource_system
)
from cybernetic_planning.utils.transportation_system import (
    Location, TransportMode, VehicleType
)


def demonstrate_resource_definitions():
    """Demonstrate the resource data structure design."""
    print("=" * 80)
    print("1. RESOURCE DATA STRUCTURE DEMONSTRATION")
    print("=" * 80)
    
    # Create example resources following the specified format
    example_resources = [
        ResourceDefinition(
            resource_id="R1.1",
            resource_name="Medicines",
            resource_type=ResourceType.MANUFACTURED_GOOD,
            producing_sector_id="S1",  # Healthcare
            base_unit="unit",
            density=1000.0,  # kg/m³
            value_per_unit=100.0,  # $/unit
            perishability=0.3,
            criticality=0.9
        ),
        ResourceDefinition(
            resource_id="R1.2",
            resource_name="PPE",
            resource_type=ResourceType.MANUFACTURED_GOOD,
            producing_sector_id="S1",  # Healthcare
            base_unit="unit",
            density=500.0,  # kg/m³
            value_per_unit=50.0,  # $/unit
            criticality=0.8
        ),
        ResourceDefinition(
            resource_id="R2",
            resource_name="Farm Equipment",
            resource_type=ResourceType.MANUFACTURED_GOOD,
            producing_sector_id="S2",  # Food and Agriculture
            base_unit="unit",
            density=2000.0,  # kg/m³
            value_per_unit=50000.0,  # $/unit
            criticality=0.7
        ),
        ResourceDefinition(
            resource_id="R3",
            resource_name="Electric Components",
            resource_type=ResourceType.MANUFACTURED_GOOD,
            producing_sector_id="S3",  # Energy
            base_unit="unit",
            density=1500.0,  # kg/m³
            value_per_unit=200.0,  # $/unit
            criticality=0.8
        )
    ]
    
    print("Example Resources Defined:")
    print("-" * 40)
    for resource in example_resources:
        print(f"Resource ID: {resource.resource_id}")
        print(f"  Name: {resource.resource_name}")
        print(f"  Type: {resource.resource_type.value}")
        print(f"  Producing Sector: {resource.producing_sector_id}")
        print(f"  Base Unit: {resource.base_unit}")
        print(f"  Density: {resource.density} kg/m³")
        print(f"  Value per Unit: ${resource.value_per_unit}")
        print(f"  Criticality: {resource.criticality}")
        print()
    
    return example_resources


def demonstrate_resource_consumption_matrix():
    """Demonstrate the Resource Input Matrix (R) design."""
    print("=" * 80)
    print("2. RESOURCE CONSUMPTION MATRIX (R) DEMONSTRATION")
    print("=" * 80)
    
    # Create resource model with example data
    model = create_example_resource_model()
    
    # Add our example resources
    example_resources = demonstrate_resource_definitions()
    model.add_resources(example_resources)
    
    # Initialize matrix with example sectors
    example_sectors = ["S1", "S2", "S3"]  # Healthcare, Food & Agriculture, Energy
    model.initialize_resource_matrix(example_sectors)
    
    # Set consumption patterns (R[i][j] = units of resource i per unit of sector j output)
    consumption_patterns = [
        # (resource_id, sector_id, consumption_rate, consumption_type)
        ("R1.1", "S1", 0.1, "direct"),    # Healthcare needs medicines
        ("R1.2", "S1", 0.05, "direct"),   # Healthcare needs PPE
        ("R2", "S2", 0.001, "direct"),    # Agriculture needs farm equipment
        ("R3", "S3", 0.2, "direct"),      # Energy needs electric components
        ("R3", "S1", 0.01, "indirect"),   # Healthcare needs some electric components
        ("R3", "S2", 0.02, "indirect"),   # Agriculture needs some electric components
    ]
    
    for resource_id, sector_id, rate, cons_type in consumption_patterns:
        model.set_resource_consumption(resource_id, sector_id, rate, consumption_type=cons_type)
    
    print("Resource Consumption Matrix (R) - Units of resource per unit of sector output:")
    print("-" * 70)
    print(f"{'Resource':<15} {'S1 (Healthcare)':<15} {'S2 (Agriculture)':<15} {'S3 (Energy)':<15}")
    print("-" * 70)
    
    for resource_id in ["R1.1", "R1.2", "R2", "R3"]:
        if resource_id in model.resources:
            row = []
            for sector_id in example_sectors:
                rate = model.get_consumption(resource_id, sector_id)
                row.append(f"{rate:.3f}")
            print(f"{resource_id:<15} {' '.join(f'{r:<15}' for r in row)}")
    
    print()
    
    # Demonstrate total demand calculation
    sector_outputs = np.array([1000, 2000, 1500])  # Example sector outputs
    print(f"Example Sector Outputs: {sector_outputs}")
    print()
    
    print("Total Resource Demand Calculation:")
    print("-" * 40)
    for resource_id in ["R1.1", "R1.2", "R2", "R3"]:
        if resource_id in model.resources:
            total_demand = model.resource_matrix.get_total_resource_demand(resource_id, sector_outputs)
            resource = model.resources[resource_id]
            print(f"{resource.resource_name} ({resource_id}): {total_demand:.1f} {resource.base_unit}")
    
    return model


def demonstrate_transportation_cost_calculation():
    """Demonstrate the transportation cost calculation framework."""
    print("=" * 80)
    print("3. TRANSPORTATION COST CALCULATION FRAMEWORK")
    print("=" * 80)
    
    # Create example locations
    locations = {
        "MINE001": Location("MINE001", "Iron Mine", 40.0, -100.0, location_type="mine"),
        "FACTORY001": Location("FACTORY001", "Steel Factory", 41.0, -99.0, location_type="factory"),
        "HOSPITAL001": Location("HOSPITAL001", "City Hospital", 40.5, -99.5, location_type="hospital"),
        "FARM001": Location("FARM001", "Agricultural Center", 40.2, -99.8, location_type="farm"),
        "POWER001": Location("POWER001", "Power Plant", 40.8, -99.2, location_type="power_plant")
    }
    
    # Create resource model
    model = create_example_resource_model()
    
    # Create transportation calculator
    transport_calc = TransportationCostCalculator()
    
    print("Example Locations:")
    print("-" * 30)
    for loc_id, location in locations.items():
        print(f"{loc_id}: {location.name} ({location.lat}, {location.lon})")
    print()
    
    # Demonstrate cost calculation for different resources and transport modes
    test_cases = [
        ("R001", "Iron Ore", "MINE001", "FACTORY001", 1000.0, "ton"),
        ("R002", "Crude Oil", "MINE001", "POWER001", 500.0, "barrel"),
        ("R005", "Medicines", "FACTORY001", "HOSPITAL001", 100.0, "unit"),
        ("R006", "Farm Equipment", "FACTORY001", "FARM001", 5.0, "unit")
    ]
    
    print("Transportation Cost Calculations:")
    print("-" * 50)
    
    for resource_id, resource_name, origin_id, dest_id, quantity, unit in test_cases:
        if resource_id in model.resources:
            resource = model.resources[resource_id]
            origin = locations[origin_id]
            destination = locations[dest_id]
            
            print(f"\nTransporting {quantity} {unit} of {resource_name} from {origin.name} to {destination.name}")
            print("-" * 60)
            
            # Calculate costs for different transport modes
            for transport_mode in [TransportMode.TRUCK, TransportMode.RAIL, TransportMode.AIRCRAFT]:
                cost_details = transport_calc.calculate_transportation_cost(
                    resource=resource,
                    quantity=quantity,
                    origin=origin,
                    destination=destination,
                    transport_mode=transport_mode
                )
                
                print(f"{transport_mode.value.upper():<8}: ${cost_details['total_cost']:>8.2f} "
                      f"({cost_details['distance_km']:.1f} km, {cost_details['emissions_kg_co2']:.1f} kg CO2)")
    
    print()
    
    # Demonstrate cost breakdown for one example
    print("Detailed Cost Breakdown Example:")
    print("-" * 40)
    iron_ore = model.resources["R001"]
    cost_details = transport_calc.calculate_transportation_cost(
        resource=iron_ore,
        quantity=1000.0,
        origin=locations["MINE001"],
        destination=locations["FACTORY001"],
        transport_mode=TransportMode.TRUCK
    )
    
    print(f"Resource: {iron_ore.resource_name}")
    print(f"Quantity: {cost_details['quantity']} {iron_ore.base_unit}")
    print(f"Distance: {cost_details['distance_km']:.1f} km")
    print(f"Transport Mode: {cost_details['transport_mode']}")
    print()
    print("Cost Breakdown:")
    print(f"  Base Cost:        ${cost_details['base_cost']:>8.2f}")
    print(f"  Fuel Cost:        ${cost_details['fuel_cost']:>8.2f}")
    print(f"  Labor Cost:       ${cost_details['labor_cost']:>8.2f}")
    print(f"  Maintenance:      ${cost_details['maintenance_cost']:>8.2f}")
    print(f"  Insurance:        ${cost_details['insurance_cost']:>8.2f}")
    print(f"  Total Cost:       ${cost_details['total_cost']:>8.2f}")
    print(f"  Cost per Unit:    ${cost_details['cost_per_unit']:>8.2f}")
    print(f"  Cost per km:      ${cost_details['cost_per_km']:>8.2f}")
    print(f"  Emissions:        {cost_details['emissions_kg_co2']:>8.1f} kg CO2")
    
    return model, locations


def demonstrate_integration_and_optimization():
    """Demonstrate integration with economic planning system."""
    print("=" * 80)
    print("4. INTEGRATION WITH ECONOMIC PLANNING SYSTEM")
    print("=" * 80)
    
    # Create integrated resource system
    integration = create_integrated_resource_system()
    
    # Example sector outputs (simulating economic plan)
    sector_outputs = np.array([1000, 2000, 1500, 500, 100, 5000, 3000, 2000])
    
    print("Example Economic Plan (Sector Outputs):")
    print("-" * 40)
    sector_names = ["Healthcare", "Agriculture", "Energy", "Pharmaceuticals", 
                   "Farm Equipment", "Oil & Gas", "Power Gen", "Metal Products"]
    for i, (name, output) in enumerate(zip(sector_names, sector_outputs)):
        print(f"S{i+1:02d} ({name}): {output:>8.0f} units")
    print()
    
    # Generate comprehensive resource flow report
    print("Generating Resource Flow Report...")
    print("-" * 40)
    
    report = integration.generate_resource_flow_report(sector_outputs)
    
    print(f"Resource Flow Report Summary:")
    print(f"  Total Resources Required: {len(report['resource_summary']['total_requirements'])}")
    print(f"  Total Resource Value: ${report['resource_summary']['total_value']:,.2f}")
    print(f"  Critical Resources: {len(report['resource_summary']['critical_resources'])}")
    print()
    
    # Show resource type summary
    print("Resource Requirements by Type:")
    print("-" * 35)
    for resource_type, summary in report['resource_summary']['resource_types'].items():
        print(f"{resource_type.replace('_', ' ').title()}:")
        print(f"  Count: {summary['count']}")
        print(f"  Total Quantity: {summary['total_quantity']:,.1f}")
        print(f"  Total Value: ${summary['total_value']:,.2f}")
        print()
    
    # Show critical resources
    if report['resource_summary']['critical_resources']:
        print("Critical Resources:")
        print("-" * 20)
        for critical in report['resource_summary']['critical_resources'][:3]:  # Show top 3
            print(f"{critical['name']} ({critical['resource_id']}):")
            print(f"  Required: {critical['required']:,.1f}")
            print(f"  Available: {critical['available']:,.1f}")
            print(f"  Criticality: {critical['criticality']:.2f}")
            if critical['shortage'] > 0:
                print(f"  SHORTAGE: {critical['shortage']:,.1f} units")
            print()
    
    # Show optimization recommendations
    if report['recommendations']:
        print("Optimization Recommendations:")
        print("-" * 35)
        for i, rec in enumerate(report['recommendations'][:5], 1):  # Show top 5
            print(f"{i}. {rec['title']} ({rec['priority']} priority)")
            print(f"   {rec['description']}")
            if 'details' in rec:
                for detail in rec['details']:
                    print(f"   - {detail}")
            print()
    
    return integration, report


def demonstrate_transportation_network_optimization():
    """Demonstrate transportation network cost optimization."""
    print("=" * 80)
    print("5. TRANSPORTATION NETWORK OPTIMIZATION")
    print("=" * 80)
    
    # Create locations for transportation network
    locations = {
        "MINE001": Location("MINE001", "Iron Mine", 40.0, -100.0, location_type="mine"),
        "OIL001": Location("OIL001", "Oil Field", 40.5, -100.5, location_type="oil_field"),
        "FACTORY001": Location("FACTORY001", "Steel Factory", 41.0, -99.0, location_type="factory"),
        "HOSPITAL001": Location("HOSPITAL001", "City Hospital", 40.5, -99.5, location_type="hospital"),
        "FARM001": Location("FARM001", "Agricultural Center", 40.2, -99.8, location_type="farm"),
        "POWER001": Location("POWER001", "Power Plant", 40.8, -99.2, location_type="power_plant")
    }
    
    # Create resource flows
    resource_flows = {
        ("R001", "FACTORY001"): 1000.0,  # Iron ore to steel factory
        ("R002", "POWER001"): 500.0,     # Oil to power plant
        ("R003", "FACTORY001"): 200.0,   # Steel to factory (internal)
        ("R005", "HOSPITAL001"): 100.0,  # Medicines to hospital
        ("R006", "FARM001"): 5.0,        # Farm equipment to farm
    }
    
    # Create integration system
    integration = create_integrated_resource_system()
    
    print("Transportation Network Analysis:")
    print("-" * 40)
    print("Resource Flows:")
    for (resource_id, dest_id), quantity in resource_flows.items():
        resource = integration.resource_model.resources.get(resource_id)
        destination = locations[dest_id]
        if resource:
            print(f"  {resource.resource_name}: {quantity} {resource.base_unit} → {destination.name}")
    print()
    
    # Calculate transportation costs
    transport_analysis = integration.calculate_transportation_network_costs(resource_flows, locations)
    
    print("Transportation Cost Analysis:")
    print("-" * 35)
    print(f"Total Transport Cost: ${transport_analysis['total_transport_cost']:,.2f}")
    print()
    
    # Show mode efficiency analysis
    if 'mode_efficiency' in transport_analysis['cost_analysis']:
        print("Transport Mode Efficiency:")
        print("-" * 30)
        for mode, efficiency in transport_analysis['cost_analysis']['mode_efficiency'].items():
            print(f"{mode.upper()}:")
            print(f"  Average Cost: ${efficiency['average_cost']:,.2f}")
            print(f"  Flow Count: {efficiency['flow_count']}")
            print()
    
    # Show optimization opportunities
    if transport_analysis['recommendations']:
        print("Transportation Optimization Recommendations:")
        print("-" * 50)
        for i, rec in enumerate(transport_analysis['recommendations'], 1):
            print(f"{i}. {rec['title']} ({rec['priority']} priority)")
            print(f"   {rec['description']}")
            if 'potential_savings' in rec:
                print(f"   Potential Savings: {rec['potential_savings']}")
            print()
    
    return transport_analysis


def main():
    """Run the complete resource flow model demonstration."""
    print("RESOURCE FLOW AND TRANSPORTATION COST MODEL DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows the complete resource flow and transportation")
    print("cost model designed for the cybernetic planning system.")
    print()
    
    try:
        # 1. Demonstrate resource data structure
        demonstrate_resource_definitions()
        
        # 2. Demonstrate resource consumption matrix
        model = demonstrate_resource_consumption_matrix()
        
        # 3. Demonstrate transportation cost calculation
        model, locations = demonstrate_transportation_cost_calculation()
        
        # 4. Demonstrate integration with economic planning
        integration, report = demonstrate_integration_and_optimization()
        
        # 5. Demonstrate transportation network optimization
        transport_analysis = demonstrate_transportation_network_optimization()
        
        print("=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("The resource flow and transportation cost model provides:")
        print("✓ Comprehensive resource definition and management")
        print("✓ Resource consumption matrix for economic planning")
        print("✓ Multi-modal transportation cost calculation")
        print("✓ Integration with existing economic planning system")
        print("✓ Optimization recommendations and cost analysis")
        print("✓ Scalable framework for resource flow modeling")
        print()
        print("This model can be extended and customized for specific")
        print("economic planning scenarios and resource requirements.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
