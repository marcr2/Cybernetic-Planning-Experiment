#!/usr/bin/env python3
"""
Unified Simulation System Demonstration

Demonstrates the complete unified simulation system that merges
spatial and economic simulation with comprehensive reporting.
"""

import sys
import os
from pathlib import Path
import time
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cybernetic_planning.core.unified_simulation_system import (
    UnifiedSimulationSystem, 
    UnifiedSimulationConfig,
    SpatialEconomicIntegration
)
from cybernetic_planning.core.unified_time_management import (
    UnifiedTimeManager, 
    TimeManagementConfig,
    TimeScale,
    UpdatePriority
)
from cybernetic_planning.core.enhanced_sector_settlement_mapper import (
    EnhancedSectorSettlementMapper,
    SectorType,
    SettlementHierarchy
)
from cybernetic_planning.core.unified_simulation_loop import (
    UnifiedSimulationLoop,
    SimulationLoopConfig
)
from cybernetic_planning.core.unified_reporting_system import (
    UnifiedReportingSystem,
    ReportConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_basic_unified_simulation():
    """Demonstrate basic unified simulation."""
    print("=" * 80)
    print("UNIFIED SIMULATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create unified simulation configuration
    config = UnifiedSimulationConfig(
        map_width=100,
        map_height=100,
        terrain_distribution={
            "flatland": 0.4,
            "forest": 0.3,
            "mountain": 0.2,
            "water": 0.1
        },
        num_cities=3,
        num_towns=8,
        total_population=500000,
        rural_population_percent=0.25,
        urban_concentration="medium",
        n_sectors=12,
        technology_density=0.4,
        resource_count=6,
        policy_goals=[
            "Increase industrial production",
            "Improve living standards",
            "Develop infrastructure",
            "Enhance disaster resilience"
        ],
        simulation_duration_months=24,  # 2-year simulation
        spatial_update_frequency="daily",
        economic_update_frequency="monthly",
        disaster_probability=0.08,
        enable_bidirectional_feedback=True,
        enable_spatial_constraints=True,
        enable_disaster_economic_impact=True,
        enable_infrastructure_economic_feedback=True
    )
    
    print("Configuration:")
    print(f"  Map size: {config.map_width}x{config.map_height}")
    print(f"  Settlements: {config.num_cities} cities, {config.num_towns} towns")
    print(f"  Population: {config.total_population:,}")
    print(f"  Economic sectors: {config.n_sectors}")
    print(f"  Simulation duration: {config.simulation_duration_months} months")
    print(f"  Spatial updates: {config.spatial_update_frequency}")
    print(f"  Economic updates: {config.economic_update_frequency}")
    print()
    
    # Create unified simulation system
    print("Creating unified simulation system...")
    unified_system = UnifiedSimulationSystem(config)
    
    # Create unified simulation
    print("Creating unified simulation...")
    creation_result = unified_system.create_unified_simulation()
    
    if not creation_result["success"]:
        print(f"❌ Failed to create unified simulation: {creation_result['error']}")
        return
    
    print("✅ Unified simulation created successfully!")
    print(f"  Spatial summary: {creation_result['spatial_summary']}")
    print(f"  Economic summary: {creation_result['economic_summary']}")
    print(f"  Integration summary: {creation_result['integration_summary']}")
    print()
    
    # Run unified simulation
    print("Running unified simulation...")
    start_time = time.time()
    
    simulation_result = unified_system.run_unified_simulation(
        duration_months=config.simulation_duration_months,
        spatial_update_frequency=config.spatial_update_frequency,
        economic_update_frequency=config.economic_update_frequency
    )
    
    end_time = time.time()
    
    if not simulation_result["success"]:
        print(f"❌ Simulation failed: {simulation_result['error']}")
        return
    
    print("✅ Simulation completed successfully!")
    print(f"  Duration: {simulation_result['simulation_duration_months']} months")
    print(f"  Total simulation time: {simulation_result['total_simulation_time']:.2f} seconds")
    print(f"  Average step time: {simulation_result['average_step_time']:.3f} seconds")
    print()
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    reporting_system = UnifiedReportingSystem()
    
    report_result = reporting_system.generate_comprehensive_report(
        simulation_result,
        {"config": config.__dict__}
    )
    
    if report_result["success"]:
        print("✅ Comprehensive report generated!")
        print(f"  Report path: {report_result['report_path']}")
        print(f"  Report sections: {', '.join(report_result['report_sections'])}")
        print()
        
        # Show metrics summary
        metrics = report_result["metrics_summary"]
        print("Key Metrics:")
        print(f"  Spatial efficiency: {metrics['spatial_efficiency']:.2%}")
        print(f"  Economic growth: {metrics['economic_growth']:.2%}")
        print(f"  Disaster resilience: {metrics['disaster_resilience']:.2%}")
        print(f"  Performance score: {metrics['performance_score']:.2%}")
    else:
        print(f"❌ Report generation failed: {report_result['error']}")
    
    print()
    print("=" * 80)
    print("BASIC UNIFIED SIMULATION DEMO COMPLETED")
    print("=" * 80)

def demo_advanced_unified_simulation():
    """Demonstrate advanced unified simulation with custom components."""
    print("\n" + "=" * 80)
    print("ADVANCED UNIFIED SIMULATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create enhanced sector-settlement mapper
    print("Setting up enhanced sector-settlement mapping...")
    mapper = EnhancedSectorSettlementMapper()
    
    # Add sectors with realistic characteristics
    sectors_data = [
        {
            "id": "agriculture",
            "name": "Agriculture",
            "labor_intensity": 0.8,
            "capital_intensity": 0.3,
            "resource_intensity": 0.9,
            "transport_sensitivity": 0.7,
            "agglomeration_preference": 0.2,
            "infrastructure_requirements": ["roads", "irrigation"],
            "environmental_impact": 0.3,
            "technology_level": 0.6,
            "market_access_importance": 0.8
        },
        {
            "id": "manufacturing",
            "name": "Manufacturing",
            "labor_intensity": 0.6,
            "capital_intensity": 0.8,
            "resource_intensity": 0.7,
            "transport_sensitivity": 0.5,
            "agglomeration_preference": 0.7,
            "infrastructure_requirements": ["roads", "railway", "utilities"],
            "environmental_impact": 0.6,
            "technology_level": 0.8,
            "market_access_importance": 0.9
        },
        {
            "id": "services",
            "name": "Services",
            "labor_intensity": 0.4,
            "capital_intensity": 0.5,
            "resource_intensity": 0.2,
            "transport_sensitivity": 0.3,
            "agglomeration_preference": 0.8,
            "infrastructure_requirements": ["roads", "utilities", "communications"],
            "environmental_impact": 0.1,
            "technology_level": 0.9,
            "market_access_importance": 1.0
        }
    ]
    
    for sector_data in sectors_data:
        result = mapper.add_sector(sector_data)
        if result["success"]:
            print(f"  ✅ Added sector: {sector_data['name']} ({result['sector_type']})")
        else:
            print(f"  ❌ Failed to add sector: {result['error']}")
    
    # Add settlements with realistic characteristics
    settlements_data = [
        {
            "id": "metropolis_1",
            "name": "Central Metropolis",
            "population": 500000,
            "economic_importance": 0.9,
            "infrastructure_quality": 0.9,
            "resource_endowment": 0.6,
            "market_access": 1.0,
            "labor_availability": 0.9,
            "capital_availability": 0.9,
            "environmental_capacity": 0.7,
            "existing_sectors": [],
            "specialization_index": 0.0
        },
        {
            "id": "city_1",
            "name": "Industrial City",
            "population": 150000,
            "economic_importance": 0.7,
            "infrastructure_quality": 0.8,
            "resource_endowment": 0.8,
            "market_access": 0.8,
            "labor_availability": 0.8,
            "capital_availability": 0.8,
            "environmental_capacity": 0.6,
            "existing_sectors": [],
            "specialization_index": 0.0
        },
        {
            "id": "town_1",
            "name": "Agricultural Town",
            "population": 25000,
            "economic_importance": 0.5,
            "infrastructure_quality": 0.6,
            "resource_endowment": 0.9,
            "market_access": 0.6,
            "labor_availability": 0.7,
            "capital_availability": 0.5,
            "environmental_capacity": 0.9,
            "existing_sectors": [],
            "specialization_index": 0.0
        }
    ]
    
    for settlement_data in settlements_data:
        result = mapper.add_settlement(settlement_data)
        if result["success"]:
            print(f"  ✅ Added settlement: {settlement_data['name']} ({result['hierarchy_level']})")
        else:
            print(f"  ❌ Failed to add settlement: {result['error']}")
    
    # Add constraints
    constraints = [
        {
            "constraint_type": "minimum_population",
            "min_value": 10000,
            "weight": 1.0,
            "description": "Minimum population for industrial sectors"
        },
        {
            "constraint_type": "maximum_sectors_per_settlement",
            "max_value": 5,
            "weight": 0.8,
            "description": "Maximum sectors per settlement"
        }
    ]
    
    for constraint_data in constraints:
        from cybernetic_planning.core.enhanced_sector_settlement_mapper import MappingConstraint
        constraint = MappingConstraint(**constraint_data)
        result = mapper.add_constraint(constraint)
        if result["success"]:
            print(f"  ✅ Added constraint: {constraint_data['description']}")
        else:
            print(f"  ❌ Failed to add constraint: {result['error']}")
    
    # Optimize mapping
    print("\nOptimizing sector-settlement mapping...")
    optimization_result = mapper.optimize_mapping(method="hungarian")
    
    if optimization_result["success"]:
        print("✅ Mapping optimization completed!")
        print(f"  Mappings created: {optimization_result['mappings_created']}")
        print(f"  Average suitability: {optimization_result['average_suitability']:.3f}")
        print(f"  Constraint violations: {optimization_result['constraint_violations']}")
        
        # Show optimization summary
        summary = optimization_result["optimization_summary"]
        print(f"  Total production capacity: {summary['total_production_capacity']:.2f}")
        print(f"  Constraint satisfaction rate: {summary['constraint_satisfaction_rate']:.2%}")
    else:
        print(f"❌ Mapping optimization failed: {optimization_result['error']}")
        return
    
    # Get mapping analysis
    print("\nAnalyzing mapping results...")
    analysis_result = mapper.get_mapping_analysis()
    
    if analysis_result["success"]:
        print("✅ Mapping analysis completed!")
        print(f"  Total mappings: {analysis_result['total_mappings']}")
        print(f"  Sector type distribution: {analysis_result['sector_type_distribution']}")
        print(f"  Hierarchy distribution: {analysis_result['hierarchy_distribution']}")
        
        efficiency = analysis_result["efficiency_analysis"]
        print(f"  Average suitability: {efficiency['average_suitability']:.3f}")
        print(f"  High efficiency mappings: {efficiency['high_efficiency_count']}")
        print(f"  Medium efficiency mappings: {efficiency['medium_efficiency_count']}")
        print(f"  Low efficiency mappings: {efficiency['low_efficiency_count']}")
    else:
        print(f"❌ Mapping analysis failed: {analysis_result['error']}")
    
    print()
    print("=" * 80)
    print("ADVANCED UNIFIED SIMULATION DEMO COMPLETED")
    print("=" * 80)

def demo_time_management():
    """Demonstrate unified time management system."""
    print("\n" + "=" * 80)
    print("UNIFIED TIME MANAGEMENT DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create time management configuration
    time_config = TimeManagementConfig(
        simulation_start_date=datetime(2024, 1, 1),
        max_simulation_days=365,  # 1 year
        enable_time_compression=True,
        enable_event_scheduling=True,
        enable_performance_monitoring=True
    )
    
    # Create time manager
    print("Creating unified time manager...")
    time_manager = UnifiedTimeManager(time_config)
    
    # Register systems with different update frequencies
    systems = [
        ("spatial_system", TimeScale.DAILY, UpdatePriority.HIGH),
        ("economic_system", TimeScale.MONTHLY, UpdatePriority.HIGH),
        ("integration_system", TimeScale.WEEKLY, UpdatePriority.MEDIUM),
        ("reporting_system", TimeScale.QUARTERLY, UpdatePriority.LOW)
    ]
    
    for system_name, time_scale, priority in systems:
        result = time_manager.register_system(system_name, time_scale, priority)
        if result["success"]:
            print(f"  ✅ Registered {system_name} with {time_scale.value} updates ({priority.value} priority)")
        else:
            print(f"  ❌ Failed to register {system_name}: {result['error']}")
    
    # Schedule some events
    print("\nScheduling temporal events...")
    
    def disaster_event_handler(**kwargs):
        return {"event": "disaster", "impact": "moderate"}
    
    def economic_review_handler(**kwargs):
        return {"event": "economic_review", "status": "completed"}
    
    events = [
        ("disaster_event", datetime(2024, 3, 15), disaster_event_handler, UpdatePriority.HIGH),
        ("economic_review", datetime(2024, 6, 1), economic_review_handler, UpdatePriority.MEDIUM),
        ("annual_report", datetime(2024, 12, 31), lambda **kwargs: {"event": "annual_report", "status": "generated"}, UpdatePriority.LOW)
    ]
    
    for event_type, scheduled_time, handler, priority in events:
        result = time_manager.schedule_event(
            event_type=event_type,
            scheduled_time=scheduled_time,
            handler_function=handler,
            priority=priority
        )
        if result["success"]:
            print(f"  ✅ Scheduled {event_type} for {scheduled_time.strftime('%Y-%m-%d')}")
        else:
            print(f"  ❌ Failed to schedule {event_type}: {result['error']}")
    
    # Start simulation
    print("\nStarting time management simulation...")
    start_result = time_manager.start_simulation()
    
    if not start_result["success"]:
        print(f"❌ Failed to start simulation: {start_result['error']}")
        return
    
    print("✅ Time management simulation started!")
    
    # Simulate time advancement
    print("\nSimulating time advancement...")
    total_days = 100  # Simulate 100 days
    
    for day in range(total_days):
        advance_result = time_manager.advance_time(1)
        
        if not advance_result["success"]:
            print(f"❌ Failed to advance time on day {day}: {advance_result['error']}")
            break
        
        # Report progress every 20 days
        if day % 20 == 0:
            systems_updated = advance_result["systems_updated"]
            events_executed = advance_result["events_executed"]
            print(f"  Day {day}: Updated {len(systems_updated)} systems, executed {len(events_executed)} events")
            
            if events_executed:
                for event in events_executed:
                    print(f"    Event: {event['event_type']} - {event['result']}")
    
    # Get final status
    print("\nFinal simulation status:")
    status = time_manager.get_system_update_status()
    
    for system_name, system_status in status.items():
        print(f"  {system_name}:")
        print(f"    Updates: {system_status['update_count']}")
        print(f"    Last update: {system_status['last_update']}")
        print(f"    Average update time: {system_status['average_update_time']:.3f}s")
    
    # Stop simulation
    stop_result = time_manager.stop_simulation()
    if stop_result["success"]:
        print("\n✅ Time management simulation completed!")
        print(f"  Final time: {stop_result['final_time']}")
        print(f"  Total simulation days: {stop_result['total_simulation_days']}")
    else:
        print(f"❌ Failed to stop simulation: {stop_result['error']}")
    
    print()
    print("=" * 80)
    print("TIME MANAGEMENT DEMO COMPLETED")
    print("=" * 80)

def demo_simulation_loop():
    """Demonstrate unified simulation loop."""
    print("\n" + "=" * 80)
    print("UNIFIED SIMULATION LOOP DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create simulation loop configuration
    loop_config = SimulationLoopConfig(
        max_concurrent_threads=2,
        step_timeout_seconds=10.0,
        enable_parallel_execution=True,
        enable_performance_monitoring=True,
        enable_error_recovery=True,
        max_consecutive_errors=3,
        checkpoint_interval_steps=50,
        enable_real_time_output=True,
        output_frequency_steps=10
    )
    
    # Create unified simulation system
    unified_config = UnifiedSimulationConfig(
        map_width=80,
        map_height=80,
        num_cities=2,
        num_towns=5,
        total_population=300000,
        n_sectors=8,
        simulation_duration_months=12,  # 1 year
        spatial_update_frequency="daily",
        economic_update_frequency="monthly"
    )
    
    unified_system = UnifiedSimulationSystem(unified_config)
    
    # Create unified simulation
    creation_result = unified_system.create_unified_simulation()
    if not creation_result["success"]:
        print(f"❌ Failed to create unified simulation: {creation_result['error']}")
        return
    
    print("✅ Unified simulation created for loop demonstration")
    
    # Create simulation loop
    print("Creating unified simulation loop...")
    simulation_loop = UnifiedSimulationLoop(unified_system, loop_config)
    
    # Add callbacks
    def step_callback(step_result):
        if step_result.step_number % 20 == 0:
            print(f"    Step {step_result.step_number}: "
                  f"Spatial updates: {len(step_result.spatial_updates)}, "
                  f"Economic updates: {len(step_result.economic_updates)}, "
                  f"Errors: {len(step_result.errors)}")
    
    def progress_callback(current_step, total_steps):
        if current_step % 30 == 0:
            print(f"    Progress: {current_step}/{total_steps} steps completed")
    
    def error_callback(step_result):
        if step_result.errors:
            print(f"    Error in step {step_result.step_number}: {step_result.errors[0]}")
    
    simulation_loop.add_step_callback(step_callback)
    simulation_loop.add_progress_callback(progress_callback)
    simulation_loop.add_error_callback(error_callback)
    
    print("✅ Simulation loop created with callbacks")
    
    # Run simulation loop
    print("\nRunning simulation loop...")
    loop_result = simulation_loop.run_simulation(
        duration_months=12,
        spatial_frequency="daily",
        economic_frequency="monthly"
    )
    
    if loop_result["success"]:
        print("✅ Simulation loop completed successfully!")
        print(f"  Duration: {loop_result['simulation_duration_months']} months")
        print(f"  Total simulation time: {loop_result['total_simulation_time']:.2f} seconds")
        print(f"  Total steps: {loop_result['total_steps']}")
        print(f"  Average step time: {loop_result['average_step_time']:.3f} seconds")
        print(f"  Error count: {loop_result['error_count']}")
        
        # Show final results
        final_results = loop_result["final_results"]
        print(f"  Final spatial metrics: {len(final_results.get('spatial_metrics', {}))} metrics")
        print(f"  Final economic metrics: {len(final_results.get('economic_metrics', {}))} metrics")
        print(f"  Final integration metrics: {len(final_results.get('integration_metrics', {}))} metrics")
    else:
        print(f"❌ Simulation loop failed: {loop_result['error']}")
    
    # Demonstrate checkpointing
    print("\nDemonstrating checkpointing...")
    checkpoint_result = simulation_loop.create_checkpoint("demo_checkpoint")
    
    if checkpoint_result["success"]:
        print(f"✅ Checkpoint created: {checkpoint_result['checkpoint_name']}")
        print(f"  Checkpoint ID: {checkpoint_result['checkpoint_id']}")
        print(f"  Timestamp: {checkpoint_result['timestamp']}")
    else:
        print(f"❌ Checkpoint creation failed: {checkpoint_result['error']}")
    
    # Get simulation status
    status = simulation_loop.get_simulation_status()
    print(f"\nFinal simulation status:")
    print(f"  Current step: {status['current_step']}")
    print(f"  Total steps completed: {status['total_steps_completed']}")
    print(f"  Error count: {status['error_count']}")
    print(f"  Checkpoints created: {status['checkpoints_created']}")
    
    print()
    print("=" * 80)
    print("SIMULATION LOOP DEMO COMPLETED")
    print("=" * 80)

def demo_comparative_analysis():
    """Demonstrate comparative analysis across multiple simulation runs."""
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create reporting system
    reporting_system = UnifiedReportingSystem()
    
    # Simulate multiple runs with different configurations
    print("Running multiple simulation scenarios...")
    simulation_runs = []
    
    scenarios = [
        {
            "name": "Baseline Scenario",
            "config": {
                "disaster_probability": 0.05,
                "urban_concentration": "medium",
                "economic_update_frequency": "monthly"
            }
        },
        {
            "name": "High Disaster Scenario",
            "config": {
                "disaster_probability": 0.15,
                "urban_concentration": "medium",
                "economic_update_frequency": "monthly"
            }
        },
        {
            "name": "High Concentration Scenario",
            "config": {
                "disaster_probability": 0.05,
                "urban_concentration": "high",
                "economic_update_frequency": "monthly"
            }
        },
        {
            "name": "Frequent Economic Updates",
            "config": {
                "disaster_probability": 0.05,
                "urban_concentration": "medium",
                "economic_update_frequency": "weekly"
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\nRunning {scenario['name']}...")
        
        # Create configuration for this scenario
        config = UnifiedSimulationConfig(
            map_width=60,
            map_height=60,
            num_cities=2,
            num_towns=4,
            total_population=200000,
            n_sectors=6,
            simulation_duration_months=6,  # 6 months for faster demo
            **scenario['config']
        )
        
        # Create and run simulation
        unified_system = UnifiedSimulationSystem(config)
        creation_result = unified_system.create_unified_simulation()
        
        if creation_result["success"]:
            simulation_result = unified_system.run_unified_simulation(
                duration_months=config.simulation_duration_months
            )
            
            if simulation_result["success"]:
                simulation_runs.append({
                    "scenario_name": scenario['name'],
                    "config": config.__dict__,
                    "results": simulation_result
                })
                print(f"  ✅ {scenario['name']} completed successfully")
            else:
                print(f"  ❌ {scenario['name']} simulation failed: {simulation_result['error']}")
        else:
            print(f"  ❌ {scenario['name']} creation failed: {creation_result['error']}")
    
    if len(simulation_runs) < 2:
        print("❌ Not enough successful simulation runs for comparison")
        return
    
    print(f"\n✅ Completed {len(simulation_runs)} simulation runs")
    
    # Generate comparative report
    print("Generating comparative analysis...")
    comparative_result = reporting_system.generate_comparative_report(
        simulation_runs,
        comparison_metrics=["spatial_efficiency", "economic_growth", "disaster_resilience", "performance_score"]
    )
    
    if comparative_result["success"]:
        print("✅ Comparative analysis completed!")
        print(f"  Report path: {comparative_result['report_path']}")
        print(f"  Number of runs compared: {comparative_result['number_of_runs']}")
        
        # Show comparative analysis summary
        analysis = comparative_result["comparative_analysis"]
        print(f"  Comparative analysis: {analysis}")
    else:
        print(f"❌ Comparative analysis failed: {comparative_result['error']}")
    
    print()
    print("=" * 80)
    print("COMPARATIVE ANALYSIS DEMO COMPLETED")
    print("=" * 80)

def main():
    """Run all demonstrations."""
    print("UNIFIED SIMULATION SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("This demonstration showcases the complete unified simulation system")
    print("that merges spatial and economic simulation with advanced features.")
    print()
    
    try:
        # Run all demonstrations
        demo_basic_unified_simulation()
        demo_advanced_unified_simulation()
        demo_time_management()
        demo_simulation_loop()
        demo_comparative_analysis()
        
        print("\n" + "=" * 80)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("The unified simulation system demonstrates:")
        print("✅ Complete integration of spatial and economic simulation")
        print("✅ Advanced sector-settlement mapping with optimization")
        print("✅ Sophisticated time management with multiple update frequencies")
        print("✅ Parallel execution and performance monitoring")
        print("✅ Comprehensive reporting and visualization")
        print("✅ Comparative analysis across multiple scenarios")
        print("✅ Checkpointing and error recovery")
        print("✅ Real-time progress monitoring and callbacks")
        print()
        print("Key features implemented:")
        print("• UnifiedSimulationSystem - Main coordination system")
        print("• SpatialEconomicIntegration - Bidirectional feedback")
        print("• UnifiedTimeManager - Multi-scale time management")
        print("• EnhancedSectorSettlementMapper - Economic geography")
        print("• UnifiedSimulationLoop - Parallel execution coordination")
        print("• UnifiedReportingSystem - Comprehensive analysis")
        print()
        print("This system provides a realistic simulation environment")
        print("for testing socialist planned economy strategies with")
        print("both geographical constraints and economic processes.")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
