#!/usr/bin/env python3
"""
Test script to verify the unified simulation fixes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cybernetic_planning.core.unified_simulation_system import UnifiedSimulationSystem, UnifiedSimulationConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unified_simulation():
    """Test the unified simulation system with fixes."""
    try:
        # Create configuration
        config = UnifiedSimulationConfig(
            n_sectors=50,
            map_width=100,
            map_height=100,
            num_cities=5,
            num_towns=10,
            total_population=1000000
        )
        
        # Create unified simulation system
        logger.info("Creating unified simulation system...")
        system = UnifiedSimulationSystem(config)
        
        # Create the simulation
        logger.info("Creating unified simulation...")
        result = system.create_unified_simulation()
        
        if not result.get("success", False):
            logger.error(f"Failed to create simulation: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info("Simulation created successfully!")
        logger.info(f"Spatial: {result.get('spatial_summary', {})}")
        logger.info(f"Economic: {result.get('economic_summary', {})}")
        
        # Debug: Check what's in the economic plan
        if hasattr(system, 'planning_system') and system.planning_system.current_plan:
            plan = system.planning_system.current_plan
            logger.info(f"Economic plan keys: {list(plan.keys())}")
            logger.info(f"Economic plan sectors: {plan.get('sectors', 'Not found')}")
            logger.info(f"Economic plan total_output: {plan.get('total_output', 'Not found')}")
        
        # Test economic simulation step
        if system.economic_simulator:
            logger.info("Testing economic simulation step...")
            economic_result = system._run_economic_step()
            logger.info(f"Economic step result: {economic_result}")
            
            if economic_result.get("total_economic_output", 0) > 0:
                logger.info("✅ Economic simulation is working - output > 0")
                return True
            else:
                logger.warning("⚠️ Economic simulation returned 0 output")
                return False
        else:
            logger.warning("⚠️ No economic simulator available")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unified_simulation()
    if success:
        print("\n✅ Test passed - Unified simulation fixes are working!")
    else:
        print("\n❌ Test failed - Issues remain with unified simulation")
