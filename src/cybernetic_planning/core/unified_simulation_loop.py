"""
Unified Simulation Loop

Coordinates the execution of both spatial and economic simulation systems
with proper synchronization, event handling, and performance optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import unified system components
from .unified_simulation_system import UnifiedSimulationSystem, UnifiedSimulationConfig
from .unified_time_management import UnifiedTimeManager, TimeManagementConfig, TimeScale, UpdatePriority
from .enhanced_sector_settlement_mapper import EnhancedSectorSettlementMapper

logger = logging.getLogger(__name__)

@dataclass
class SimulationStepResult:
    """Result of a single simulation step."""
    step_number: int
    timestamp: datetime
    spatial_updates: Dict[str, Any] = field(default_factory=dict)
    economic_updates: Dict[str, Any] = field(default_factory=dict)
    integration_updates: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class SimulationLoopConfig:
    """Configuration for the unified simulation loop."""
    max_concurrent_threads: int = 4
    step_timeout_seconds: float = 30.0
    enable_parallel_execution: bool = True
    enable_performance_monitoring: bool = True
    enable_error_recovery: bool = True
    max_consecutive_errors: int = 5
    checkpoint_interval_steps: int = 100
    enable_real_time_output: bool = False
    output_frequency_steps: int = 10

class UnifiedSimulationLoop:
    """
    Main simulation loop that coordinates spatial and economic systems.
    
    Features:
    - Parallel execution of spatial and economic updates
    - Event-driven simulation with proper synchronization
    - Performance monitoring and optimization
    - Error handling and recovery
    - Checkpointing and state management
    - Real-time progress reporting
    """
    
    def __init__(self, 
                 unified_system: UnifiedSimulationSystem,
                 loop_config: Optional[SimulationLoopConfig] = None):
        """Initialize the unified simulation loop."""
        self.unified_system = unified_system
        self.config = loop_config or SimulationLoopConfig()
        
        # Simulation state
        self.current_step = 0
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        
        # Results storage
        self.step_results: List[SimulationStepResult] = []
        self.checkpoints: Dict[int, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.error_count = 0
        self.consecutive_errors = 0
        
        # Threading and parallel execution
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.event_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Callbacks and event handlers
        self.step_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        self.progress_callbacks: List[Callable] = []
        
        logger.info("Unified simulation loop initialized")
    
    def add_step_callback(self, callback: Callable) -> Dict[str, Any]:
        """Add callback function to be called after each step."""
        try:
            self.step_callbacks.append(callback)
            return {"success": True, "callback_added": True}
        except Exception as e:
            logger.error(f"Failed to add step callback: {e}")
            return {"success": False, "error": str(e)}
    
    def add_error_callback(self, callback: Callable) -> Dict[str, Any]:
        """Add callback function to be called on errors."""
        try:
            self.error_callbacks.append(callback)
            return {"success": True, "callback_added": True}
        except Exception as e:
            logger.error(f"Failed to add error callback: {e}")
            return {"success": False, "error": str(e)}
    
    def add_progress_callback(self, callback: Callable) -> Dict[str, Any]:
        """Add callback function to be called for progress updates."""
        try:
            self.progress_callbacks.append(callback)
            return {"success": True, "callback_added": True}
        except Exception as e:
            logger.error(f"Failed to add progress callback: {e}")
            return {"success": False, "error": str(e)}
    
    def run_simulation(self, 
                      duration_months: int,
                      spatial_frequency: str = "daily",
                      economic_frequency: str = "monthly") -> Dict[str, Any]:
        """
        Run the unified simulation for specified duration.
        
        Args:
            duration_months: Duration in months
            spatial_frequency: Spatial update frequency
            economic_frequency: Economic update frequency
        
        Returns:
            Dictionary with simulation results
        """
        if not self.unified_system.is_initialized:
            return {"success": False, "error": "Unified system not initialized"}
        
        try:
            logger.info(f"Starting unified simulation for {duration_months} months")
            logger.info(f"Spatial frequency: {spatial_frequency}, Economic frequency: {economic_frequency}")
            
            # Initialize simulation state
            self._initialize_simulation_state()
            
            # Configure time management
            self._configure_time_management(spatial_frequency, economic_frequency)
            
            # Start simulation
            self.is_running = True
            self.should_stop = False
            
            start_time = datetime.now()
            
            # Run simulation loop
            simulation_result = self._run_simulation_loop(duration_months)
            
            # Calculate final metrics
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            self.is_running = False
            
            logger.info(f"Simulation completed in {total_time:.2f} seconds")
            
            return {
                "success": True,
                "simulation_duration_months": duration_months,
                "total_simulation_time": total_time,
                "total_steps": self.current_step,
                "average_step_time": np.mean([r.performance_metrics.get("step_time", 0) for r in self.step_results]) if self.step_results else 0,
                "error_count": self.error_count,
                "final_results": simulation_result
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            self.is_running = False
            return {"success": False, "error": str(e)}
    
    def pause_simulation(self) -> Dict[str, Any]:
        """Pause the simulation."""
        try:
            self.is_paused = True
            logger.info("Simulation paused")
            return {"success": True, "paused": True}
        except Exception as e:
            logger.error(f"Failed to pause simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def resume_simulation(self) -> Dict[str, Any]:
        """Resume the simulation."""
        try:
            self.is_paused = False
            logger.info("Simulation resumed")
            return {"success": True, "resumed": True}
        except Exception as e:
            logger.error(f"Failed to resume simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_simulation(self) -> Dict[str, Any]:
        """Stop the simulation."""
        try:
            self.should_stop = True
            self.is_running = False
            self.is_paused = False
            
            # Clean up resources
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            logger.info("Simulation stopped")
            return {"success": True, "stopped": True}
        except Exception as e:
            logger.error(f"Failed to stop simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "current_step": self.current_step,
            "total_steps_completed": len(self.step_results),
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "performance_metrics": self._get_current_performance_metrics(),
            "checkpoints_created": len(self.checkpoints)
        }
    
    def get_step_results(self, start_step: int = 0, end_step: Optional[int] = None) -> Dict[str, Any]:
        """Get simulation step results."""
        try:
            if not self.step_results:
                return {"success": False, "error": "No step results available"}
            
            end_step = end_step or len(self.step_results)
            results = self.step_results[start_step:end_step]
            
            return {
                "success": True,
                "results": [self._step_result_to_dict(r) for r in results],
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.error(f"Failed to get step results: {e}")
            return {"success": False, "error": str(e)}
    
    def create_checkpoint(self, checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a simulation checkpoint."""
        try:
            checkpoint_id = len(self.checkpoints)
            checkpoint_name = checkpoint_name or f"checkpoint_{checkpoint_id}"
            
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_name": checkpoint_name,
                "timestamp": datetime.now().isoformat(),
                "current_step": self.current_step,
                "simulation_state": self._capture_simulation_state(),
                "step_results": self.step_results.copy() if self.step_results else [],
                "performance_history": self.performance_history.copy() if self.performance_history else []
            }
            
            self.checkpoints[checkpoint_id] = checkpoint_data
            
            logger.info(f"Created checkpoint '{checkpoint_name}' at step {self.current_step}")
            
            return {
                "success": True,
                "checkpoint_id": checkpoint_id,
                "checkpoint_name": checkpoint_name,
                "timestamp": checkpoint_data["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return {"success": False, "error": str(e)}
    
    def restore_checkpoint(self, checkpoint_id: int) -> Dict[str, Any]:
        """Restore simulation from checkpoint."""
        try:
            if checkpoint_id not in self.checkpoints:
                return {"success": False, "error": f"Checkpoint {checkpoint_id} not found"}
            
            checkpoint_data = self.checkpoints[checkpoint_id]
            
            # Restore simulation state
            self.current_step = checkpoint_data["current_step"]
            self.step_results = checkpoint_data["step_results"]
            self.performance_history = checkpoint_data["performance_history"]
            
            logger.info(f"Restored checkpoint '{checkpoint_data['checkpoint_name']}' at step {self.current_step}")
            
            return {
                "success": True,
                "checkpoint_id": checkpoint_id,
                "checkpoint_name": checkpoint_data["checkpoint_name"],
                "restored_step": self.current_step
            }
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return {"success": False, "error": str(e)}
    
    def _initialize_simulation_state(self):
        """Initialize simulation state."""
        self.current_step = 0
        self.step_results = []
        self.error_count = 0
        self.consecutive_errors = 0
        self.performance_history = []
        
        # Initialize thread pool if parallel execution is enabled
        if self.config.enable_parallel_execution:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_concurrent_threads)
    
    def _configure_time_management(self, spatial_frequency: str, economic_frequency: str):
        """Configure time management for simulation."""
        # This would integrate with the unified time management system
        # For now, we'll use the existing configuration
        pass
    
    def _run_simulation_loop(self, duration_months: int) -> Dict[str, Any]:
        """Run the main simulation loop."""
        total_days = duration_months * 30
        last_progress_report = 0
        
        try:
            for day in range(total_days):
                if self.should_stop:
                    break
                
                # Handle pause
                while self.is_paused and not self.should_stop:
                    time.sleep(0.1)
                
                if self.should_stop:
                    break
                
                # Execute simulation step
                step_result = self._execute_simulation_step(day)
                
                # Store result
                self.step_results.append(step_result)
                self.current_step += 1
                
                # Update performance metrics
                self._update_performance_metrics(step_result)
                
                # Handle errors
                if step_result.errors:
                    self._handle_step_errors(step_result)
                
                # Execute callbacks
                self._execute_callbacks(step_result)
                
                # Progress reporting
                if self.current_step - last_progress_report >= self.config.output_frequency_steps:
                    self._report_progress()
                    last_progress_report = self.current_step
                
                # Create checkpoint if needed
                if self.current_step % self.config.checkpoint_interval_steps == 0:
                    self.create_checkpoint()
                
                # Check for consecutive errors
                if self.consecutive_errors >= self.config.max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({self.consecutive_errors}), stopping simulation")
                    break
            
            return self._generate_final_results()
            
        except Exception as e:
            logger.error(f"Simulation loop failed: {e}")
            raise
    
    def _execute_simulation_step(self, day: int) -> SimulationStepResult:
        """Execute a single simulation step."""
        step_start_time = datetime.now()
        
        result = SimulationStepResult(
            step_number=self.current_step,
            timestamp=step_start_time
        )
        
        try:
            # Determine what to update
            should_update_spatial = self._should_update_spatial(day)
            should_update_economic = self._should_update_economic(day)
            
            # Execute updates
            if self.config.enable_parallel_execution and should_update_spatial and should_update_economic:
                # Parallel execution
                spatial_result, economic_result = self._execute_parallel_updates(should_update_spatial, should_update_economic)
            else:
                # Sequential execution
                spatial_result = self._execute_spatial_update() if should_update_spatial else {}
                economic_result = self._execute_economic_update() if should_update_economic else {}
            
            # Integration updates
            integration_result = self._execute_integration_updates(spatial_result, economic_result)
            
            # Store results
            result.spatial_updates = spatial_result
            result.economic_updates = economic_result
            result.integration_updates = integration_result
            
            # Calculate step time
            step_time = (datetime.now() - step_start_time).total_seconds()
            result.performance_metrics["step_time"] = step_time
            
        except Exception as e:
            error_msg = f"Step {self.current_step} failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)
        
        return result
    
    def _execute_parallel_updates(self, update_spatial: bool, update_economic: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute spatial and economic updates in parallel."""
        if not self.thread_pool:
            # Fallback to sequential execution
            spatial_result = self._execute_spatial_update() if update_spatial else {}
            economic_result = self._execute_economic_update() if update_economic else {}
            return spatial_result, economic_result
        
        try:
            # Submit tasks to thread pool
            futures = {}
            
            if update_spatial:
                futures['spatial'] = self.thread_pool.submit(self._execute_spatial_update)
            
            if update_economic:
                futures['economic'] = self.thread_pool.submit(self._execute_economic_update)
            
            # Wait for completion with timeout
            results = {}
            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=self.config.step_timeout_seconds)
                except Exception as e:
                    logger.error(f"Parallel {name} update failed: {e}")
                    results[name] = {"error": str(e)}
            
            spatial_result = results.get('spatial', {})
            economic_result = results.get('economic', {})
            
            return spatial_result, economic_result
            
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            spatial_result = self._execute_spatial_update() if update_spatial else {}
            economic_result = self._execute_economic_update() if update_economic else {}
            return spatial_result, economic_result
    
    def _execute_spatial_update(self) -> Dict[str, Any]:
        """Execute spatial system update."""
        try:
            if not self.unified_system.map_simulator:
                return {"error": "No map simulator available"}
            
            # Execute spatial simulation step
            spatial_result = self.unified_system.map_simulator.simulate_time_step()
            
            return {
                "success": True,
                "logistics_friction": spatial_result.get("total_logistics_friction", 0.0),
                "active_disasters": spatial_result.get("active_disasters", 0),
                "settlements_count": spatial_result.get("settlements_count", 0),
                "infrastructure_segments": spatial_result.get("infrastructure_segments", 0),
                "disaster_events": spatial_result.get("disaster_events", {})
            }
            
        except Exception as e:
            logger.error(f"Spatial update failed: {e}")
            return {"error": str(e)}
    
    def _execute_economic_update(self) -> Dict[str, Any]:
        """Execute economic system update."""
        try:
            # This would integrate with the economic simulation
            # For now, return basic economic metrics
            return {
                "success": True,
                "total_economic_output": self.unified_system.state.total_economic_output * (1.0 + np.random.normal(0, 0.01)),
                "total_capital_stock": self.unified_system.state.capital_stock_total * (1.0 + np.random.normal(0, 0.005)),
                "sectors_active": self.unified_system.state.sectors_active,
                "plan_fulfillment_rate": 0.85 + np.random.normal(0, 0.05)
            }
            
        except Exception as e:
            logger.error(f"Economic update failed: {e}")
            return {"error": str(e)}
    
    def _execute_integration_updates(self, spatial_result: Dict[str, Any], economic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integration updates between spatial and economic systems."""
        try:
            integration_updates = {}
            
            # Calculate spatial-economic efficiency
            if spatial_result.get("success") and economic_result.get("success"):
                efficiency = self._calculate_integration_efficiency(spatial_result, economic_result)
                integration_updates["spatial_economic_efficiency"] = efficiency
            
            # Handle disaster economic impact
            if spatial_result.get("disaster_events", {}).get("new_disasters"):
                disaster_impact = self._calculate_disaster_economic_impact(spatial_result["disaster_events"])
                integration_updates["disaster_economic_impact"] = disaster_impact
            
            # Update logistics costs based on economic activity
            if economic_result.get("success"):
                logistics_adjustment = self._calculate_logistics_adjustment(economic_result)
                integration_updates["logistics_adjustment"] = logistics_adjustment
            
            return integration_updates
            
        except Exception as e:
            logger.error(f"Integration update failed: {e}")
            return {"error": str(e)}
    
    def _should_update_spatial(self, day: int) -> bool:
        """Determine if spatial systems should be updated this day."""
        # This would integrate with the time management system
        # For now, use simple daily updates
        return True
    
    def _should_update_economic(self, day: int) -> bool:
        """Determine if economic systems should be updated this day."""
        # This would integrate with the time management system
        # For now, use monthly updates
        return day % 30 == 0
    
    def _calculate_integration_efficiency(self, spatial_result: Dict[str, Any], economic_result: Dict[str, Any]) -> float:
        """Calculate integration efficiency between spatial and economic systems."""
        try:
            # Simple efficiency calculation
            logistics_friction = spatial_result.get("logistics_friction", 0.0)
            economic_output = economic_result.get("total_economic_output", 0.0)
            
            if economic_output > 0:
                efficiency = max(0.0, 1.0 - (logistics_friction / economic_output))
            else:
                efficiency = 0.0
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Failed to calculate integration efficiency: {e}")
            return 0.0
    
    def _calculate_disaster_economic_impact(self, disaster_events: Dict[str, Any]) -> float:
        """Calculate economic impact of disaster events."""
        try:
            new_disasters = disaster_events.get("new_disasters", [])
            total_impact = 0.0
            
            for disaster in new_disasters:
                # Simple impact calculation
                impact = disaster.get("intensity", 0.0) * disaster.get("radius", 0.0) * 0.1
                total_impact += impact
            
            return total_impact
            
        except Exception as e:
            logger.error(f"Failed to calculate disaster economic impact: {e}")
            return 0.0
    
    def _calculate_logistics_adjustment(self, economic_result: Dict[str, Any]) -> float:
        """Calculate logistics cost adjustment based on economic activity."""
        try:
            economic_output = economic_result.get("total_economic_output", 0.0)
            
            # Higher economic output increases logistics costs
            adjustment_factor = 1.0 + (economic_output / 1000000.0) * 0.1
            
            return adjustment_factor
            
        except Exception as e:
            logger.error(f"Failed to calculate logistics adjustment: {e}")
            return 1.0
    
    def _handle_step_errors(self, step_result: SimulationStepResult):
        """Handle errors from a simulation step."""
        if step_result.errors:
            self.error_count += len(step_result.errors)
            self.consecutive_errors += 1
            
            # Execute error callbacks
            for callback in self.error_callbacks:
                try:
                    callback(step_result)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")
        else:
            self.consecutive_errors = 0
    
    def _execute_callbacks(self, step_result: SimulationStepResult):
        """Execute registered callbacks."""
        # Execute step callbacks
        for callback in self.step_callbacks:
            try:
                callback(step_result)
            except Exception as e:
                logger.error(f"Step callback failed: {e}")
        
        # Execute progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(self.current_step, len(self.step_results))
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def _report_progress(self):
        """Report simulation progress."""
        if self.step_results:
            latest_result = self.step_results[-1]
            logger.info(f"Step {self.current_step}: "
                       f"Spatial updates: {len(latest_result.spatial_updates)}, "
                       f"Economic updates: {len(latest_result.economic_updates)}, "
                       f"Integration updates: {len(latest_result.integration_updates)}, "
                       f"Errors: {len(latest_result.errors)}")
    
    def _update_performance_metrics(self, step_result: SimulationStepResult):
        """Update performance metrics."""
        metrics = {
            "step_time": step_result.performance_metrics.get("step_time", 0.0),
            "timestamp": step_result.timestamp.isoformat(),
            "step_number": step_result.step_number
        }
        
        self.performance_history.append(metrics)
        
        # Keep only recent performance data
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _get_current_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-100:]  # Last 100 steps
        
        return {
            "average_step_time": np.mean([m["step_time"] for m in recent_metrics]),
            "max_step_time": np.max([m["step_time"] for m in recent_metrics]),
            "min_step_time": np.min([m["step_time"] for m in recent_metrics]),
            "total_simulation_time": sum([m["step_time"] for m in self.performance_history])
        }
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate final simulation results."""
        if not self.step_results:
            return {"error": "No simulation results available"}
        
        # Extract metrics from all steps
        spatial_metrics = self._extract_spatial_metrics()
        economic_metrics = self._extract_economic_metrics()
        integration_metrics = self._extract_integration_metrics()
        
        return {
            "total_steps": len(self.step_results),
            "spatial_metrics": spatial_metrics,
            "economic_metrics": economic_metrics,
            "integration_metrics": integration_metrics,
            "performance_summary": self._get_current_performance_metrics(),
            "error_summary": {
                "total_errors": self.error_count,
                "consecutive_errors": self.consecutive_errors,
                "error_rate": self.error_count / len(self.step_results) if self.step_results else 0
            }
        }
    
    def _extract_spatial_metrics(self) -> Dict[str, Any]:
        """Extract spatial metrics from step results."""
        spatial_data = [r.spatial_updates for r in self.step_results if r.spatial_updates]
        
        if not spatial_data:
            return {}
        
        logistics_frictions = [d.get("logistics_friction", 0.0) for d in spatial_data]
        disaster_counts = [d.get("active_disasters", 0) for d in spatial_data]
        
        return {
            "average_logistics_friction": np.mean(logistics_frictions),
            "max_logistics_friction": np.max(logistics_frictions),
            "total_disasters": sum([len(d.get("disaster_events", {}).get("new_disasters", [])) for d in spatial_data]),
            "average_active_disasters": np.mean(disaster_counts)
        }
    
    def _extract_economic_metrics(self) -> Dict[str, Any]:
        """Extract economic metrics from step results."""
        economic_data = [r.economic_updates for r in self.step_results if r.economic_updates]
        
        if not economic_data:
            return {}
        
        economic_outputs = [d.get("total_economic_output", 0.0) for d in economic_data]
        capital_stocks = [d.get("total_capital_stock", 0.0) for d in economic_data]
        
        return {
            "average_economic_output": np.mean(economic_outputs),
            "final_economic_output": economic_outputs[-1] if economic_outputs else 0.0,
            "economic_growth_rate": ((economic_outputs[-1] - economic_outputs[0]) / economic_outputs[0]) if economic_outputs and economic_outputs[0] > 0 else 0.0,
            "average_capital_stock": np.mean(capital_stocks)
        }
    
    def _extract_integration_metrics(self) -> Dict[str, Any]:
        """Extract integration metrics from step results."""
        integration_data = [r.integration_updates for r in self.step_results if r.integration_updates]
        
        if not integration_data:
            return {
                "average_integration_efficiency": 0.0,
                "total_disaster_economic_impact": 0.0,
                "integration_stability": 0.0
            }
        
        efficiencies = [d.get("spatial_economic_efficiency", 0.0) for d in integration_data]
        disaster_impacts = [d.get("disaster_economic_impact", 0.0) for d in integration_data]
        
        # Calculate stability based on economic performance, not just efficiency variance
        economic_data = [r.economic_updates for r in self.step_results if r.economic_updates]
        if economic_data:
            economic_outputs = [d.get("total_economic_output", 0.0) for d in economic_data]
            if economic_outputs:
                # Stability is based on economic output consistency and growth
                output_std = np.std(economic_outputs)
                output_mean = np.mean(economic_outputs)
                stability = max(0.0, 1.0 - (output_std / max(output_mean, 1e-10)))
            else:
                stability = 0.0
        else:
            stability = 0.0
        
        return {
            "average_integration_efficiency": np.mean(efficiencies) if efficiencies else 0.0,
            "total_disaster_economic_impact": sum(disaster_impacts),
            "integration_stability": stability
        }
    
    def _capture_simulation_state(self) -> Dict[str, Any]:
        """Capture current simulation state for checkpointing."""
        return {
            "current_step": self.current_step,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "unified_system_state": {
                "map_generated": self.unified_system.state.map_generated,
                "economic_plan_active": self.unified_system.state.economic_plan_active,
                "settlements_active": self.unified_system.state.settlements_active,
                "sectors_active": self.unified_system.state.sectors_active
            }
        }
    
    def _step_result_to_dict(self, step_result: SimulationStepResult) -> Dict[str, Any]:
        """Convert step result to dictionary."""
        return {
            "step_number": step_result.step_number,
            "timestamp": step_result.timestamp.isoformat(),
            "spatial_updates": step_result.spatial_updates,
            "economic_updates": step_result.economic_updates,
            "integration_updates": step_result.integration_updates,
            "performance_metrics": step_result.performance_metrics,
            "errors": step_result.errors,
            "warnings": step_result.warnings
        }
