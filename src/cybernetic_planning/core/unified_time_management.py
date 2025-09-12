"""
Unified Time Management System

Manages time synchronization between spatial and economic simulation systems
with different update frequencies and temporal resolution requirements.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TimeScale(Enum):
    """Different time scales for simulation updates."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"

class UpdatePriority(Enum):
    """Priority levels for system updates."""
    CRITICAL = "critical"      # Must update every step
    HIGH = "high"              # Update frequently
    MEDIUM = "medium"          # Update moderately
    LOW = "low"               # Update infrequently
    BACKGROUND = "background"  # Update when resources available

@dataclass
class TemporalEvent:
    """Represents a temporal event in the simulation."""
    event_id: str
    event_type: str
    scheduled_time: datetime
    priority: UpdatePriority
    handler_function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    recurring: bool = False
    recurrence_interval: Optional[timedelta] = None
    executed: bool = False
    execution_count: int = 0

@dataclass
class SystemUpdateSchedule:
    """Defines update schedule for a simulation system."""
    system_name: str
    time_scale: TimeScale
    priority: UpdatePriority
    last_update: Optional[datetime] = None
    next_update: Optional[datetime] = None
    update_count: int = 0
    average_update_time: float = 0.0
    enabled: bool = True

@dataclass
class TimeManagementConfig:
    """Configuration for time management system."""
    simulation_start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    simulation_end_date: Optional[datetime] = None
    time_acceleration_factor: float = 1.0  # 1.0 = real time, 10.0 = 10x speed
    max_simulation_days: int = 1825  # 5 years
    enable_time_compression: bool = True
    enable_event_scheduling: bool = True
    enable_performance_monitoring: bool = True
    default_update_frequency: TimeScale = TimeScale.DAILY

class UnifiedTimeManager:
    """
    Manages time synchronization and scheduling for unified simulation.
    
    Features:
    - Multi-scale time management (daily, weekly, monthly, etc.)
    - Event scheduling and execution
    - Performance monitoring and optimization
    - Time compression for long simulations
    - Synchronization between spatial and economic systems
    """
    
    def __init__(self, config: Optional[TimeManagementConfig] = None):
        """Initialize the unified time manager."""
        self.config = config or TimeManagementConfig()
        
        # Time state
        self.current_time = self.config.simulation_start_date
        self.simulation_day = 0
        self.simulation_month = 1
        self.simulation_year = self.config.simulation_start_date.year
        
        # System update schedules
        self.update_schedules: Dict[str, SystemUpdateSchedule] = {}
        
        # Temporal events
        self.scheduled_events: List[TemporalEvent] = []
        self.event_counter = 0
        
        # Performance tracking
        self.update_times: Dict[str, List[float]] = {}
        self.time_compression_active = False
        
        # System state
        self.is_running = False
        self.pause_requested = False
        
        logger.info(f"Unified time manager initialized starting from {self.current_time}")
    
    def register_system(self, 
                       system_name: str, 
                       time_scale: TimeScale,
                       priority: UpdatePriority = UpdatePriority.MEDIUM) -> Dict[str, Any]:
        """
        Register a simulation system for time management.
        
        Args:
            system_name: Name of the system
            time_scale: How frequently the system should update
            priority: Priority level for updates
        
        Returns:
            Dictionary with registration results
        """
        try:
            if system_name in self.update_schedules:
                return {"success": False, "error": f"System '{system_name}' already registered"}
            
            # Create update schedule
            schedule = SystemUpdateSchedule(
                system_name=system_name,
                time_scale=time_scale,
                priority=priority,
                next_update=self._calculate_next_update_time(time_scale)
            )
            
            self.update_schedules[system_name] = schedule
            self.update_times[system_name] = []
            
            logger.info(f"Registered system '{system_name}' with {time_scale.value} updates")
            
            return {
                "success": True,
                "system_name": system_name,
                "time_scale": time_scale.value,
                "priority": priority.value,
                "next_update": schedule.next_update.isoformat() if schedule.next_update else None
            }
            
        except Exception as e:
            logger.error(f"Failed to register system '{system_name}': {e}")
            return {"success": False, "error": str(e)}
    
    def schedule_event(self,
                      event_type: str,
                      scheduled_time: datetime,
                      handler_function: Callable,
                      parameters: Optional[Dict[str, Any]] = None,
                      priority: UpdatePriority = UpdatePriority.MEDIUM,
                      recurring: bool = False,
                      recurrence_interval: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Schedule a temporal event.
        
        Args:
            event_type: Type of event
            scheduled_time: When the event should occur
            handler_function: Function to call when event occurs
            parameters: Parameters to pass to handler
            priority: Event priority
            recurring: Whether event should recur
            recurrence_interval: Interval for recurring events
        
        Returns:
            Dictionary with scheduling results
        """
        try:
            self.event_counter += 1
            event_id = f"event_{self.event_counter}_{event_type}"
            
            event = TemporalEvent(
                event_id=event_id,
                event_type=event_type,
                scheduled_time=scheduled_time,
                priority=priority,
                handler_function=handler_function,
                parameters=parameters or {},
                recurring=recurring,
                recurrence_interval=recurrence_interval
            )
            
            self.scheduled_events.append(event)
            
            logger.info(f"Scheduled event '{event_type}' for {scheduled_time}")
            
            return {
                "success": True,
                "event_id": event_id,
                "event_type": event_type,
                "scheduled_time": scheduled_time.isoformat(),
                "priority": priority.value
            }
            
        except Exception as e:
            logger.error(f"Failed to schedule event '{event_type}': {e}")
            return {"success": False, "error": str(e)}
    
    def advance_time(self, days: int = 1) -> Dict[str, Any]:
        """
        Advance simulation time by specified number of days.
        
        Args:
            days: Number of days to advance
        
        Returns:
            Dictionary with time advancement results
        """
        try:
            if not self.is_running:
                return {"success": False, "error": "Time manager not running"}
            
            results = {
                "success": True,
                "time_advanced": days,
                "systems_updated": [],
                "events_executed": [],
                "performance_metrics": {}
            }
            
            # Advance time
            old_time = self.current_time
            self.current_time += timedelta(days=days)
            self.simulation_day += days
            
            # Update month and year
            self.simulation_month = ((self.simulation_day - 1) // 30) + 1
            self.simulation_year = self.config.simulation_start_date.year + (self.simulation_day // 365)
            
            # Check which systems need updates
            systems_to_update = self._get_systems_to_update()
            
            # Execute system updates
            for system_name in systems_to_update:
                update_result = self._execute_system_update(system_name)
                if update_result["success"]:
                    results["systems_updated"].append(system_name)
            
            # Execute scheduled events
            events_executed = self._execute_scheduled_events()
            results["events_executed"] = events_executed
            
            # Update performance metrics
            results["performance_metrics"] = self._calculate_performance_metrics()
            
            # Check for time compression opportunities
            if self.config.enable_time_compression:
                self._check_time_compression()
            
            logger.debug(f"Advanced time from {old_time} to {self.current_time}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to advance time: {e}")
            return {"success": False, "error": str(e)}
    
    def get_current_time_info(self) -> Dict[str, Any]:
        """Get current time information."""
        return {
            "current_time": self.current_time.isoformat(),
            "simulation_day": self.simulation_day,
            "simulation_month": self.simulation_month,
            "simulation_year": self.simulation_year,
            "time_acceleration_factor": self.config.time_acceleration_factor,
            "time_compression_active": self.time_compression_active,
            "registered_systems": len(self.update_schedules),
            "scheduled_events": len(self.scheduled_events),
            "pending_events": len([e for e in self.scheduled_events if not e.executed])
        }
    
    def get_system_update_status(self) -> Dict[str, Any]:
        """Get status of all registered systems."""
        status = {}
        
        for system_name, schedule in self.update_schedules.items():
            status[system_name] = {
                "time_scale": schedule.time_scale.value,
                "priority": schedule.priority.value,
                "enabled": schedule.enabled,
                "last_update": schedule.last_update.isoformat() if schedule.last_update else None,
                "next_update": schedule.next_update.isoformat() if schedule.next_update else None,
                "update_count": schedule.update_count,
                "average_update_time": schedule.average_update_time,
                "days_since_last_update": self._days_since_last_update(schedule)
            }
        
        return status
    
    def enable_time_compression(self, factor: float = 10.0) -> Dict[str, Any]:
        """
        Enable time compression for faster simulation.
        
        Args:
            factor: Time acceleration factor
        
        Returns:
            Dictionary with compression results
        """
        try:
            if factor <= 0:
                return {"success": False, "error": "Time acceleration factor must be positive"}
            
            self.config.time_acceleration_factor = factor
            self.time_compression_active = True
            
            logger.info(f"Enabled time compression with factor {factor}")
            
            return {
                "success": True,
                "time_acceleration_factor": factor,
                "time_compression_active": True
            }
            
        except Exception as e:
            logger.error(f"Failed to enable time compression: {e}")
            return {"success": False, "error": str(e)}
    
    def disable_time_compression(self) -> Dict[str, Any]:
        """Disable time compression."""
        try:
            self.config.time_acceleration_factor = 1.0
            self.time_compression_active = False
            
            logger.info("Disabled time compression")
            
            return {
                "success": True,
                "time_acceleration_factor": 1.0,
                "time_compression_active": False
            }
            
        except Exception as e:
            logger.error(f"Failed to disable time compression: {e}")
            return {"success": False, "error": str(e)}
    
    def pause_simulation(self) -> Dict[str, Any]:
        """Pause the simulation."""
        try:
            self.pause_requested = True
            self.is_running = False
            
            logger.info("Simulation paused")
            
            return {"success": True, "paused": True}
            
        except Exception as e:
            logger.error(f"Failed to pause simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def resume_simulation(self) -> Dict[str, Any]:
        """Resume the simulation."""
        try:
            self.pause_requested = False
            self.is_running = True
            
            logger.info("Simulation resumed")
            
            return {"success": True, "resumed": True}
            
        except Exception as e:
            logger.error(f"Failed to resume simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def start_simulation(self) -> Dict[str, Any]:
        """Start the time management system."""
        try:
            self.is_running = True
            self.pause_requested = False
            
            logger.info("Time management system started")
            
            return {
                "success": True,
                "started": True,
                "current_time": self.current_time.isoformat(),
                "registered_systems": len(self.update_schedules),
                "scheduled_events": len(self.scheduled_events)
            }
            
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_simulation(self) -> Dict[str, Any]:
        """Stop the time management system."""
        try:
            self.is_running = False
            self.pause_requested = False
            
            # Generate final performance report
            performance_report = self._generate_performance_report()
            
            logger.info("Time management system stopped")
            
            return {
                "success": True,
                "stopped": True,
                "final_time": self.current_time.isoformat(),
                "total_simulation_days": self.simulation_day,
                "performance_report": performance_report
            }
            
        except Exception as e:
            logger.error(f"Failed to stop simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_next_update_time(self, time_scale: TimeScale) -> datetime:
        """Calculate next update time for a given time scale."""
        if time_scale == TimeScale.DAILY:
            return self.current_time + timedelta(days=1)
        elif time_scale == TimeScale.WEEKLY:
            return self.current_time + timedelta(weeks=1)
        elif time_scale == TimeScale.MONTHLY:
            return self.current_time + timedelta(days=30)
        elif time_scale == TimeScale.QUARTERLY:
            return self.current_time + timedelta(days=90)
        elif time_scale == TimeScale.ANNUAL:
            return self.current_time + timedelta(days=365)
        else:
            return self.current_time + timedelta(days=1)
    
    def _get_systems_to_update(self) -> List[str]:
        """Get list of systems that need updates at current time."""
        systems_to_update = []
        
        for system_name, schedule in self.update_schedules.items():
            if not schedule.enabled:
                continue
            
            if schedule.next_update and self.current_time >= schedule.next_update:
                systems_to_update.append(system_name)
        
        # Sort by priority
        systems_to_update.sort(key=lambda s: self.update_schedules[s].priority.value)
        
        return systems_to_update
    
    def _execute_system_update(self, system_name: str) -> Dict[str, Any]:
        """Execute update for a specific system."""
        try:
            schedule = self.update_schedules[system_name]
            start_time = datetime.now()
            
            # Update schedule
            schedule.last_update = self.current_time
            schedule.next_update = self._calculate_next_update_time(schedule.time_scale)
            schedule.update_count += 1
            
            # Record update time
            update_time = (datetime.now() - start_time).total_seconds()
            self.update_times[system_name].append(update_time)
            
            # Update average update time
            schedule.average_update_time = np.mean(self.update_times[system_name])
            
            logger.debug(f"Updated system '{system_name}' in {update_time:.3f}s")
            
            return {
                "success": True,
                "system_name": system_name,
                "update_time": update_time,
                "next_update": schedule.next_update.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute update for system '{system_name}': {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_scheduled_events(self) -> List[Dict[str, Any]]:
        """Execute scheduled events that are due."""
        executed_events = []
        
        for event in self.scheduled_events:
            if event.executed:
                continue
            
            if self.current_time >= event.scheduled_time:
                try:
                    # Execute event
                    result = event.handler_function(**event.parameters)
                    event.executed = True
                    event.execution_count += 1
                    
                    executed_events.append({
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "executed_time": self.current_time.isoformat(),
                        "result": result
                    })
                    
                    # Schedule recurring event if applicable
                    if event.recurring and event.recurrence_interval:
                        new_event = TemporalEvent(
                            event_id=f"{event.event_id}_recur_{event.execution_count}",
                            event_type=event.event_type,
                            scheduled_time=self.current_time + event.recurrence_interval,
                            priority=event.priority,
                            handler_function=event.handler_function,
                            parameters=event.parameters,
                            recurring=event.recurring,
                            recurrence_interval=event.recurrence_interval
                        )
                        self.scheduled_events.append(new_event)
                    
                    logger.debug(f"Executed event '{event.event_type}' at {self.current_time}")
                    
                except Exception as e:
                    logger.error(f"Failed to execute event '{event.event_type}': {e}")
        
        return executed_events
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for time management."""
        metrics = {
            "total_systems": len(self.update_schedules),
            "active_systems": len([s for s in self.update_schedules.values() if s.enabled]),
            "total_events": len(self.scheduled_events),
            "pending_events": len([e for e in self.scheduled_events if not e.executed]),
            "time_acceleration_factor": self.config.time_acceleration_factor,
            "time_compression_active": self.time_compression_active
        }
        
        # Calculate average update times
        if self.update_times:
            all_update_times = []
            for system_times in self.update_times.values():
                all_update_times.extend(system_times)
            
            if all_update_times:
                metrics["average_update_time"] = np.mean(all_update_times)
                metrics["max_update_time"] = np.max(all_update_times)
                metrics["min_update_time"] = np.min(all_update_times)
        
        return metrics
    
    def _check_time_compression(self):
        """Check if time compression should be activated."""
        if not self.config.enable_time_compression:
            return
        
        # Simple heuristic: compress time if simulation is running slowly
        if self.update_times:
            recent_update_times = []
            for system_times in self.update_times.values():
                if system_times:
                    recent_update_times.extend(system_times[-10:])  # Last 10 updates
            
            if recent_update_times and np.mean(recent_update_times) > 1.0:  # > 1 second per update
                if not self.time_compression_active:
                    self.enable_time_compression(factor=5.0)
    
    def _days_since_last_update(self, schedule: SystemUpdateSchedule) -> int:
        """Calculate days since last update for a system."""
        if not schedule.last_update:
            return 0
        
        delta = self.current_time - schedule.last_update
        return delta.days
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "simulation_duration": {
                "start_time": self.config.simulation_start_date.isoformat(),
                "end_time": self.current_time.isoformat(),
                "total_days": self.simulation_day
            },
            "system_performance": {},
            "event_performance": {
                "total_events": len(self.scheduled_events),
                "executed_events": len([e for e in self.scheduled_events if e.executed]),
                "recurring_events": len([e for e in self.scheduled_events if e.recurring])
            },
            "time_management": {
                "time_acceleration_factor": self.config.time_acceleration_factor,
                "time_compression_active": self.time_compression_active,
                "pause_count": 1 if self.pause_requested else 0
            }
        }
        
        # System performance details
        for system_name, schedule in self.update_schedules.items():
            report["system_performance"][system_name] = {
                "time_scale": schedule.time_scale.value,
                "priority": schedule.priority.value,
                "update_count": schedule.update_count,
                "average_update_time": schedule.average_update_time,
                "total_update_time": sum(self.update_times.get(system_name, [])),
                "enabled": schedule.enabled
            }
        
        return report

class TemporalSynchronizer:
    """
    Synchronizes time between different simulation systems.
    
    Ensures that systems with different update frequencies
    remain synchronized and consistent.
    """
    
    def __init__(self, time_manager: UnifiedTimeManager):
        """Initialize temporal synchronizer."""
        self.time_manager = time_manager
        self.synchronization_points: List[datetime] = []
        self.sync_tolerance = timedelta(minutes=1)  # 1 minute tolerance
    
    def add_synchronization_point(self, sync_time: datetime) -> Dict[str, Any]:
        """Add a synchronization point."""
        try:
            self.synchronization_points.append(sync_time)
            self.synchronization_points.sort()
            
            return {
                "success": True,
                "sync_time": sync_time.isoformat(),
                "total_sync_points": len(self.synchronization_points)
            }
            
        except Exception as e:
            logger.error(f"Failed to add synchronization point: {e}")
            return {"success": False, "error": str(e)}
    
    def check_synchronization(self) -> Dict[str, Any]:
        """Check if systems are synchronized."""
        try:
            current_time = self.time_manager.current_time
            
            # Find nearest synchronization point
            nearest_sync = None
            min_distance = timedelta.max
            
            for sync_point in self.synchronization_points:
                distance = abs(current_time - sync_point)
                if distance < min_distance:
                    min_distance = distance
                    nearest_sync = sync_point
            
            is_synchronized = min_distance <= self.sync_tolerance if nearest_sync else True
            
            return {
                "success": True,
                "is_synchronized": is_synchronized,
                "current_time": current_time.isoformat(),
                "nearest_sync_point": nearest_sync.isoformat() if nearest_sync else None,
                "sync_distance": min_distance.total_seconds() if nearest_sync else 0,
                "sync_tolerance": self.sync_tolerance.total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Failed to check synchronization: {e}")
            return {"success": False, "error": str(e)}
    
    def force_synchronization(self) -> Dict[str, Any]:
        """Force synchronization of all systems."""
        try:
            # This would trigger updates for all systems regardless of schedule
            systems_updated = []
            
            for system_name in self.time_manager.update_schedules.keys():
                update_result = self.time_manager._execute_system_update(system_name)
                if update_result["success"]:
                    systems_updated.append(system_name)
            
            return {
                "success": True,
                "systems_updated": systems_updated,
                "sync_time": self.time_manager.current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to force synchronization: {e}")
            return {"success": False, "error": str(e)}
