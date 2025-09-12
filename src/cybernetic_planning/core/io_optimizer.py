"""
I/O Optimization Module

Implements optimized file I/O operations and data serialization for the cybernetic planning simulation.
This module provides efficient data storage, compression, and background saving capabilities.
"""

import numpy as np
import json
import pickle
import gzip
import threading
import queue
import time
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import warnings
from dataclasses import dataclass, asdict
import h5py
import zlib
from concurrent.futures import ThreadPoolExecutor
import os


@dataclass
class IOConfig:
    """Configuration for I/O operations."""
    use_compression: bool = True
    use_binary_format: bool = True
    background_saving: bool = True
    save_interval_seconds: int = 30
    max_file_size_mb: int = 100
    backup_count: int = 3


class CompressedSerializer:
    """
    Efficient data serialization with compression support.
    """
    
    def __init__(self, use_compression: bool = True, use_binary: bool = True):
        """
        Initialize serializer.
        
        Args:
            use_compression: Whether to use compression
            use_binary: Whether to use binary format (pickle) or JSON
        """
        self.use_compression = use_compression
        self.use_binary = use_binary
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        if self.use_binary:
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            serialized = json.dumps(data, default=self._json_serializer).encode('utf-8')
        
        if self.use_compression:
            return gzip.compress(serialized, compresslevel=6)
        else:
            return serialized
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize data from bytes."""
        if self.use_compression:
            decompressed = gzip.decompress(data)
        else:
            decompressed = data
        
        if self.use_binary:
            return pickle.loads(decompressed)
        else:
            return json.loads(decompressed.decode('utf-8'))
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return asdict(obj)
        else:
            return str(obj)


class HDF5Storage:
    """
    Efficient HDF5-based storage for large datasets.
    """
    
    def __init__(self, file_path: str, mode: str = 'a'):
        """
        Initialize HDF5 storage.
        
        Args:
            file_path: Path to HDF5 file
            mode: File mode ('r', 'w', 'a')
        """
        self.file_path = file_path
        self.mode = mode
        self._file = None
    
    def __enter__(self):
        """Context manager entry."""
        self._file = h5py.File(self.file_path, self.mode)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file:
            self._file.close()
    
    def store_simulation_data(self, group_name: str, data: Dict[str, Any]) -> None:
        """Store simulation data in HDF5 format."""
        if not self._file:
            raise RuntimeError("HDF5 file not open")
        
        group = self._file.create_group(group_name) if group_name not in self._file else self._file[group_name]
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression='gzip', compression_opts=6)
            elif isinstance(value, (int, float)):
                group.attrs[key] = value
            elif isinstance(value, str):
                group.attrs[key] = value
            else:
                # Convert to string for complex objects
                group.attrs[key] = str(value)
    
    def load_simulation_data(self, group_name: str) -> Dict[str, Any]:
        """Load simulation data from HDF5 format."""
        if not self._file:
            raise RuntimeError("HDF5 file not open")
        
        if group_name not in self._file:
            return {}
        
        group = self._file[group_name]
        data = {}
        
        # Load datasets
        for key in group.keys():
            data[key] = group[key][:]
        
        # Load attributes
        for key in group.attrs.keys():
            data[key] = group.attrs[key]
        
        return data


class BackgroundSaver:
    """
    Background saving system for simulation data.
    """
    
    def __init__(self, config: IOConfig):
        """
        Initialize background saver.
        
        Args:
            config: I/O configuration
        """
        self.config = config
        self.save_queue = queue.Queue()
        self.save_thread = None
        self.running = False
        self.last_save_time = 0
        self.serializer = CompressedSerializer(
            use_compression=config.use_compression,
            use_binary=config.use_binary_format
        )
    
    def start(self) -> None:
        """Start background saving thread."""
        if self.running:
            return
        
        self.running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
    
    def stop(self) -> None:
        """Stop background saving thread."""
        self.running = False
        if self.save_thread:
            self.save_thread.join(timeout=5.0)
    
    def queue_save(self, file_path: str, data: Any, priority: int = 0) -> None:
        """Queue data for background saving."""
        if not self.running:
            self.start()
        
        save_task = {
            'file_path': file_path,
            'data': data,
            'priority': priority,
            'timestamp': time.time()
        }
        
        self.save_queue.put(save_task)
    
    def _save_worker(self) -> None:
        """Background worker for saving data."""
        while self.running:
            try:
                # Get save task with timeout
                save_task = self.save_queue.get(timeout=1.0)
                
                # Check if enough time has passed since last save
                current_time = time.time()
                if current_time - self.last_save_time < self.config.save_interval_seconds:
                    # Put task back and wait
                    self.save_queue.put(save_task)
                    time.sleep(1.0)
                    continue
                
                # Save the data
                self._save_data(save_task['file_path'], save_task['data'])
                self.last_save_time = current_time
                
                # Clean up old backups if needed
                self._cleanup_old_backups(save_task['file_path'])
                
            except queue.Empty:
                continue
            except Exception as e:
                warnings.warn(f"Background save failed: {e}")
    
    def _save_data(self, file_path: str, data: Any) -> None:
        """Save data to file."""
        try:
            # Create backup if file exists
            if os.path.exists(file_path):
                backup_path = f"{file_path}.backup"
                os.rename(file_path, backup_path)
            
            # Serialize and save data
            serialized_data = self.serializer.serialize(data)
            
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
        except Exception as e:
            warnings.warn(f"Failed to save {file_path}: {e}")
    
    def _cleanup_old_backups(self, file_path: str) -> None:
        """Clean up old backup files."""
        try:
            backup_files = []
            base_path = str(Path(file_path).with_suffix(''))
            
            for i in range(1, self.config.backup_count + 1):
                backup_file = f"{base_path}.backup.{i}"
                if os.path.exists(backup_file):
                    backup_files.append((backup_file, os.path.getmtime(backup_file)))
            
            # Sort by modification time and remove oldest
            backup_files.sort(key=lambda x: x[1])
            
            while len(backup_files) >= self.config.backup_count:
                old_backup = backup_files.pop(0)
                os.remove(old_backup[0])
                
        except Exception as e:
            warnings.warn(f"Failed to cleanup backups: {e}")


class IOOptimizer:
    """
    Main I/O optimization manager.
    """
    
    def __init__(self, config: Optional[IOConfig] = None):
        """
        Initialize I/O optimizer.
        
        Args:
            config: I/O configuration
        """
        self.config = config or IOConfig()
        self.serializer = CompressedSerializer(
            use_compression=self.config.use_compression,
            use_binary=self.config.use_binary_format
        )
        self.background_saver = BackgroundSaver(self.config) if self.config.background_saving else None
        
        # Performance monitoring
        self.save_times = []
        self.load_times = []
        self.compression_ratios = []
    
    def start_background_saving(self) -> None:
        """Start background saving system."""
        if self.background_saver:
            self.background_saver.start()
    
    def stop_background_saving(self) -> None:
        """Stop background saving system."""
        if self.background_saver:
            self.background_saver.stop()
    
    def save_simulation_data(
        self, 
        file_path: str, 
        data: Any, 
        use_background: bool = True
    ) -> Dict[str, Any]:
        """
        Save simulation data with optimization.
        
        Args:
            file_path: Path to save file
            data: Data to save
            use_background: Whether to use background saving
            
        Returns:
            Save statistics
        """
        start_time = time.time()
        
        if use_background and self.background_saver:
            # Queue for background saving
            self.background_saver.queue_save(file_path, data)
            return {'status': 'queued', 'file_path': file_path}
        else:
            # Save immediately
            return self._save_immediately(file_path, data, start_time)
    
    def _save_immediately(self, file_path: str, data: Any, start_time: float) -> Dict[str, Any]:
        """Save data immediately."""
        try:
            # Create backup if file exists
            if os.path.exists(file_path):
                backup_path = f"{file_path}.backup"
                os.rename(file_path, backup_path)
            
            # Serialize data
            serialized_data = self.serializer.serialize(data)
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            save_time = time.time() - start_time
            self.save_times.append(save_time)
            
            # Calculate compression ratio
            original_size = len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            compressed_size = len(serialized_data)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            self.compression_ratios.append(compression_ratio)
            
            return {
                'status': 'saved',
                'file_path': file_path,
                'save_time': save_time,
                'file_size_mb': len(serialized_data) / (1024 * 1024),
                'compression_ratio': compression_ratio
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'file_path': file_path,
                'error': str(e)
            }
    
    def load_simulation_data(self, file_path: str) -> Any:
        """
        Load simulation data with optimization.
        
        Args:
            file_path: Path to load file from
            
        Returns:
            Loaded data
        """
        start_time = time.time()
        
        try:
            with open(file_path, 'rb') as f:
                serialized_data = f.read()
            
            data = self.serializer.deserialize(serialized_data)
            
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            
            return data
            
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
            return None
    
    def save_incremental(
        self, 
        base_file_path: str, 
        new_data: Dict[str, Any], 
        existing_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save only changed data incrementally.
        
        Args:
            base_file_path: Base file path
            new_data: New data to save
            existing_data: Existing data to compare against
            
        Returns:
            Save statistics
        """
        if existing_data is None:
            # Load existing data if not provided
            existing_data = self.load_simulation_data(base_file_path) or {}
        
        # Find changed data
        changed_data = {}
        for key, value in new_data.items():
            if key not in existing_data or existing_data[key] != value:
                changed_data[key] = value
        
        if not changed_data:
            return {'status': 'no_changes', 'file_path': base_file_path}
        
        # Save only changed data
        return self.save_simulation_data(base_file_path, changed_data, use_background=False)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get I/O performance statistics."""
        stats = {
            'average_save_time': np.mean(self.save_times) if self.save_times else 0.0,
            'average_load_time': np.mean(self.load_times) if self.load_times else 0.0,
            'total_saves': len(self.save_times),
            'total_loads': len(self.load_times),
            'average_compression_ratio': np.mean(self.compression_ratios) if self.compression_ratios else 1.0,
            'background_saving_enabled': self.background_saver is not None
        }
        
        return stats
    
    def optimize_file_size(self, file_path: str) -> Dict[str, Any]:
        """
        Optimize file size by recompressing and cleaning up.
        
        Args:
            file_path: Path to file to optimize
            
        Returns:
            Optimization statistics
        """
        try:
            # Load data
            data = self.load_simulation_data(file_path)
            if data is None:
                return {'status': 'error', 'message': 'Could not load file'}
            
            # Get original size
            original_size = os.path.getsize(file_path)
            
            # Re-save with current compression settings
            result = self._save_immediately(file_path, data, time.time())
            
            # Get new size
            new_size = os.path.getsize(file_path)
            
            return {
                'status': 'optimized',
                'original_size_mb': original_size / (1024 * 1024),
                'new_size_mb': new_size / (1024 * 1024),
                'size_reduction_mb': (original_size - new_size) / (1024 * 1024),
                'compression_improvement': (original_size - new_size) / original_size if original_size > 0 else 0.0
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


class SimulationIO:
    """
    High-level I/O manager for simulation data.
    """
    
    def __init__(self, config: Optional[IOConfig] = None):
        """
        Initialize simulation I/O manager.
        
        Args:
            config: I/O configuration
        """
        self.config = config or IOConfig()
        self.io_optimizer = IOOptimizer(self.config)
        
        # Start background saving if enabled
        if self.config.background_saving:
            self.io_optimizer.start_background_saving()
    
    def save_simulation_state(
        self, 
        file_path: str, 
        simulation_data: Dict[str, Any],
        use_background: bool = True
    ) -> Dict[str, Any]:
        """Save complete simulation state."""
        return self.io_optimizer.save_simulation_data(file_path, simulation_data, use_background)
    
    def load_simulation_state(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load complete simulation state."""
        return self.io_optimizer.load_simulation_data(file_path)
    
    def save_simulation_results(
        self, 
        file_path: str, 
        results: List[Dict[str, Any]],
        use_background: bool = True
    ) -> Dict[str, Any]:
        """Save simulation results with optimization."""
        # Convert results to more efficient format
        optimized_results = self._optimize_results_format(results)
        return self.io_optimizer.save_simulation_data(file_path, optimized_results, use_background)
    
    def _optimize_results_format(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize results format for storage."""
        if not results:
            return {}
        
        # Extract common structure
        first_result = results[0]
        common_keys = set(first_result.keys())
        
        # Find keys that are consistent across all results
        for result in results[1:]:
            common_keys &= set(result.keys())
        
        # Create optimized structure
        optimized = {
            'metadata': {
                'total_results': len(results),
                'common_keys': list(common_keys)
            },
            'data': {}
        }
        
        # Group data by key for better compression
        for key in common_keys:
            optimized['data'][key] = [result[key] for result in results]
        
        return optimized
    
    def get_io_performance(self) -> Dict[str, Any]:
        """Get I/O performance statistics."""
        return self.io_optimizer.get_performance_stats()
    
    def cleanup(self) -> None:
        """Cleanup I/O resources."""
        self.io_optimizer.stop_background_saving()
