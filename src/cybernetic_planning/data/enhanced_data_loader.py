"""
Enhanced Data Loader

Integrates real resource constraint data with existing BEA Input-Output data
to provide comprehensive economic planning data for the cybernetic system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

from .io_parser import IOParser
from .resource_matrix_builder import ResourceMatrixBuilder
from .sector_mapper import SectorMapper
from .web_scrapers.data_collector import ResourceDataCollector


class EnhancedDataLoader:
    """
    Enhanced data loader that integrates real resource data with BEA I-O data.
    
    Provides comprehensive economic planning data including:
    - BEA Input-Output tables (technology matrix, final demand)
    - Real resource constraint data (energy, materials, labor, environmental)
    - Sector mapping and data validation
    - Data quality assessment and reporting
    """
    
    def __init__(self, 
                 eia_api_key: Optional[str] = None,
                 data_dir: str = "data",
                 cache_dir: str = "cache"):
        """
        Initialize the enhanced data loader.
        
        Args:
            eia_api_key: EIA API key for enhanced data access
            data_dir: Directory for data files
            cache_dir: Directory for cached data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.io_parser = IOParser()
        self.sector_mapper = SectorMapper()
        self.data_collector = ResourceDataCollector(
            eia_api_key=eia_api_key,
            cache_dir=str(self.cache_dir),
            output_dir=str(self.data_dir)
        )
        self.matrix_builder = ResourceMatrixBuilder(
            sector_mapper=self.sector_mapper,
            data_collector=self.data_collector
        )
        
        # Data storage
        self.current_data = {}
        self.resource_matrices = {}
        self.metadata = {
            'load_timestamp': datetime.now().isoformat(),
            'data_sources': [],
            'sector_count': 175,
            'resource_count': 0,
            'data_quality': {}
        }
    
    def load_comprehensive_data(self, 
                               year: int = 2024,
                               use_real_data: bool = True,
                               bea_data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load comprehensive economic planning data.
        
        Args:
            year: Year for data collection
            use_real_data: Whether to collect real resource data
            bea_data_file: Path to BEA data file (if None, uses default)
            
        Returns:
            Comprehensive data dictionary
        """
        print(f"Loading comprehensive economic planning data for {year}...")
        
        # Load BEA Input-Output data
        print("Loading BEA Input-Output data...")
        bea_data = self._load_bea_data(bea_data_file)
        
        # Load or collect resource data
        if use_real_data:
            print("Collecting real resource constraint data...")
            resource_data = self._load_real_resource_data(year)
        else:
            print("No real data available - skipping resource data collection...")
            resource_data = self._create_empty_resource_data()
        
        # Build resource matrices
        print("Building resource constraint matrices...")
        resource_matrices = self.matrix_builder.build_all_matrices(resource_data, year)
        
        # Combine all data
        comprehensive_data = {
            'year': year,
            'bea_data': bea_data,
            'resource_data': resource_data,
            'resource_matrices': resource_matrices,
            'metadata': self._build_comprehensive_metadata(bea_data, resource_data, resource_matrices)
        }
        
        # Validate data quality
        print("Validating data quality...")
        validation_results = self._validate_comprehensive_data(comprehensive_data)
        comprehensive_data['validation_results'] = validation_results
        
        # Store current data
        self.current_data = comprehensive_data
        self.resource_matrices = resource_matrices
        
        print("Comprehensive data loading completed successfully")
        return comprehensive_data
    
    def _load_bea_data(self, bea_data_file: Optional[str] = None) -> Dict[str, Any]:
        """Load BEA Input-Output data."""
        if bea_data_file is None:
            # Try to find existing BEA data file
            bea_files = list(self.data_dir.glob("*bea*.json")) + list(self.data_dir.glob("*io*.json"))
            if bea_files:
                bea_data_file = bea_files[0]
            else:
                # Use default BEA data file
                bea_data_file = self.data_dir / "usa_io_data_2024.json"
        
        if Path(bea_data_file).exists():
            try:
                bea_data = self.io_parser.parse_file(bea_data_file)
                print(f"✓ Loaded BEA data from {bea_data_file}")
                return bea_data
            except Exception as e:
                print(f"✗ Error loading BEA data: {e}")
                raise ValueError(f"Failed to load BEA data: {e}")
        else:
            print("No BEA data file found")
            raise FileNotFoundError(f"BEA data file not found: {bea_data_file}")
    
    def _load_real_resource_data(self, year: int) -> Dict[str, Any]:
        """Load real resource constraint data."""
        try:
            resource_data = self.data_collector.collect_all_resource_data(year)
            self.metadata['data_sources'].extend(resource_data.get('metadata', {}).get('data_sources', []))
            return resource_data
        except Exception as e:
            print(f"Error collecting real resource data: {e}")
            return self._create_empty_resource_data()
    
    def _create_empty_resource_data(self) -> Dict[str, Any]:
        """Create empty resource data when no real data is available."""
        return {
            'year': 2024,
            'energy_data': {},
            'material_data': {},
            'labor_data': {},
            'environmental_data': {},
            'metadata': {
                'collection_timestamp': datetime.now().isoformat(),
                'data_sources': [],
                'data_quality': 'none'
            }
        }
    
    
    def _build_comprehensive_metadata(self, 
                                    bea_data: Dict[str, Any],
                                    resource_data: Dict[str, Any],
                                    resource_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive metadata."""
        metadata = {
            'load_timestamp': datetime.now().isoformat(),
            'year': resource_data.get('year', 2024),
            'sector_count': bea_data.get('sector_count', 175),
            'data_sources': {
                'bea': bea_data.get('data_source', 'unknown'),
                'resources': resource_data.get('metadata', {}).get('data_sources', [])
            },
            'resource_matrices': {
                'count': len([m for m in resource_matrices.values() if isinstance(m, np.ndarray)]),
                'types': list(resource_matrices.keys())
            },
            'data_quality': self._assess_data_quality(bea_data, resource_data, resource_matrices)
        }
        
        return metadata
    
    def _assess_data_quality(self, 
                           bea_data: Dict[str, Any],
                           resource_data: Dict[str, Any],
                           resource_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality = {
            'bea_data_quality': 'high' if bea_data.get('data_source') != 'synthetic' else 'none',
            'resource_data_quality': resource_data.get('metadata', {}).get('data_quality', 'unknown'),
            'matrix_completeness': 0.0,
            'sector_coverage': 0.0,
            'overall_quality': 'unknown'
        }
        
        # Calculate matrix completeness
        matrix_count = len([m for m in resource_matrices.values() if isinstance(m, np.ndarray)])
        expected_matrices = 4  # energy, material, labor, environmental
        quality['matrix_completeness'] = matrix_count / expected_matrices
        
        # Calculate sector coverage
        if 'combined_resource_matrix' in resource_matrices:
            matrix = resource_matrices['combined_resource_matrix']
            non_zero_sectors = np.count_nonzero(np.sum(matrix, axis=0))
            quality['sector_coverage'] = non_zero_sectors / matrix.shape[1]
        
        # Determine overall quality
        if quality['bea_data_quality'] == 'high' and quality['resource_data_quality'] == 'high':
            quality['overall_quality'] = 'high'
        elif quality['bea_data_quality'] == 'high' or quality['resource_data_quality'] == 'high':
            quality['overall_quality'] = 'medium'
        else:
            quality['overall_quality'] = 'low'
        
        return quality
    
    def _validate_comprehensive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate comprehensive data for consistency and completeness."""
        validation_results = {
            'overall_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Validate BEA data
        bea_data = data.get('bea_data', {})
        if not bea_data:
            validation_results['errors'].append("Missing BEA data")
            validation_results['overall_valid'] = False
        else:
            if 'technology_matrix' not in bea_data:
                validation_results['errors'].append("Missing technology matrix in BEA data")
                validation_results['overall_valid'] = False
            
            if 'final_demand' not in bea_data:
                validation_results['errors'].append("Missing final demand vector in BEA data")
                validation_results['overall_valid'] = False
        
        # Validate resource matrices
        resource_matrices = data.get('resource_matrices', {})
        if not resource_matrices:
            validation_results['warnings'].append("No resource matrices available")
        else:
            required_matrices = ['energy_matrix', 'material_matrix', 'labor_matrix', 'environmental_matrix']
            missing_matrices = [m for m in required_matrices if m not in resource_matrices]
            if missing_matrices:
                validation_results['warnings'].append(f"Missing resource matrices: {missing_matrices}")
        
        # Validate matrix dimensions
        if 'technology_matrix' in bea_data:
            tech_matrix = bea_data['technology_matrix']
            if isinstance(tech_matrix, np.ndarray):
                expected_sectors = 175
                if tech_matrix.shape[0] != expected_sectors or tech_matrix.shape[1] != expected_sectors:
                    validation_results['errors'].append(f"Technology matrix dimensions {tech_matrix.shape} don't match expected {expected_sectors}x{expected_sectors}")
                    validation_results['overall_valid'] = False
        
        # Add recommendations
        if validation_results['overall_valid']:
            validation_results['recommendations'].append("Data validation passed successfully")
        else:
            validation_results['recommendations'].append("Fix validation errors before using data")
        
        return validation_results
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        if not self.current_data:
            return {'error': 'No data loaded'}
        
        summary = {
            'year': self.current_data.get('year', 'unknown'),
            'sector_count': self.current_data.get('bea_data', {}).get('sector_count', 0),
            'data_sources': self.current_data.get('metadata', {}).get('data_sources', {}),
            'resource_matrices': self.current_data.get('metadata', {}).get('resource_matrices', {}),
            'data_quality': self.current_data.get('metadata', {}).get('data_quality', {}),
            'validation_status': self.current_data.get('validation_results', {}).get('overall_valid', False)
        }
        
        return summary
    
    def save_comprehensive_data(self, output_file: Optional[str] = None) -> str:
        """Save comprehensive data to file."""
        if not self.current_data:
            raise ValueError("No data to save")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"comprehensive_economic_data_{timestamp}.json"
        else:
            output_file = Path(output_file)
        
        # Prepare data for JSON serialization
        json_data = self._prepare_for_json(self.current_data)
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"Comprehensive data saved to {output_file}")
        return str(output_file)
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def load_comprehensive_data_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load comprehensive data from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        self._convert_lists_to_arrays(data)
        
        self.current_data = data
        return data
    
    def _convert_lists_to_arrays(self, obj: Any) -> None:
        """Convert lists back to numpy arrays in loaded data."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ['technology_matrix', 'final_demand', 'labor_input'] and isinstance(value, list):
                    obj[key] = np.array(value)
                elif isinstance(value, (dict, list)):
                    self._convert_lists_to_arrays(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._convert_lists_to_arrays(item)
    
    def get_resource_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get resource constraint matrix and constraints vector.
        
        Returns:
            Tuple of (resource_matrix, max_resources)
        """
        if not self.current_data:
            raise ValueError("No data loaded")
        
        resource_matrices = self.current_data.get('resource_matrices', {})
        
        if 'combined_resource_matrix' in resource_matrices:
            resource_matrix = resource_matrices['combined_resource_matrix']
        else:
            # Combine individual matrices
            matrix_list = []
            for name, matrix in resource_matrices.items():
                if isinstance(matrix, np.ndarray) and name != 'resource_constraints':
                    matrix_list.append(matrix)
            
            if matrix_list:
                resource_matrix = np.vstack(matrix_list)
            else:
                raise ValueError("No resource matrices available")
        
        if 'resource_constraints' in resource_matrices:
            max_resources = resource_matrices['resource_constraints']
        else:
            # Estimate constraints
            max_resources = np.max(resource_matrix, axis=1) * 1.5  # 50% buffer
        
        return resource_matrix, max_resources
