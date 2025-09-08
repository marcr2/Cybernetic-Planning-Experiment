"""
Resource Matrix Builder

Builds resource constraint matrices from collected data and integrates
them with the existing BEA Input-Output data structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

from .sector_mapper import SectorMapper
from .web_scrapers.data_collector import ResourceDataCollector


class ResourceMatrixBuilder:
    """
    Builds resource constraint matrices for the cybernetic planning system.
    
    Creates matrices for:
    - Energy resources (coal, natural gas, petroleum, nuclear, renewable)
    - Material resources (critical materials, minerals)
    - Labor resources (by skill category)
    - Environmental resources (emissions, water, waste)
    """
    
    def __init__(self, 
                 sector_mapper: Optional[SectorMapper] = None,
                 data_collector: Optional[ResourceDataCollector] = None):
        """
        Initialize the resource matrix builder.
        
        Args:
            sector_mapper: Sector mapping instance
            data_collector: Data collection instance
        """
        self.sector_mapper = sector_mapper or SectorMapper()
        self.data_collector = data_collector or ResourceDataCollector()
        
        # Matrix specifications
        self.n_sectors = 175
        self.resource_specifications = self._load_resource_specifications()
        
        # Built matrices
        self.matrices = {}
        self.metadata = {
            'build_timestamp': datetime.now().isoformat(),
            'sector_count': self.n_sectors,
            'resource_types': [],
            'data_sources': [],
            'quality_metrics': {}
        }
    
    def _load_resource_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Load resource specifications and constraints."""
        return {
            'energy': {
                'types': ['coal', 'natural_gas', 'petroleum', 'nuclear', 'renewable'],
                'units': 'btu',
                'constraint_type': 'consumption',
                'description': 'Energy consumption by sector'
            },
            'materials': {
                'types': ['lithium', 'cobalt', 'rare_earth', 'copper', 'aluminum', 'steel'],
                'units': 'kg',
                'constraint_type': 'consumption',
                'description': 'Material consumption by sector'
            },
            'labor': {
                'types': ['high_skilled', 'medium_skilled', 'low_skilled', 'technical', 'management'],
                'units': 'hours',
                'constraint_type': 'capacity',
                'description': 'Labor capacity by skill category'
            },
            'environmental': {
                'types': ['carbon_emissions', 'water_usage', 'waste_generation'],
                'units': 'tons_co2_equivalent',
                'constraint_type': 'impact',
                'description': 'Environmental impact by sector'
            }
        }
    
    def build_all_matrices(self, 
                          collected_data: Optional[Dict[str, Any]] = None,
                          year: int = 2024) -> Dict[str, Any]:
        """
        Build all resource constraint matrices.
        
        Args:
            collected_data: Pre-collected data (if None, will collect fresh data)
            year: Year for data collection
            
        Returns:
            Dictionary containing all resource matrices
        """
        print("Building resource constraint matrices...")
        
        # Collect data if not provided
        if collected_data is None:
            print("Collecting fresh resource data...")
            collected_data = self.data_collector.collect_all_resource_data(year)
        
        # Build individual matrices
        matrices = {}
        
        # Energy matrix
        print("Building energy resource matrix...")
        energy_matrix = self._build_energy_matrix(collected_data.get('energy_data', {}))
        if energy_matrix is not None:
            matrices['energy_matrix'] = energy_matrix
        
        # Material matrix
        print("Building material resource matrix...")
        material_matrix = self._build_material_matrix(collected_data.get('material_data', {}))
        if material_matrix is not None:
            matrices['material_matrix'] = material_matrix
        
        # Labor matrix
        print("Building labor resource matrix...")
        labor_matrix = self._build_labor_matrix(collected_data.get('labor_data', {}))
        if labor_matrix is not None:
            matrices['labor_matrix'] = labor_matrix
        
        # Environmental matrix
        print("Building environmental resource matrix...")
        environmental_matrix = self._build_environmental_matrix(collected_data.get('environmental_data', {}))
        if environmental_matrix is not None:
            matrices['environmental_matrix'] = environmental_matrix
        
        # Combined matrix
        print("Building combined resource matrix...")
        combined_matrix = self._build_combined_matrix(matrices)
        if combined_matrix is not None:
            matrices['combined_resource_matrix'] = combined_matrix
        
        # Resource constraints vector
        print("Building resource constraints vector...")
        constraints_vector = self._build_constraints_vector(matrices, collected_data)
        if constraints_vector is not None:
            matrices['resource_constraints'] = constraints_vector
        
        # Metadata
        matrices['metadata'] = self._build_metadata(matrices, collected_data)
        
        self.matrices = matrices
        print(f"Successfully built {len(matrices)} resource matrices")
        
        return matrices
    
    def _build_energy_matrix(self, energy_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build energy resource constraint matrix."""
        try:
            energy_types = self.resource_specifications['energy']['types']
            n_energy_types = len(energy_types)
            
            # Initialize matrix
            matrix = np.zeros((n_energy_types, self.n_sectors))
            
            # Map energy data to BEA sectors
            if 'energy_intensity_by_sector' in energy_data:
                intensity_data = energy_data['energy_intensity_by_sector'].get('data', {})
                
                if 'sector_intensities' in intensity_data:
                    for sector_id, intensities in intensity_data['sector_intensities'].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1
                            
                            for i, energy_type in enumerate(energy_types):
                                if energy_type in intensities:
                                    matrix[i, sector_idx] = intensities[energy_type]
            
            # If no intensity data, use consumption data
            elif 'energy_consumption_by_sector' in energy_data:
                consumption_data = energy_data['energy_consumption_by_sector'].get('data', {})
                
                if 'energy_data' in consumption_data and 'sector_consumption' in consumption_data['energy_data']:
                    sector_consumption = consumption_data['energy_data']['sector_consumption']
                    
                    # Distribute consumption across energy types
                    for sector_idx, total_consumption in enumerate(sector_consumption):
                        if sector_idx < self.n_sectors:
                            # Distribute evenly across energy types (simplified)
                            for i in range(n_energy_types):
                                matrix[i, sector_idx] = total_consumption / n_energy_types
            
            # If still no data, return None
            if np.all(matrix == 0):
                return None
            
            return matrix
            
        except Exception as e:
            print(f"Error building energy matrix: {e}")
            return None
    
    def _build_material_matrix(self, material_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build material resource constraint matrix."""
        try:
            materials = self.resource_specifications['materials']['types']
            n_materials = len(materials)
            
            # Initialize matrix
            matrix = np.zeros((n_materials, self.n_sectors))
            
            # Map material data to BEA sectors
            if 'material_intensity' in material_data:
                intensity_data = material_data['material_intensity'].get('data', {})
                
                if 'sectors' in intensity_data:
                    for sector_id, intensities in intensity_data['sectors'].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1
                            
                            for i, material in enumerate(materials):
                                if material in intensities:
                                    matrix[i, sector_idx] = intensities[material]
            
            # If no intensity data, use consumption data
            elif 'material_consumption' in material_data:
                consumption_data = material_data['material_consumption'].get('data', {})
                
                if 'sectors' in consumption_data:
                    for sector_id, consumption in consumption_data['sectors'].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1
                            
                            for i, material in enumerate(materials):
                                if material in consumption:
                                    matrix[i, sector_idx] = consumption[material]
            
            # If still no data, return None
            if np.all(matrix == 0):
                return None
            
            return matrix
            
        except Exception as e:
            print(f"Error building material matrix: {e}")
            return None
    
    def _build_labor_matrix(self, labor_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build labor resource constraint matrix."""
        try:
            labor_categories = self.resource_specifications['labor']['types']
            n_categories = len(labor_categories)
            
            # Initialize matrix
            matrix = np.zeros((n_categories, self.n_sectors))
            
            # Map labor data to BEA sectors
            if 'labor_intensity' in labor_data:
                intensity_data = labor_data['labor_intensity'].get('data', {})
                
                if 'sectors' in intensity_data:
                    for sector_id, sector_data in intensity_data['sectors'].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1
                            
                            base_intensity = sector_data.get('labor_intensity', 0)
                            skill_requirements = sector_data.get('skill_requirements', {})
                            
                            for i, category in enumerate(labor_categories):
                                if category in skill_requirements:
                                    matrix[i, sector_idx] = base_intensity * skill_requirements[category]
            
            # If no intensity data, use employment data
            elif 'employment_by_sector' in labor_data:
                employment_data = labor_data['employment_by_sector'].get('data', {})
                
                if 'sectors' in employment_data:
                    for sector_id, sector_data in employment_data['sectors'].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1
                            
                            total_employment = sector_data.get('total_employment', 0)
                            skill_distribution = sector_data.get('skill_distribution', {})
                            
                            for i, category in enumerate(labor_categories):
                                if category in skill_distribution:
                                    matrix[i, sector_idx] = total_employment * skill_distribution[category]
            
            # If still no data, return None
            if np.all(matrix == 0):
                return None
            
            return matrix
            
        except Exception as e:
            print(f"Error building labor matrix: {e}")
            return None
    
    def _build_environmental_matrix(self, environmental_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build environmental resource constraint matrix."""
        try:
            env_factors = self.resource_specifications['environmental']['types']
            n_factors = len(env_factors)
            
            # Initialize matrix
            matrix = np.zeros((n_factors, self.n_sectors))
            
            # Map environmental data to BEA sectors
            if 'environmental_intensity' in environmental_data:
                intensity_data = environmental_data['environmental_intensity'].get('data', {})
                
                if 'sectors' in intensity_data:
                    for sector_id, intensities in intensity_data['sectors'].items():
                        if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                            sector_idx = sector_id - 1
                            
                            for i, factor in enumerate(env_factors):
                                if factor in intensities:
                                    matrix[i, sector_idx] = intensities[factor]
            
            # If no intensity data, use impact data
            for factor in env_factors:
                if factor in environmental_data:
                    impact_data = environmental_data[factor].get('data', {})
                    
                    if 'sectors' in impact_data:
                        for sector_id, impact_value in impact_data['sectors'].items():
                            if isinstance(sector_id, int) and 1 <= sector_id <= 175:
                                sector_idx = sector_id - 1
                                factor_idx = env_factors.index(factor)
                                matrix[factor_idx, sector_idx] = impact_value
            
            # If still no data, return None
            if np.all(matrix == 0):
                return None
            
            return matrix
            
        except Exception as e:
            print(f"Error building environmental matrix: {e}")
            return None
    
    def _build_combined_matrix(self, matrices: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Build combined resource constraint matrix."""
        try:
            matrix_list = []
            resource_names = []
            
            for matrix_name, matrix in matrices.items():
                if matrix is not None and matrix.size > 0 and matrix_name != 'combined_resource_matrix':
                    matrix_list.append(matrix)
                    resource_names.append(matrix_name)
            
            if not matrix_list:
                return None
            
            # Stack matrices vertically
            combined_matrix = np.vstack(matrix_list)
            
            # Store metadata
            self.metadata['resource_types'] = resource_names
            self.metadata['total_resources'] = combined_matrix.shape[0]
            
            return combined_matrix
            
        except Exception as e:
            print(f"Error building combined matrix: {e}")
            return None
    
    def _build_constraints_vector(self, matrices: Dict[str, np.ndarray], collected_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build resource constraints vector."""
        try:
            if 'combined_resource_matrix' not in matrices:
                return None
            
            combined_matrix = matrices['combined_resource_matrix']
            n_resources = combined_matrix.shape[0]
            
            # Initialize constraints vector
            constraints = np.zeros(n_resources)
            
            # Estimate maximum available resources
            # This would typically come from actual resource availability data
            for i in range(n_resources):
                # Use maximum sector demand as constraint (simplified)
                max_demand = np.max(combined_matrix[i, :])
                constraints[i] = max_demand * np.random.uniform(1.2, 2.0)  # 20-100% buffer
            
            return constraints
            
        except Exception as e:
            print(f"Error building constraints vector: {e}")
            return None
    
    def _build_metadata(self, matrices: Dict[str, np.ndarray], collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata for the matrices."""
        metadata = {
            'build_timestamp': datetime.now().isoformat(),
            'sector_count': self.n_sectors,
            'resource_types': list(matrices.keys()),
            'data_sources': collected_data.get('metadata', {}).get('data_sources', []),
            'year': collected_data.get('year', 2024),
            'matrix_shapes': {},
            'quality_metrics': self._calculate_quality_metrics(matrices)
        }
        
        # Add matrix shapes
        for name, matrix in matrices.items():
            if isinstance(matrix, np.ndarray):
                metadata['matrix_shapes'][name] = matrix.shape
        
        return metadata
    
    def _calculate_quality_metrics(self, matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate quality metrics for the matrices."""
        metrics = {}
        
        for name, matrix in matrices.items():
            if isinstance(matrix, np.ndarray):
                metrics[name] = {
                    'shape': matrix.shape,
                    'sparsity': np.count_nonzero(matrix == 0) / matrix.size,
                    'mean_value': np.mean(matrix),
                    'std_value': np.std(matrix),
                    'max_value': np.max(matrix),
                    'min_value': np.min(matrix)
                }
        
        return metrics
    
    
    def save_matrices(self, output_dir: str = "data") -> None:
        """Save built matrices to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual matrices
        for name, matrix in self.matrices.items():
            if isinstance(matrix, np.ndarray):
                filename = f"{name}_{timestamp}.npy"
                np.save(output_path / filename, matrix)
                print(f"Saved {name} to {filename}")
        
        # Save metadata
        metadata_file = f"resource_matrices_metadata_{timestamp}.json"
        with open(output_path / metadata_file, 'w') as f:
            json.dump(self.matrices.get('metadata', {}), f, indent=2, default=str)
        
        print(f"Saved metadata to {metadata_file}")
    
    def load_matrices(self, input_dir: str = "data") -> Dict[str, Any]:
        """Load matrices from files."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        
        matrices = {}
        
        # Load .npy files
        for npy_file in input_path.glob("*.npy"):
            matrix_name = npy_file.stem.split('_')[0]  # Remove timestamp
            matrices[matrix_name] = np.load(npy_file)
        
        # Load metadata
        metadata_files = list(input_path.glob("*metadata*.json"))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                matrices['metadata'] = json.load(f)
        
        self.matrices = matrices
        return matrices
    
    def get_matrix_summary(self) -> Dict[str, Any]:
        """Get summary of built matrices."""
        if not self.matrices:
            return {'error': 'No matrices built'}
        
        summary = {
            'total_matrices': len([m for m in self.matrices.values() if isinstance(m, np.ndarray)]),
            'sector_count': self.n_sectors,
            'matrices': {}
        }
        
        for name, matrix in self.matrices.items():
            if isinstance(matrix, np.ndarray):
                summary['matrices'][name] = {
                    'shape': matrix.shape,
                    'sparsity': np.count_nonzero(matrix == 0) / matrix.size,
                    'mean_value': float(np.mean(matrix)),
                    'std_value': float(np.std(matrix))
                }
        
        return summary
