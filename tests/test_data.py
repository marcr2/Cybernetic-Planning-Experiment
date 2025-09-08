"""
Unit tests for data processing utilities.
"""

import pytest
import numpy as np
import pandas as pd
from src.cybernetic_planning.data import IOParser, MatrixBuilder, DataValidator


class TestIOParser:
    """Test cases for IOParser class."""
    
    def test_initialization(self):
        """Test IOParser initialization."""
        parser = IOParser()
        assert isinstance(parser.supported_formats, list)
        assert '.csv' in parser.supported_formats
    
    def test_validate_io_data(self):
        """Test I-O data validation."""
        parser = IOParser()
        
        data = {
            'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
            'sectors': ['Sector1', 'Sector2'],
            'final_demand': np.array([100, 200]),
            'labor_input': np.array([0.5, 0.8])
        }
        
        result = parser.validate_io_data(data)
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
    
    def test_process_dataframe(self):
        """Test DataFrame processing."""
        parser = IOParser()
        
        df = pd.DataFrame({
            'Sector1': [0.1, 0.3],
            'Sector2': [0.2, 0.1]
        }, index=['Sector1', 'Sector2'])
        
        result = parser._process_dataframe(df, 'test')
        assert 'technology_matrix' in result
        assert 'sectors' in result
        assert result['sector_count'] == 2


class TestMatrixBuilder:
    """Test cases for MatrixBuilder class."""
    
    def test_initialization(self):
        """Test MatrixBuilder initialization."""
        builder = MatrixBuilder()
        assert isinstance(builder.matrices, dict)
        assert isinstance(builder.vectors, dict)
    
    def test_create_technology_matrix(self):
        """Test technology matrix creation."""
        builder = MatrixBuilder()
        
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        sectors = ['Sector1', 'Sector2']
        
        result = builder.create_technology_matrix(A, sectors)
        
        assert np.array_equal(result, A)
        assert 'technology_matrix' in builder.matrices
        assert builder.matrices['technology_matrix']['sectors'] == sectors
    
    def test_create_final_demand_vector(self):
        """Test final demand vector creation."""
        builder = MatrixBuilder()
        
        d = np.array([100, 200])
        sectors = ['Sector1', 'Sector2']
        
        result = builder.create_final_demand_vector(d, sectors)
        
        assert np.array_equal(result, d)
        assert 'final_demand' in builder.vectors
        assert builder.vectors['final_demand']['sectors'] == sectors
    
    def test_create_synthetic_data(self):
        """Test synthetic data creation."""
        builder = MatrixBuilder()
        
        data = builder.create_synthetic_data(n_sectors=5, technology_density=0.3)
        
        assert 'technology_matrix' in data
        assert 'final_demand' in data
        assert 'labor_input' in data
        assert data['technology_matrix'].shape == (5, 5)
        assert len(data['final_demand']) == 5
        assert len(data['labor_input']) == 5
    
    def test_validate_matrix_dimensions(self):
        """Test matrix dimension validation."""
        builder = MatrixBuilder()
        
        # Create compatible matrix and vector
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        d = np.array([100, 200])
        
        builder.create_technology_matrix(A)
        builder.create_final_demand_vector(d)
        
        assert builder.validate_matrix_dimensions('technology_matrix', 'final_demand')
    
    def test_export_to_dataframe(self):
        """Test DataFrame export."""
        builder = MatrixBuilder()
        
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        sectors = ['Sector1', 'Sector2']
        
        builder.create_technology_matrix(A, sectors)
        df = builder.export_to_dataframe('technology_matrix')
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.index) == sectors
        assert list(df.columns) == sectors


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        assert isinstance(validator.validation_rules, dict)
        assert 'technology_matrix' in validator.validation_rules
    
    def test_validate_technology_matrix(self):
        """Test technology matrix validation."""
        validator = DataValidator()
        
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        result = validator.validate_technology_matrix(A)
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'metrics' in result
        assert result['valid']
    
    def test_validate_final_demand(self):
        """Test final demand validation."""
        validator = DataValidator()
        
        d = np.array([100, 200])
        result = validator.validate_final_demand(d)
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'metrics' in result
        assert result['valid']
    
    def test_validate_labor_input(self):
        """Test labor input validation."""
        validator = DataValidator()
        
        l = np.array([0.5, 0.8])
        result = validator.validate_labor_input(l)
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'metrics' in result
        assert result['valid']
    
    def test_validate_data_consistency(self):
        """Test data consistency validation."""
        validator = DataValidator()
        
        data = {
            'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
            'final_demand': np.array([100, 200]),
            'labor_input': np.array([0.5, 0.8])
        }
        
        result = validator.validate_data_consistency(data)
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
    
    def test_validate_plan_feasibility(self):
        """Test plan feasibility validation."""
        validator = DataValidator()
        
        plan = {
            'total_output': np.array([100, 200]),
            'total_labor_cost': 150.0
        }
        
        result = validator.validate_plan_feasibility(plan)
        assert 'feasible' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'metrics' in result
    
    def test_validate_all(self):
        """Test comprehensive validation."""
        validator = DataValidator()
        
        data = {
            'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
            'final_demand': np.array([100, 200]),
            'labor_input': np.array([0.5, 0.8])
        }
        
        result = validator.validate_all(data)
        assert 'overall_valid' in result
        assert 'component_results' in result
        assert 'consistency_results' in result
        assert 'summary' in result
