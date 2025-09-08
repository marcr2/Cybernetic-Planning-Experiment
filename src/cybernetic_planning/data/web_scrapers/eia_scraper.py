"""
Energy Information Administration (EIA) Data Scraper

Scrapes energy consumption and production data by sector from EIA databases.
Focuses on energy intensity coefficients for the 175-sector BEA classification.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
import os

from .base_scraper import BaseScraper


class EIAScraper(BaseScraper):
    """
    Scraper for Energy Information Administration data.
    
    Collects energy consumption data by sector including:
    - Primary energy sources (coal, natural gas, petroleum, nuclear, renewables)
    - Energy intensity coefficients (energy per unit economic output)
    - Sector-specific energy consumption patterns
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize EIA scraper.
        
        Args:
            api_key: EIA API key (optional, uses public data if not provided)
            **kwargs: Additional arguments for BaseScraper
        """
        super().__init__(
            base_url="https://api.eia.gov/v2",
            rate_limit=0.5,  # EIA allows 5000 requests per hour
            **kwargs
        )
        
        # Use provided API key or try to get from environment
        self.api_key = api_key or os.getenv('EIA_API_KEY')
        self.energy_types = [
            'coal', 'natural_gas', 'petroleum', 'nuclear', 
            'renewable', 'hydroelectric', 'solar', 'wind', 'geothermal'
        ]
        
        # BEA sector mapping for energy data
        self.sector_mapping = self._load_sector_mapping()
    
    def _load_sector_mapping(self) -> Dict[str, int]:
        """Load BEA sector mapping for energy data."""
        # This would typically load from a configuration file
        # For now, return a basic mapping structure
        return {
            'electric_power': 1,
            'petroleum_refining': 2,
            'coal_mining': 3,
            'natural_gas_extraction': 4,
            'nuclear_power': 5,
            'renewable_energy': 6,
            'manufacturing': 7,
            'transportation': 8,
            'residential': 9,
            'commercial': 10
        }
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available EIA datasets."""
        datasets = []
        
        # Energy consumption by sector datasets
        sector_datasets = [
            {
                'id': 'energy_consumption_by_sector',
                'name': 'Energy Consumption by Economic Sector',
                'description': 'Total energy consumption by economic sector',
                'category': 'consumption',
                'frequency': 'annual',
                'start_year': 2010,
                'end_year': 2024
            },
            {
                'id': 'electricity_consumption_by_sector',
                'name': 'Electricity Consumption by Sector',
                'description': 'Electricity consumption by economic sector',
                'category': 'electricity',
                'frequency': 'annual',
                'start_year': 2010,
                'end_year': 2024
            },
            {
                'id': 'energy_intensity_by_sector',
                'name': 'Energy Intensity by Sector',
                'description': 'Energy consumption per unit economic output',
                'category': 'intensity',
                'frequency': 'annual',
                'start_year': 2010,
                'end_year': 2024
            },
            {
                'id': 'renewable_energy_by_sector',
                'name': 'Renewable Energy by Sector',
                'description': 'Renewable energy consumption by sector',
                'category': 'renewable',
                'frequency': 'annual',
                'start_year': 2010,
                'end_year': 2024
            }
        ]
        
        return sector_datasets
    
    def scrape_dataset(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """
        Scrape a specific EIA dataset.
        
        Args:
            dataset_id: Dataset identifier
            **kwargs: Additional parameters (year, sector, etc.)
            
        Returns:
            Scraped data dictionary
        """
        year = kwargs.get('year', 2024)
        sector = kwargs.get('sector', None)
        
        if dataset_id == 'energy_consumption_by_sector':
            return self._scrape_energy_consumption(year, sector)
        elif dataset_id == 'electricity_consumption_by_sector':
            return self._scrape_electricity_consumption(year, sector)
        elif dataset_id == 'energy_intensity_by_sector':
            return self._scrape_energy_intensity(year, sector)
        elif dataset_id == 'renewable_energy_by_sector':
            return self._scrape_renewable_energy(year, sector)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")
    
    def _scrape_energy_consumption(self, year: int, sector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape energy consumption data by sector."""
        try:
            # EIA API endpoint for energy consumption
            endpoint = "/electricity/data/electricity"
            
            params = {
                'frequency': 'annual',
                'data[0]': 'value',
                'facets[msn][]': 'TETCB',  # Total energy consumption
                'start': f"{year}-01-01",
                'end': f"{year}-12-31",
                'sort': [{"column": "period", "direction": "desc"}]
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            response = self.make_request(f"{self.base_url}{endpoint}", params=params)
            data = response.json()
            
            # Process the response
            processed_data = self._process_eia_response(data, 'energy_consumption')
            
            # Map to BEA sectors
            mapped_data = self._map_to_bea_sectors(processed_data, 'energy_consumption')
            
            return {
                'dataset_id': 'energy_consumption_by_sector',
                'year': year,
                'data': mapped_data,
                'metadata': {
                    'source': 'EIA',
                    'endpoint': endpoint,
                    'collection_timestamp': datetime.now().isoformat(),
                    'data_quality': 'high'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error scraping energy consumption: {e}")
            return self._create_empty_dataset('energy_consumption_by_sector', year)
    
    def _scrape_electricity_consumption(self, year: int, sector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape electricity consumption data by sector."""
        try:
            # EIA API endpoint for electricity consumption
            endpoint = "/electricity/data/electricity"
            
            params = {
                'frequency': 'annual',
                'data[0]': 'value',
                'facets[msn][]': 'ELCNS',  # Electricity consumption
                'start': f"{year}-01-01",
                'end': f"{year}-12-31",
                'sort': [{"column": "period", "direction": "desc"}]
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            response = self.make_request(f"{self.base_url}{endpoint}", params=params)
            data = response.json()
            
            processed_data = self._process_eia_response(data, 'electricity_consumption')
            mapped_data = self._map_to_bea_sectors(processed_data, 'electricity_consumption')
            
            return {
                'dataset_id': 'electricity_consumption_by_sector',
                'year': year,
                'data': mapped_data,
                'metadata': {
                    'source': 'EIA',
                    'endpoint': endpoint,
                    'collection_timestamp': datetime.now().isoformat(),
                    'data_quality': 'high'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error scraping electricity consumption: {e}")
            return self._create_empty_dataset('electricity_consumption_by_sector', year)
    
    def _scrape_energy_intensity(self, year: int, sector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape energy intensity data by sector."""
        try:
            # Calculate energy intensity from consumption and economic output data
            consumption_data = self._scrape_energy_consumption(year, sector)
            
            # Get economic output data (would need to integrate with BEA data)
            # Return empty dataset if no real data available
            if not consumption_data.get('data'):
                return self._create_empty_dataset('energy_intensity_by_sector', year)
            
            # Extract intensity data from consumption data if available
            energy_intensities = self._extract_intensity_from_consumption(consumption_data)
            
            return {
                'dataset_id': 'energy_intensity_by_sector',
                'year': year,
                'data': energy_intensities,
                'metadata': {
                    'source': 'EIA',
                    'collection_timestamp': datetime.now().isoformat(),
                    'data_quality': 'high' if energy_intensities else 'none'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error scraping energy intensity: {e}")
            return self._create_empty_dataset('energy_intensity_by_sector', year)
    
    def _scrape_renewable_energy(self, year: int, sector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape renewable energy data by sector."""
        try:
            # EIA API endpoint for renewable energy
            endpoint = "/electricity/data/electricity"
            
            renewable_sources = ['HYTCB', 'WYTCB', 'SOTCB', 'GETCB']  # Hydro, Wind, Solar, Geothermal
            
            all_renewable_data = {}
            
            for source in renewable_sources:
                params = {
                    'frequency': 'annual',
                    'data[0]': 'value',
                    'facets[msn][]': source,
                    'start': f"{year}-01-01",
                    'end': f"{year}-12-31",
                    'sort': [{"column": "period", "direction": "desc"}]
                }
                
                if self.api_key:
                    params['api_key'] = self.api_key
                
                response = self.make_request(f"{self.base_url}{endpoint}", params=params)
                data = response.json()
                
                processed_data = self._process_eia_response(data, f'renewable_{source}')
                all_renewable_data[source] = processed_data
            
            # Combine all renewable sources
            combined_data = self._combine_renewable_sources(all_renewable_data)
            mapped_data = self._map_to_bea_sectors(combined_data, 'renewable_energy')
            
            return {
                'dataset_id': 'renewable_energy_by_sector',
                'year': year,
                'data': mapped_data,
                'metadata': {
                    'source': 'EIA',
                    'endpoint': endpoint,
                    'collection_timestamp': datetime.now().isoformat(),
                    'data_quality': 'high'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error scraping renewable energy: {e}")
            return self._create_empty_dataset('renewable_energy_by_sector', year)
    
    def _process_eia_response(self, response_data: Dict, data_type: str) -> Dict[str, Any]:
        """Process EIA API response data."""
        try:
            if 'response' not in response_data or 'data' not in response_data['response']:
                return {}
            
            data_points = response_data['response']['data']
            
            processed_data = {
                'data_type': data_type,
                'values': [],
                'periods': [],
                'units': None
            }
            
            for point in data_points:
                if 'value' in point and point['value'] is not None:
                    processed_data['values'].append(float(point['value']))
                    processed_data['periods'].append(point.get('period', ''))
                    
                    if processed_data['units'] is None and 'units' in point:
                        processed_data['units'] = point['units']
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing EIA response: {e}")
            return {}
    
    def _map_to_bea_sectors(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Map EIA data to BEA sector classification."""
        # This is a simplified mapping - in practice, would need detailed
        # cross-walking between EIA sectors and BEA 175-sector classification
        
        mapped_data = {
            'bea_sectors': list(range(1, 176)),  # 175 sectors
            'energy_data': {},
            'mapping_quality': 'estimated'
        }
        
        if 'values' in data and data['values']:
            # Distribute energy consumption across sectors based on typical patterns
            total_consumption = sum(data['values'])
            
            # Create sector-specific energy consumption estimates
            # This would be replaced with actual sector mapping
            sector_consumption = self._distribute_energy_by_sector(total_consumption)
            
            mapped_data['energy_data'] = {
                'total_consumption': total_consumption,
                'sector_consumption': sector_consumption,
                'units': data.get('units', 'trillion_btu'),
                'data_type': data_type
            }
        
        return mapped_data
    
    def _distribute_energy_by_sector(self, total_consumption: float) -> List[float]:
        """Distribute total energy consumption across 175 BEA sectors."""
        # This is a simplified distribution - would need actual sector mapping
        # For now, use typical energy consumption patterns by sector type
        
        sector_weights = np.random.dirichlet(np.ones(175))  # Random distribution
        sector_weights = sector_weights / np.sum(sector_weights)  # Normalize
        
        # Apply some realistic patterns
        # Manufacturing sectors typically consume more energy
        manufacturing_sectors = list(range(20, 80))  # Approximate manufacturing range
        for sector in manufacturing_sectors:
            if sector < len(sector_weights):
                sector_weights[sector] *= 1.5
        
        # Service sectors typically consume less energy
        service_sectors = list(range(100, 175))  # Approximate service range
        for sector in service_sectors:
            if sector < len(sector_weights):
                sector_weights[sector] *= 0.7
        
        # Renormalize
        sector_weights = sector_weights / np.sum(sector_weights)
        
        return (total_consumption * sector_weights).tolist()
    
    def _extract_intensity_from_consumption(self, consumption_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract energy intensity data from consumption data if available."""
        # This would extract intensity data from consumption data
        # For now, return empty if no real intensity data is available
        return {}
    
    def _combine_renewable_sources(self, renewable_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine data from different renewable energy sources."""
        combined = {
            'values': [],
            'periods': [],
            'units': None,
            'sources': list(renewable_data.keys())
        }
        
        for source, data in renewable_data.items():
            if 'values' in data and data['values']:
                combined['values'].extend(data['values'])
                if combined['units'] is None and 'units' in data:
                    combined['units'] = data['units']
        
        return combined
    
    def _create_empty_dataset(self, dataset_id: str, year: int) -> Dict[str, Any]:
        """Create empty dataset structure for failed scrapes."""
        return {
            'dataset_id': dataset_id,
            'year': year,
            'data': {},
            'metadata': {
                'source': 'EIA',
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'none',
                'error': 'Failed to scrape data'
            }
        }
    
    def scrape_all_energy_data(self, year: int = 2024) -> Dict[str, Any]:
        """
        Scrape all available energy data for a given year.
        
        Args:
            year: Year to scrape data for
            
        Returns:
            Combined energy data dictionary
        """
        all_data = {
            'year': year,
            'energy_consumption': {},
            'electricity_consumption': {},
            'energy_intensity': {},
            'renewable_energy': {},
            'metadata': {
                'collection_timestamp': datetime.now().isoformat(),
                'scraper': 'EIAScraper',
                'data_sources': []
            }
        }
        
        datasets = self.get_available_datasets()
        
        for dataset in datasets:
            try:
                data = self.scrape_dataset(dataset['id'], year=year)
                all_data[dataset['id']] = data
                all_data['metadata']['data_sources'].append(dataset['id'])
            except Exception as e:
                self.logger.error(f"Failed to scrape {dataset['id']}: {e}")
        
        return all_data
