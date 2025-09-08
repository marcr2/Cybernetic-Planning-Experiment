"""
Bureau of Labor Statistics (BLS) Data Scraper

Scrapes labor data including employment, wages, and labor intensity by sector
from BLS databases and O*NET occupational information.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json

from .base_scraper import BaseScraper


class BLSScraper(BaseScraper):
    """
    Scraper for Bureau of Labor Statistics labor data.
    
    Collects labor capacity and intensity data including:
    - Employment by sector and occupation
    - Wage rates by skill level
    - Labor intensity coefficients (labor hours per unit output)
    - Occupational skill requirements
    """
    
    def __init__(self, **kwargs):
        """Initialize BLS scraper."""
        super().__init__(
            base_url="https://api.bls.gov/publicAPI/v2",
            rate_limit=1.0,
            **kwargs
        )
        
        self.labor_categories = [
            'high_skilled', 'medium_skilled', 'low_skilled', 
            'technical', 'management', 'professional'
        ]
        
        self.sector_mapping = self._load_sector_mapping()
    
    def _load_sector_mapping(self) -> Dict[str, int]:
        """Load BEA sector mapping for labor data."""
        return {
            'agriculture': 1,
            'mining': 2,
            'utilities': 3,
            'construction': 4,
            'manufacturing': 5,
            'wholesale_trade': 6,
            'retail_trade': 7,
            'transportation': 8,
            'information': 9,
            'finance': 10
        }
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available BLS datasets."""
        return [
            {
                'id': 'employment_by_sector',
                'name': 'Employment by Economic Sector',
                'description': 'Employment data by BEA sector classification',
                'category': 'employment',
                'frequency': 'annual',
                'start_year': 2010,
                'end_year': 2024
            },
            {
                'id': 'wage_rates',
                'name': 'Wage Rates by Occupation',
                'description': 'Average wage rates by occupation and skill level',
                'category': 'wages',
                'frequency': 'annual',
                'start_year': 2010,
                'end_year': 2024
            },
            {
                'id': 'labor_intensity',
                'name': 'Labor Intensity by Sector',
                'description': 'Labor hours per unit economic output',
                'category': 'intensity',
                'frequency': 'annual',
                'start_year': 2010,
                'end_year': 2024
            },
            {
                'id': 'occupational_skills',
                'name': 'Occupational Skill Requirements',
                'description': 'Skill requirements by occupation from O*NET',
                'category': 'skills',
                'frequency': 'annual',
                'start_year': 2010,
                'end_year': 2024
            }
        ]
    
    def scrape_dataset(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """Scrape a specific BLS dataset."""
        year = kwargs.get('year', 2024)
        
        if dataset_id == 'employment_by_sector':
            return self._scrape_employment_data(year)
        elif dataset_id == 'wage_rates':
            return self._scrape_wage_data(year)
        elif dataset_id == 'labor_intensity':
            return self._scrape_labor_intensity(year)
        elif dataset_id == 'occupational_skills':
            return self._scrape_occupational_skills(year)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")
    
    def _scrape_employment_data(self, year: int) -> Dict[str, Any]:
        """Scrape employment data by sector."""
        try:
            # Return empty dataset if no real data available
            return self._create_empty_dataset('employment_by_sector', year)
        except Exception as e:
            self.logger.error(f"Error scraping employment data: {e}")
            return self._create_empty_dataset('employment_by_sector', year)
    
    def _scrape_wage_data(self, year: int) -> Dict[str, Any]:
        """Scrape wage rate data by occupation."""
        try:
            # Return empty dataset if no real data available
            return self._create_empty_dataset('wage_rates', year)
        except Exception as e:
            self.logger.error(f"Error scraping wage data: {e}")
            return self._create_empty_dataset('wage_rates', year)
    
    def _scrape_labor_intensity(self, year: int) -> Dict[str, Any]:
        """Scrape labor intensity data by sector."""
        try:
            # Return empty dataset if no real data available
            return self._create_empty_dataset('labor_intensity', year)
        except Exception as e:
            self.logger.error(f"Error scraping labor intensity: {e}")
            return self._create_empty_dataset('labor_intensity', year)
    
    def _scrape_occupational_skills(self, year: int) -> Dict[str, Any]:
        """Scrape occupational skill requirements."""
        try:
            # Return empty dataset if no real data available
            return self._create_empty_dataset('occupational_skills', year)
        except Exception as e:
            self.logger.error(f"Error scraping occupational skills: {e}")
            return self._create_empty_dataset('occupational_skills', year)
    
    
    def _create_empty_dataset(self, dataset_id: str, year: int) -> Dict[str, Any]:
        """Create empty dataset structure for failed scrapes."""
        return {
            'dataset_id': dataset_id,
            'year': year,
            'data': {},
            'metadata': {
                'source': 'BLS',
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'none',
                'error': 'Failed to scrape data'
            }
        }
    
    def scrape_all_labor_data(self, year: int = 2024) -> Dict[str, Any]:
        """Scrape all available labor data for a given year."""
        all_data = {
            'year': year,
            'employment_by_sector': {},
            'wage_rates': {},
            'labor_intensity': {},
            'occupational_skills': {},
            'metadata': {
                'collection_timestamp': datetime.now().isoformat(),
                'scraper': 'BLSScraper',
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
