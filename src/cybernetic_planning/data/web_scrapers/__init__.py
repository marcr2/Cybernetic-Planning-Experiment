"""
Web Scraping Module for Resource Data Collection

This module provides web scrapers for collecting real resource constraint data
from various government agencies and data sources to replace synthetic data
in the cybernetic planning system.
"""

from .base_scraper import BaseScraper
from .eia_scraper import EIAScraper
from .usgs_scraper import USGSScraper
from .bls_scraper import BLSScraper
from .epa_scraper import EPAScraper
from .data_collector import ResourceDataCollector
from .international_scrapers import InternationalDataCollector

__all__ = [
    "BaseScraper",
    "EIAScraper",
    "USGSScraper",
    "BLSScraper",
    "EPAScraper",
    "ResourceDataCollector",
    "InternationalDataCollector",
]
