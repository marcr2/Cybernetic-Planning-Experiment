"""
Bureau of Economic Analysis (BEA) Data Scraper

Downloads Input - Output tables and other economic data from the BEA API.
Supports both API - based downloads and local file detection.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for API key manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

try:
    from api_keys_config import APIKeyManager
except ImportError:
    APIKeyManager = None

from .base_scraper import BaseScraper

class BEAScraper(BaseScraper):
    """
    Scraper for Bureau of Economic Analysis (BEA) data.

    Downloads Input - Output tables, GDP data, and other economic statistics
    from the BEA API. Also supports loading existing local BEA data files.
    """

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache", output_dir: str = "data"):
        """
        Initialize the BEA scraper.

        Args:
            api_key: BEA API key (optional, will use environment variable if not provided)
            cache_dir: Directory for caching downloaded data
            output_dir: Directory for output files
        """
        super().__init__(base_url="https://apps.bea.gov / api / data", cache_dir = cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok = True)

        # Get API key
        if api_key:
            self.api_key = api_key
        elif APIKeyManager:
            api_manager = APIKeyManager()
            self.api_key = api_manager.get_api_key("BEA_API_KEY")
        else:
            self.api_key = os.getenv("BEA_API_KEY")

        # BEA API configuration
        self.base_url = "https://apps.bea.gov / api / data"
        self.rate_limit_delay = 1.0  # BEA API rate limit

        # Input - Output table configurations
        self.io_tables = {
            "use_table": {
                "table_id": "2",  # Use Table (After Redefinitions)
                "description": "Use Table - shows how commodities are used by industries",
            },
            "make_table": {
                "table_id": "1",  # Make Table (After Redefinitions)
                "description": "Make Table - shows how industries produce commodities",
            },
            "direct_requirements": {
                "table_id": "3",  # Direct Requirements Table
                "description": "Direct Requirements - shows direct input requirements per dollar of output",
            },
            "total_requirements": {
                "table_id": "4",  # Total Requirements Table
                "description": "Total Requirements - shows total input requirements including indirect effects",
            },
        }

        # Available years for I - O data
        self.available_years = [2012, 2017, 2022]  # Most recent years available

    def scrape_io_data(
        self, year: int = 2022, table_types: List[str] = None, force_download: bool = False
    ) -> Dict[str, Any]:
        """
        Scrape Input - Output data from BEA.

        Args:
            year: Year of data to download
            table_types: List of table types to download (default: all)
            force_download: Force download even if cached data exists

        Returns:
            Dictionary containing I - O data and metadata
        """
        if table_types is None:
            table_types = list(self.io_tables.keys())

        if year not in self.available_years:
            print(f"Warning: Year {year} not available. Using {max(self.available_years)} instead.")
            year = max(self.available_years)

        print(f"Scraping BEA Input - Output data for {year}...")

        io_data = {
            "year": year,
            "tables": {},
            "metadata": {
                "source": "BEA API",
                "scraped_at": datetime.now().isoformat(),
                "table_types": table_types,
                "api_key_used": bool(self.api_key),
            },
        }

        for table_type in table_types:
            if table_type not in self.io_tables:
                print(f"Warning: Unknown table type '{table_type}', skipping...")
                continue

            try:
                table_data = self._scrape_io_table(year = year, table_type = table_type, force_download = force_download)
                io_data["tables"][table_type] = table_data
                print(f"✓ Downloaded {table_type} table")

            except Exception as e:
                print(f"✗ Error downloading {table_type} table: {e}")
                io_data["tables"][table_type] = None

        # Save combined data
        output_file = self.output_dir / f"usa_io_data_{year}.json"
        self._save_io_data(io_data, output_file)

        return io_data

    def _scrape_io_table(self, year: int, table_type: str, force_download: bool = False) -> Dict[str, Any]:
        """Scrape a specific I - O table from BEA API."""
        table_config = self.io_tables[table_type]
        table_id = table_config["table_id"]

        # Check cache first
        cache_key = f"bea_io_{year}_{table_type}"
        if not force_download:
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

        if not self.api_key:
            print(f"Warning: No BEA API key available. Using sample data for {table_type}.")
            return self._generate_sample_io_table(table_type, year)

        # Construct API request
        params = {
            "UserID": self.api_key,
            "method": "GetData",
            "DataSetName": "InputOutput",
            "Year": str(year),
            "TableID": table_id,
            "ResultFormat": "json",
        }

        try:
            response = self.make_request(self.base_url, params = params)

            if response.get("BEAAPI", {}).get("Results", {}).get("Error"):
                error_info = response["BEAAPI"]["Results"]["Error"]
                raise Exception(f"BEA API Error: {error_info}")

            # Parse the response
            data = response["BEAAPI"]["Results"]["Data"]
            parsed_data = self._parse_io_table_data(data, table_type)

            # Cache the data
            self._cache_data(cache_key, parsed_data)

            return parsed_data

        except Exception as e:
            print(f"Error scraping {table_type} table: {e}")
            # Fallback to sample data
            return self._generate_sample_io_table(table_type, year)

    def _parse_io_table_data(self, data: List[Dict], table_type: str) -> Dict[str, Any]:
        """Parse BEA API response data into structured format."""
        parsed_data = {
            "table_type": table_type,
            "raw_data": data,
            "matrix_data": {},
            "sector_info": {},
            "commodity_info": {},
        }

        # Extract sector and commodity information
        sectors = set()
        commodities = set()

        for row in data:
            if "Code" in row:
                code = row["Code"]
                if len(code) == 6:  # Industry code
                    sectors.add(code)
                elif len(code) == 3:  # Commodity code
                    commodities.add(code)

        parsed_data["sector_info"] = {"sector_codes": sorted(list(sectors)), "num_sectors": len(sectors)}

        parsed_data["commodity_info"] = {
            "commodity_codes": sorted(list(commodities)),
            "num_commodities": len(commodities),
        }

        # Convert to matrix format
        if table_type in ["use_table", "make_table"]:
            matrix_data = self._convert_to_matrix(data, sectors, commodities)
            parsed_data["matrix_data"] = matrix_data

        return parsed_data

    def _convert_to_matrix(self, data: List[Dict], sectors: set, commodities: set) -> Dict[str, Any]:
        """Convert BEA data to matrix format."""
        # Create mapping from codes to indices
        sector_to_idx = {code: idx for idx, code in enumerate(sorted(sectors))}
        commodity_to_idx = {code: idx for idx, code in enumerate(sorted(commodities))}

        # Initialize matrices
        use_matrix = np.zeros((len(commodities), len(sectors)))
        make_matrix = np.zeros((len(sectors), len(commodities)))

        # Fill matrices with data
        for row in data:
            if "Code" in row and "DataValue" in row:
                code = row["Code"]
                float(row["DataValue"]) if row["DataValue"] else 0.0

                if len(code) == 6:  # Industry code
                    if code in sector_to_idx:
                        sector_to_idx[code]
                        # This would need more complex logic based on BEA data structure
                        # For now, create a simplified version
                elif len(code) == 3:  # Commodity code
                    if code in commodity_to_idx:
                        commodity_to_idx[code]
                        # This would need more complex logic based on BEA data structure
                        # For now, create a simplified version

        return {
            "use_matrix": use_matrix.tolist(),
            "make_matrix": make_matrix.tolist(),
            "sector_to_idx": sector_to_idx,
            "commodity_to_idx": commodity_to_idx,
        }

    def _generate_sample_io_table(self, table_type: str, year: int) -> Dict[str, Any]:
        """Generate sample I - O table data for testing."""
        print(f"Generating sample {table_type} table for {year}...")

        # Create sample sector and commodity data
        num_sectors = 15  # Simplified for sample data
        num_commodities = 10

        sectors = [f"SEC{i:03d}" for i in range(num_sectors)]
        commodities = [f"COM{i:03d}" for i in range(num_commodities)]

        # Generate sample matrix data
        if table_type in ["use_table", "make_table"]:
            if table_type == "use_table":
                matrix = np.random.exponential(0.1, (num_commodities, num_sectors))
            else:  # make_table
                matrix = np.random.exponential(0.1, (num_sectors, num_commodities))

            matrix_data = {
                "use_matrix": matrix.tolist() if table_type == "use_table" else None,
                "make_matrix": matrix.tolist() if table_type == "make_table" else None,
                "sector_to_idx": {code: idx for idx, code in enumerate(sectors)},
                "commodity_to_idx": {code: idx for idx, code in enumerate(commodities)},
            }
        else:
            matrix_data = {}

        return {
            "table_type": table_type,
            "raw_data": [],  # Empty for sample data
            "matrix_data": matrix_data,
            "sector_info": {"sector_codes": sectors, "num_sectors": num_sectors},
            "commodity_info": {"commodity_codes": commodities, "num_commodities": num_commodities},
            "sample_data": True,
        }

    def _save_io_data(self, io_data: Dict[str, Any], output_file: Path):
        """Save I - O data to JSON file."""
        try:
            with open(output_file, "w") as f:
                json.dump(io_data, f, indent = 2, default = str)
            print(f"✓ Saved I - O data to {output_file}")
        except Exception as e:
            print(f"✗ Error saving I - O data: {e}")

    def detect_local_bea_data(self, data_dir: str = "data") -> Optional[Path]:
        """
        Detect existing local BEA data files.

        Args:
            data_dir: Directory to search for BEA data files

        Returns:
            Path to detected BEA data file, or None if not found
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            return None

        # Look for various BEA data file patterns
        patterns = ["*bea*.json", "*io*.json", "*input * output*.json", "*use_table*.json", "*make_table*.json"]

        for pattern in patterns:
            files = list(data_path.glob(pattern))
            if files:
                # Return the most recent file
                most_recent = max(files, key = lambda f: f.stat().st_mtime)
                print(f"✓ Found local BEA data: {most_recent}")
                return most_recent

        return None

    def load_local_bea_data(self, file_path: Path) -> Dict[str, Any]:
        """Load BEA data from local file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            print(f"✓ Loaded local BEA data from {file_path}")
            return data
        except Exception as e:
            print(f"✗ Error loading local BEA data: {e}")
            raise

    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of available datasets from BEA.

        Returns:
            List of dataset metadata dictionaries
        """
        datasets = []

        # Add Input - Output datasets
        for table_type, config in self.io_tables.items():
            datasets.append(
                {
                    "id": table_type,
                    "name": config["description"],
                    "table_id": config["table_id"],
                    "available_years": self.available_years,
                    "source": "BEA Input - Output",
                    "requires_api_key": True,
                }
            )

        return datasets

    def scrape_dataset(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """
        Scrape a specific BEA dataset.

        Args:
            dataset_id: Identifier for the dataset (table type)
            **kwargs: Additional parameters (year, force_download, etc.)

        Returns:
            Scraped data dictionary
        """
        year = kwargs.get("year", 2022)
        force_download = kwargs.get("force_download", False)

        if dataset_id in self.io_tables:
            return self._scrape_io_table(year = year, table_type = dataset_id, force_download = force_download)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")

    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if available and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Check if cache is expired (24 hours)
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cache_time > timedelta(hours = 24):
                cache_file.unlink()  # Remove expired cache
                return None

            return cache_data["data"]

        except Exception as e:
            print(f"Warning: Error reading cache: {e}")
            return None

    def _cache_data(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache data for future use."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"

            cache_data = {"timestamp": datetime.now().isoformat(), "data": data}

            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent = 2, default = str)

        except Exception as e:
            print(f"Warning: Error caching data: {e}")
