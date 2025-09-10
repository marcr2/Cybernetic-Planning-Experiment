"""
Base Web Scraper Class

Provides common functionality for all web scrapers including error handling,
rate limiting, data validation, and common utilities.
"""

import requests
import time
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from pathlib import Path
import json
from datetime import datetime, timedelta
import hashlib

class BaseScraper(ABC):
    """
    Base class for all web scrapers.

    Provides common functionality including:
    - Rate limiting and retry logic - Error handling and logging - Data validation and cleaning - Caching mechanisms - Progress tracking
    """

    def __init__(
        self,
        base_url: str,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        cache_dir: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the base scraper.

        Args:
            base_url: Base URL for the data source
            rate_limit: Minimum seconds between requests
            max_retries: Maximum number of retry attempts
            cache_dir: Directory for caching responses
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(exist_ok = True)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({"User - Agent": "Cybernetic Planning System Data Collector 1.0"})

        # Data storage
        self.collected_data = {}
        self.metadata = {
            "scraper_name": self.__class__.__name__,
            "base_url": self.base_url,
            "collection_timestamp": None,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "data_sources": [],
        }

    @abstractmethod
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of available datasets from the source.

        Returns:
            List of dataset metadata dictionaries
        """

    @abstractmethod
    def scrape_dataset(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """
        Scrape a specific dataset.

        Args:
            dataset_id: Identifier for the dataset
            **kwargs: Additional parameters for scraping

        Returns:
            Scraped data dictionary
        """

    def make_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        method: str = "GET",
        json_data: Optional[Dict] = None,
    ) -> requests.Response:
        """
        Make a rate - limited HTTP request with retry logic.

        Args:
            url: URL to request
            params: Query parameters
            headers: Additional headers
            method: HTTP method (GET, POST, etc.)
            json_data: JSON data for POST requests

        Returns:
            Response object

        Raises:
            requests.RequestException: If all retry attempts fail
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        # Prepare request
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)

        # Check cache first
        cache_key = self._generate_cache_key(url, params)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.logger.debug(f"Using cached response for {url}")
            return cached_response

        # Make request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")

                if method.upper() == "POST" and json_data:
                    response = self.session.post(url, json = json_data, headers = request_headers, timeout = self.timeout)
                elif method.upper() == "POST":
                    response = self.session.post(url, data = params, headers = request_headers, timeout = self.timeout)
                else:
                    response = self.session.get(url, params = params, headers = request_headers, timeout = self.timeout)

                self.last_request_time = time.time()
                self.metadata["total_requests"] += 1

                # Check for successful response
                if response.status_code == 200:
                    self.metadata["successful_requests"] += 1
                    self._cache_response(cache_key, response)
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2**attempt
                    self.logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [404, 500]:  # Not found or server error
                    self.logger.warning(f"HTTP {response.status_code}: {response.text[:200]}")
                    # For 404 / 500 errors, don't retry as they're likely permanent
                    self.metadata["failed_requests"] += 1
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text[:200]}")
                else:
                    self.logger.warning(f"HTTP {response.status_code}: {response.text[:200]}")

            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    self.metadata["failed_requests"] += 1
                    raise
                time.sleep(2**attempt)  # Exponential backoff

        self.metadata["failed_requests"] += 1
        raise requests.RequestException(f"All {self.max_retries + 1} attempts failed")

    def _generate_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key for the request."""
        key_string = f"{url}:{json.dumps(params or {}, sort_keys = True)}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[requests.Response]:
        """Get cached response if available and not expired."""
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

            # Recreate response object
            response = requests.Response()
            response.status_code = cache_data["status_code"]
            response._content = cache_data["content"].encode()
            response.headers.update(cache_data["headers"])

            return response

        except Exception as e:
            self.logger.warning(f"Error reading cache: {e}")
            return None

    def _cache_response(self, cache_key: str, response: requests.Response) -> None:
        """Cache the response for future use."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "status_code": response.status_code,
                "content": response.text,
                "headers": dict(response.headers),
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

        except Exception as e:
            self.logger.warning(f"Error caching response: {e}")

    def validate_data(self, data: Dict[str, Any], required_fields: List[str] = None) -> Dict[str, Any]:
        """
        Validate scraped data for completeness and quality.

        Args:
            data: Data to validate
            required_fields: List of required field names

        Returns:
            Validation results dictionary
        """
        validation_results = {"valid": True, "errors": [], "warnings": [], "data_quality_score": 0.0}

        if not data:
            validation_results["valid"] = False
            validation_results["errors"].append("No data provided")
            return validation_results

        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in data:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Missing required field: {field}")

        # Check data types and values
        for key, value in data.items():
            if value is None:
                validation_results["warnings"].append(f"Field '{key}' is None")
            elif isinstance(value, (list, np.ndarray)) and len(value) == 0:
                validation_results["warnings"].append(f"Field '{key}' is empty")
            elif isinstance(value, str) and not value.strip():
                validation_results["warnings"].append(f"Field '{key}' is empty string")

        # Calculate data quality score
        total_fields = len(data)
        valid_fields = sum(1 for v in data.values() if v is not None and (not isinstance(v, (list, str)) or len(v) > 0))

        if total_fields > 0:
            validation_results["data_quality_score"] = valid_fields / total_fields

        return validation_results

    def clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize scraped data.

        Args:
            data: Raw data to clean

        Returns:
            Cleaned data dictionary
        """
        cleaned_data = {}

        for key, value in data.items():
            if value is None:
                continue

            # Clean string values
            if isinstance(value, str):
                cleaned_value = value.strip()
                if cleaned_value:
                    cleaned_data[key] = cleaned_value

            # Clean numeric values
            elif isinstance(value, (int, float)):
                if not np.isnan(value) and np.isfinite(value):
                    cleaned_data[key] = float(value)

            # Clean arrays and lists
            elif isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    # Convert to numpy array and remove NaN values
                    arr = np.array(value)
                    if arr.dtype in [np.float64, np.float32]:
                        arr = arr[~np.isnan(arr)]
                    if len(arr) > 0:
                        cleaned_data[key] = arr.tolist() if isinstance(value, list) else arr

            # Clean dictionaries
            elif isinstance(value, dict):
                cleaned_dict = self.clean_data(value)
                if cleaned_dict:
                    cleaned_data[key] = cleaned_dict

            else:
                cleaned_data[key] = value

        return cleaned_data

    def save_data(self, data: Dict[str, Any], filename: Optional[str] = None, format_type: str = "json") -> Path:
        """
        Save collected data to file.

        Args:
            data: Data to save
            filename: Output filename (auto - generated if None)
            format_type: Output format ('json', 'csv', 'excel')

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.__class__.__name__}_{timestamp}.{format_type}"

        output_path = self.cache_dir / filename

        if format_type == "json":
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._prepare_for_json(data)
            with open(output_path, "w") as f:
                json.dump(json_data, f, indent = 2, default = str)

        elif format_type == "csv":
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
                df.to_csv(output_path, index = False)
            else:
                # Convert dict to DataFrame
                df = pd.DataFrame([data])
                df.to_csv(output_path, index = False)

        elif format_type == "excel":
            with pd.ExcelWriter(output_path) as writer:
                if isinstance(data, dict) and "data" in data:
                    df = pd.DataFrame(data["data"])
                    df.to_excel(writer, sheet_name="Data", index = False)
                else:
                    # Convert dict to DataFrame
                    df = pd.DataFrame([data])
                    df.to_excel(writer, sheet_name="Data", index = False)

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        self.logger.info(f"Data saved to {output_path}")
        return output_path

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

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of data collection session."""
        return {
            "scraper_name": self.__class__.__name__,
            "base_url": self.base_url,
            "collection_timestamp": self.metadata["collection_timestamp"],
            "total_requests": self.metadata["total_requests"],
            "successful_requests": self.metadata["successful_requests"],
            "failed_requests": self.metadata["failed_requests"],
            "success_rate": (self.metadata["successful_requests"] / max(self.metadata["total_requests"], 1)),
            "data_sources": self.metadata["data_sources"],
            "collected_datasets": len(self.collected_data),
        }

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        self.logger.info("Cache cleared")

    def reset_session(self) -> None:
        """Reset the scraper session."""
        self.session.close()
        self.session = requests.Session()
        self.session.headers.update({"User - Agent": "Cybernetic Planning System Data Collector 1.0"})
        self.collected_data = {}
        self.metadata["collection_timestamp"] = datetime.now().isoformat()
        self.metadata["total_requests"] = 0
        self.metadata["successful_requests"] = 0
        self.metadata["failed_requests"] = 0
        self.metadata["data_sources"] = []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
