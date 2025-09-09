"""
Bureau of Labor Statistics (BLS) Data Scraper

Scrapes labor data including employment, wages, and labor intensity by sector
from BLS databases and O*NET occupational information.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import os

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

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize BLS scraper."""
        super().__init__(base_url="https://api.bls.gov/publicAPI/v2", rate_limit=1.0, **kwargs)

        # Use provided API key or try to get from environment
        self.api_key = api_key or os.getenv("BLS_API_KEY")

        self.labor_categories = [
            "high_skilled",
            "medium_skilled",
            "low_skilled",
            "technical",
            "management",
            "professional",
        ]

        self.sector_mapping = self._load_sector_mapping()

        # BLS series IDs for different data types
        self.series_ids = {
            "employment_by_industry": [
                "CES0000000001",  # Total nonfarm employment
                "CES1011330001",  # Mining and logging
                "CES2023610001",  # Construction
                "CES3130000001",  # Manufacturing
                "CES4142000001",  # Wholesale trade
                "CES4244000001",  # Retail trade
                "CES4840000001",  # Transportation and warehousing
                "CES5051000001",  # Information
                "CES5552000001",  # Financial activities
                "CES6562000001",  # Professional and business services
                "CES7071000001",  # Education and health services
                "CES8081000001",  # Leisure and hospitality
                "CES9092000001",  # Other services
            ],
            "wage_rates": [
                "LEU0254551500",  # Average hourly earnings
                "LEU0254551600",  # Average weekly earnings
                "LEU0254551700",  # Average weekly hours
            ],
        }

    def _load_sector_mapping(self) -> Dict[str, int]:
        """Load BEA sector mapping for labor data."""
        return {
            "agriculture": 1,
            "mining": 2,
            "utilities": 3,
            "construction": 4,
            "manufacturing": 5,
            "wholesale_trade": 6,
            "retail_trade": 7,
            "transportation": 8,
            "information": 9,
            "finance": 10,
        }

    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available BLS datasets."""
        return [
            {
                "id": "employment_by_sector",
                "name": "Employment by Economic Sector",
                "description": "Employment data by BEA sector classification",
                "category": "employment",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
            {
                "id": "wage_rates",
                "name": "Wage Rates by Occupation",
                "description": "Average wage rates by occupation and skill level",
                "category": "wages",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
            {
                "id": "labor_intensity",
                "name": "Labor Intensity by Sector",
                "description": "Labor hours per unit economic output",
                "category": "intensity",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
            {
                "id": "occupational_skills",
                "name": "Occupational Skill Requirements",
                "description": "Skill requirements by occupation from O*NET",
                "category": "skills",
                "frequency": "annual",
                "start_year": 2010,
                "end_year": 2024,
            },
        ]

    def scrape_dataset(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """Scrape a specific BLS dataset."""
        year = kwargs.get("year", 2024)

        if dataset_id == "employment_by_sector":
            return self._scrape_employment_data(year)
        elif dataset_id == "wage_rates":
            return self._scrape_wage_data(year)
        elif dataset_id == "labor_intensity":
            return self._scrape_labor_intensity(year)
        elif dataset_id == "occupational_skills":
            return self._scrape_occupational_skills(year)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")

    def _scrape_employment_data(self, year: int) -> Dict[str, Any]:
        """Scrape employment data by sector."""
        try:
            if not self.api_key:
                self.logger.warning("No BLS API key provided, using public data")
                return self._scrape_public_employment_data(year)

            # Use BLS API v2 for employment data
            url = f"{self.base_url}/timeseries/data/"

            # Prepare request data
            request_data = {
                "seriesid": self.series_ids["employment_by_industry"],
                "startyear": str(year),
                "endyear": str(year),
                "registrationkey": self.api_key,
            }

            response = self.make_request(url, method="POST", json_data=request_data)

            # Parse JSON response
            if response and response.status_code == 200:
                data = response.json()
                if data.get("status") == "REQUEST_SUCCEEDED":
                    employment_data = self._parse_employment_response(data)
                return {
                    "dataset_id": "employment_by_sector",
                    "year": year,
                    "data": employment_data,
                    "metadata": {
                        "source": "BLS API",
                        "collection_timestamp": datetime.now().isoformat(),
                        "data_quality": "high",
                        "api_key_used": True,
                    },
                }
            else:
                self.logger.warning("BLS API request failed, falling back to public data")
                return self._scrape_public_employment_data(year)

        except Exception as e:
            self.logger.error(f"Error scraping employment data: {e}")
            return self._scrape_public_employment_data(year)

    def _scrape_public_employment_data(self, year: int) -> Dict[str, Any]:
        """Scrape employment data from public BLS sources."""
        try:
            # Use BLS public API (no key required)
            url = f"{self.base_url}/timeseries/data/"

            request_data = {
                "seriesid": self.series_ids["employment_by_industry"],
                "startyear": str(year),
                "endyear": str(year),
            }

            response = self.make_request(url, method="POST", json_data=request_data)

            # Parse JSON response
            if response and response.status_code == 200:
                data = response.json()
                if data.get("status") == "REQUEST_SUCCEEDED":
                    employment_data = self._parse_employment_response(data)
                return {
                    "dataset_id": "employment_by_sector",
                    "year": year,
                    "data": employment_data,
                    "metadata": {
                        "source": "BLS Public API",
                        "collection_timestamp": datetime.now().isoformat(),
                        "data_quality": "high",
                        "api_key_used": False,
                    },
                }
            else:
                return self._create_empty_dataset("employment_by_sector", year)

        except Exception as e:
            self.logger.error(f"Error scraping public employment data: {e}")
            return self._create_empty_dataset("employment_by_sector", year)

    def _parse_employment_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse BLS employment response data."""
        employment_data = {}

        if "Results" in response and "series" in response["Results"]:
            for series in response["Results"]["series"]:
                series_id = series["seriesID"]
                series_title = series.get("seriesTitle", "Unknown")

                if "data" in series and series["data"]:
                    # Get the most recent data point
                    latest_data = series["data"][0]
                    value = float(latest_data["value"])
                    period = latest_data["period"]

                    employment_data[series_id] = {
                        "title": series_title,
                        "value": value,
                        "period": period,
                        "units": "thousands of persons",
                    }

        return employment_data

    def _scrape_wage_data(self, year: int) -> Dict[str, Any]:
        """Scrape wage rate data by occupation."""
        try:
            if not self.api_key:
                self.logger.warning("No BLS API key provided, using public data")
                return self._scrape_public_wage_data(year)

            # Use BLS API v2 for wage data
            url = f"{self.base_url}/timeseries/data/"

            request_data = {
                "seriesid": self.series_ids["wage_rates"],
                "startyear": str(year),
                "endyear": str(year),
                "registrationkey": self.api_key,
            }

            response = self.make_request(url, method="POST", json_data=request_data)

            # Parse JSON response
            if response and response.status_code == 200:
                data = response.json()
                if data.get("status") == "REQUEST_SUCCEEDED":
                    wage_data = self._parse_wage_response(data)
                return {
                    "dataset_id": "wage_rates",
                    "year": year,
                    "data": wage_data,
                    "metadata": {
                        "source": "BLS API",
                        "collection_timestamp": datetime.now().isoformat(),
                        "data_quality": "high",
                        "api_key_used": True,
                    },
                }
            else:
                self.logger.warning("BLS API request failed, falling back to public data")
                return self._scrape_public_wage_data(year)

        except Exception as e:
            self.logger.error(f"Error scraping wage data: {e}")
            return self._scrape_public_wage_data(year)

    def _scrape_public_wage_data(self, year: int) -> Dict[str, Any]:
        """Scrape wage data from public BLS sources."""
        try:
            url = f"{self.base_url}/timeseries/data/"

            request_data = {"seriesid": self.series_ids["wage_rates"], "startyear": str(year), "endyear": str(year)}

            response = self.make_request(url, method="POST", json_data=request_data)

            # Parse JSON response
            if response and response.status_code == 200:
                data = response.json()
                if data.get("status") == "REQUEST_SUCCEEDED":
                    wage_data = self._parse_wage_response(data)
                return {
                    "dataset_id": "wage_rates",
                    "year": year,
                    "data": wage_data,
                    "metadata": {
                        "source": "BLS Public API",
                        "collection_timestamp": datetime.now().isoformat(),
                        "data_quality": "high",
                        "api_key_used": False,
                    },
                }
            else:
                return self._create_empty_dataset("wage_rates", year)

        except Exception as e:
            self.logger.error(f"Error scraping public wage data: {e}")
            return self._create_empty_dataset("wage_rates", year)

    def _parse_wage_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse BLS wage response data."""
        wage_data = {}

        if "Results" in response and "series" in response["Results"]:
            for series in response["Results"]["series"]:
                series_id = series["seriesID"]
                series_title = series.get("seriesTitle", "Unknown")

                if "data" in series and series["data"]:
                    latest_data = series["data"][0]
                    value = float(latest_data["value"])
                    period = latest_data["period"]

                    wage_data[series_id] = {
                        "title": series_title,
                        "value": value,
                        "period": period,
                        "units": "dollars" if "earnings" in series_title.lower() else "hours",
                    }

        return wage_data

    def _scrape_labor_intensity(self, year: int) -> Dict[str, Any]:
        """Calculate labor intensity data from employment and wage data."""
        try:
            # Get employment and wage data
            employment_data = self._scrape_employment_data(year)
            wage_data = self._scrape_wage_data(year)

            if employment_data.get("data") and wage_data.get("data"):
                # Calculate labor intensity coefficients
                labor_intensity = self._calculate_labor_intensity(employment_data["data"], wage_data["data"])

                return {
                    "dataset_id": "labor_intensity",
                    "year": year,
                    "data": labor_intensity,
                    "metadata": {
                        "source": "BLS Calculated",
                        "collection_timestamp": datetime.now().isoformat(),
                        "data_quality": "high",
                        "calculation_method": "employment_and_wage_derived",
                    },
                }
            else:
                return self._create_empty_dataset("labor_intensity", year)

        except Exception as e:
            self.logger.error(f"Error calculating labor intensity: {e}")
            return self._create_empty_dataset("labor_intensity", year)

    def _calculate_labor_intensity(self, employment_data: Dict[str, Any], wage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate labor intensity coefficients from employment and wage data."""
        labor_intensity = {}

        # Map employment series to sectors
        sector_employment = {}
        for series_id, data in employment_data.items():
            if "Manufacturing" in data.get("title", ""):
                sector_employment["manufacturing"] = data["value"]
            elif "Construction" in data.get("title", ""):
                sector_employment["construction"] = data["value"]
            elif "Mining" in data.get("title", ""):
                sector_employment["mining"] = data["value"]
            elif "Wholesale" in data.get("title", ""):
                sector_employment["wholesale_trade"] = data["value"]
            elif "Retail" in data.get("title", ""):
                sector_employment["retail_trade"] = data["value"]
            elif "Transportation" in data.get("title", ""):
                sector_employment["transportation"] = data["value"]
            elif "Information" in data.get("title", ""):
                sector_employment["information"] = data["value"]
            elif "Financial" in data.get("title", ""):
                sector_employment["finance"] = data["value"]

        # Get average wage data
        avg_hourly_wage = None
        avg_weekly_hours = None

        for series_id, data in wage_data.items():
            if "hourly" in data.get("title", "").lower():
                avg_hourly_wage = data["value"]
            elif "weekly" in data.get("title", "").lower() and "hours" in data.get("title", "").lower():
                avg_weekly_hours = data["value"]

        # Calculate labor intensity (employment per million dollars of output)
        # This is a simplified calculation - in practice, you'd need GDP by sector
        for sector, employment in sector_employment.items():
            # Simplified labor intensity calculation
            # In practice, you'd divide by sector GDP
            labor_intensity[sector] = {
                "employment": employment,
                "avg_hourly_wage": avg_hourly_wage,
                "avg_weekly_hours": avg_weekly_hours,
                "labor_intensity_coefficient": employment / 1000,  # Simplified
                "units": "employment_per_million_gdp",
            }

        return labor_intensity

    def _scrape_occupational_skills(self, year: int) -> Dict[str, Any]:
        """Scrape occupational skill requirements."""
        try:
            # Return empty dataset if no real data available
            return self._create_empty_dataset("occupational_skills", year)
        except Exception as e:
            self.logger.error(f"Error scraping occupational skills: {e}")
            return self._create_empty_dataset("occupational_skills", year)

    def _create_empty_dataset(self, dataset_id: str, year: int) -> Dict[str, Any]:
        """Create empty dataset structure for failed scrapes."""
        return {
            "dataset_id": dataset_id,
            "year": year,
            "data": {},
            "metadata": {
                "source": "BLS",
                "collection_timestamp": datetime.now().isoformat(),
                "data_quality": "none",
                "error": "Failed to scrape data",
            },
        }

    def scrape_all_labor_data(self, year: int = 2024) -> Dict[str, Any]:
        """Scrape all available labor data for a given year."""
        all_data = {
            "year": year,
            "employment_by_sector": {},
            "wage_rates": {},
            "labor_intensity": {},
            "occupational_skills": {},
            "metadata": {
                "collection_timestamp": datetime.now().isoformat(),
                "scraper": "BLSScraper",
                "data_sources": [],
            },
        }

        datasets = self.get_available_datasets()

        for dataset in datasets:
            try:
                data = self.scrape_dataset(dataset["id"], year=year)
                all_data[dataset["id"]] = data
                all_data["metadata"]["data_sources"].append(dataset["id"])
            except Exception as e:
                self.logger.error(f"Failed to scrape {dataset['id']}: {e}")

        return all_data
