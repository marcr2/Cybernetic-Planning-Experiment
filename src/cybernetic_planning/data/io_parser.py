"""
Input-Output Data Parser

Parses Input-Output tables from various sources and formats them
for use in the cybernetic planning system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import json
import csv
from pathlib import Path


class IOParser:
    """
    Parser for Input-Output tables from various sources.

    Supports parsing from CSV, Excel, JSON, and other common formats
    used by statistical offices and economic data providers.
    """

    def __init__(self):
        """Initialize the I-O parser."""
        self.supported_formats = [".csv", ".xlsx", ".xls", ".json"]
        self.parsed_data = {}

    def parse_file(self, file_path: Union[str, Path], format_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse an I-O table from a file.

        Args:
            file_path: Path to the I-O table file
            format_type: File format (auto-detected if None)

        Returns:
            Parsed I-O data dictionary
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if format_type is None:
            format_type = file_path.suffix.lower()

        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")

        if format_type == ".csv":
            return self._parse_csv(file_path)
        elif format_type in [".xlsx", ".xls"]:
            return self._parse_excel(file_path)
        elif format_type == ".json":
            return self._parse_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _parse_csv(self, file_path: Path) -> Dict[str, Any]:
        """Parse CSV file containing I-O data."""
        try:
            # Try to read with pandas first
            df = pd.read_csv(file_path, index_col=0)
            return self._process_dataframe(df, file_path.stem)
        except Exception as e:
            # Fall back to manual CSV parsing
            return self._parse_csv_manual(file_path)

    def _parse_excel(self, file_path: Path) -> Dict[str, Any]:
        """Parse Excel file containing I-O data."""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets_data = {}

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
                sheets_data[sheet_name] = self._process_dataframe(df, sheet_name)

            return sheets_data
        except Exception as e:
            raise ValueError(f"Error parsing Excel file: {e}")

    def _parse_json(self, file_path: Path) -> Dict[str, Any]:
        """Parse JSON file containing I-O data."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert to numpy arrays if needed
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, list):
                    processed_data[key] = np.array(value)
                else:
                    processed_data[key] = value

            return processed_data
        except Exception as e:
            raise ValueError(f"Error parsing JSON file: {e}")

    def _parse_csv_manual(self, file_path: Path) -> Dict[str, Any]:
        """Manually parse CSV file."""
        try:
            with open(file_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) < 2:
                raise ValueError("CSV file must have at least 2 rows")

            # Extract headers and data
            headers = rows[0]
            data_rows = rows[1:]

            # Convert to numpy array
            data_array = np.array([[float(cell) for cell in row] for row in data_rows])

            # Create DataFrame
            df = pd.DataFrame(data_array, columns=headers[1:], index=headers[1:])

            return self._process_dataframe(df, file_path.stem)
        except Exception as e:
            raise ValueError(f"Error manually parsing CSV file: {e}")

    def _process_dataframe(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """
        Process a DataFrame into I-O data format.

        Args:
            df: Input DataFrame
            name: Name for the dataset

        Returns:
            Processed I-O data dictionary
        """
        # Ensure data is numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Fill NaN values with 0
        df = df.fillna(0)

        # Extract matrices and vectors
        data = {"name": name, "sectors": df.index.tolist(), "technology_matrix": df.values, "sector_count": len(df)}

        # Try to identify final demand and labor vectors
        if "final_demand" in df.columns:
            data["final_demand"] = df["final_demand"].values
        if "labor_input" in df.columns:
            data["labor_input"] = df["labor_input"].values

        return data

    def parse_bea_format(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse BEA (Bureau of Economic Analysis) format I-O tables.

        Args:
            file_path: Path to BEA format file

        Returns:
            Parsed I-O data in standard format
        """
        # This would implement BEA-specific parsing
        # For now, use generic CSV parsing
        return self.parse_file(file_path, ".csv")

    def parse_eurostat_format(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse Eurostat format I-O tables.

        Args:
            file_path: Path to Eurostat format file

        Returns:
            Parsed I-O data in standard format
        """
        # This would implement Eurostat-specific parsing
        # For now, use generic CSV parsing
        return self.parse_file(file_path, ".csv")

    def parse_custom_format(self, file_path: Union[str, Path], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse I-O table using custom format configuration.

        Args:
            file_path: Path to I-O table file
            config: Custom format configuration

        Returns:
            Parsed I-O data in standard format
        """
        # Load data based on format
        if config["format"] == "csv":
            df = pd.read_csv(file_path, **config.get("csv_options", {}))
        elif config["format"] == "excel":
            df = pd.read_excel(file_path, **config.get("excel_options", {}))
        else:
            raise ValueError(f"Unsupported custom format: {config['format']}")

        # Apply custom processing
        if "column_mapping" in config:
            df = df.rename(columns=config["column_mapping"])

        if "index_mapping" in config:
            df = df.rename(index=config["index_mapping"])

        # Extract specified columns
        technology_cols = config.get("technology_columns", [])
        final_demand_col = config.get("final_demand_column")
        labor_col = config.get("labor_column")

        data = {"name": config.get("name", "custom_format"), "sectors": df.index.tolist(), "sector_count": len(df)}

        if technology_cols:
            data["technology_matrix"] = df[technology_cols].values

        if final_demand_col:
            data["final_demand"] = df[final_demand_col].values

        if labor_col:
            data["labor_input"] = df[labor_col].values

        return data

    def validate_io_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parsed I-O data for consistency and completeness.

        Args:
            data: Parsed I-O data dictionary

        Returns:
            Validation results dictionary
        """
        validation_results = {"valid": True, "errors": [], "warnings": [], "recommendations": []}

        # Check required fields
        required_fields = ["technology_matrix", "sectors"]
        for field in required_fields:
            if field not in data:
                validation_results["errors"].append(f"Missing required field: {field}")
                validation_results["valid"] = False

        if not validation_results["valid"]:
            return validation_results

        # Validate technology matrix
        tech_matrix = data["technology_matrix"]
        if not isinstance(tech_matrix, np.ndarray):
            validation_results["errors"].append("Technology matrix must be a numpy array")
            validation_results["valid"] = False
        elif tech_matrix.ndim != 2:
            validation_results["errors"].append("Technology matrix must be 2-dimensional")
            validation_results["valid"] = False
        elif tech_matrix.shape[0] != tech_matrix.shape[1]:
            validation_results["errors"].append("Technology matrix must be square")
            validation_results["valid"] = False

        # Check for negative values
        if np.any(tech_matrix < 0):
            validation_results["warnings"].append("Technology matrix contains negative values")

        # Check spectral radius
        if "technology_matrix" in data:
            try:
                eigenvals = np.linalg.eigvals(tech_matrix)
                spectral_radius = np.max(np.abs(eigenvals))
                if spectral_radius >= 1.0:
                    validation_results["warnings"].append(
                        f"Spectral radius ({spectral_radius:.4f}) >= 1, economy may not be productive"
                    )
            except:
                validation_results["warnings"].append("Could not calculate spectral radius")

        # Validate final demand
        if "final_demand" in data:
            final_demand = data["final_demand"]
            if len(final_demand) != tech_matrix.shape[0]:
                validation_results["errors"].append(
                    "Final demand vector length must match technology matrix dimensions"
                )
                validation_results["valid"] = False
            elif np.any(final_demand < 0):
                validation_results["warnings"].append("Final demand contains negative values")

        # Validate labor input
        if "labor_input" in data:
            labor_input = data["labor_input"]
            if len(labor_input) != tech_matrix.shape[0]:
                validation_results["errors"].append("Labor input vector length must match technology matrix dimensions")
                validation_results["valid"] = False
            elif np.any(labor_input < 0):
                validation_results["warnings"].append("Labor input contains negative values")

        # Add recommendations
        if "final_demand" not in data:
            validation_results["recommendations"].append("Consider adding final demand vector")

        if "labor_input" not in data:
            validation_results["recommendations"].append("Consider adding labor input vector")

        return validation_results

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return self.supported_formats.copy()

    def export_data(self, data: Dict[str, Any], output_path: Union[str, Path], format_type: str = "csv") -> None:
        """
        Export I-O data to a file.

        Args:
            data: I-O data dictionary
            output_path: Output file path
            format_type: Output format ('csv', 'excel', 'json')
        """
        output_path = Path(output_path)

        if format_type == "csv":
            self._export_csv(data, output_path)
        elif format_type == "excel":
            self._export_excel(data, output_path)
        elif format_type == "json":
            self._export_json(data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_csv(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data to CSV format."""
        df = pd.DataFrame(data["technology_matrix"], index=data["sectors"], columns=data["sectors"])

        # Add additional columns if available
        if "final_demand" in data:
            df["final_demand"] = data["final_demand"]
        if "labor_input" in data:
            df["labor_input"] = data["labor_input"]

        df.to_csv(output_path)

    def _export_excel(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data to Excel format."""
        with pd.ExcelWriter(output_path) as writer:
            # Technology matrix sheet
            tech_df = pd.DataFrame(data["technology_matrix"], index=data["sectors"], columns=data["sectors"])
            tech_df.to_excel(writer, sheet_name="Technology_Matrix")

            # Additional data sheets
            if "final_demand" in data:
                final_demand_df = pd.DataFrame(data["final_demand"], index=data["sectors"], columns=["Final_Demand"])
                final_demand_df.to_excel(writer, sheet_name="Final_Demand")

            if "labor_input" in data:
                labor_df = pd.DataFrame(data["labor_input"], index=data["sectors"], columns=["Labor_Input"])
                labor_df.to_excel(writer, sheet_name="Labor_Input")

    def _export_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data to JSON format."""
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

    def get_parsed_data(self) -> Dict[str, Any]:
        """Get all parsed data."""
        return self.parsed_data.copy()

    def clear_parsed_data(self) -> None:
        """Clear all parsed data."""
        self.parsed_data = {}
