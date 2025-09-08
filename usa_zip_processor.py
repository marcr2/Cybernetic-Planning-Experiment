#!/usr/bin/env python3
"""
USA Data Zip File Processor
Automatically processes USA I-O data from zip files
Detects relevant files, selects appropriate years, and creates ready-to-use data
"""

import zipfile
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class USAZipProcessor:
    """
    Processes USA I-O data from zip files automatically
    """
    
    def __init__(self):
        self.temp_dir = None
        self.extracted_files = {}
        self.available_years = []
        self.selected_year = None
        
    def extract_zip(self, zip_path: str) -> bool:
        """
        Extract zip file to temporary directory
        """
        try:
            self.temp_dir = tempfile.mkdtemp()
            print(f"📦 Extracting zip file to: {self.temp_dir}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            
            print("✅ Zip file extracted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting zip file: {e}")
            return False
    
    def detect_usa_files(self) -> Dict[str, str]:
        """
        Automatically detect USA I-O files in the extracted directory
        """
        print("🔍 Detecting USA I-O files...")
        
        detected_files = {}
        
        # Walk through all extracted files
        for root, dirs, files in os.walk(self.temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                # Detect USE table
                if 'use' in file_lower and file.endswith('.xlsx'):
                    detected_files['use'] = file_path
                    print(f"📊 Found USE table: {file}")
                
                # Detect Final Demand
                elif 'fd' in file_lower and file.endswith('.xlsx') and 'fdagg' not in file_lower:
                    detected_files['final_demand'] = file_path
                    print(f"📊 Found Final Demand: {file}")
                
                # Detect Make table
                elif 'make' in file_lower and file.endswith('.xlsx'):
                    detected_files['make'] = file_path
                    print(f"📊 Found Make table: {file}")
                
                # Detect aggregated final demand
                elif 'fdagg' in file_lower and file.endswith('.xlsx'):
                    detected_files['final_demand_agg'] = file_path
                    print(f"📊 Found Aggregated Final Demand: {file}")
        
        self.extracted_files = detected_files
        return detected_files
    
    def detect_available_years(self, file_path: str) -> List[str]:
        """
        Detect available years in an Excel file
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            year_sheets = [s for s in excel_file.sheet_names if s.isdigit() and len(s) == 4]
            year_sheets.sort()
            return year_sheets
        except Exception as e:
            print(f"⚠️  Could not detect years in {file_path}: {e}")
            return []
    
    def select_best_year(self) -> str:
        """
        Select the best year to use (prefer most recent)
        """
        print("📅 Detecting available years...")
        
        # Check USE table for years
        if 'use' in self.extracted_files:
            years = self.detect_available_years(self.extracted_files['use'])
            if years:
                self.available_years = years
                self.selected_year = max(years)  # Most recent year
                print(f"📅 Available years: {years}")
                print(f"📅 Selected year: {self.selected_year}")
                return self.selected_year
        
        # Fallback to any other file
        for file_type, file_path in self.extracted_files.items():
            years = self.detect_available_years(file_path)
            if years:
                self.available_years = years
                self.selected_year = max(years)
                print(f"📅 Available years: {years}")
                print(f"📅 Selected year: {self.selected_year}")
                return self.selected_year
        
        print("⚠️  No years detected, using first available sheet")
        return None
    
    def load_use_table(self, file_path: str, year: str) -> Optional[Dict]:
        """
        Load USE table for specific year
        """
        try:
            print(f"📊 Loading USE table for {year}...")
            
            excel_file = pd.ExcelFile(file_path)
            
            if year and year in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=year, index_col=0)
            else:
                df = pd.read_excel(file_path, sheet_name=0, index_col=0)
            
            print(f"📏 Original shape: {df.shape}")
            
            # Clean the data
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(0)
            
            # Remove empty rows/columns
            df = df.loc[~(df == 0).all(axis=1)]
            df = df.loc[:, ~(df == 0).all(axis=0)]
            
            # Make square if needed
            if df.shape[0] != df.shape[1]:
                min_dim = min(df.shape[0], df.shape[1])
                df = df.iloc[:min_dim, :min_dim]
                print(f"📏 Adjusted to square: {df.shape}")
            
            return {
                'sectors': df.index.tolist(),
                'technology_matrix': df.values.tolist(),
                'shape': df.shape
            }
            
        except Exception as e:
            print(f"❌ Error loading USE table: {e}")
            return None
    
    def load_final_demand(self, file_path: str, year: str, target_length: int) -> Optional[np.ndarray]:
        """
        Load final demand for specific year
        """
        try:
            print(f"📊 Loading Final Demand for {year}...")
            
            excel_file = pd.ExcelFile(file_path)
            
            if year and year in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=year, index_col=0)
            else:
                df = pd.read_excel(file_path, sheet_name=0, index_col=0)
            
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(0)
            
            # Take first column or sum all columns
            if df.shape[1] > 1:
                final_demand = df.iloc[:, 0].values
            else:
                final_demand = df.iloc[:, 0].values
            
            # Adjust length to match target
            if len(final_demand) != target_length:
                if len(final_demand) > target_length:
                    final_demand = final_demand[:target_length]
                else:
                    # Pad with zeros
                    padding = np.zeros(target_length - len(final_demand))
                    final_demand = np.concatenate([final_demand, padding])
            
            return final_demand
            
        except Exception as e:
            print(f"❌ Error loading Final Demand: {e}")
            return None
    
    def create_labor_input(self, n_sectors: int) -> np.ndarray:
        """
        Create synthetic labor input vector
        """
        # Create more realistic labor input based on sector types
        labor_input = np.random.uniform(0.1, 2.0, n_sectors)
        
        # Make some sectors more labor-intensive
        for i in range(n_sectors):
            if i % 3 == 0:  # Every third sector
                labor_input[i] *= 1.5
        
        return labor_input
    
    def create_resource_data(self, n_sectors: int) -> Dict[str, Any]:
        """
        Create synthetic resource constraint data
        """
        n_resources = 3  # Energy, Materials, Labor Capacity
        
        # Create resource matrix (resources x sectors)
        # Each resource has different requirements per sector
        resource_matrix = np.random.uniform(0.1, 2.0, (n_resources, n_sectors))
        
        # Make resource requirements more realistic
        # Energy: Higher for manufacturing sectors
        for i in range(n_sectors):
            if i < n_sectors * 0.3:  # First 30% are likely manufacturing
                resource_matrix[0, i] *= 2.0  # Energy
        
        # Materials: Higher for construction and manufacturing
        for i in range(n_sectors):
            if i < n_sectors * 0.4:  # First 40% are likely material-intensive
                resource_matrix[1, i] *= 1.5  # Materials
        
        # Labor Capacity: More uniform but varies by sector
        resource_matrix[2, :] = np.random.uniform(0.5, 1.5, n_sectors)
        
        # Create max resources vector based on total economic activity
        # This ensures reasonable constraint levels
        total_economic_activity = n_sectors * 1000  # Base activity level
        max_resources = np.array([
            total_economic_activity * 0.4,  # Energy: 40% of total activity
            total_economic_activity * 0.3,  # Materials: 30% of total activity
            total_economic_activity * 0.2   # Labor: 20% of total activity
        ])
        
        resource_names = ['Energy', 'Materials', 'Labor Capacity']
        
        return {
            'resource_matrix': resource_matrix.tolist(),
            'max_resources': max_resources.tolist(),
            'resource_names': resource_names
        }
    
    def get_sector_names(self, sector_numbers: List[int], extracted_dir: str = None) -> Dict[int, str]:
        """
        Get sector names for given sector numbers
        """
        # Import the sector mapper
        try:
            from usa_sector_mapper import USASectorMapper
            mapper = USASectorMapper()
            return mapper.get_sector_names(sector_numbers, extracted_dir)
        except ImportError:
            # Fallback to basic mapping
            print("⚠️  Sector mapper not available, using basic mapping")
            return {sector: f"Sector {sector}" for sector in sector_numbers}
    
    def process_zip_file(self, zip_path: str, output_file: str = None) -> Optional[str]:
        """
        Main function to process zip file and create I-O data
        """
        print("🇺🇸 USA Zip File Processor")
        print("=" * 50)
        
        # Extract zip file
        if not self.extract_zip(zip_path):
            return None
        
        # Detect files
        detected_files = self.detect_usa_files()
        if not detected_files:
            print("❌ No USA I-O files detected in zip")
            return None
        
        # Select year
        year = self.select_best_year()
        
        # Load USE table
        if 'use' not in detected_files:
            print("❌ No USE table found")
            return None
        
        use_data = self.load_use_table(detected_files['use'], year)
        if not use_data:
            return None
        
        n_sectors = len(use_data['sectors'])
        print(f"📊 Processing {n_sectors} sectors")
        
        # Load final demand
        final_demand = None
        if 'final_demand' in detected_files:
            final_demand = self.load_final_demand(
                detected_files['final_demand'], year, n_sectors
            )
        
        if final_demand is None and 'final_demand_agg' in detected_files:
            final_demand = self.load_final_demand(
                detected_files['final_demand_agg'], year, n_sectors
            )
        
        # Create synthetic final demand if needed
        if final_demand is None:
            print("⚠️  Creating synthetic final demand")
            final_demand = np.random.uniform(50, 200, n_sectors)
        
        # Create labor input
        labor_input = self.create_labor_input(n_sectors)
        
        # Create resource constraint data
        resource_data = self.create_resource_data(n_sectors)
        
        # Get sector names
        sector_names = self.get_sector_names(use_data['sectors'], self.temp_dir)
        
        # Create complete I-O data
        io_data = {
            'sectors': use_data['sectors'],
            'sector_names': [sector_names.get(sector, f"Sector {sector}") for sector in use_data['sectors']],
            'technology_matrix': use_data['technology_matrix'],
            'final_demand': final_demand.tolist(),
            'labor_input': labor_input.tolist(),
            'resource_matrix': resource_data['resource_matrix'],
            'max_resources': resource_data['max_resources'],
            'resource_names': resource_data['resource_names'],
            'source': f'USA BEA Data {year}' if year else 'USA BEA Data',
            'year': year,
            'sector_count': n_sectors,
            'processed_files': list(detected_files.keys())
        }
        
        # Save to file in data/ folder
        if output_file is None:
            # Ensure data directory exists
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            output_file = data_dir / f"usa_io_data_{year if year else 'processed'}.json"
        else:
            output_file = Path(output_file)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(io_data, f, indent=2)
            
            print(f"✅ Successfully created USA I-O data")
            print(f"📁 Saved to: {output_file}")
            print(f"📊 Sectors: {n_sectors}")
            print(f"📊 Matrix size: {use_data['shape'][0]}x{use_data['shape'][1]}")
            print(f"📊 Year: {year}")
            print(f"📊 Files used: {', '.join(detected_files.keys())}")
            
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return None
    
    def cleanup(self):
        """
        Clean up temporary files
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("🧹 Cleaned up temporary files")

def main():
    """
    Main function for command line usage
    """
    if len(sys.argv) < 2:
        print("Usage: python usa_zip_processor.py <zip_file_path> [output_file]")
        print("Example: python usa_zip_processor.py usa_data.zip usa_2024.json")
        return
    
    zip_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        return
    
    processor = USAZipProcessor()
    
    try:
        result = processor.process_zip_file(zip_path, output_file)
        if result:
            print(f"\n🎉 Processing complete!")
            print(f"📁 Output file: {result}")
            print(f"💡 You can now load this file in the GUI")
        else:
            print("❌ Processing failed")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
