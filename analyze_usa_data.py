#!/usr/bin/env python3
"""
Analyze USA Input-Output data files to determine which ones to load
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_file(file_path):
    """Analyze a single data file"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(file_path)
        print(f"📋 Sheets found: {len(excel_file.sheet_names)}")
        
        for i, sheet_name in enumerate(excel_file.sheet_names):
            print(f"  {i+1}. {sheet_name}")
        
        # Analyze each sheet
        for sheet_name in excel_file.sheet_names:
            print(f"\n📄 Sheet: '{sheet_name}'")
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show first few column names
                cols = list(df.columns)[:10]
                print(f"   First 10 columns: {cols}")
                
                # Check if it looks like I-O data
                if df.shape[0] == df.shape[1]:
                    print("   ✅ Square matrix - good for I-O data")
                else:
                    print("   ⚠️  Not square - may need processing")
                
                # Check for numeric data
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                print(f"   Numeric columns: {len(numeric_cols)}/{len(df.columns)}")
                
                # Show sample data
                print("   Sample data (first 3x3):")
                print(df.iloc[:3, :3].to_string())
                
            except Exception as e:
                print(f"   ❌ Error reading sheet: {e}")
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def main():
    """Main analysis function"""
    print("🇺🇸 USA Input-Output Data Analysis")
    print("=" * 60)
    
    # Define the data directory
    data_dir = Path("raw_data/USA_1997-2024")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Key files to analyze
    key_files = [
        "IONom/NOMINAL_USE.xlsx",
        "IONom/NOMINAL_MAKE.xlsx", 
        "IONom/NOMINAL_FD.xlsx",
        "IONom/NOMINAL_FDAGG.xlsx",
        "IONom/NOMINAL_COMOUTPUT.xlsx",
        "IONom/NOMINAL_INDOUTPUT.xlsx",
        "IOReal/REAL_USE.xlsx",
        "IOReal/REAL_MAKE.xlsx",
        "IOReal/REAL_FD.xlsx",
        "IOReal/REAL_FDAGG.xlsx",
        "IOReal/REAL_COMOUTPUT.xlsx",
        "IOReal/REAL_INDOUTPUT.xlsx"
    ]
    
    print("🔍 Analyzing key USA I-O data files...")
    
    for file_path in key_files:
        full_path = data_dir / file_path
        if full_path.exists():
            analyze_file(full_path)
        else:
            print(f"\n❌ File not found: {file_path}")
    
    print(f"\n{'='*60}")
    print("📋 RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("Based on the analysis, you should load these files:")
    print()
    print("1. NOMINAL_USE.xlsx - Use table (intermediate consumption)")
    print("2. NOMINAL_MAKE.xlsx - Make table (production by industry)")
    print("3. NOMINAL_FD.xlsx - Final demand")
    print("4. NOMINAL_FDAGG.xlsx - Aggregated final demand")
    print()
    print("These files contain the core Input-Output data needed for:")
    print("- Technology matrix (A) - derived from USE table")
    print("- Final demand vector (d) - from FD table")
    print("- Output vector (x) - from MAKE table")
    print()
    print("The REAL_* files contain the same data but in real (constant) prices.")
    print("Choose NOMINAL_* for current prices or REAL_* for constant prices.")

if __name__ == "__main__":
    main()
