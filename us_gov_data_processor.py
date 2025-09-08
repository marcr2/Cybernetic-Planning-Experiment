#!/usr/bin/env python3
"""
US Government Data Processor for Cybernetic Planning System

This script helps process US government Input-Output data from sources like:
- Bureau of Economic Analysis (BEA)
- Census Bureau
- Other US statistical agencies
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def analyze_government_file(file_path):
    """Analyze a US government data file and show its structure."""
    print(f"🔍 Analyzing US government data file: {file_path}")
    print("=" * 60)
    
    try:
        # Try different file formats
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            analyze_excel_file(file_path)
        elif file_path.endswith('.csv'):
            analyze_csv_file(file_path)
        else:
            print("❌ Unsupported file format. Please use .xlsx, .xls, or .csv")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

def analyze_excel_file(file_path):
    """Analyze Excel file from US government."""
    print("📊 Excel file detected")
    
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        print(f"📋 Found {len(excel_file.sheet_names)} sheets:")
        
        for i, sheet_name in enumerate(excel_file.sheet_names):
            print(f"  {i+1}. {sheet_name}")
        
        # Analyze each sheet
        for sheet_name in excel_file.sheet_names:
            print(f"\n📄 Analyzing sheet: '{sheet_name}'")
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"   Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
                print(f"   First few rows:")
                print(df.head(3).to_string())
                
                # Check if it looks like I-O data
                if is_io_data(df):
                    print("   ✅ This looks like Input-Output data!")
                    suggest_processing_steps(df, sheet_name)
                else:
                    print("   ⚠️  This doesn't look like standard I-O data")
                    
            except Exception as e:
                print(f"   ❌ Error reading sheet: {e}")
                
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")

def analyze_csv_file(file_path):
    """Analyze CSV file from US government."""
    print("📊 CSV file detected")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ Successfully read with {encoding} encoding")
                break
            except:
                continue
        
        if df is None:
            print("❌ Could not read CSV file with any encoding")
            return
            
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
        print(f"   First few rows:")
        print(df.head(3).to_string())
        
        # Check if it looks like I-O data
        if is_io_data(df):
            print("   ✅ This looks like Input-Output data!")
            suggest_processing_steps(df, "CSV")
        else:
            print("   ⚠️  This doesn't look like standard I-O data")
            
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")

def is_io_data(df):
    """Check if DataFrame looks like Input-Output data."""
    # Check for square-like structure
    if df.shape[0] != df.shape[1]:
        return False
    
    # Check for numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < df.shape[1] * 0.8:  # At least 80% numeric
        return False
    
    # Check for reasonable values (typical I-O coefficients are 0-1)
    numeric_data = df.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        max_val = numeric_data.max().max()
        min_val = numeric_data.min().min()
        if max_val > 10 or min_val < -1:  # Unusual for I-O data
            return False
    
    return True

def suggest_processing_steps(df, sheet_name):
    """Suggest how to process the data for the cybernetic planning system."""
    print(f"\n💡 Processing suggestions for '{sheet_name}':")
    
    # Check if we can identify the structure
    print("   1. Data structure analysis:")
    print(f"      - Rows: {df.shape[0]}")
    print(f"      - Columns: {df.shape[1]}")
    
    # Look for sector names
    if df.index.name or any('sector' in str(col).lower() for col in df.columns):
        print("      - ✅ Sector names detected")
    else:
        print("      - ⚠️  No obvious sector names found")
    
    # Check for required components
    print("\n   2. Required components check:")
    
    # Technology matrix
    if df.shape[0] == df.shape[1]:
        print("      - ✅ Square matrix (good for technology matrix)")
    else:
        print("      - ❌ Not square - need to identify technology matrix portion")
    
    # Look for final demand
    final_demand_cols = [col for col in df.columns if any(keyword in str(col).lower() 
                        for keyword in ['final', 'demand', 'consumption', 'household'])]
    if final_demand_cols:
        print(f"      - ✅ Final demand columns found: {final_demand_cols}")
    else:
        print("      - ⚠️  No obvious final demand columns")
    
    # Look for labor input
    labor_cols = [col for col in df.columns if any(keyword in str(col).lower() 
                  for keyword in ['labor', 'employment', 'hours', 'wages'])]
    if labor_cols:
        print(f"      - ✅ Labor-related columns found: {labor_cols}")
    else:
        print("      - ⚠️  No obvious labor input columns")

def create_processed_data(file_path, output_path=None):
    """Process US government data into cybernetic planning format."""
    print(f"\n🔄 Processing {file_path} for cybernetic planning system...")
    
    if output_path is None:
        output_path = "processed_government_data.json"
    
    try:
        # Load the data
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            # For Excel, try to find the main data sheet
            excel_file = pd.ExcelFile(file_path)
            main_sheet = excel_file.sheet_names[0]  # Use first sheet
            df = pd.read_excel(file_path, sheet_name=main_sheet, index_col=0)
        else:
            df = pd.read_csv(file_path, index_col=0)
        
        # Process the data
        processed_data = process_io_data(df)
        
        # Save processed data
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2, default=convert_numpy)
        
        print(f"✅ Processed data saved to: {output_path}")
        print(f"📊 Processed data summary:")
        print(f"   - Sectors: {len(processed_data.get('sectors', []))}")
        
        # Handle both numpy arrays and lists for shape display
        tech_matrix = processed_data.get('technology_matrix', [])
        if tech_matrix:
            if hasattr(tech_matrix, 'shape'):
                print(f"   - Technology matrix: {tech_matrix.shape}")
            else:
                # It's a list, calculate dimensions
                rows = len(tech_matrix)
                cols = len(tech_matrix[0]) if rows > 0 else 0
                print(f"   - Technology matrix: ({rows}, {cols})")
        else:
            print(f"   - Technology matrix: Not found")
            
        print(f"   - Final demand: {'Found' if 'final_demand' in processed_data else 'Not found'}")
        print(f"   - Labor input: {'Found' if 'labor_input' in processed_data else 'Not found'}")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_io_data(df):
    """Process DataFrame into cybernetic planning format."""
    # Ensure data is numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    
    # Extract sectors
    sectors = df.index.tolist()
    
    # Assume the main matrix is the technology matrix
    tech_matrix = df.values
    
    # Look for final demand and labor input
    final_demand = None
    labor_input = None
    
    # Check for additional columns that might be final demand or labor
    for col in df.columns:
        if any(keyword in str(col).lower() for keyword in ['final', 'demand', 'consumption']):
            final_demand = df[col].values
        elif any(keyword in str(col).lower() for keyword in ['labor', 'employment', 'hours']):
            labor_input = df[col].values
    
    # If not found, create synthetic vectors
    if final_demand is None:
        print("   ⚠️  Final demand not found, creating synthetic vector")
        final_demand = np.random.uniform(50, 200, len(sectors))
    
    if labor_input is None:
        print("   ⚠️  Labor input not found, creating synthetic vector")
        labor_input = np.random.uniform(0.2, 1.5, len(sectors))
    
    # Create processed data
    processed_data = {
        'sectors': sectors,
        'technology_matrix': tech_matrix.tolist(),
        'final_demand': final_demand.tolist(),
        'labor_input': labor_input.tolist(),
        'source': 'US Government Data',
        'processed_date': pd.Timestamp.now().isoformat()
    }
    
    return processed_data

def convert_numpy(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def test_processed_data(data_file):
    """Test if the processed data works with the cybernetic planning system."""
    print(f"\n🧪 Testing processed data with cybernetic planning system...")
    
    try:
        from cybernetic_planning.planning_system import CyberneticPlanningSystem
        
        # Load the system
        system = CyberneticPlanningSystem()
        
        # Load the processed data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        for key in ['technology_matrix', 'final_demand', 'labor_input']:
            if key in data:
                data[key] = np.array(data[key])
        
        # Load into system
        system.load_data_from_dict(data)
        print("✅ Data loaded successfully into planning system")
        
        # Test creating a simple plan
        plan = system.create_plan(use_optimization=False, max_iterations=1)
        print("✅ Plan creation test successful")
        
        # Show summary
        summary = system.get_plan_summary()
        print(f"📊 Plan summary:")
        print(f"   - Total economic output: {summary['total_economic_output']:,.2f}")
        print(f"   - Total labor cost: {summary['total_labor_cost']:,.2f}")
        print(f"   - Labor efficiency: {summary['labor_efficiency']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing with planning system: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for processing US government data."""
    print("🇺🇸 US Government Data Processor for Cybernetic Planning")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("📁 Enter path to your US government data file: ").strip()
    
    if not file_path or not os.path.exists(file_path):
        print("❌ File not found. Please check the path.")
        return
    
    # Analyze the file
    analyze_government_file(file_path)
    
    # Ask if user wants to process it
    process = input("\n🔄 Do you want to process this data for the cybernetic planning system? (y/n): ").strip().lower()
    
    if process == 'y':
        processed_data = create_processed_data(file_path)
        
        if processed_data:
            # Test the processed data
            test_processed_data("processed_government_data.json")
            
            print(f"\n✅ Processing complete!")
            print(f"📁 You can now use 'processed_government_data.json' with the cybernetic planning system")
            print(f"💡 In the GUI, use 'Load from File' and select the processed JSON file")

def detect_data_type(file_path):
    """Detect the type of data file based on filename and content."""
    filename = os.path.basename(file_path).lower()
    
    # Check filename patterns
    if any(keyword in filename for keyword in ['usa', 'us', 'united_states', 'nominal_indoutput']):
        return 'USA'
    elif any(keyword in filename for keyword in ['eu', 'europe', 'eurostat']):
        return 'EU'
    elif any(keyword in filename for keyword in ['china', 'chinese']):
        return 'China'
    elif any(keyword in filename for keyword in ['russia', 'russian']):
        return 'Russia'
    elif any(keyword in filename for keyword in ['india', 'indian']):
        return 'India'
    elif any(keyword in filename for keyword in ['japan', 'japanese']):
        return 'Japan'
    elif any(keyword in filename for keyword in ['germany', 'german']):
        return 'Germany'
    elif any(keyword in filename for keyword in ['uk', 'britain', 'british']):
        return 'UK'
    else:
        # Try to detect from content
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path, nrows=5)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=5)
            else:
                return 'Unknown'
            
            # Check column names for country indicators
            columns_str = ' '.join([str(col).lower() for col in df.columns])
            if any(keyword in columns_str for keyword in ['usa', 'united states', 'us']):
                return 'USA'
            elif any(keyword in columns_str for keyword in ['eu', 'europe']):
                return 'EU'
            elif any(keyword in columns_str for keyword in ['china', 'chinese']):
                return 'China'
            else:
                return 'Generic'
        except:
            return 'Unknown'

def process_file(file_path, output_path):
    """Process a data file and save the result."""
    try:
        data_type = detect_data_type(file_path)
        print(f"🔍 Detected data type: {data_type}")
        
        # Process based on file type
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return process_excel_file(file_path, output_path, data_type)
        elif file_path.endswith('.csv'):
            return process_csv_file(file_path, output_path, data_type)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

def process_excel_file(file_path, output_path, data_type):
    """Process an Excel file."""
    print(f"📊 Processing Excel file: {file_path}")
    
    # Read the Excel file
    df = analyze_excel_file(file_path)
    if df is None:
        return None
    
    # Process the data
    processed_data = process_io_data(df)
    processed_data['data_type'] = data_type
    processed_data['source_file'] = os.path.basename(file_path)
    
    # Save processed data
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2, default=convert_numpy)
    
    print(f"✅ Processed data saved to: {output_path}")
    return processed_data

def process_csv_file(file_path, output_path, data_type):
    """Process a CSV file."""
    print(f"📊 Processing CSV file: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Process the data
    processed_data = process_io_data(df)
    processed_data['data_type'] = data_type
    processed_data['source_file'] = os.path.basename(file_path)
    
    # Save processed data
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2, default=convert_numpy)
    
    print(f"✅ Processed data saved to: {output_path}")
    return processed_data

if __name__ == "__main__":
    main()
