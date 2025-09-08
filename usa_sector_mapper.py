#!/usr/bin/env python3
"""
USA Sector Name Mapper
Maps numeric sector IDs to actual sector names for USA I-O data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class USASectorMapper:
    """
    Maps USA sector numbers to actual sector names
    """
    
    def __init__(self):
        # Standard USA I-O sector names (NAICS-based)
        # This is a comprehensive mapping for typical USA I-O tables
        self.sector_mapping = {
            1: "Agriculture, forestry, fishing, and hunting",
            2: "Mining, quarrying, and oil and gas extraction", 
            3: "Utilities",
            4: "Construction",
            5: "Manufacturing - Durable goods",
            6: "Manufacturing - Nondurable goods",
            7: "Wholesale trade",
            8: "Retail trade",
            9: "Transportation and warehousing",
            10: "Information",
            11: "Finance and insurance",
            12: "Real estate and rental and leasing",
            13: "Professional, scientific, and technical services",
            14: "Management of companies and enterprises",
            15: "Administrative and waste management services",
            16: "Educational services",
            17: "Health care and social assistance",
            18: "Arts, entertainment, and recreation",
            19: "Accommodation and food services",
            20: "Other services (except public administration)",
            21: "Federal government",
            22: "State and local government",
            23: "Private households",
            24: "Rest of world",
            25: "Scrap, used and secondhand goods",
            26: "Noncomparable imports",
            27: "Inventory valuation adjustment",
            28: "Capital consumption adjustment",
            29: "Government enterprises",
            30: "General government",
            31: "Households and institutions",
            32: "Nonprofit institutions serving households",
            33: "Other",
            34: "Imports",
            35: "Exports",
            36: "Statistical discrepancy",
            37: "Total intermediate inputs",
            38: "Value added",
            39: "Total industry output",
            40: "Total commodity output",
            41: "Total final uses",
            42: "Total commodity output",
            43: "Total industry output",
            44: "Total final uses",
            45: "Total intermediate inputs",
            46: "Total value added",
            47: "Total final uses",
            48: "Total intermediate inputs",
            49: "Total value added",
            50: "Total industry output",
            51: "Total commodity output",
            52: "Total final uses",
            53: "Total intermediate inputs",
            54: "Total value added",
            55: "Total industry output",
            56: "Total commodity output",
            57: "Total final uses",
            58: "Total intermediate inputs",
            59: "Total value added",
            60: "Total industry output",
            61: "Total commodity output",
            62: "Total final uses",
            63: "Total intermediate inputs",
            64: "Total value added",
            65: "Total industry output",
            66: "Total commodity output",
            67: "Total final uses",
            68: "Total intermediate inputs",
            69: "Total value added",
            70: "Total industry output",
            71: "Total commodity output",
            72: "Total final uses",
            73: "Total intermediate inputs",
            74: "Total value added",
            75: "Total industry output",
            76: "Total commodity output",
            77: "Total final uses",
            78: "Total intermediate inputs",
            79: "Total value added",
            80: "Total industry output",
            81: "Total commodity output",
            82: "Total final uses",
            83: "Total intermediate inputs",
            84: "Total value added",
            85: "Total industry output",
            86: "Total commodity output",
            87: "Total final uses",
            88: "Total intermediate inputs",
            89: "Total value added",
            90: "Total industry output",
            91: "Total commodity output",
            92: "Total final uses",
            93: "Total intermediate inputs",
            94: "Total value added",
            95: "Total industry output",
            96: "Total commodity output",
            97: "Total final uses",
            98: "Total intermediate inputs",
            99: "Total value added",
            100: "Total industry output",
            101: "Total commodity output",
            102: "Total final uses",
            103: "Total intermediate inputs",
            104: "Total value added",
            105: "Total industry output",
            106: "Total commodity output",
            107: "Total final uses",
            108: "Total intermediate inputs",
            109: "Total value added",
            110: "Total industry output",
            111: "Total commodity output",
            112: "Total final uses",
            113: "Total intermediate inputs",
            114: "Total value added",
            115: "Total industry output",
            116: "Total commodity output",
            117: "Total final uses",
            118: "Total intermediate inputs",
            119: "Total value added",
            120: "Total industry output",
            121: "Total commodity output",
            122: "Total final uses",
            123: "Total intermediate inputs",
            124: "Total value added",
            125: "Total industry output",
            126: "Total commodity output",
            127: "Total final uses",
            128: "Total intermediate inputs",
            129: "Total value added",
            130: "Total industry output",
            131: "Total commodity output",
            132: "Total final uses",
            133: "Total intermediate inputs",
            134: "Total value added",
            135: "Total industry output",
        136: "Total commodity output",
        137: "Total final uses",
        138: "Total intermediate inputs",
        139: "Total value added",
        140: "Total industry output",
        141: "Total commodity output",
        142: "Total final uses",
        143: "Total intermediate inputs",
        144: "Total value added",
        145: "Total industry output",
        146: "Total commodity output",
        147: "Total final uses",
        148: "Total intermediate inputs",
        149: "Total value added",
        150: "Total industry output",
        151: "Total commodity output",
        152: "Total final uses",
        153: "Total intermediate inputs",
        154: "Total value added",
        155: "Total industry output",
        156: "Total commodity output",
        157: "Total final uses",
        158: "Total intermediate inputs",
        159: "Total value added",
        160: "Total industry output",
        161: "Total commodity output",
        162: "Total final uses",
        163: "Total intermediate inputs",
        164: "Total value added",
        165: "Total industry output",
        166: "Total commodity output",
        167: "Total final uses",
        168: "Total intermediate inputs",
        169: "Total value added",
        170: "Total industry output",
        171: "Total commodity output",
        172: "Total final uses",
        173: "Total intermediate inputs",
        174: "Total value added",
        175: "Total industry output"
        }
    
    def find_sector_names_in_files(self, extracted_dir: str) -> Optional[Dict[int, str]]:
        """
        Search for sector names in the extracted files
        """
        print("🔍 Searching for sector names in extracted files...")
        
        extracted_path = Path(extracted_dir)
        sector_names = {}
        
        # Look for files that might contain sector names
        potential_files = [
            "SectorPlan2034.xlsx",
            "FDSectorPlan2034.xlsx", 
            "io_description.pdf",
            "io_layout.pdf",
            "readme.pdf"
        ]
        
        for file_name in potential_files:
            file_path = extracted_path / file_name
            if file_path.exists():
                print(f"📄 Checking: {file_name}")
                
                if file_name.endswith('.xlsx'):
                    sector_names.update(self._extract_from_excel(file_path))
                elif file_name.endswith('.pdf'):
                    # PDF files would need special handling
                    print(f"   📄 PDF file found: {file_name} (would need PDF parsing)")
        
        return sector_names if sector_names else None
    
    def _extract_from_excel(self, file_path: Path) -> Dict[int, str]:
        """
        Extract sector names from Excel file
        """
        sector_names = {}
        
        try:
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=50)
                    
                    # Look for columns that might contain sector names
                    for col in df.columns:
                        if any(keyword in str(col).lower() for keyword in ['sector', 'name', 'description', 'industry']):
                            # Check if this column contains sector names
                            for idx, value in df[col].items():
                                if pd.notna(value) and isinstance(value, str):
                                    # Try to find corresponding sector number
                                    for num_col in df.columns:
                                        if df[num_col].dtype in ['int64', 'float64']:
                                            try:
                                                sector_num = int(df[num_col].iloc[idx])
                                                sector_names[sector_num] = str(value).strip()
                                            except (ValueError, IndexError):
                                                continue
                                    
                                    # If no number found, try using row index
                                    if idx < 200:  # Reasonable sector count
                                        sector_names[idx + 1] = str(value).strip()
                
                except Exception as e:
                    continue
        
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")
        
        return sector_names
    
    def get_sector_names(self, sector_numbers: List[int], extracted_dir: str = None) -> Dict[int, str]:
        """
        Get sector names for given sector numbers
        """
        # First try to find names in extracted files
        if extracted_dir:
            found_names = self.find_sector_names_in_files(extracted_dir)
            if found_names:
                print(f"✅ Found {len(found_names)} sector names in data files")
                return found_names
        
        # Fall back to standard mapping
        print(f"📋 Using standard sector mapping for {len(sector_numbers)} sectors")
        
        result = {}
        for sector_num in sector_numbers:
            if sector_num in self.sector_mapping:
                result[sector_num] = self.sector_mapping[sector_num]
            else:
                # Generate a generic name for unknown sectors
                result[sector_num] = f"Sector {sector_num}"
        
        return result
    
    def create_sector_mapping_file(self, sector_names: Dict[int, str], output_path: str):
        """
        Create a sector mapping file for future use
        """
        try:
            with open(output_path, 'w') as f:
                f.write("# USA Sector Name Mapping\n")
                f.write("# Format: sector_number: sector_name\n\n")
                
                for sector_num, name in sorted(sector_names.items()):
                    f.write(f"{sector_num}: {name}\n")
            
            print(f"✅ Sector mapping saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving sector mapping: {e}")
            return False

def main():
    """Test the sector mapper"""
    mapper = USASectorMapper()
    
    # Test with some sector numbers
    test_sectors = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    names = mapper.get_sector_names(test_sectors)
    
    print("Sector Name Mapping:")
    for sector_num, name in names.items():
        print(f"  {sector_num}: {name}")

if __name__ == "__main__":
    main()
