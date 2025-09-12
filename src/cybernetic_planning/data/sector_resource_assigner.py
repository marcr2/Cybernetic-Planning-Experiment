"""
Sector-Resource Assignment System

This module provides intelligent assignment of resources to sectors based on
sector characteristics, economic logic, and real-world resource consumption patterns.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from enum import Enum

class SectorCategory(Enum):
    """Categories of economic sectors."""
    PRIMARY = "primary"           # Agriculture, mining, extraction
    SECONDARY = "secondary"       # Manufacturing, construction
    TERTIARY = "tertiary"         # Services, retail, healthcare
    QUATERNARY = "quaternary"     # Technology, R&D, information
    QUINARY = "quinary"           # Government, education, culture

class ResourceIntensity(Enum):
    """Resource consumption intensity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4

class SectorResourceAssigner:
    """
    Intelligent sector-resource assignment system.
    
    Assigns resources to sectors based on:
    1. Sector category and characteristics
    2. Economic logic and real-world patterns
    3. Resource type and sector compatibility
    4. Intensity levels based on sector needs
    """
    
    def __init__(self):
        """Initialize the sector-resource assigner."""
        self.sector_resource_patterns = self._load_sector_resource_patterns()
        self.resource_categories = self._load_resource_categories()
        
    def _load_sector_resource_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load sector-resource consumption patterns based on real-world data."""
        return {
            # PRIMARY SECTORS (Agriculture, Mining, Extraction)
            "agriculture": {
                "energy": ResourceIntensity.HIGH,
                "water": ResourceIntensity.VERY_HIGH,
                "fertilizers": ResourceIntensity.HIGH,
                "machinery": ResourceIntensity.MEDIUM,
                "labor": ResourceIntensity.HIGH,
                "land": ResourceIntensity.VERY_HIGH,
                "seeds": ResourceIntensity.MEDIUM,
                "pesticides": ResourceIntensity.MEDIUM,
                "fuel": ResourceIntensity.HIGH,
                "electricity": ResourceIntensity.MEDIUM
            },
            "mining": {
                "energy": ResourceIntensity.VERY_HIGH,
                "machinery": ResourceIntensity.VERY_HIGH,
                "labor": ResourceIntensity.MEDIUM,
                "explosives": ResourceIntensity.HIGH,
                "water": ResourceIntensity.MEDIUM,
                "fuel": ResourceIntensity.VERY_HIGH,
                "electricity": ResourceIntensity.HIGH,
                "steel": ResourceIntensity.HIGH,
                "lubricants": ResourceIntensity.MEDIUM
            },
            "oil_gas": {
                "energy": ResourceIntensity.VERY_HIGH,
                "machinery": ResourceIntensity.VERY_HIGH,
                "labor": ResourceIntensity.MEDIUM,
                "water": ResourceIntensity.HIGH,
                "chemicals": ResourceIntensity.HIGH,
                "steel": ResourceIntensity.HIGH,
                "fuel": ResourceIntensity.VERY_HIGH,
                "electricity": ResourceIntensity.HIGH
            },
            
            # SECONDARY SECTORS (Manufacturing, Construction)
            "manufacturing": {
                "energy": ResourceIntensity.HIGH,
                "raw_materials": ResourceIntensity.VERY_HIGH,
                "machinery": ResourceIntensity.HIGH,
                "labor": ResourceIntensity.HIGH,
                "water": ResourceIntensity.MEDIUM,
                "electricity": ResourceIntensity.HIGH,
                "steel": ResourceIntensity.HIGH,
                "aluminum": ResourceIntensity.MEDIUM,
                "plastics": ResourceIntensity.MEDIUM,
                "chemicals": ResourceIntensity.MEDIUM
            },
            "construction": {
                "energy": ResourceIntensity.MEDIUM,
                "cement": ResourceIntensity.VERY_HIGH,
                "steel": ResourceIntensity.HIGH,
                "lumber": ResourceIntensity.HIGH,
                "labor": ResourceIntensity.HIGH,
                "machinery": ResourceIntensity.HIGH,
                "sand": ResourceIntensity.HIGH,
                "gravel": ResourceIntensity.HIGH,
                "water": ResourceIntensity.MEDIUM,
                "electricity": ResourceIntensity.MEDIUM
            },
            "automotive": {
                "energy": ResourceIntensity.HIGH,
                "steel": ResourceIntensity.VERY_HIGH,
                "aluminum": ResourceIntensity.HIGH,
                "plastics": ResourceIntensity.HIGH,
                "labor": ResourceIntensity.HIGH,
                "machinery": ResourceIntensity.HIGH,
                "electronics": ResourceIntensity.MEDIUM,
                "rubber": ResourceIntensity.MEDIUM,
                "glass": ResourceIntensity.MEDIUM,
                "electricity": ResourceIntensity.HIGH
            },
            
            # TERTIARY SECTORS (Services, Retail, Healthcare)
            "healthcare": {
                "energy": ResourceIntensity.MEDIUM,
                "electricity": ResourceIntensity.HIGH,
                "water": ResourceIntensity.MEDIUM,
                "labor": ResourceIntensity.VERY_HIGH,
                "medical_supplies": ResourceIntensity.HIGH,
                "pharmaceuticals": ResourceIntensity.HIGH,
                "equipment": ResourceIntensity.MEDIUM,
                "technology": ResourceIntensity.MEDIUM,
                "transportation": ResourceIntensity.MEDIUM
            },
            "retail": {
                "energy": ResourceIntensity.MEDIUM,
                "electricity": ResourceIntensity.MEDIUM,
                "labor": ResourceIntensity.MEDIUM,
                "transportation": ResourceIntensity.HIGH,
                "packaging": ResourceIntensity.MEDIUM,
                "technology": ResourceIntensity.LOW,
                "water": ResourceIntensity.LOW,
                "fuel": ResourceIntensity.MEDIUM
            },
            "education": {
                "energy": ResourceIntensity.LOW,
                "electricity": ResourceIntensity.MEDIUM,
                "labor": ResourceIntensity.VERY_HIGH,
                "technology": ResourceIntensity.MEDIUM,
                "water": ResourceIntensity.LOW,
                "transportation": ResourceIntensity.MEDIUM,
                "materials": ResourceIntensity.LOW
            },
            
            # QUATERNARY SECTORS (Technology, R&D, Information)
            "technology": {
                "energy": ResourceIntensity.MEDIUM,
                "electricity": ResourceIntensity.HIGH,
                "labor": ResourceIntensity.HIGH,
                "technology": ResourceIntensity.VERY_HIGH,
                "data_centers": ResourceIntensity.HIGH,
                "software": ResourceIntensity.HIGH,
                "hardware": ResourceIntensity.MEDIUM,
                "water": ResourceIntensity.LOW,
                "transportation": ResourceIntensity.LOW
            },
            "research": {
                "energy": ResourceIntensity.MEDIUM,
                "electricity": ResourceIntensity.HIGH,
                "labor": ResourceIntensity.VERY_HIGH,
                "technology": ResourceIntensity.VERY_HIGH,
                "equipment": ResourceIntensity.HIGH,
                "materials": ResourceIntensity.MEDIUM,
                "water": ResourceIntensity.LOW,
                "transportation": ResourceIntensity.LOW
            },
            
            # QUINARY SECTORS (Government, Culture, Public Services)
            "government": {
                "energy": ResourceIntensity.LOW,
                "electricity": ResourceIntensity.MEDIUM,
                "labor": ResourceIntensity.HIGH,
                "technology": ResourceIntensity.MEDIUM,
                "transportation": ResourceIntensity.MEDIUM,
                "water": ResourceIntensity.LOW,
                "materials": ResourceIntensity.LOW
            },
            "transportation": {
                "energy": ResourceIntensity.VERY_HIGH,
                "fuel": ResourceIntensity.VERY_HIGH,
                "labor": ResourceIntensity.MEDIUM,
                "machinery": ResourceIntensity.HIGH,
                "steel": ResourceIntensity.MEDIUM,
                "electricity": ResourceIntensity.MEDIUM,
                "water": ResourceIntensity.LOW,
                "maintenance": ResourceIntensity.HIGH
            }
        }
    
    def _load_resource_categories(self) -> Dict[str, List[str]]:
        """Load resource categories and their typical sector assignments."""
        return {
            "raw_materials": [
                "Iron Ore", "Coal", "Crude Oil", "Natural Gas", "Copper Ore", 
                "Bauxite", "Limestone", "Sand", "Gravel", "Timber", "Water"
            ],
            "processed_materials": [
                "Steel", "Aluminum", "Concrete", "Glass", "Plastic", "Paper",
                "Textiles", "Chemicals", "Fertilizers", "Petroleum Products"
            ],
            "energy": [
                "Electricity", "Gasoline", "Diesel", "Heating Oil", "Propane",
                "Solar Power", "Wind Power", "Hydroelectric", "Nuclear Power"
            ],
            "manufactured_goods": [
                "Machinery", "Vehicles", "Electronics", "Appliances", "Tools",
                "Equipment", "Components", "Parts", "Devices", "Instruments"
            ],
            "agricultural": [
                "Wheat", "Corn", "Rice", "Soybeans", "Cotton", "Livestock",
                "Dairy Products", "Fruits", "Vegetables", "Seeds"
            ],
            "services": [
                "Labor Hours", "Transportation", "Healthcare", "Education",
                "Financial Services", "Communication", "Utilities", "Maintenance"
            ],
            "technology": [
                "Software", "Data Storage", "Computing Power", "Bandwidth",
                "Processing Units", "Memory", "Storage", "Network Capacity"
            ],
            "construction": [
                "Cement", "Steel Beams", "Lumber", "Insulation", "Roofing",
                "Flooring", "Windows", "Doors", "Pipes", "Wiring"
            ]
        }
    
    def classify_sector(self, sector_name: str) -> str:
        """Classify a sector into its category based on name and characteristics."""
        sector_lower = sector_name.lower()
        
        # Primary sectors
        if any(keyword in sector_lower for keyword in ["agriculture", "farming", "crop", "livestock", "fishing"]):
            return "agriculture"
        elif any(keyword in sector_lower for keyword in ["mining", "extraction", "quarry", "drilling"]):
            return "mining"
        elif any(keyword in sector_lower for keyword in ["oil", "gas", "petroleum", "energy", "fuel"]):
            return "oil_gas"
        
        # Secondary sectors
        elif any(keyword in sector_lower for keyword in ["manufacturing", "production", "factory", "plant"]):
            return "manufacturing"
        elif any(keyword in sector_lower for keyword in ["construction", "building", "infrastructure"]):
            return "construction"
        elif any(keyword in sector_lower for keyword in ["automotive", "vehicle", "car", "truck"]):
            return "automotive"
        
        # Tertiary sectors
        elif any(keyword in sector_lower for keyword in ["healthcare", "medical", "hospital", "pharmaceutical"]):
            return "healthcare"
        elif any(keyword in sector_lower for keyword in ["retail", "store", "shop", "commerce", "trade"]):
            return "retail"
        elif any(keyword in sector_lower for keyword in ["education", "school", "university", "training"]):
            return "education"
        
        # Quaternary sectors
        elif any(keyword in sector_lower for keyword in ["technology", "software", "computer", "digital", "tech"]):
            return "technology"
        elif any(keyword in sector_lower for keyword in ["research", "development", "r&d", "innovation"]):
            return "research"
        
        # Quinary sectors
        elif any(keyword in sector_lower for keyword in ["government", "public", "administration", "policy"]):
            return "government"
        elif any(keyword in sector_lower for keyword in ["transportation", "logistics", "shipping", "freight"]):
            return "transportation"
        
        # Default classification
        else:
            return "manufacturing"  # Default to manufacturing for unknown sectors
    
    def get_resource_intensity(self, sector_name: str, resource_name: str) -> ResourceIntensity:
        """Get resource intensity for a specific sector-resource pair."""
        sector_category = self.classify_sector(sector_name)
        
        if sector_category in self.sector_resource_patterns:
            patterns = self.sector_resource_patterns[sector_category]
            
            # Direct match
            if resource_name.lower() in patterns:
                return patterns[resource_name.lower()]
            
            # Category-based matching
            for category, resources in self.resource_categories.items():
                if resource_name in resources:
                    category_key = category.replace("_", " ")
                    if category_key in patterns:
                        return patterns[category_key]
            
            # Generic resource matching
            if "energy" in resource_name.lower() or "electricity" in resource_name.lower():
                return patterns.get("energy", ResourceIntensity.MEDIUM)
            elif "labor" in resource_name.lower() or "work" in resource_name.lower():
                return patterns.get("labor", ResourceIntensity.MEDIUM)
            elif "water" in resource_name.lower():
                return patterns.get("water", ResourceIntensity.LOW)
            elif "machinery" in resource_name.lower() or "equipment" in resource_name.lower():
                return patterns.get("machinery", ResourceIntensity.MEDIUM)
        
        # Default intensity
        return ResourceIntensity.LOW
    
    def assign_resources_to_sector(self, sector_name: str, available_resources: List[str]) -> Dict[str, float]:
        """Assign resources to a sector based on its characteristics."""
        assignments = {}
        
        for resource_name in available_resources:
            intensity = self.get_resource_intensity(sector_name, resource_name)
            
            # Convert intensity to consumption value
            if intensity == ResourceIntensity.NONE:
                consumption = 0.0
            elif intensity == ResourceIntensity.LOW:
                consumption = np.random.uniform(0.01, 0.1)
            elif intensity == ResourceIntensity.MEDIUM:
                consumption = np.random.uniform(0.1, 0.5)
            elif intensity == ResourceIntensity.HIGH:
                consumption = np.random.uniform(0.5, 2.0)
            elif intensity == ResourceIntensity.VERY_HIGH:
                consumption = np.random.uniform(2.0, 5.0)
            
            # Add some randomness to make it more realistic
            consumption *= np.random.uniform(0.8, 1.2)
            
            if consumption > 0:
                assignments[resource_name] = consumption
        
        # Ensure at least one resource is assigned
        if not assignments:
            # Assign a basic resource (usually electricity or labor)
            basic_resource = "Electricity" if "Electricity" in available_resources else available_resources[0]
            assignments[basic_resource] = np.random.uniform(0.1, 0.5)
        
        return assignments
    
    def generate_realistic_resource_matrix(self, 
                                         sector_names: List[str], 
                                         resource_names: List[str]) -> np.ndarray:
        """Generate a realistic resource matrix based on sector characteristics."""
        n_resources = len(resource_names)
        n_sectors = len(sector_names)
        
        # Initialize matrix
        matrix = np.zeros((n_resources, n_sectors))
        
        # Create resource name to index mapping
        resource_to_index = {name: i for i, name in enumerate(resource_names)}
        
        # Assign resources to each sector
        for sector_idx, sector_name in enumerate(sector_names):
            assignments = self.assign_resources_to_sector(sector_name, resource_names)
            
            for resource_name, consumption in assignments.items():
                if resource_name in resource_to_index:
                    resource_idx = resource_to_index[resource_name]
                    matrix[resource_idx, sector_idx] = consumption
        
        return matrix
    
    def get_sector_resource_summary(self, sector_name: str, resource_names: List[str]) -> Dict[str, Any]:
        """Get a summary of resource assignments for a sector."""
        assignments = self.assign_resources_to_sector(sector_name, resource_names)
        sector_category = self.classify_sector(sector_name)
        
        # Categorize assignments by intensity
        intensity_categories = {
            "high_consumption": [],
            "medium_consumption": [],
            "low_consumption": []
        }
        
        for resource, consumption in assignments.items():
            if consumption >= 1.0:
                intensity_categories["high_consumption"].append((resource, consumption))
            elif consumption >= 0.3:
                intensity_categories["medium_consumption"].append((resource, consumption))
            else:
                intensity_categories["low_consumption"].append((resource, consumption))
        
        return {
            "sector_name": sector_name,
            "sector_category": sector_category,
            "total_resources": len(assignments),
            "total_consumption": sum(assignments.values()),
            "intensity_categories": intensity_categories,
            "assignments": assignments
        }


def create_realistic_resource_matrix(sector_names: List[str], 
                                   resource_names: List[str]) -> np.ndarray:
    """Convenience function to create a realistic resource matrix."""
    assigner = SectorResourceAssigner()
    return assigner.generate_realistic_resource_matrix(sector_names, resource_names)


if __name__ == "__main__":
    # Example usage
    assigner = SectorResourceAssigner()
    
    # Example sectors
    sectors = [
        "Agriculture", "Mining", "Manufacturing", "Construction", 
        "Healthcare", "Education", "Technology", "Transportation"
    ]
    
    # Example resources
    resources = [
        "Iron Ore", "Steel", "Electricity", "Labor Hours", "Water",
        "Machinery", "Fuel", "Chemicals", "Technology", "Transportation"
    ]
    
    print("Sector-Resource Assignment Analysis:")
    print("=" * 50)
    
    for sector in sectors:
        summary = assigner.get_sector_resource_summary(sector, resources)
        print(f"\n{sector} ({summary['sector_category']}):")
        print(f"  Total resources: {summary['total_resources']}")
        print(f"  High consumption: {[r[0] for r in summary['intensity_categories']['high_consumption']]}")
        print(f"  Medium consumption: {[r[0] for r in summary['intensity_categories']['medium_consumption']]}")
        print(f"  Low consumption: {[r[0] for r in summary['intensity_categories']['low_consumption']]}")
