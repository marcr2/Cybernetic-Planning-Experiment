#!/usr/bin/env python3
"""
API Keys Configuration

Secure API key management for the cybernetic planning system.
This module handles API keys using environment variables for security
and provides links to obtain the required API keys.
"""

import os
import sys
from typing import Dict, Optional, List
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class APIKeyManager:
    """
    Secure API key manager for the cybernetic planning system.
    
    Handles API keys for various data sources including:
    - EIA (Energy Information Administration)
    - USGS (US Geological Survey) 
    - BLS (Bureau of Labor Statistics)
    - EPA (Environmental Protection Agency)
    - Eurostat (EU statistics)
    - Other international data sources
    """
    
    def __init__(self):
        """Initialize the API key manager."""
        self.logger = logging.getLogger(__name__)
        
        # API key configuration with source information
        self.api_configs = {
            'EIA_API_KEY': {
                'name': 'Energy Information Administration',
                'description': 'US energy consumption, production, and pricing data',
                'website': 'https://www.eia.gov/opendata/register.php',
                'required_for': ['Energy data collection', 'Electricity consumption', 'Renewable energy data'],
                'environment_variable': 'EIA_API_KEY',
                'optional': False
            },
            'USGS_API_KEY': {
                'name': 'US Geological Survey',
                'description': 'Mineral production, consumption, and critical materials data',
                'website': 'https://www.usgs.gov/apis',
                'required_for': ['Mineral production data', 'Critical materials assessment', 'Supply chain analysis'],
                'environment_variable': 'USGS_API_KEY',
                'optional': True
            },
            'BLS_API_KEY': {
                'name': 'Bureau of Labor Statistics',
                'description': 'Employment, wages, and labor market data',
                'website': 'https://data.bls.gov/registrationEngine/',
                'required_for': ['Employment statistics', 'Wage data', 'Labor intensity analysis'],
                'environment_variable': 'BLS_API_KEY',
                'optional': True
            },
            'EPA_API_KEY': {
                'name': 'Environmental Protection Agency',
                'description': 'Environmental data including emissions and pollution',
                'website': 'https://www.epa.gov/developer',
                'required_for': ['Carbon emissions data', 'Environmental impact assessment', 'Pollution data'],
                'environment_variable': 'EPA_API_KEY',
                'optional': True
            },
            'EUROSTAT_API_KEY': {
                'name': 'Eurostat',
                'description': 'European Union statistics and data',
                'website': 'https://ec.europa.eu/eurostat/web/main/data/api',
                'required_for': ['EU energy data', 'EU material flows', 'EU labor statistics'],
                'environment_variable': 'EUROSTAT_API_KEY',
                'optional': True
            },
            'FRED_API_KEY': {
                'name': 'Federal Reserve Economic Data',
                'description': 'Economic and financial data from the Federal Reserve',
                'website': 'https://fred.stlouisfed.org/docs/api/api_key.html',
                'required_for': ['Economic indicators', 'Financial data', 'Macroeconomic analysis'],
                'environment_variable': 'FRED_API_KEY',
                'optional': True
            }
        }
        
        # Load API keys from environment variables
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        """Load API keys from environment variables."""
        keys = {}
        
        for config_name, config in self.api_configs.items():
            env_var = config['environment_variable']
            key_value = os.getenv(env_var)
            keys[config_name] = key_value
            
            if key_value:
                self.logger.info(f"Loaded API key for {config['name']}")
            else:
                if not config['optional']:
                    self.logger.warning(f"Missing required API key: {config['name']}")
                else:
                    self.logger.info(f"Optional API key not set: {config['name']}")
        
        return keys
    
    def get_api_key(self, service_name: str) -> Optional[str]:
        """
        Get API key for a specific service.
        
        Args:
            service_name: Name of the service (e.g., 'EIA_API_KEY')
            
        Returns:
            API key if available, None otherwise
        """
        return self.api_keys.get(service_name)
    
    def is_api_key_available(self, service_name: str) -> bool:
        """
        Check if API key is available for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if API key is available, False otherwise
        """
        return self.api_keys.get(service_name) is not None
    
    def get_required_api_keys(self) -> List[str]:
        """Get list of required API keys that are missing."""
        missing_keys = []
        
        for config_name, config in self.api_configs.items():
            if not config['optional'] and not self.is_api_key_available(config_name):
                missing_keys.append(config_name)
        
        return missing_keys
    
    def get_optional_api_keys(self) -> List[str]:
        """Get list of optional API keys that are missing."""
        missing_keys = []
        
        for config_name, config in self.api_configs.items():
            if config['optional'] and not self.is_api_key_available(config_name):
                missing_keys.append(config_name)
        
        return missing_keys
    
    def get_api_key_info(self, service_name: str) -> Optional[Dict[str, str]]:
        """
        Get information about a specific API key service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Dictionary with service information
        """
        if service_name not in self.api_configs:
            return None
        
        config = self.api_configs[service_name]
        return {
            'name': config['name'],
            'description': config['description'],
            'website': config['website'],
            'required_for': ', '.join(config['required_for']),
            'environment_variable': config['environment_variable'],
            'is_available': self.is_api_key_available(service_name),
            'is_optional': config['optional']
        }
    
    def print_api_key_status(self) -> None:
        """Print the status of all API keys."""
        print("🔑 API Key Status")
        print("=" * 60)
        
        for config_name, config in self.api_configs.items():
            status = "✅ Available" if self.is_api_key_available(config_name) else "❌ Missing"
            required = "Required" if not config['optional'] else "Optional"
            
            print(f"\n{config['name']} ({config_name})")
            print(f"  Status: {status} ({required})")
            print(f"  Description: {config['description']}")
            print(f"  Website: {config['website']}")
            print(f"  Environment Variable: {config['environment_variable']}")
            print(f"  Used for: {', '.join(config['required_for'])}")
    
    def print_setup_instructions(self) -> None:
        """Print instructions for setting up API keys."""
        print("\n🔧 API Key Setup Instructions")
        print("=" * 60)
        
        print("\n1. Required API Keys:")
        required_missing = self.get_required_api_keys()
        if required_missing:
            for key in required_missing:
                config = self.api_configs[key]
                print(f"\n   {config['name']} ({key}):")
                print(f"   - Visit: {config['website']}")
                print(f"   - Register and obtain your API key")
                print(f"   - Set environment variable: {config['environment_variable']}")
        else:
            print("   ✅ All required API keys are configured")
        
        print("\n2. Optional API Keys (for enhanced functionality):")
        optional_missing = self.get_optional_api_keys()
        if optional_missing:
            for key in optional_missing:
                config = self.api_configs[key]
                print(f"\n   {config['name']} ({key}):")
                print(f"   - Visit: {config['website']}")
                print(f"   - Register and obtain your API key")
                print(f"   - Set environment variable: {config['environment_variable']}")
        else:
            print("   ✅ All optional API keys are configured")
        
        print("\n3. Setting Environment Variables:")
        print("   Windows (PowerShell):")
        print("   $env:EIA_API_KEY='your_api_key_here'")
        print("   $env:USGS_API_KEY='your_api_key_here'")
        print("   # Add to your PowerShell profile for persistence")
        print("   [Environment]::SetEnvironmentVariable('EIA_API_KEY', 'your_key', 'User')")
        
        print("\n   Windows (Command Prompt):")
        print("   set EIA_API_KEY=your_api_key_here")
        print("   set USGS_API_KEY=your_api_key_here")
        
        print("\n   Linux/macOS:")
        print("   export EIA_API_KEY='your_api_key_here'")
        print("   export USGS_API_KEY='your_api_key_here'")
        print("   # Add to ~/.bashrc or ~/.zshrc for persistence")
        
        print("\n4. Security Best Practices:")
        print("   ✅ Never commit API keys to version control")
        print("   ✅ Use environment variables for storage")
        print("   ✅ Rotate API keys regularly")
        print("   ✅ Monitor API key usage")
        print("   ✅ Use least privilege principle")
    
    def create_env_template(self, output_file: str = ".env.template") -> None:
        """
        Create a template .env file with all required environment variables.
        
        Args:
            output_file: Path to the template file
        """
        template_content = "# API Keys Configuration Template\n"
        template_content += "# Copy this file to .env and fill in your actual API keys\n"
        template_content += "# Never commit the .env file to version control\n\n"
        
        for config_name, config in self.api_configs.items():
            template_content += f"# {config['name']}\n"
            template_content += f"# {config['description']}\n"
            template_content += f"# Website: {config['website']}\n"
            template_content += f"# Required for: {', '.join(config['required_for'])}\n"
            template_content += f"{config['environment_variable']}=your_api_key_here\n\n"
        
        with open(output_file, 'w') as f:
            f.write(template_content)
        
        print(f"📁 Environment template created: {output_file}")
        print("   Copy this file to .env and fill in your actual API keys")
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate that all API keys are properly formatted.
        
        Returns:
            Dictionary with validation results for each key
        """
        validation_results = {}
        
        for config_name, key_value in self.api_keys.items():
            if key_value is None:
                validation_results[config_name] = False
                continue
            
            # Basic validation - check if key looks reasonable
            if len(key_value) < 10:  # Most API keys are longer than 10 characters
                validation_results[config_name] = False
                self.logger.warning(f"API key for {config_name} seems too short")
            elif ' ' in key_value:  # API keys typically don't contain spaces
                validation_results[config_name] = False
                self.logger.warning(f"API key for {config_name} contains spaces")
            else:
                validation_results[config_name] = True
        
        return validation_results
    
    def check_api_key_status(self) -> Dict[str, any]:
        """
        Check the status of all API keys for GUI display.
        
        Returns:
            Dictionary with status information for the GUI
        """
        # Get missing keys
        required_missing = self.get_required_api_keys()
        optional_missing = self.get_optional_api_keys()
        
        # Create detailed status for each key
        key_details = {}
        for config_name, config in self.api_configs.items():
            key_value = self.api_keys.get(config_name)
            is_available = key_value is not None
            
            key_details[config_name] = {
                'name': config['name'],
                'description': config['description'],
                'website': config['website'],
                'required_for': config['required_for'],
                'environment_variable': config['environment_variable'],
                'is_available': is_available,
                'is_optional': config['optional'],
                'is_required': not config['optional'],
                'status_text': 'Available' if is_available else 'Missing',
                'status_icon': '✅' if is_available else '❌'
            }
        
        return {
            'api_manager_available': True,
            'required_keys': required_missing,
            'optional_keys': optional_missing,
            'key_details': key_details,
            'total_keys': len(self.api_configs),
            'available_keys': len(self.api_configs) - len(required_missing) - len(optional_missing),
            'missing_required': len(required_missing),
            'missing_optional': len(optional_missing)
        }
    
    def get_data_collection_capabilities(self) -> Dict[str, List[str]]:
        """
        Get data collection capabilities based on available API keys.
        
        Returns:
            Dictionary mapping data types to available capabilities
        """
        capabilities = {
            'energy_data': [],
            'material_data': [],
            'labor_data': [],
            'environmental_data': [],
            'economic_data': []
        }
        
        # Energy data capabilities
        if self.is_api_key_available('EIA_API_KEY'):
            capabilities['energy_data'].extend([
                'Energy consumption by sector',
                'Electricity generation and consumption',
                'Renewable energy production',
                'Energy prices and costs'
            ])
        
        # Material data capabilities
        if self.is_api_key_available('USGS_API_KEY'):
            capabilities['material_data'].extend([
                'Mineral production statistics',
                'Critical materials assessment',
                'Supply chain analysis',
                'Material consumption data'
            ])
        
        # Labor data capabilities
        if self.is_api_key_available('BLS_API_KEY'):
            capabilities['labor_data'].extend([
                'Employment by sector',
                'Wage and salary data',
                'Labor productivity metrics',
                'Occupational statistics'
            ])
        
        # Environmental data capabilities
        if self.is_api_key_available('EPA_API_KEY'):
            capabilities['environmental_data'].extend([
                'Carbon emissions by sector',
                'Air and water pollution data',
                'Environmental impact assessment',
                'Waste generation statistics'
            ])
        
        # Economic data capabilities
        if self.is_api_key_available('FRED_API_KEY'):
            capabilities['economic_data'].extend([
                'Economic indicators',
                'Financial market data',
                'Macroeconomic statistics',
                'Interest rates and inflation'
            ])
        
        return capabilities


def main():
    """Main function to demonstrate API key management."""
    print("🔑 Cybernetic Planning System - API Key Manager")
    print("=" * 60)
    
    # Initialize API key manager
    manager = APIKeyManager()
    
    # Print current status
    manager.print_api_key_status()
    
    # Print setup instructions
    manager.print_setup_instructions()
    
    # Create environment template
    manager.create_env_template()
    
    # Show data collection capabilities
    print("\n📊 Data Collection Capabilities")
    print("=" * 60)
    
    capabilities = manager.get_data_collection_capabilities()
    for data_type, features in capabilities.items():
        if features:
            print(f"\n{data_type.replace('_', ' ').title()}:")
            for feature in features:
                print(f"  ✅ {feature}")
        else:
            print(f"\n{data_type.replace('_', ' ').title()}:")
            print("  ❌ No capabilities available (API key required)")
    
    # Validate API keys
    print("\n🔍 API Key Validation")
    print("=" * 60)
    
    validation_results = manager.validate_api_keys()
    for key_name, is_valid in validation_results.items():
        if manager.api_keys[key_name] is not None:
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"{key_name}: {status}")
    
    print("\n💡 Next Steps:")
    print("1. Obtain required API keys from the websites listed above")
    print("2. Set environment variables with your API keys")
    print("3. Restart the application to load the new keys")
    print("4. Run the data collection system to test the keys")


if __name__ == "__main__":
    main()
