#!/usr/bin/env python3
"""
API Keys Configuration

Secure API key management for the cybernetic planning system.
This module handles API keys using environment variables for security
and provides links to obtain the required API keys.
"""

import os
import sys
import json
import base64
from typing import Dict, Optional, List
from pathlib import Path
import logging
from cryptography.fernet import Fernet

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
    
    def __init__(self, keys_file: str = "keys.json", use_encryption: bool = True):
        """
        Initialize the API key manager.
        
        Args:
            keys_file: Path to the JSON file containing API keys
            use_encryption: Whether to encrypt the keys file
        """
        self.logger = logging.getLogger(__name__)
        self.keys_file = Path(keys_file)
        self.use_encryption = use_encryption
        self.encryption_key = None
        
        # Initialize encryption if enabled
        if self.use_encryption:
            self._initialize_encryption()
        
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
            'BLS_API_KEY': {
                'name': 'Bureau of Labor Statistics',
                'description': 'Employment, wages, and labor market data',
                'website': 'https://data.bls.gov/registrationEngine/',
                'required_for': ['Employment statistics', 'Wage data', 'Labor intensity analysis'],
                'environment_variable': 'BLS_API_KEY',
                'optional': True
            },
            'USGS_API_KEY': {
                'name': 'US Geological Survey',
                'description': 'Mineral production, consumption, and critical materials data',
                'website': 'https://mrdata.usgs.gov/',
                'required_for': ['Mineral production data', 'Critical materials assessment', 'Supply chain analysis'],
                'environment_variable': 'USGS_API_KEY',
                'optional': True
            },
            'BEA_API_KEY': {
                'name': 'Bureau of Economic Analysis',
                'description': 'Input-Output tables, GDP data, and economic statistics',
                'website': 'https://apps.bea.gov/api/signup/',
                'required_for': ['Input-Output analysis', 'Economic planning', 'Sector analysis'],
                'environment_variable': 'BEA_API_KEY',
                'optional': True
            },
            'GOOGLE_API_KEY': {
                'name': 'Google Gemini AI',
                'description': 'AI agents for economic plan analysis and review',
                'website': 'https://console.cloud.google.com/',
                'required_for': ['AI plan analysis', 'Economic review agents', 'Multi-agent coordination'],
                'environment_variable': 'GOOGLE_API_KEY',
                'optional': True
            }
        }
        
        # Load API keys from both JSON file and environment variables
        self.api_keys = self._load_api_keys()
    
    def _initialize_encryption(self):
        """Initialize encryption for secure key storage."""
        try:
            # Try to load existing encryption key
            key_file = self.keys_file.parent / ".encryption_key"
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new encryption key
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                # Set restrictive permissions on key file
                key_file.chmod(0o600)
        except Exception as e:
            self.logger.warning(f"Failed to initialize encryption: {e}")
            self.use_encryption = False
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt data using Fernet."""
        if not self.use_encryption or not self.encryption_key:
            return data
        
        try:
            f = Fernet(self.encryption_key)
            encrypted_data = f.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using Fernet."""
        if not self.use_encryption or not self.encryption_key:
            return encrypted_data
        
        try:
            f = Fernet(self.encryption_key)
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = f.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        """Load API keys from JSON file and environment variables."""
        keys = {}
        
        # First, try to load from JSON file
        json_keys = self._load_keys_from_json()
        
        for config_name, config in self.api_configs.items():
            # Priority: Environment variable > JSON file
            env_var = config['environment_variable']
            env_value = os.getenv(env_var)
            json_value = json_keys.get(config_name, "")
            
            # Use environment variable if available, otherwise use JSON value
            key_value = env_value if env_value else (json_value if json_value else None)
            keys[config_name] = key_value
            
            if key_value:
                source = "environment" if env_value else "JSON file"
                self.logger.info(f"Loaded API key for {config['name']} from {source}")
            else:
                if not config['optional']:
                    self.logger.warning(f"Missing required API key: {config['name']}")
                else:
                    self.logger.info(f"Optional API key not set: {config['name']}")
        
        return keys
    
    def _load_keys_from_json(self) -> Dict[str, str]:
        """Load API keys from JSON file."""
        if not self.keys_file.exists():
            self.logger.info(f"Keys file {self.keys_file} not found, using environment variables only")
            return {}
        
        try:
            with open(self.keys_file, 'r') as f:
                data = json.load(f)
            
            api_keys = data.get('api_keys', {})
            
            # Decrypt keys if encryption is enabled
            if self.use_encryption:
                decrypted_keys = {}
                for key_name, key_value in api_keys.items():
                    if key_value:  # Only decrypt non-empty values
                        decrypted_keys[key_name] = self._decrypt_data(key_value)
                    else:
                        decrypted_keys[key_name] = key_value
                return decrypted_keys
            
            return api_keys
            
        except Exception as e:
            self.logger.error(f"Failed to load keys from JSON file: {e}")
            return {}
    
    def save_keys_to_json(self, keys: Dict[str, str]) -> bool:
        """Save API keys to JSON file."""
        try:
            # Load existing file structure
            if self.keys_file.exists():
                with open(self.keys_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    "api_keys": {},
                    "metadata": {
                        "version": "1.0",
                        "created": "2024-01-01",
                        "last_updated": "2024-01-01",
                        "description": "Centralized API key storage for Cybernetic Planning System",
                        "security_note": "This file contains sensitive API keys. Keep it secure and never commit to version control."
                    },
                    "api_info": {}
                }
            
            # Update API keys
            api_keys = {}
            for key_name, key_value in keys.items():
                if key_value:  # Only save non-empty values
                    if self.use_encryption:
                        api_keys[key_name] = self._encrypt_data(key_value)
                    else:
                        api_keys[key_name] = key_value
                else:
                    api_keys[key_name] = ""
            
            data['api_keys'] = api_keys
            data['metadata']['last_updated'] = "2024-01-01"  # You might want to use actual timestamp
            
            # Write to file
            with open(self.keys_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Set restrictive permissions
            self.keys_file.chmod(0o600)
            
            self.logger.info(f"API keys saved to {self.keys_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save keys to JSON file: {e}")
            return False
    
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
    
    def get_keys_for_gui(self) -> Dict[str, Dict[str, any]]:
        """Get API keys formatted for GUI display."""
        gui_keys = {}
        
        for config_name, config in self.api_configs.items():
            key_value = self.api_keys.get(config_name, "")
            
            gui_keys[config_name] = {
                'name': config['name'],
                'description': config['description'],
                'website': config['website'],
                'required_for': config['required_for'],
                'environment_variable': config['environment_variable'],
                'is_optional': config['optional'],
                'is_required': not config['optional'],
                'current_value': key_value,
                'is_set': bool(key_value),
                'status': 'Available' if key_value else 'Missing',
                'status_icon': '✅' if key_value else '❌'
            }
        
        return gui_keys
    
    def update_api_key(self, key_name: str, key_value: str) -> bool:
        """Update a specific API key."""
        if key_name not in self.api_configs:
            self.logger.error(f"Unknown API key: {key_name}")
            return False
        
        # Update in memory
        self.api_keys[key_name] = key_value
        
        # Save to JSON file
        return self.save_keys_to_json(self.api_keys)
    
    def clear_api_key(self, key_name: str) -> bool:
        """Clear a specific API key."""
        return self.update_api_key(key_name, "")
    
    def export_keys_template(self, output_file: str = "keys_template.json") -> bool:
        """Export a template JSON file with empty keys."""
        try:
            template_data = {
                "api_keys": {},
                "metadata": {
                    "version": "1.0",
                    "created": "2024-01-01",
                    "last_updated": "2024-01-01",
                    "description": "API Keys Template - Fill in your actual keys",
                    "security_note": "This is a template file. Copy to keys.json and fill in your actual API keys."
                },
                "api_info": {}
            }
            
            # Add empty keys for all services
            for config_name, config in self.api_configs.items():
                template_data["api_keys"][config_name] = ""
                template_data["api_info"][config_name] = {
                    "name": config['name'],
                    "description": config['description'],
                    "website": config['website'],
                    "required": not config['optional']
                }
            
            with open(output_file, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            self.logger.info(f"Keys template exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export keys template: {e}")
            return False
    
    def regenerate_encryption_key(self) -> bool:
        """Regenerate the encryption key (will make existing encrypted data unreadable)."""
        if not self.use_encryption:
            self.logger.warning("Encryption is not enabled")
            return False
        
        try:
            # Generate new encryption key
            new_key = Fernet.generate_key()
            
            # Save new key
            key_file = self.keys_file.parent / ".encryption_key"
            with open(key_file, 'wb') as f:
                f.write(new_key)
            key_file.chmod(0o600)
            
            # Update current key
            self.encryption_key = new_key
            
            self.logger.info("Encryption key regenerated successfully")
            self.logger.warning("Existing encrypted data in keys.json will be unreadable")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to regenerate encryption key: {e}")
            return False
    
    def disable_encryption(self) -> bool:
        """Disable encryption for the keys file."""
        try:
            self.use_encryption = False
            self.encryption_key = None
            
            # Remove encryption key file
            key_file = self.keys_file.parent / ".encryption_key"
            if key_file.exists():
                key_file.unlink()
            
            self.logger.info("Encryption disabled")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to disable encryption: {e}")
            return False
    
    def enable_encryption(self) -> bool:
        """Enable encryption for the keys file."""
        try:
            self.use_encryption = True
            self._initialize_encryption()
            
            if self.encryption_key:
                self.logger.info("Encryption enabled")
                return True
            else:
                self.logger.error("Failed to initialize encryption")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to enable encryption: {e}")
            return False


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
