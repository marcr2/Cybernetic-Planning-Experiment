"""
API Keys Configuration for Cybernetic Planning System

This file contains API key configurations for various services.
Copy this file to api_keys_config.py and fill in your actual API keys.

IMPORTANT: Never commit actual API keys to version control!
Add api_keys_config.py to your .gitignore file.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Google AI / Generative AI API
GOOGLE_API_KEY = "your_google_api_key_here"

# OpenAI API (if using OpenAI services)
OPENAI_API_KEY = "your_openai_api_key_here"

# Other API keys can be added here as needed
# Example:
# CUSTOM_API_KEY = "your_custom_api_key_here"

# API Configuration
API_CONFIG = {
    "google": {
        "api_key": GOOGLE_API_KEY,
        "model": "gemini - pro",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "model": "gpt - 3.5 - turbo",
        "max_tokens": 1000,
        "temperature": 0.7
    }
}

# Rate limiting configuration
RATE_LIMITS = {
    "google": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    },
    "openai": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    }
}

class APIKeyManager:
    """Manage API keys and credentials for the Cybernetic Planning System."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the API key manager."""
        if project_root is None:
            # Try to find project root by looking for pyproject.toml
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent:
                if (current_dir / "pyproject.toml").exists():
                    project_root = current_dir
                    break
                current_dir = current_dir.parent

            if project_root is None:
                project_root = Path(__file__).parent

        self.project_root = project_root
        self.keys_file = project_root / "keys.json"
        self.api_keys_file = project_root / "api_keys_config.py"

        # Define required and optional API keys
        self.required_keys = [
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY"
        ]

        self.optional_keys = [
            "EIA_API_KEY",
            "BLS_API_KEY",
            "USGS_API_KEY",
            "BEA_API_KEY",
            "EPA_API_KEY",
            "CUSTOM_API_KEY"
        ]

        # Load existing keys
        self._load_keys()

    def _load_keys(self):
        """Load API keys from keys.json file."""
        self.keys = {}
        if self.keys_file.exists():
            try:
                with open(self.keys_file, 'r') as f:
                    data = json.load(f)
                    self.keys = data.get("api_keys", {})
            except Exception as e:
                print(f"Warning: Could not load keys from {self.keys_file}: {e}")

    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get an API key by name."""
        # First try to get from keys.json
        if key_name in self.keys and self.keys[key_name]:
            return self.keys[key_name]

        # Fallback to environment variables
        env_key = os.getenv(key_name)
        if env_key:
            return env_key

        # Fallback to api_keys_config.py constants
        try:
            if key_name == "GOOGLE_API_KEY":
                return GOOGLE_API_KEY if GOOGLE_API_KEY != "your_google_api_key_here" else None
            elif key_name == "OPENAI_API_KEY":
                return OPENAI_API_KEY if OPENAI_API_KEY != "your_openai_api_key_here" else None
        except NameError:
            pass

        return None

    def check_api_key_status(self) -> Dict[str, Any]:
        """Check the status of all API keys."""
        status = {
            "total_keys": len(self.required_keys) + len(self.optional_keys),
            "required_keys_status": {},
            "optional_keys_status": {},
            "overall_status": "unknown"
        }

        # Check required keys
        required_available = 0
        for key in self.required_keys:
            key_value = self.get_api_key(key)
            is_available = key_value is not None and key_value != f"your_{key.lower()}_here"
            status["required_keys_status"][key] = {
                "available": is_available,
                "configured": is_available
            }
            if is_available:
                required_available += 1

        # Check optional keys
        optional_available = 0
        for key in self.optional_keys:
            key_value = self.get_api_key(key)
            is_available = key_value is not None and key_value != ""
            status["optional_keys_status"][key] = {
                "available": is_available,
                "configured": is_available
            }
            if is_available:
                optional_available += 1

        # Determine overall status
        if required_available == len(self.required_keys):
            status["overall_status"] = "excellent"
        elif required_available > 0:
            status["overall_status"] = "partial"
        else:
            status["overall_status"] = "none"

        status["required_available"] = required_available
        status["optional_available"] = optional_available

        return status

    def get_required_api_keys(self) -> List[str]:
        """Get list of required API keys."""
        return self.required_keys.copy()

    def get_optional_api_keys(self) -> List[str]:
        """Get list of optional API keys."""
        return self.optional_keys.copy()

    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate all API keys."""
        validation_results = {}

        for key in self.required_keys + self.optional_keys:
            key_value = self.get_api_key(key)
            is_valid = (
                key_value is not None and
                key_value != "" and
                key_value != f"your_{key.lower()}_here"
            )
            validation_results[key] = is_valid

        return validation_results

    def get_data_collection_capabilities(self) -> Dict[str, bool]:
        """Get data collection capabilities based on available API keys."""
        capabilities = {
            "google_ai": False,
            "openai": False,
            "eia_data": False,
            "bls_data": False,
            "usgs_data": False,
            "bea_data": False,
            "epa_data": False
        }

        # Check for specific API keys
        if self.get_api_key("GOOGLE_API_KEY"):
            capabilities["google_ai"] = True

        if self.get_api_key("OPENAI_API_KEY"):
            capabilities["openai"] = True

        if self.get_api_key("EIA_API_KEY"):
            capabilities["eia_data"] = True

        if self.get_api_key("BLS_API_KEY"):
            capabilities["bls_data"] = True

        if self.get_api_key("USGS_API_KEY"):
            capabilities["usgs_data"] = True

        if self.get_api_key("BEA_API_KEY"):
            capabilities["bea_data"] = True

        if self.get_api_key("EPA_API_KEY"):
            capabilities["epa_data"] = True

        return capabilities

    def print_setup_instructions(self):
        """Print setup instructions for API keys."""
        print("\n" + "="*60)
        print("API KEY SETUP INSTRUCTIONS")
        print("="*60)
        print("\nRequired API Keys:")
        for key in self.required_keys:
            status = "✓ Configured" if self.get_api_key(key) else "✗ Missing"
            print(f"  {key}: {status}")

        print("\nOptional API Keys:")
        for key in self.optional_keys:
            status = "✓ Configured" if self.get_api_key(key) else "○ Not configured"
            print(f"  {key}: {status}")

        print("\nSetup Methods:")
        print("1. Use the GUI to configure keys interactively")
        print("2. Edit keys.json file directly")
        print("3. Set environment variables")
        print("4. Edit api_keys_config.py file")
        print("\n" + "="*60)

    def get_keys_for_gui(self) -> Dict[str, str]:
        """Get API keys formatted for GUI display."""
        gui_keys = {}

        for key in self.required_keys + self.optional_keys:
            value = self.get_api_key(key)
            if value and value != f"your_{key.lower()}_here":
                # Mask the key for display
                masked_value = value[:8] + "*" * (len(value) - 12) + value[-4:] if len(value) > 12 else "*" * len(value)
                gui_keys[key] = masked_value
            else:
                gui_keys[key] = ""

        return gui_keys

    def save_keys_to_json(self, keys: Dict[str, str]) -> bool:
        """Save API keys to keys.json file."""
        try:
            # Load existing data
            data = {}
            if self.keys_file.exists():
                with open(self.keys_file, 'r') as f:
                    data = json.load(f)

            # Update API keys
            data["api_keys"] = keys
            data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Save to file
            with open(self.keys_file, 'w') as f:
                json.dump(data, f, indent = 2)

            # Update internal keys
            self.keys = keys

            return True
        except Exception as e:
            print(f"Error saving keys to JSON: {e}")
            return False

    def export_keys_template(self, filename: str) -> bool:
        """Export a template file for API keys."""
        try:
            template = {
                "api_keys": {
                    key: "" for key in self.required_keys + self.optional_keys
                },
                "instructions": {
                    "required_keys": self.required_keys,
                    "optional_keys": self.optional_keys,
                    "note": "Fill in your actual API keys and save as keys.json"
                },
                "created": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(filename, 'w') as f:
                json.dump(template, f, indent = 2)

            return True
        except Exception as e:
            print(f"Error exporting keys template: {e}")
            return False

    def create_env_template(self) -> bool:
        """Create a .env template file."""
        try:
            env_content = "# API Keys for Cybernetic Planning System\n"
            env_content += "# Copy this file to .env and fill in your actual API keys\n\n"

            for key in self.required_keys:
                env_content += f"{key}=your_{key.lower()}_here\n"

            env_content += "\n# Optional API Keys\n"
            for key in self.optional_keys:
                env_content += f"# {key}=your_{key.lower()}_here\n"

            with open(".env.template", 'w') as f:
                f.write(env_content)

            return True
        except Exception as e:
            print(f"Error creating .env template: {e}")
            return False
