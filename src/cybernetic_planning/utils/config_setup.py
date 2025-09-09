"""
Configuration setup utilities for the Cybernetic Planning System.

This module handles the creation and management of configuration files,
API keys, and system settings.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import secrets
import string

class ConfigurationManager:
    """Manage system configuration and settings."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.config_file = self.config_dir / "system_config.json"
        self.api_keys_file = project_root / "api_keys_config.py"
        self.keys_file = project_root / "keys.json"
        
    def create_config_directory(self) -> bool:
        """Create configuration directory structure."""
        try:
            self.config_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"Failed to create config directory: {e}")
            return False
    
    def generate_default_config(self) -> Dict[str, Any]:
        """Generate default system configuration."""
        return {
            "system": {
                "name": "Cybernetic Planning System",
                "version": "0.1.0",
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "installation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "paths": {
                "data_directory": "data",
                "exports_directory": "exports",
                "logs_directory": "logs",
                "cache_directory": "cache",
                "outputs_directory": "outputs",
                "config_directory": "config"
            },
            "planning": {
                "default_sectors": 8,
                "default_years": 5,
                "convergence_threshold": 0.005,
                "max_iterations": 15,
                "optimization_solver": "ECOS",
                "technology_density": 0.4
            },
            "data_processing": {
                "max_file_size_mb": 100,
                "supported_formats": ["csv", "xlsx", "json"],
                "auto_validate": True,
                "backup_original": True
            },
            "visualization": {
                "default_style": "seaborn-v0_8",
                "figure_size": [12, 8],
                "dpi": 300,
                "save_formats": ["png", "pdf", "svg"]
            },
            "security": {
                "encrypt_sensitive_data": True,
                "log_api_usage": True,
                "max_api_calls_per_hour": 1000
            },
            "logging": {
                "level": "INFO",
                "max_file_size_mb": 10,
                "backup_count": 5,
                "log_to_console": True
            }
        }
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            self.create_config_directory()
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        try:
            if not self.config_file.exists():
                return None
            
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return None
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        config = self.load_config() or self.generate_default_config()
        
        # Deep merge updates
        for key, value in updates.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
        
        config["system"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return self.save_config(config)

class APIKeyManager:
    """Manage API keys and credentials."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.api_keys_file = project_root / "api_keys_config.py"
        self.keys_file = project_root / "keys.json"
        self.encryption_key_file = project_root / ".encryption_key"
        self.security_key_file = project_root / ".security_key"
    
    def generate_encryption_key(self) -> str:
        """Generate a secure encryption key."""
        return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
    
    def create_api_keys_template(self) -> bool:
        """Create API keys configuration template."""
        template = '''"""
API Keys Configuration for Cybernetic Planning System

This file contains API key configurations for various services.
Copy this file to api_keys_config.py and fill in your actual API keys.

IMPORTANT: Never commit actual API keys to version control!
Add api_keys_config.py to your .gitignore file.
"""

# Google AI/Generative AI API
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
        "model": "gemini-pro",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "model": "gpt-3.5-turbo",
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
'''
        
        try:
            with open(self.api_keys_file, 'w') as f:
                f.write(template)
            return True
        except Exception as e:
            print(f"Failed to create API keys template: {e}")
            return False
    
    def create_keys_template(self) -> bool:
        """Create keys.json template."""
        template = {
            "encryption_key": self.generate_encryption_key(),
            "security_key": self.generate_encryption_key(),
            "api_keys": {
                "google": "",
                "openai": "",
                "custom": ""
            },
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "This file contains sensitive information. Keep it secure and never commit to version control."
        }
        
        try:
            with open(self.keys_file, 'w') as f:
                json.dump(template, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to create keys template: {e}")
            return False
    
    def create_encryption_keys(self) -> bool:
        """Create encryption key files."""
        try:
            # Create .encryption_key
            with open(self.encryption_key_file, 'w') as f:
                f.write(self.generate_encryption_key())
            
            # Create .security_key
            with open(self.security_key_file, 'w') as f:
                f.write(self.generate_encryption_key())
            
            # Set restrictive permissions (Unix only)
            if os.name != 'nt':
                os.chmod(self.encryption_key_file, 0o600)
                os.chmod(self.security_key_file, 0o600)
            
            return True
        except Exception as e:
            print(f"Failed to create encryption keys: {e}")
            return False
    
    def setup_security(self) -> bool:
        """Setup security configuration."""
        success = True
        
        # Create API keys template
        if not self.create_api_keys_template():
            success = False
        
        # Create keys.json template
        if not self.create_keys_template():
            success = False
        
        # Create encryption keys
        if not self.create_encryption_keys():
            success = False
        
        return success

class DirectorySetup:
    """Setup project directory structure."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.required_directories = [
            "data",
            "exports",
            "logs",
            "cache",
            "outputs",
            "config",
            "tests",
            "temp"
        ]
    
    def create_directories(self) -> bool:
        """Create all required directories."""
        try:
            for directory in self.required_directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                
                # Create .gitkeep files to ensure directories are tracked
                gitkeep_file = dir_path / ".gitkeep"
                if not gitkeep_file.exists():
                    gitkeep_file.touch()
            
            return True
        except Exception as e:
            print(f"Failed to create directories: {e}")
            return False
    
    def create_gitignore(self) -> bool:
        """Create or update .gitignore file."""
        gitignore_content = '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Project specific
config/
*.log
*.tmp
temp/
cache/
data/raw/
data/processed/
exports/
outputs/
logs/

# API keys and sensitive data
api_keys_config.py
keys.json
.encryption_key
.security_key
*.key
*.pem

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE specific
.vscode/
.idea/
*.swp
*.swo
*~

# Backup files
*.bak
*.backup
*.old
'''
        
        try:
            gitignore_file = self.project_root / ".gitignore"
            
            # Read existing .gitignore if it exists
            existing_content = ""
            if gitignore_file.exists():
                with open(gitignore_file, 'r') as f:
                    existing_content = f.read()
            
            # Only update if content is different
            if existing_content != gitignore_content:
                with open(gitignore_file, 'w') as f:
                    f.write(gitignore_content)
            
            return True
        except Exception as e:
            print(f"Failed to create .gitignore: {e}")
            return False

class SystemInitializer:
    """Initialize the complete system configuration."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_manager = ConfigurationManager(project_root)
        self.api_key_manager = APIKeyManager(project_root)
        self.directory_setup = DirectorySetup(project_root)
    
    def initialize_system(self) -> Dict[str, bool]:
        """Initialize the complete system."""
        results = {
            'directories': False,
            'configuration': False,
            'security': False,
            'gitignore': False
        }
        
        # Create directories
        results['directories'] = self.directory_setup.create_directories()
        
        # Create configuration
        config = self.config_manager.generate_default_config()
        results['configuration'] = self.config_manager.save_config(config)
        
        # Setup security
        results['security'] = self.api_key_manager.setup_security()
        
        # Create .gitignore
        results['gitignore'] = self.directory_setup.create_gitignore()
        
        return results
    
    def get_initialization_report(self) -> str:
        """Get a report of the initialization process."""
        results = self.initialize_system()
        
        report = "System Initialization Report\n"
        report += "=" * 40 + "\n\n"
        
        for component, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            report += f"{component.capitalize()}: {status}\n"
        
        report += f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        overall_success = all(results.values())
        report += f"Overall Status: {'✓ SUCCESS' if overall_success else '✗ FAILED'}\n"
        
        return report
