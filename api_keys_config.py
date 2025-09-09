"""
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
