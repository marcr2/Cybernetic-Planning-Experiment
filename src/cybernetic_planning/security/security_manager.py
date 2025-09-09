"""
Security Manager

Comprehensive security features for the economic plan review system,
including API key protection, data validation, and access control.
"""

import os
import hashlib
import hmac
import time
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    max_plan_size: int = 1_000_000  # 1MB max plan size
    max_api_calls_per_hour: int = 100
    api_key_min_length: int = 32
    session_timeout: int = 3600  # 1 hour
    encryption_key_file: str = ".security_key"
    rate_limit_window: int = 3600  # 1 hour
    max_failed_attempts: int = 5
    lockout_duration: int = 1800  # 30 minutes


class InputValidator:
    """Validates and sanitizes user inputs."""

    def __init__(self):
        self.max_text_length = 1_000_000
        self.dangerous_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"<iframe[^>]*>.*?</iframe>",
        ]

    def validate_economic_plan(self, plan_text: str) -> Tuple[bool, str]:
        """
        Validate economic plan input.

        Args:
            plan_text: The economic plan text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check length
            if len(plan_text) > self.max_text_length:
                return False, f"Plan text too long. Maximum {self.max_text_length:,} characters allowed."

            if len(plan_text.strip()) < 100:
                return False, "Plan text too short. Minimum 100 characters required."

            # Check for dangerous patterns
            for pattern in self.dangerous_patterns:
                if re.search(pattern, plan_text, re.IGNORECASE):
                    return False, "Plan text contains potentially dangerous content."

            # Check encoding
            try:
                plan_text.encode("utf-8")
            except UnicodeEncodeError:
                return False, "Plan text contains invalid characters."

            return True, "Valid"

        except Exception as e:
            logger.error(f"Error validating plan text: {str(e)}")
            return False, "Validation error occurred."

    def validate_api_key(self, api_key: str) -> Tuple[bool, str]:
        """
        Validate API key format.

        Args:
            api_key: The API key to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not api_key or not isinstance(api_key, str):
                return False, "API key is required."

            api_key = api_key.strip()

            if len(api_key) < 32:
                return False, "API key too short. Minimum 32 characters required."

            if len(api_key) > 200:
                return False, "API key too long. Maximum 200 characters allowed."

            # Check for valid characters (alphanumeric, hyphens, underscores)
            if not re.match(r"^[A-Za-z0-9\-_]+$", api_key):
                return False, "API key contains invalid characters."

            return True, "Valid"

        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            return False, "API key validation error."

    def sanitize_text(self, text: str) -> str:
        """Sanitize text input by removing dangerous patterns."""
        try:
            sanitized = text

            # Remove dangerous patterns
            for pattern in self.dangerous_patterns:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

            # Remove null bytes
            sanitized = sanitized.replace("\x00", "")

            # Normalize whitespace
            sanitized = re.sub(r"\s+", " ", sanitized).strip()

            return sanitized

        except Exception as e:
            logger.error(f"Error sanitizing text: {str(e)}")
            return ""


class RateLimiter:
    """Rate limiting for API calls and user actions."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.call_history: Dict[str, List[float]] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.locked_users: Dict[str, float] = {}

    def check_rate_limit(self, user_id: str, action: str = "api_call") -> Tuple[bool, str]:
        """
        Check if user is within rate limits.

        Args:
            user_id: Identifier for the user
            action: Type of action being performed

        Returns:
            Tuple of (is_allowed, message)
        """
        try:
            current_time = time.time()

            # Check if user is locked out
            if user_id in self.locked_users:
                lockout_end = self.locked_users[user_id]
                if current_time < lockout_end:
                    remaining = int(lockout_end - current_time)
                    return False, f"Account locked. Try again in {remaining} seconds."
                else:
                    # Lockout expired, remove from locked users
                    del self.locked_users[user_id]

            # Initialize user history if needed
            if user_id not in self.call_history:
                self.call_history[user_id] = []

            # Clean old entries
            window_start = current_time - self.config.rate_limit_window
            self.call_history[user_id] = [
                call_time for call_time in self.call_history[user_id] if call_time > window_start
            ]

            # Check rate limit
            if len(self.call_history[user_id]) >= self.config.max_api_calls_per_hour:
                return False, "Rate limit exceeded. Please wait before making more requests."

            # Record this call
            self.call_history[user_id].append(current_time)

            return True, "OK"

        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            return False, "Rate limit check failed."

    def record_failed_attempt(self, user_id: str):
        """Record a failed authentication/validation attempt."""
        try:
            current_time = time.time()

            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = []

            # Clean old attempts
            window_start = current_time - self.config.rate_limit_window
            self.failed_attempts[user_id] = [
                attempt_time for attempt_time in self.failed_attempts[user_id] if attempt_time > window_start
            ]

            # Record this attempt
            self.failed_attempts[user_id].append(current_time)

            # Check if should lock user
            if len(self.failed_attempts[user_id]) >= self.config.max_failed_attempts:
                self.locked_users[user_id] = current_time + self.config.lockout_duration
                logger.warning(f"User {user_id} locked due to too many failed attempts")

        except Exception as e:
            logger.error(f"Error recording failed attempt: {str(e)}")

    def clear_failed_attempts(self, user_id: str):
        """Clear failed attempts for a user (after successful authentication)."""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]


class SecureAPIKeyManager:
    """Secure API key management with encryption."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.key_file = config.encryption_key_file
        self._fernet = None
        self._initialize_encryption()

    def _initialize_encryption(self):
        """Initialize encryption system."""
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, "rb") as f:
                    key = f.read()
            else:
                # Generate new key
                password = os.urandom(32)
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))

                # Save key securely
                with open(self.key_file, "wb") as f:
                    f.write(key)

                # Set restrictive permissions
                os.chmod(self.key_file, 0o600)

            self._fernet = Fernet(key)
            logger.info("Encryption system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize encryption: {str(e)}")
            raise

    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for secure storage."""
        try:
            encrypted = self._fernet.encrypt(api_key.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt API key: {str(e)}")
            raise

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key from secure storage."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_key.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {str(e)}")
            raise

    def validate_and_store_key(self, api_key: str, key_name: str) -> Tuple[bool, str]:
        """Validate and securely store API key."""
        try:
            validator = InputValidator()
            is_valid, message = validator.validate_api_key(api_key)

            if not is_valid:
                return False, message

            # Encrypt and store
            encrypted_key = self.encrypt_api_key(api_key)

            # Store in secure config file
            config_file = f".{key_name}_config.json"
            config_data = {
                "encrypted_key": encrypted_key,
                "created_time": time.time(),
                "key_hash": hashlib.sha256(api_key.encode()).hexdigest()[:16],  # For verification
            }

            with open(config_file, "w") as f:
                json.dump(config_data, f)

            # Set restrictive permissions
            os.chmod(config_file, 0o600)

            return True, "API key stored securely"

        except Exception as e:
            logger.error(f"Failed to store API key: {str(e)}")
            return False, "Failed to store API key securely"

    def load_api_key(self, key_name: str) -> Optional[str]:
        """Load and decrypt API key."""
        try:
            config_file = f".{key_name}_config.json"

            if not os.path.exists(config_file):
                return None

            with open(config_file, "r") as f:
                config_data = json.load(f)

            encrypted_key = config_data["encrypted_key"]
            api_key = self.decrypt_api_key(encrypted_key)

            # Verify integrity
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            if key_hash != config_data.get("key_hash"):
                logger.warning("API key integrity check failed")
                return None

            return api_key

        except Exception as e:
            logger.error(f"Failed to load API key: {str(e)}")
            return None


class SessionManager:
    """Secure session management."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str) -> str:
        """Create a new secure session."""
        try:
            session_id = self._generate_session_id()
            current_time = time.time()

            self.active_sessions[session_id] = {
                "user_id": user_id,
                "created_time": current_time,
                "last_activity": current_time,
                "is_active": True,
            }

            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise

    def validate_session(self, session_id: str) -> Tuple[bool, str]:
        """Validate session and check timeout."""
        try:
            if session_id not in self.active_sessions:
                return False, "Invalid session"

            session = self.active_sessions[session_id]
            current_time = time.time()

            # Check if session is active
            if not session["is_active"]:
                return False, "Session is inactive"

            # Check timeout
            if current_time - session["last_activity"] > self.config.session_timeout:
                session["is_active"] = False
                return False, "Session expired"

            # Update last activity
            session["last_activity"] = current_time

            return True, "Valid session"

        except Exception as e:
            logger.error(f"Error validating session: {str(e)}")
            return False, "Session validation error"

    def end_session(self, session_id: str):
        """End a session."""
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["is_active"] = False
                logger.info(f"Ended session {session_id}")
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            current_time = time.time()
            expired_sessions = []

            for session_id, session in self.active_sessions.items():
                if current_time - session["last_activity"] > self.config.session_timeout:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                self.active_sessions[session_id]["is_active"] = False

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        except Exception as e:
            logger.error(f"Error cleaning up sessions: {str(e)}")

    def _generate_session_id(self) -> str:
        """Generate a secure session ID."""
        random_bytes = os.urandom(32)
        return hashlib.sha256(random_bytes).hexdigest()


class SecurityAuditor:
    """Security auditing and monitoring."""

    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.setup_audit_logging()

    def setup_audit_logging(self):
        """Set up security audit logging."""
        try:
            audit_logger = logging.getLogger("security_audit")
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
            audit_logger.setLevel(logging.INFO)
            self.audit_logger = audit_logger
        except Exception as e:
            logger.error(f"Failed to setup audit logging: {str(e)}")
            self.audit_logger = logger

    def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log a security event."""
        try:
            event_data = {"event_type": event_type, "user_id": user_id, "timestamp": time.time(), "details": details}

            self.audit_logger.info(f"SECURITY_EVENT: {json.dumps(event_data)}")

        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")

    def log_api_call(self, user_id: str, endpoint: str, success: bool, response_time: float):
        """Log API call for monitoring."""
        try:
            self.log_security_event(
                "api_call", user_id, {"endpoint": endpoint, "success": success, "response_time": response_time}
            )
        except Exception as e:
            logger.error(f"Failed to log API call: {str(e)}")

    def log_authentication_attempt(self, user_id: str, success: bool, method: str):
        """Log authentication attempt."""
        try:
            self.log_security_event("authentication", user_id, {"success": success, "method": method})
        except Exception as e:
            logger.error(f"Failed to log authentication attempt: {str(e)}")

    def log_data_access(self, user_id: str, data_type: str, action: str):
        """Log data access event."""
        try:
            self.log_security_event("data_access", user_id, {"data_type": data_type, "action": action})
        except Exception as e:
            logger.error(f"Failed to log data access: {str(e)}")


class SecurityManager:
    """Main security manager coordinating all security components."""

    def __init__(self, config: SecurityConfig = None):
        """Initialize security manager."""
        self.config = config or SecurityConfig()
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter(self.config)
        self.api_key_manager = SecureAPIKeyManager(self.config)
        self.session_manager = SessionManager(self.config)
        self.auditor = SecurityAuditor()

        logger.info("Security manager initialized")

    def validate_review_request(self, user_id: str, plan_text: str, api_key: str) -> Tuple[bool, str]:
        """Validate a complete review request."""
        try:
            # Check rate limits
            rate_ok, rate_msg = self.rate_limiter.check_rate_limit(user_id)
            if not rate_ok:
                self.auditor.log_security_event("rate_limit_exceeded", user_id, {"message": rate_msg})
                return False, rate_msg

            # Validate API key
            key_ok, key_msg = self.input_validator.validate_api_key(api_key)
            if not key_ok:
                self.rate_limiter.record_failed_attempt(user_id)
                self.auditor.log_authentication_attempt(user_id, False, "api_key")
                return False, key_msg

            # Validate plan text
            plan_ok, plan_msg = self.input_validator.validate_economic_plan(plan_text)
            if not plan_ok:
                self.auditor.log_security_event("invalid_input", user_id, {"error": plan_msg})
                return False, plan_msg

            # Log successful validation
            self.auditor.log_authentication_attempt(user_id, True, "api_key")
            self.auditor.log_data_access(user_id, "economic_plan", "review_request")

            return True, "Request validated successfully"

        except Exception as e:
            logger.error(f"Error validating review request: {str(e)}")
            self.auditor.log_security_event("validation_error", user_id, {"error": str(e)})
            return False, "Security validation failed"

    def create_secure_session(self, user_id: str) -> Optional[str]:
        """Create a secure session for a user."""
        try:
            session_id = self.session_manager.create_session(user_id)
            self.auditor.log_security_event("session_created", user_id, {"session_id": session_id})
            return session_id
        except Exception as e:
            logger.error(f"Failed to create secure session: {str(e)}")
            return None

    def cleanup_security_resources(self):
        """Clean up security resources and expired sessions."""
        try:
            self.session_manager.cleanup_expired_sessions()
            logger.info("Security resource cleanup completed")
        except Exception as e:
            logger.error(f"Error during security cleanup: {str(e)}")

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        try:
            return {
                "active_sessions": len([s for s in self.session_manager.active_sessions.values() if s["is_active"]]),
                "total_sessions": len(self.session_manager.active_sessions),
                "rate_limited_users": len(self.rate_limiter.locked_users),
                "encryption_available": self.api_key_manager._fernet is not None,
                "audit_logging_enabled": True,
            }
        except Exception as e:
            logger.error(f"Error getting security status: {str(e)}")
            return {"error": "Unable to get security status"}


# Utility functions
def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for logging/comparison."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def generate_secure_token() -> str:
    """Generate a secure random token."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")


def verify_data_integrity(data: str, expected_hash: str) -> bool:
    """Verify data integrity using hash comparison."""
    actual_hash = hashlib.sha256(data.encode()).hexdigest()
    return hmac.compare_digest(actual_hash, expected_hash)
