"""
Error Handling and Recovery

Comprehensive error handling system for the economic plan review system,
providing graceful error recovery and detailed error reporting.
"""

import sys
import traceback
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    SECURITY_ERROR = "security_error"
    DATA_ERROR = "data_error"
    SYSTEM_ERROR = "system_error"
    USER_ERROR = "user_error"
    AGENT_ERROR = "agent_error"
    COMMUNICATION_ERROR = "communication_error"


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: float
    traceback_info: Optional[str]
    context: Dict[str, Any]
    suggested_action: str
    recoverable: bool


class ErrorHandler:
    """Central error handling and recovery system."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.max_history = 1000
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies[ErrorCategory.API_ERROR] = self._recover_api_error
        self.recovery_strategies[ErrorCategory.NETWORK_ERROR] = self._recover_network_error
        self.recovery_strategies[ErrorCategory.VALIDATION_ERROR] = self._recover_validation_error
        self.recovery_strategies[ErrorCategory.AGENT_ERROR] = self._recover_agent_error
        self.recovery_strategies[ErrorCategory.COMMUNICATION_ERROR] = self._recover_communication_error
    
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None,
                    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    recoverable: bool = True) -> ErrorInfo:
        """
        Handle an error with comprehensive logging and recovery.
        
        Args:
            exception: The exception that occurred
            context: Additional context information
            category: Error category
            severity: Error severity level
            recoverable: Whether the error is recoverable
            
        Returns:
            ErrorInfo object with error details
        """
        try:
            error_id = self._generate_error_id()
            context = context or {}
            
            # Extract error details
            error_message = str(exception)
            error_type = type(exception).__name__
            traceback_str = traceback.format_exc()
            
            # Determine suggested action
            suggested_action = self._get_suggested_action(category, error_type, error_message)
            
            # Create error info
            error_info = ErrorInfo(
                error_id=error_id,
                category=category,
                severity=severity,
                message=error_message,
                details={
                    'error_type': error_type,
                    'module': context.get('module', 'unknown'),
                    'function': context.get('function', 'unknown'),
                    'user_id': context.get('user_id', 'unknown')
                },
                timestamp=time.time(),
                traceback_info=traceback_str,
                context=context,
                suggested_action=suggested_action,
                recoverable=recoverable
            )
            
            # Log error
            self._log_error(error_info)
            
            # Store in history
            self._store_error(error_info)
            
            # Attempt recovery if possible
            if recoverable and category in self.recovery_strategies:
                try:
                    self.recovery_strategies[category](error_info)
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
            
            return error_info
            
        except Exception as handler_error:
            logger.critical(f"Error handler itself failed: {handler_error}")
            # Return minimal error info
            return ErrorInfo(
                error_id="handler_error",
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.CRITICAL,
                message="Error handler failure",
                details={'original_error': str(exception)},
                timestamp=time.time(),
                traceback_info=None,
                context={},
                suggested_action="Contact system administrator",
                recoverable=False
            )
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import uuid
        return f"ERR_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    def _get_suggested_action(self, category: ErrorCategory, error_type: str, message: str) -> str:
        """Get suggested action based on error details."""
        suggestions = {
            ErrorCategory.API_ERROR: {
                'default': 'Check API key validity and network connection',
                'AuthenticationError': 'Verify API key is correct and active',
                'RateLimitError': 'Wait before making more API calls',
                'QuotaExceededError': 'Check API usage limits and billing'
            },
            ErrorCategory.VALIDATION_ERROR: {
                'default': 'Check input format and requirements',
                'ValueError': 'Verify input data types and ranges',
                'TypeError': 'Check data type compatibility'
            },
            ErrorCategory.NETWORK_ERROR: {
                'default': 'Check internet connection and retry',
                'ConnectionError': 'Verify network connectivity',
                'TimeoutError': 'Retry with longer timeout'
            },
            ErrorCategory.SECURITY_ERROR: {
                'default': 'Review security settings and permissions',
                'PermissionError': 'Check file and directory permissions',
                'AccessDeniedError': 'Verify user access rights'
            },
            ErrorCategory.AGENT_ERROR: {
                'default': 'Review agent configuration and retry',
                'ConfigurationError': 'Check agent initialization parameters',
                'AnalysisError': 'Verify input data quality'
            }
        }
        
        category_suggestions = suggestions.get(category, {})
        return category_suggestions.get(error_type, category_suggestions.get('default', 'Contact support'))
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_message = f"[{error_info.error_id}] {error_info.category.value}: {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log detailed information at debug level
        logger.debug(f"Error details: {json.dumps(error_info.details, indent=2)}")
        if error_info.traceback_info:
            logger.debug(f"Traceback:\n{error_info.traceback_info}")
    
    def _store_error(self, error_info: ErrorInfo):
        """Store error in history."""
        self.error_history.append(error_info)
        
        # Update error counts
        error_key = f"{error_info.category.value}_{error_info.details.get('error_type', 'unknown')}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Maintain history size
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def _recover_api_error(self, error_info: ErrorInfo):
        """Recovery strategy for API errors."""
        logger.info(f"Attempting API error recovery for {error_info.error_id}")
        
        # Implement exponential backoff for retries
        if 'retry_count' not in error_info.context:
            error_info.context['retry_count'] = 0
        
        retry_count = error_info.context['retry_count']
        if retry_count < 3:
            wait_time = 2 ** retry_count
            logger.info(f"Scheduling retry in {wait_time} seconds")
            error_info.context['retry_count'] = retry_count + 1
            error_info.context['next_retry'] = time.time() + wait_time
    
    def _recover_network_error(self, error_info: ErrorInfo):
        """Recovery strategy for network errors."""
        logger.info(f"Attempting network error recovery for {error_info.error_id}")
        
        # Check network connectivity
        try:
            import requests
            response = requests.get('https://www.google.com', timeout=5)
            if response.status_code == 200:
                logger.info("Network connectivity confirmed")
            else:
                logger.warning("Network connectivity issues detected")
        except Exception:
            logger.warning("Cannot verify network connectivity")
    
    def _recover_validation_error(self, error_info: ErrorInfo):
        """Recovery strategy for validation errors."""
        logger.info(f"Validation error recovery for {error_info.error_id}")
        
        # Suggest data cleaning or format correction
        error_info.context['recovery_suggestion'] = "Review and correct input data format"
    
    def _recover_agent_error(self, error_info: ErrorInfo):
        """Recovery strategy for agent errors."""
        logger.info(f"Agent error recovery for {error_info.error_id}")
        
        # Suggest agent reinitialization
        error_info.context['recovery_suggestion'] = "Reinitialize agent with correct parameters"
    
    def _recover_communication_error(self, error_info: ErrorInfo):
        """Recovery strategy for communication errors."""
        logger.info(f"Communication error recovery for {error_info.error_id}")
        
        # Reset communication channels
        error_info.context['recovery_suggestion'] = "Reset communication channels and retry"
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        try:
            total_errors = len(self.error_history)
            if total_errors == 0:
                return {'total_errors': 0}
            
            # Count by category
            category_counts = {}
            severity_counts = {}
            recent_errors = 0
            current_time = time.time()
            
            for error in self.error_history:
                # Category counts
                category = error.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Severity counts
                severity = error.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Recent errors (last hour)
                if current_time - error.timestamp < 3600:
                    recent_errors += 1
            
            return {
                'total_errors': total_errors,
                'recent_errors_1h': recent_errors,
                'category_breakdown': category_counts,
                'severity_breakdown': severity_counts,
                'most_common_errors': dict(sorted(self.error_counts.items(), 
                                                key=lambda x: x[1], reverse=True)[:5])
            }
            
        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {'error': 'Unable to generate statistics'}
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorInfo]:
        """Get recent errors."""
        return self.error_history[-limit:] if self.error_history else []
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error history cleared")


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_exceptions(category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     recoverable: bool = True,
                     reraise: bool = False):
    """
    Decorator for automatic exception handling.
    
    Args:
        category: Error category
        severity: Error severity
        recoverable: Whether error is recoverable
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                error_info = global_error_handler.handle_error(
                    e, context, category, severity, recoverable
                )
                
                if reraise:
                    raise
                
                # Return error info instead of raising
                return {'error': True, 'error_info': error_info}
        
        return wrapper
    return decorator


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry and exponential backoff.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    # Final attempt failed, log and reraise
                    logger.error(f"All {self.max_retries} retry attempts failed for {func.__name__}")
                    global_error_handler.handle_error(
                        e, 
                        {'function': func.__name__, 'attempt': attempt + 1},
                        ErrorCategory.SYSTEM_ERROR,
                        ErrorSeverity.HIGH
                    )
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.1f}s")
                
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception


def safe_execute(func: Callable, default_return: Any = None, 
                error_category: ErrorCategory = ErrorCategory.SYSTEM_ERROR) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_return: Default return value on error
        error_category: Error category for logging
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        global_error_handler.handle_error(
            e,
            {'function': getattr(func, '__name__', 'unknown')},
            error_category,
            ErrorSeverity.LOW
        )
        return default_return


def create_error_report(error_info: ErrorInfo) -> str:
    """Create a formatted error report."""
    try:
        report = f"""
ERROR REPORT
============

Error ID: {error_info.error_id}
Category: {error_info.category.value}
Severity: {error_info.severity.value}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(error_info.timestamp))}
Recoverable: {error_info.recoverable}

Message:
{error_info.message}

Details:
{json.dumps(error_info.details, indent=2)}

Context:
{json.dumps(error_info.context, indent=2)}

Suggested Action:
{error_info.suggested_action}

Traceback:
{error_info.traceback_info or 'Not available'}
"""
        return report.strip()
        
    except Exception as e:
        return f"Error generating report: {e}"


# Utility functions for common error scenarios
def validate_api_response(response: Any, expected_fields: List[str] = None) -> bool:
    """Validate API response format."""
    try:
        if response is None:
            raise ValueError("API response is None")
        
        if expected_fields:
            if not isinstance(response, dict):
                raise TypeError("API response is not a dictionary")
            
            missing_fields = [field for field in expected_fields if field not in response]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
        
        return True
        
    except Exception as e:
        global_error_handler.handle_error(
            e,
            {'response_type': type(response).__name__, 'expected_fields': expected_fields},
            ErrorCategory.API_ERROR,
            ErrorSeverity.MEDIUM
        )
        return False


def handle_file_operation(operation: Callable, file_path: str) -> bool:
    """Handle file operations with proper error handling."""
    try:
        operation()
        return True
    except FileNotFoundError:
        global_error_handler.handle_error(
            FileNotFoundError(f"File not found: {file_path}"),
            {'file_path': file_path, 'operation': operation.__name__},
            ErrorCategory.DATA_ERROR,
            ErrorSeverity.MEDIUM
        )
        return False
    except PermissionError:
        global_error_handler.handle_error(
            PermissionError(f"Permission denied: {file_path}"),
            {'file_path': file_path, 'operation': operation.__name__},
            ErrorCategory.SECURITY_ERROR,
            ErrorSeverity.HIGH
        )
        return False
    except Exception as e:
        global_error_handler.handle_error(
            e,
            {'file_path': file_path, 'operation': operation.__name__},
            ErrorCategory.SYSTEM_ERROR,
            ErrorSeverity.MEDIUM
        )
        return False