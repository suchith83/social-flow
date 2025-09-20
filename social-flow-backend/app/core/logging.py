"""
Logging configuration and setup.

This module handles structured logging configuration for the Social Flow backend.
"""

import logging
import logging.config
import sys
from typing import Dict, Any

from app.core.config import settings


def setup_logging() -> None:
    """Setup structured logging configuration."""
    
    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "default",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "detailed",
                "filename": "logs/social_flow.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
            "json_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "json",
                "filename": "logs/social_flow.json",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file", "json_file"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "redis": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "boto3": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "botocore": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
    }
    
    # Apply logging configuration
    logging.config.dictConfig(log_config)
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("logs", exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance with proper configuration.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(f"app.{name}")


class StructuredLogger:
    """Structured logger for business events."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_event(
        self,
        event_type: str,
        user_id: str = None,
        **kwargs: Any,
    ) -> None:
        """
        Log business event with structured data.
        
        Args:
            event_type: Type of event
            user_id: User ID if applicable
            **kwargs: Additional event data
        """
        event_data = {
            "event_type": event_type,
            "user_id": user_id,
            **kwargs,
        }
        
        self.logger.info(f"Business event: {event_type}", extra=event_data)
    
    def log_error(
        self,
        error: Exception,
        context: str = None,
        user_id: str = None,
        **kwargs: Any,
    ) -> None:
        """
        Log error with structured data.
        
        Args:
            error: Exception instance
            context: Error context
            user_id: User ID if applicable
            **kwargs: Additional error data
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "user_id": user_id,
            **kwargs,
        }
        
        self.logger.error(f"Error in {context}: {error}", extra=error_data)
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        user_id: str = None,
        **kwargs: Any,
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            user_id: User ID if applicable
            **kwargs: Additional performance data
        """
        perf_data = {
            "operation": operation,
            "duration": duration,
            "user_id": user_id,
            **kwargs,
        }
        
        self.logger.info(f"Performance: {operation} took {duration:.3f}s", extra=perf_data)
