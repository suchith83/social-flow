"""
Structured Logging Configuration with structlog.

Provides JSON-formatted logs with request ID tracking, context propagation,
and log correlation across services.
"""

import logging
import sys
from typing import Any
from contextvars import ContextVar

import structlog
from structlog.types import EventDict, Processor

from app.core.config import settings


# Context variable for request ID tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def add_request_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add request ID to log context."""
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_app_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application info to log context."""
    event_dict["app"] = "social-flow-backend"
    event_dict["environment"] = settings.ENVIRONMENT
    return event_dict


def drop_color_message_key(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Remove color_message key from event dict (used by rich handler)."""
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    
    Sets up:
    - JSON-formatted logs for production
    - Human-readable console logs for development
    - Request ID tracking
    - Context propagation
    - Log correlation
    """
    # Determine log processors based on environment
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_request_id,
        add_app_info,
    ]
    
    if settings.ENVIRONMENT == "production":
        # Production: JSON formatting
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Human-readable formatting
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            drop_color_message_key,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG,
    )
    
    # Set log levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.getLogger("stripe").setLevel(logging.INFO)
    logging.getLogger("celery").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
        
    Example:
        logger = get_logger(__name__)
        logger.info("user_registered", user_id=user.id, email=user.email)
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, **kwargs: Any):
        """Initialize log context with key-value pairs."""
        self.context = kwargs
        self.token = None
    
    def __enter__(self) -> "LogContext":
        """Enter context and bind context variables."""
        structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and unbind context variables."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def set_request_id(request_id: str) -> None:
    """
    Set request ID for the current context.
    
    Args:
        request_id: Unique request identifier
    """
    request_id_var.set(request_id)


def get_request_id() -> str:
    """
    Get request ID from current context.
    
    Returns:
        Current request ID or empty string
    """
    return request_id_var.get()


# Example usage patterns
"""
# Basic logging
logger = get_logger(__name__)
logger.info("user_login", user_id=user.id, ip_address=request.client.host)

# With context manager
with LogContext(user_id=user.id, operation="video_upload"):
    logger.info("upload_started", filename=file.filename)
    # ... do work ...
    logger.info("upload_completed", size=file.size)

# Request tracking
set_request_id(str(uuid.uuid4()))
logger.info("api_request", method="POST", path="/api/v1/videos")
"""
