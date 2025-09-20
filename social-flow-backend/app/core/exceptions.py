"""
Custom exceptions for the Social Flow backend.

This module defines custom exception classes for different types of errors
that can occur in the application.
"""

from typing import Any, Dict, Optional


class SocialFlowException(Exception):
    """Base exception for Social Flow application."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SOCIAL_FLOW_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(SocialFlowException):
    """Validation error exception."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class AuthenticationError(SocialFlowException):
    """Authentication error exception."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
        )


class AuthorizationError(SocialFlowException):
    """Authorization error exception."""
    
    def __init__(self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details,
        )


class NotFoundError(SocialFlowException):
    """Resource not found error exception."""
    
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="NOT_FOUND_ERROR",
            status_code=404,
            details=details,
        )


class ConflictError(SocialFlowException):
    """Resource conflict error exception."""
    
    def __init__(self, message: str = "Resource conflict", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFLICT_ERROR",
            status_code=409,
            details=details,
        )


class RateLimitError(SocialFlowException):
    """Rate limit exceeded error exception."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429,
            details=details,
        )


class InternalServerError(SocialFlowException):
    """Internal server error exception."""
    
    def __init__(self, message: str = "Internal server error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="INTERNAL_SERVER_ERROR",
            status_code=500,
            details=details,
        )


class ServiceUnavailableError(SocialFlowException):
    """Service unavailable error exception."""
    
    def __init__(self, message: str = "Service unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE_ERROR",
            status_code=503,
            details=details,
        )


class DatabaseError(SocialFlowException):
    """Database error exception."""
    
    def __init__(self, message: str = "Database error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details,
        )


class CacheError(SocialFlowException):
    """Cache error exception."""
    
    def __init__(self, message: str = "Cache error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details=details,
        )


class ExternalServiceError(SocialFlowException):
    """External service error exception."""
    
    def __init__(self, message: str = "External service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details=details,
        )


class FileUploadError(SocialFlowException):
    """File upload error exception."""
    
    def __init__(self, message: str = "File upload error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="FILE_UPLOAD_ERROR",
            status_code=400,
            details=details,
        )


class VideoProcessingError(SocialFlowException):
    """Video processing error exception."""
    
    def __init__(self, message: str = "Video processing error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VIDEO_PROCESSING_ERROR",
            status_code=500,
            details=details,
        )


class PaymentError(SocialFlowException):
    """Payment processing error exception."""
    
    def __init__(self, message: str = "Payment processing error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PAYMENT_ERROR",
            status_code=400,
            details=details,
        )


class NotificationError(SocialFlowException):
    """Notification error exception."""
    
    def __init__(self, message: str = "Notification error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="NOTIFICATION_ERROR",
            status_code=500,
            details=details,
        )
