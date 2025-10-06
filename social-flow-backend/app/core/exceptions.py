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


class StorageServiceError(SocialFlowException):
    """Storage service error exception."""
    
    def __init__(self, message: str = "Storage service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="STORAGE_SERVICE_ERROR",
            status_code=500,
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


class VideoServiceError(SocialFlowException):
    """Video service error exception."""
    
    def __init__(self, message: str = "Video service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VIDEO_SERVICE_ERROR",
            status_code=500,
            details=details,
        )


class LiveStreamingServiceError(SocialFlowException):
    """Live streaming service error exception."""
    
    def __init__(self, message: str = "Live streaming service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LIVE_STREAMING_SERVICE_ERROR",
            status_code=500,
            details=details,
        )


class AdsServiceError(SocialFlowException):
    """Ads service error exception."""
    
    def __init__(self, message: str = "Ads service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ADS_SERVICE_ERROR",
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


class PaymentServiceError(SocialFlowException):
    """Payment service error exception."""
    
    def __init__(self, message: str = "Payment service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PAYMENT_SERVICE_ERROR",
            status_code=500,
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


class NotificationServiceError(SocialFlowException):
    """Notification service error exception."""
    
    def __init__(self, message: str = "Notification service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="NOTIFICATION_SERVICE_ERROR",
            status_code=500,
            details=details,
        )


class AnalyticsServiceError(SocialFlowException):
    """Analytics service error exception."""
    
    def __init__(self, message: str = "Analytics service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ANALYTICS_SERVICE_ERROR",
            status_code=500,
            details=details,
        )


class MLServiceError(SocialFlowException):
    """ML service error exception."""
    
    def __init__(self, message: str = "ML service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ML_SERVICE_ERROR",
            status_code=500,
            details=details,
        )


class ModelLoadError(MLServiceError):
    """Raised when an ML/AI model fails to load or initialize.

    Referenced in AI_ML_ARCHITECTURE.md (Model Orchestration & Lazy Loading section).
    """

    def __init__(self, message: str = "Model load failure", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)
        self.error_code = "MODEL_LOAD_ERROR"


class InferenceError(MLServiceError):
    """Raised when inference/execution of a model or pipeline fails.

    Used to distinguish runtime inference issues from initialization problems.
    """

    def __init__(self, message: str = "Inference failure", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)
        self.error_code = "INFERENCE_ERROR"


class PipelineError(MLServiceError):
    """Raised when a multi-step ML pipeline orchestration fails mid-execution.

    Enables callers to implement compensating actions or fallbacks.
    """

    def __init__(self, message: str = "Pipeline execution failure", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)
        self.error_code = "PIPELINE_ERROR"


class AuthServiceError(SocialFlowException):
    """Auth service error exception."""
    
    def __init__(self, message: str = "Auth service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTH_SERVICE_ERROR",
            status_code=500,
            details=details,
        )


class PostServiceError(SocialFlowException):
    """Post service error exception."""
    
    def __init__(self, message: str = "Post service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="POST_SERVICE_ERROR",
            status_code=500,
            details=details,
        )
