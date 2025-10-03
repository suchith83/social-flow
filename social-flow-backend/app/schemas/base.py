"""
Base Pydantic schemas for API request/response validation.

This module provides base schemas and common fields used across all models.
"""

from datetime import datetime
from typing import Optional, Generic, TypeVar, List
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# Generic type variable for paginated responses
T = TypeVar('T')


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {}
        }
    )


class TimestampSchema(BaseSchema):
    """Schema with timestamp fields."""
    
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SoftDeleteSchema(TimestampSchema):
    """Schema with soft delete support."""
    
    deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp (null if not deleted)")
    is_deleted: bool = Field(default=False, description="Whether the record is deleted")


class IDSchema(BaseSchema):
    """Schema with UUID identifier."""
    
    id: UUID = Field(..., description="Unique identifier")


class BaseDBSchema(IDSchema, SoftDeleteSchema):
    """Complete base schema for database models."""
    
    pass


# Pagination schemas
class PaginationParams(BaseSchema):
    """Pagination parameters for list endpoints."""
    
    skip: int = Field(default=0, ge=0, description="Number of records to skip")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of records to return")


class PaginatedResponse(BaseSchema, Generic[T]):
    """Paginated response wrapper."""
    
    total: int = Field(..., description="Total number of records")
    skip: int = Field(..., description="Number of records skipped")
    limit: int = Field(..., description="Maximum number of records returned")
    items: List[T] = Field(..., description="List of items")


# Response wrappers
class SuccessResponse(BaseSchema):
    """Generic success response."""
    
    success: bool = Field(default=True, description="Indicates successful operation")
    message: str = Field(..., description="Success message")
    data: Optional[dict] = Field(None, description="Optional response data")


class ErrorResponse(BaseSchema):
    """Generic error response."""
    
    success: bool = Field(default=False, description="Indicates failed operation")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")


# Common field validators
class EmailField(BaseSchema):
    """Email field with validation."""
    
    email: str = Field(..., description="Email address", pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


class URLField(BaseSchema):
    """URL field with validation."""
    
    url: str = Field(..., description="URL", pattern=r'^https?://.*')


class PhoneField(BaseSchema):
    """Phone field with validation."""
    
    phone: str = Field(..., description="Phone number", pattern=r'^\+?[1-9]\d{1,14}$')


# Sorting and filtering
class SortParams(BaseSchema):
    """Sorting parameters."""
    
    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_order: str = Field(default="desc", pattern=r'^(asc|desc)$', description="Sort order: asc or desc")


class FilterParams(BaseSchema):
    """Base filter parameters."""
    
    search: Optional[str] = Field(None, description="Search query")
    created_after: Optional[datetime] = Field(None, description="Filter records created after this date")
    created_before: Optional[datetime] = Field(None, description="Filter records created before this date")
