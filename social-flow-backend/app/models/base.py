"""
Base models and mixins for the Social Flow backend.

This module provides base classes, mixins, and utilities for all database models.
All models should inherit from these base classes to ensure consistency.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Column, DateTime, func, String
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase

from app.models.types import UUID


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.
    
    This class provides common functionality for all models including:
    - Automatic table name generation
    - UUID primary key
    - Timestamp fields
    - Soft delete support
    - Dictionary conversion
    """
    
    # Generate __tablename__ automatically from class name
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name (snake_case)."""
        # Convert CamelCase to snake_case
        name = cls.__name__
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_') + 's'
    
    def to_dict(self, exclude: Optional[list] = None) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.
        
        Args:
            exclude: List of fields to exclude from the dictionary
            
        Returns:
            Dictionary representation of the model
        """
        exclude = exclude or []
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                # Convert datetime to ISO format
                if isinstance(value, datetime):
                    result[column.name] = value.isoformat()
                # Convert UUID to string
                elif isinstance(value, uuid.UUID):
                    result[column.name] = str(value)
                else:
                    result[column.name] = value
                    
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[list] = None) -> None:
        """
        Update model instance from dictionary.
        
        Args:
            data: Dictionary containing field values
            exclude: List of fields to exclude from update
        """
        exclude = exclude or ['id', 'created_at']
        
        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)


class UUIDMixin:
    """Mixin for UUID primary key."""
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        index=True,
        doc="Unique identifier for the record"
    )


class TimestampMixin:
    """
    Mixin for created_at and updated_at timestamp fields.
    
    Automatically sets created_at on insert and updated_at on update.
    Uses database server time for consistency.
    """
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="Timestamp when the record was created"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Timestamp when the record was last updated"
    )


class SoftDeleteMixin:
    """
    Mixin for soft delete functionality.
    
    Records are marked as deleted instead of being physically removed.
    This allows for data recovery and audit trails.
    """
    
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp when the record was soft deleted (NULL if not deleted)"
    )
    
    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Flag indicating if the record is soft deleted"
    )
    
    def soft_delete(self) -> None:
        """Mark the record as deleted."""
        self.is_deleted = True
        self.deleted_at = func.now()
    
    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """
    Mixin for audit trail fields.
    
    Tracks who created, updated, and deleted records.
    """
    
    created_by_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        doc="ID of the user who created the record"
    )
    
    updated_by_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        doc="ID of the user who last updated the record"
    )
    
    deleted_by_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        doc="ID of the user who deleted the record"
    )


class MetadataMixin:
    """
    Mixin for JSONB metadata field.
    
    Allows storing arbitrary JSON data without schema changes.
    """
    
    @declared_attr
    def extra_metadata(cls):
        """Extra metadata column (JSONB/JSON depending on database)."""
        from app.models.types import JSONB
        return Column(
            JSONB,
            default={},
            nullable=False,
            doc="Flexible JSONB field for storing additional metadata"
        )


# Common base model that combines all mixins
class CommonBase(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """
    Common base model with UUID, timestamps, and soft delete support.
    
    Most models should inherit from this class.
    
    Example:
        class User(CommonBase):
            __tablename__ = "users"
            username = Column(String(50), unique=True, nullable=False)
    """
    
    __abstract__ = True


class AuditedBase(CommonBase, AuditMixin):
    """
    Audited base model with full audit trail.
    
    Use this for models that require tracking who performed actions.
    
    Example:
        class Payment(AuditedBase):
            __tablename__ = "payments"
            amount = Column(Float, nullable=False)
    """
    
    __abstract__ = True


class FlexibleBase(AuditedBase, MetadataMixin):
    """
    Flexible base model with all features including JSONB metadata.
    
    Use this for models that need maximum flexibility.
    
    Example:
        class ExternalIntegration(FlexibleBase):
            __tablename__ = "external_integrations"
            provider = Column(String(50), nullable=False)
    """
    
    __abstract__ = True


# Utility functions for model operations
def soft_delete_filter(query, model_class):
    """
    Apply soft delete filter to a query.
    
    Args:
        query: SQLAlchemy query object
        model_class: Model class to filter
        
    Returns:
        Filtered query excluding soft-deleted records
        
    Example:
        query = session.query(User)
        query = soft_delete_filter(query, User)
        users = await query.all()
    """
    return query.filter(model_class.is_deleted == False)  # noqa: E712


def include_deleted_filter(query, model_class, include_deleted: bool = False):
    """
    Conditionally apply soft delete filter.
    
    Args:
        query: SQLAlchemy query object
        model_class: Model class to filter
        include_deleted: If False, exclude deleted records
        
    Returns:
        Filtered query
        
    Example:
        query = session.query(User)
        query = include_deleted_filter(query, User, include_deleted=False)
    """
    if not include_deleted:
        return soft_delete_filter(query, model_class)
    return query


# Export all base classes and mixins
__all__ = [
    'Base',
    'CommonBase',
    'AuditedBase',
    'FlexibleBase',
    'UUIDMixin',
    'TimestampMixin',
    'SoftDeleteMixin',
    'AuditMixin',
    'MetadataMixin',
    'soft_delete_filter',
    'include_deleted_filter',
]
