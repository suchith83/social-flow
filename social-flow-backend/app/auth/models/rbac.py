"""
Role-Based Access Control (RBAC) models.

This module defines models for roles, permissions, and their relationships.
"""

import uuid
from datetime import datetime
from typing import List

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Table, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


# Association table for many-to-many relationship between roles and permissions
role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("permission_id", UUID(as_uuid=True), ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
)


# Association table for many-to-many relationship between users and roles
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
)


class Permission(Base):
    """Permission model for granular access control."""
    
    __tablename__ = "permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Permission details
    name = Column(String(100), unique=True, nullable=False, index=True)  # e.g., "video:create", "user:ban"
    description = Column(Text, nullable=True)
    resource = Column(String(50), nullable=False, index=True)  # e.g., "video", "user", "post"
    action = Column(String(50), nullable=False, index=True)  # e.g., "create", "read", "update", "delete"
    
    # Permission status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
    
    def __repr__(self) -> str:
        return f"<Permission(id={self.id}, name={self.name})>"


class Role(Base):
    """Role model for user roles."""
    
    __tablename__ = "roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Role details
    name = Column(String(50), unique=True, nullable=False, index=True)  # e.g., "admin", "moderator", "creator", "viewer"
    display_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Role hierarchy (optional, for role inheritance)
    priority = Column(String(10), default=0, nullable=False)  # Higher priority = more privileges
    
    # Role status
    is_active = Column(Boolean, default=True, nullable=False)
    is_system = Column(Boolean, default=False, nullable=False)  # System roles cannot be deleted
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    users = relationship("User", secondary=user_roles, back_populates="roles")
    
    def __repr__(self) -> str:
        return f"<Role(id={self.id}, name={self.name})>"
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if role has a specific permission."""
        return any(p.name == permission_name and p.is_active for p in self.permissions)
    
    def get_permission_names(self) -> List[str]:
        """Get list of permission names."""
        return [p.name for p in self.permissions if p.is_active]
