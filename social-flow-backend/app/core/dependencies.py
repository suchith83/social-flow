"""
Authentication and authorization dependencies.

FastAPI dependencies for authentication, authorization, and permission checks.
"""

from typing import Optional, List, AsyncGenerator
import uuid

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
from app.core.redis import get_redis
from app.core.security import verify_token
from app.models.user import User
from app.auth.services.enhanced_auth_service import EnhancedAuthService
from app.application.services import (
    UserApplicationService,
    VideoApplicationService,
    PostApplicationService,
)


# HTTP Bearer token security scheme
security = HTTPBearer()


async def get_auth_service(
    db: AsyncSession = Depends(get_db),
    redis = Depends(get_redis),
) -> EnhancedAuthService:
    """Get enhanced authentication service."""
    return EnhancedAuthService(db=db, redis=redis)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
    redis = Depends(get_redis),
) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    # Testing shortcut: accept tokens of form 'test_token_<user_id>'
    if settings.TESTING and token.startswith("test_token_"):
        user_id = token.split("test_token_", 1)[1]
        try:
            user_uuid = uuid.UUID(user_id)
        except Exception:
            # Not a valid UUID
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid test token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = await db.get(User, user_uuid)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Test user not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    
    # Verify token
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if token is blacklisted
    jti = payload.get("jti")
    if jti:
        auth_service = EnhancedAuthService(db=db, redis=redis)
        is_blacklisted = await auth_service.is_token_blacklisted(jti)
        if is_blacklisted:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Get user from database
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active and not banned/suspended
    if not user.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is not active",
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user (not banned or suspended)."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    if current_user.is_banned:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is banned",
        )
    
    if current_user.is_suspended_now:
        suspension_msg = "User account is suspended"
        if current_user.suspension_ends_at:
            suspension_msg += f" until {current_user.suspension_ends_at.isoformat()}"
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=suspension_msg,
        )
    
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Get current verified user."""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required",
        )
    
    return current_user


def require_permission(permission: str):
    """
    Dependency to require specific permission.
    
    Usage:
        @router.post("/admin/action", dependencies=[Depends(require_permission("admin:all"))])
        async def admin_action():
            ...
    """
    async def permission_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission} required",
            )
        return current_user
    
    return permission_checker


def require_role(role: str):
    """
    Dependency to require specific role.
    
    Usage:
        @router.post("/admin/action", dependencies=[Depends(require_role("admin"))])
        async def admin_action():
            ...
    """
    async def role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if not current_user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: {role} role required",
            )
        return current_user
    
    return role_checker


def require_any_role(roles: List[str]):
    """
    Dependency to require any of the specified roles.
    
    Usage:
        @router.post("/action", dependencies=[Depends(require_any_role(["admin", "moderator"]))])
        async def action():
            ...
    """
    async def any_role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        user_roles = current_user.get_role_names()
        if not any(role in user_roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: one of {roles} roles required",
            )
        return current_user
    
    return any_role_checker


def require_any_permission(permissions: List[str]):
    """
    Dependency to require any of the specified permissions.
    
    Usage:
        @router.post("/action", dependencies=[Depends(require_any_permission(["video:create", "video:update"]))])
        async def action():
            ...
    """
    async def any_permission_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        user_permissions = current_user.get_permissions()
        if not any(perm in user_permissions for perm in permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: one of {permissions} permissions required",
            )
        return current_user
    
    return any_permission_checker


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis = Depends(get_redis),
) -> Optional[User]:
    """Get current user if authenticated, otherwise None (for optional auth)."""
    # Try to get token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    
    # Testing shortcut: accept tokens of form 'test_token_<user_id>'
    if settings.TESTING and token.startswith("test_token_"):
        user_id = token.split("test_token_", 1)[1]
        try:
            user_uuid = uuid.UUID(user_id)
        except Exception:
            return None
        user = await db.get(User, user_uuid)
        return user
    
    # Verify token
    payload = verify_token(token)
    if not payload or payload.get("type") != "access":
        return None
    
    # Check if token is blacklisted
    jti = payload.get("jti")
    if jti:
        auth_service = EnhancedAuthService(db=db, redis=redis)
        is_blacklisted = await auth_service.is_token_blacklisted(jti)
        if is_blacklisted:
            return None
    
    # Get user
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    user = await db.get(User, user_id)
    if not user or not user.is_authenticated:
        return None
    
    return user


# Admin-only dependencies
require_admin = require_role("admin")
require_moderator = require_any_role(["admin", "moderator"])
require_creator = require_any_role(["admin", "moderator", "creator"])


# Application Service Dependencies


async def get_user_service(
    session: AsyncSession = Depends(get_db),
) -> AsyncGenerator[UserApplicationService, None]:
    """
    Get UserApplicationService instance.
    
    Args:
        session: Database session
        
    Yields:
        UserApplicationService instance
    """
    service = UserApplicationService(session)
    try:
        yield service
    finally:
        # Cleanup if needed
        pass


async def get_video_service(
    session: AsyncSession = Depends(get_db),
) -> AsyncGenerator[VideoApplicationService, None]:
    """
    Get VideoApplicationService instance.
    
    Args:
        session: Database session
        
    Yields:
        VideoApplicationService instance
    """
    service = VideoApplicationService(session)
    try:
        yield service
    finally:
        # Cleanup if needed
        pass


async def get_post_service(
    session: AsyncSession = Depends(get_db),
) -> AsyncGenerator[PostApplicationService, None]:
    """
    Get PostApplicationService instance.
    
    Args:
        session: Database session
        
    Yields:
        PostApplicationService instance
    """
    service = PostApplicationService(session)
    try:
        yield service
    finally:
        # Cleanup if needed
        pass

