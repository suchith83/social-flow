"""
FastAPI dependencies for authentication, database sessions, and permission checking.

This module provides reusable dependencies for:
- Database session management
- Current user authentication
- Permission checking
- Rate limiting
"""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.infrastructure.crud import user as crud_user
from app.models.user import User, UserRole, UserStatus

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False,
)

# Optional OAuth2 scheme (doesn't raise error if token is missing)
optional_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False,
)


# Export get_db from database module for use in endpoints
# This is already defined in app.core.database, so we just re-export it
__all__ = ["get_db", "get_current_user", "get_current_user_optional", "get_current_active_user",
           "get_current_verified_user", "require_role", "require_admin", "require_creator",
           "require_moderator", "require_ownership", "RateLimitChecker", "oauth2_scheme"]


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme),
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        db: Database session
        token: JWT access token
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
        
    Example:
        @app.get("/me")
        async def get_me(user: User = Depends(get_current_user)):
            return user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = await crud_user.get(db, UUID(user_id))
    if user is None:
        raise credentials_exception
    
    # Check if user is active
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User account is {user.status.value}",
        )
    
    return user


async def get_current_user_optional(
    db: AsyncSession = Depends(get_db),
    token: Optional[str] = Depends(optional_oauth2_scheme),
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    
    Useful for endpoints that work differently for authenticated vs anonymous users.
    
    Args:
        db: Database session
        token: JWT access token (optional)
        
    Returns:
        User or None: Current user if authenticated, None otherwise
        
    Example:
        @app.get("/posts")
        async def list_posts(user: Optional[User] = Depends(get_current_user_optional)):
            # Show different content based on whether user is logged in
            ...
    """
    if not token:
        return None
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
            
    except JWTError:
        return None
    
    user = await crud_user.get(db, UUID(user_id))
    if user is None or user.status != UserStatus.ACTIVE:
        return None
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Ensure current user is active.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Active user
        
    Raises:
        HTTPException: If user is not active
    """
    if current_user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Ensure current user has verified their email.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Verified user
        
    Raises:
        HTTPException: If user email is not verified
    """
    if not current_user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please verify your email to access this resource.",
        )
    return current_user


def require_role(*allowed_roles: UserRole):
    """
    Dependency factory for role-based access control.
    
    Args:
        *allowed_roles: Roles that are allowed to access the endpoint
        
    Returns:
        Dependency function that checks user role
        
    Example:
        @app.get("/admin/users")
        async def list_all_users(
            user: User = Depends(require_role(UserRole.ADMIN))
        ):
            ...
    """
    async def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required role: {[r.value for r in allowed_roles]}",
            )
        return current_user
    
    return role_checker


async def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Require user to be an admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Admin user
        
    Raises:
        HTTPException: If user is not an admin
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def require_creator(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Require user to be a creator or admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Creator or admin user
        
    Raises:
        HTTPException: If user is not a creator or admin
    """
    if current_user.role not in [UserRole.CREATOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Creator access required",
        )
    return current_user


async def require_moderator(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Require user to be a moderator or admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Moderator or admin user
        
    Raises:
        HTTPException: If user is not a moderator or admin
    """
    if current_user.role not in [UserRole.MODERATOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Moderator access required",
        )
    return current_user


def require_ownership(resource_user_id_field: str = "user_id"):
    """
    Dependency factory to ensure user owns the resource or is admin.
    
    Args:
        resource_user_id_field: Name of the field containing the owner's user_id
        
    Returns:
        Dependency function that checks ownership
        
    Example:
        @app.delete("/posts/{post_id}")
        async def delete_post(
            post: Post,
            user: User = Depends(require_ownership("user_id"))
        ):
            ...
    """
    async def ownership_checker(
        resource: dict,
        current_user: User = Depends(get_current_user),
    ) -> User:
        resource_user_id = resource.get(resource_user_id_field)
        
        # Admin can access everything
        if current_user.role == UserRole.ADMIN:
            return current_user
        
        # Check ownership
        if resource_user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this resource",
            )
        
        return current_user
    
    return ownership_checker


class RateLimitChecker:
    """
    Rate limiting dependency.
    
    Example:
        rate_limit = RateLimitChecker(max_requests=100, window_seconds=60)
        
        @app.post("/posts")
        async def create_post(
            _: None = Depends(rate_limit),
            user: User = Depends(get_current_user)
        ):
            ...
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # TODO: Implement Redis-based rate limiting
    
    async def __call__(
        self,
        current_user: User = Depends(get_current_user),
    ):
        """
        Check rate limit for current user.
        
        Raises:
            HTTPException: If rate limit exceeded
        """
        # TODO: Implement actual rate limiting logic with Redis
        # For now, just return to allow all requests
        return None


# Commonly used rate limiters
rate_limit_strict = RateLimitChecker(max_requests=10, window_seconds=60)
rate_limit_moderate = RateLimitChecker(max_requests=30, window_seconds=60)
rate_limit_relaxed = RateLimitChecker(max_requests=100, window_seconds=60)
