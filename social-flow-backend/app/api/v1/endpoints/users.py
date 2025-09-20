"""
User endpoints.

This module contains all user-related API endpoints.
"""

from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import NotFoundError
from app.models.user import User
from app.schemas.auth import UserResponse, UserUpdate
from app.services.auth import AuthService
from app.api.v1.endpoints.auth import get_current_active_user

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get current user profile."""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update current user profile."""
    auth_service = AuthService(db)
    updated_user = await auth_service.update_user(str(current_user.id), user_data)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return updated_user


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get user by ID."""
    auth_service = AuthService(db)
    user = await auth_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return user


@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    search: str = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get list of users with optional search."""
    try:
        auth_service = AuthService(db)
        users = await auth_service.get_users(skip=skip, limit=limit, search=search)
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get users")


@router.get("/{user_id}/followers", response_model=List[UserResponse])
async def get_user_followers(
    user_id: str,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get user's followers."""
    try:
        auth_service = AuthService(db)
        followers = await auth_service.get_user_followers(user_id, skip=skip, limit=limit)
        return followers
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get followers")


@router.get("/{user_id}/following", response_model=List[UserResponse])
async def get_user_following(
    user_id: str,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get users that the user is following."""
    try:
        auth_service = AuthService(db)
        following = await auth_service.get_user_following(user_id, skip=skip, limit=limit)
        return following
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get following")


@router.post("/{user_id}/follow")
async def follow_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Follow a user."""
    try:
        auth_service = AuthService(db)
        success = await auth_service.follow_user(str(current_user.id), user_id)
        
        if success:
            return {"message": "Successfully followed user"}
        else:
            raise HTTPException(status_code=400, detail="Failed to follow user")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to follow user")


@router.delete("/{user_id}/follow")
async def unfollow_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Unfollow a user."""
    try:
        auth_service = AuthService(db)
        success = await auth_service.unfollow_user(str(current_user.id), user_id)
        
        if success:
            return {"message": "Successfully unfollowed user"}
        else:
            raise HTTPException(status_code=400, detail="Failed to unfollow user")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to unfollow user")


@router.get("/{user_id}/is-following")
async def is_following_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Check if current user is following the specified user."""
    try:
        auth_service = AuthService(db)
        is_following = await auth_service.is_following(str(current_user.id), user_id)
        
        return {"is_following": is_following}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to check follow status")


@router.get("/{user_id}/profile", response_model=dict)
async def get_user_profile(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get comprehensive user profile."""
    try:
        auth_service = AuthService(db)
        profile = await auth_service.get_user_profile(user_id)
        
        if profile:
            return profile
        else:
            raise HTTPException(status_code=404, detail="User profile not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get user profile")


@router.put("/me/preferences")
async def update_user_preferences(
    preferences: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update user preferences."""
    try:
        auth_service = AuthService(db)
        success = await auth_service.update_user_preferences(str(current_user.id), preferences)
        
        if success:
            return {"message": "Preferences updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to update preferences")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update preferences")
