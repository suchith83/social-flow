"""
User API endpoints using clean architecture.

This module provides user management endpoints using UserApplicationService.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.application.services import UserApplicationService
from app.core.dependencies import get_user_service, get_current_active_user
from app.models.user import User as UserModel

router = APIRouter()


# Request/Response Models


class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(..., min_length=8)
    display_name: str = Field(..., min_length=1, max_length=50)


class LoginRequest(BaseModel):
    """User login request."""
    username_or_email: str
    password: str


class ProfileUpdateRequest(BaseModel):
    """Profile update request."""
    display_name: str | None = Field(None, max_length=50)
    bio: str | None = Field(None, max_length=500)
    website: str | None = Field(None, max_length=200)
    location: str | None = Field(None, max_length=100)


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User response model."""
    id: UUID
    username: str
    email: str
    display_name: str
    bio: str | None
    avatar_url: str | None
    is_verified: bool
    is_active: bool
    follower_count: int
    following_count: int
    created_at: str
    
    class Config:
        from_attributes = True


# Routes


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    data: RegisterRequest,
    service: UserApplicationService = Depends(get_user_service),
):
    """
    Register a new user.
    
    - **username**: Unique username (3-30 characters)
    - **email**: Valid email address
    - **password**: Strong password (min 8 characters)
    - **display_name**: Display name (1-50 characters)
    """
    try:
        user = await service.register_user(
            username=data.username,
            email=data.email,
            password=data.password,
            display_name=data.display_name,
        )
        
        return UserResponse(
            id=user.id,
            username=user.username.value,
            email=user.email.value,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            is_verified=user.is_verified,
            is_active=user.is_active,
            follower_count=user.followers_count,
            following_count=user.following_count,
            created_at=user.created_at.isoformat(),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login")
async def login_user(
    data: LoginRequest,
    service: UserApplicationService = Depends(get_user_service),
):
    """
    Authenticate user and return JWT token.
    
    - **username_or_email**: Username or email address
    - **password**: User password
    """
    user = await service.authenticate_user(
        username_or_email=data.username_or_email,
        password=data.password,
    )
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    
    # TODO: Generate JWT token (integrate with existing auth system)
    return {
        "access_token": "token_placeholder",
        "token_type": "bearer",
        "user": UserResponse(
            id=user.id,
            username=user.username.value,
            email=user.email.value,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            is_verified=user.is_verified,
            is_active=user.is_active,
            follower_count=user.followers_count,
            following_count=user.following_count,
            created_at=user.created_at.isoformat(),
        ),
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: UserModel = Depends(get_current_active_user),
    service: UserApplicationService = Depends(get_user_service),
):
    """Get current authenticated user profile."""
    user = await service.get_user_by_id(current_user.id)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return UserResponse(
        id=user.id,
        username=user.username.value,
        email=user.email.value,
        display_name=user.display_name,
        bio=user.bio,
        avatar_url=user.avatar_url,
        is_verified=user.is_verified,
        is_active=user.is_active,
        follower_count=user.followers_count,
        following_count=user.following_count,
        created_at=user.created_at.isoformat(),
    )


@router.put("/me", response_model=UserResponse)
async def update_profile(
    data: ProfileUpdateRequest,
    current_user: UserModel = Depends(get_current_active_user),
    service: UserApplicationService = Depends(get_user_service),
):
    """
    Update current user profile.
    
    - **display_name**: New display name (optional)
    - **bio**: User bio (optional)
    - **website**: Website URL (optional)
    - **location**: Location (optional)
    """
    try:
        user = await service.update_user_profile(
            user_id=current_user.id,
            display_name=data.display_name,
            bio=data.bio,
            website=data.website,
            location=data.location,
        )
        
        return UserResponse(
            id=user.id,
            username=user.username.value,
            email=user.email.value,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            is_verified=user.is_verified,
            is_active=user.is_active,
            follower_count=user.followers_count,
            following_count=user.following_count,
            created_at=user.created_at.isoformat(),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/me/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    data: PasswordChangeRequest,
    current_user: UserModel = Depends(get_current_active_user),
    service: UserApplicationService = Depends(get_user_service),
):
    """
    Change current user password.
    
    - **current_password**: Current password for verification
    - **new_password**: New password (min 8 characters)
    """
    try:
        await service.change_password(
            user_id=current_user.id,
            current_password=data.current_password,
            new_password=data.new_password,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    service: UserApplicationService = Depends(get_user_service),
):
    """Get user by ID."""
    user = await service.get_user_by_id(user_id)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return UserResponse(
        id=user.id,
        username=user.username.value,
        email=user.email.value,
        display_name=user.display_name,
        bio=user.bio,
        avatar_url=user.avatar_url,
        is_verified=user.is_verified,
        is_active=user.is_active,
        follower_count=user.followers_count,
        following_count=user.following_count,
        created_at=user.created_at.isoformat(),
    )


@router.get("/username/{username}", response_model=UserResponse)
async def get_user_by_username(
    username: str,
    service: UserApplicationService = Depends(get_user_service),
):
    """Get user by username."""
    user = await service.get_user_by_username(username)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return UserResponse(
        id=user.id,
        username=user.username.value,
        email=user.email.value,
        display_name=user.display_name,
        bio=user.bio,
        avatar_url=user.avatar_url,
        is_verified=user.is_verified,
        is_active=user.is_active,
        follower_count=user.followers_count,
        following_count=user.following_count,
        created_at=user.created_at.isoformat(),
    )


@router.get("/", response_model=List[UserResponse])
async def search_users(
    query: str,
    skip: int = 0,
    limit: int = 20,
    service: UserApplicationService = Depends(get_user_service),
):
    """
    Search users by username.
    
    - **query**: Search query
    - **skip**: Number of results to skip (pagination)
    - **limit**: Maximum results to return (max 100)
    """
    if limit > 100:
        limit = 100
    
    users = await service.search_users(query=query, skip=skip, limit=limit)
    
    return [
        UserResponse(
            id=user.id,
            username=user.username.value,
            email=user.email.value,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            is_verified=user.is_verified,
            is_active=user.is_active,
            follower_count=user.followers_count,
            following_count=user.following_count,
            created_at=user.created_at.isoformat(),
        )
        for user in users
    ]


@router.get("/{user_id}/followers", response_model=List[UserResponse])
async def get_followers(
    user_id: UUID,
    skip: int = 0,
    limit: int = 20,
    service: UserApplicationService = Depends(get_user_service),
):
    """Get user's followers."""
    if limit > 100:
        limit = 100
    
    users = await service.get_followers(user_id=user_id, skip=skip, limit=limit)
    
    return [
        UserResponse(
            id=user.id,
            username=user.username.value,
            email=user.email.value,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            is_verified=user.is_verified,
            is_active=user.is_active,
            follower_count=user.followers_count,
            following_count=user.following_count,
            created_at=user.created_at.isoformat(),
        )
        for user in users
    ]


@router.get("/{user_id}/following", response_model=List[UserResponse])
async def get_following(
    user_id: UUID,
    skip: int = 0,
    limit: int = 20,
    service: UserApplicationService = Depends(get_user_service),
):
    """Get users that this user follows."""
    if limit > 100:
        limit = 100
    
    users = await service.get_following(user_id=user_id, skip=skip, limit=limit)
    
    return [
        UserResponse(
            id=user.id,
            username=user.username.value,
            email=user.email.value,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            is_verified=user.is_verified,
            is_active=user.is_active,
            follower_count=user.followers_count,
            following_count=user.following_count,
            created_at=user.created_at.isoformat(),
        )
        for user in users
    ]

