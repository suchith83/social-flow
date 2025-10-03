"""
User management endpoints for profile management, user search, and social features.
"""

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_current_user,
    get_current_user_optional,
    get_db,
    require_admin,
)
from app.core.security import get_password_hash, verify_password
from app.infrastructure.crud import user as crud_user, follow as crud_follow
from app.models.user import User, UserRole, UserStatus
from app.schemas.base import PaginatedResponse, SuccessResponse
from app.schemas.user import (
    UserResponse,
    UserDetailResponse,
    UserPublicResponse,
    UserUpdate,
    UserUpdatePassword,
    UserAdminUpdate,
)
from app.schemas.social import (
    FollowResponse,
)

router = APIRouter()


@router.get("/me", response_model=UserDetailResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get current user's detailed profile.
    
    Returns complete profile information for the authenticated user.
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    user_in: UserUpdate,
) -> Any:
    """
    Update current user's profile.
    
    - **full_name**: User's full name
    - **bio**: User biography/description
    - **website_url**: Personal website URL
    - **location**: User's location
    - **avatar_url**: Profile picture URL
    - **cover_url**: Cover/banner image URL
    """
    # UserUpdate schema doesn't include username or email changes
    # Those require separate endpoints for security
    
    user = await crud_user.update(db, db_obj=current_user, obj_in=user_in)
    return user


@router.put("/me/password", response_model=SuccessResponse)
async def change_password(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    password_data: UserUpdatePassword,
) -> Any:
    """
    Change current user's password.
    
    - **current_password**: Current password for verification
    - **new_password**: New password (min 8 chars, uppercase, lowercase, digit)
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password",
        )
    
    # Update password
    await crud_user.update(
        db,
        db_obj=current_user,
        obj_in={"password_hash": get_password_hash(password_data.new_password)},
    )
    
    return {"success": True, "message": "Password updated successfully"}


@router.get("", response_model=PaginatedResponse[UserPublicResponse])
async def list_users(
    *,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    role: Optional[UserRole] = None,
    status: Optional[UserStatus] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    List users with pagination and filtering.
    
    - **skip**: Number of users to skip (pagination)
    - **limit**: Maximum number of users to return (1-100)
    - **role**: Filter by user role
    - **status**: Filter by user status
    
    Returns paginated list of user profiles.
    """
    filters = {}
    if role:
        filters["role"] = role
    if status:
        filters["status"] = status
    
    users = await crud_user.get_multi(
        db,
        skip=skip,
        limit=limit,
        filters=filters,
        order_by="created_at",
        order_desc=True,
    )
    
    total = await crud_user.count(db, filters=filters)
    
    return {
        "items": users,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/search", response_model=PaginatedResponse[UserPublicResponse])
async def search_users(
    *,
    db: AsyncSession = Depends(get_db),
    q: str = Query(..., min_length=1, max_length=100),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> Any:
    """
    Search users by username, full name, or email.
    
    - **q**: Search query (min 1 char)
    - **skip**: Number of results to skip
    - **limit**: Maximum number of results (1-100)
    """
    from sqlalchemy import select, or_, func
    from app.models.user import User as UserModel
    
    # Build search query
    search_pattern = f"%{q}%"
    query = select(UserModel).where(
        or_(
            UserModel.username.ilike(search_pattern),
            UserModel.display_name.ilike(search_pattern),
            UserModel.email.ilike(search_pattern),
        )
    )
    
    # Get results
    query = query.order_by(UserModel.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    users = list(result.scalars().all())
    
    # Get total count
    count_query = select(func.count()).select_from(UserModel).where(
        or_(
            UserModel.username.ilike(search_pattern),
            UserModel.display_name.ilike(search_pattern),
            UserModel.email.ilike(search_pattern),
        )
    )
    count_result = await db.execute(count_query)
    total = count_result.scalar_one()
    
    return {
        "items": users,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/{user_id}", response_model=UserPublicResponse)
async def get_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    Get user profile by ID.
    
    Returns public profile information for any user.
    If authenticated user requests their own profile, returns detailed info.
    """
    user = await crud_user.get(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Return detailed info if user is viewing their own profile
    if current_user and current_user.id == user_id:
        return UserDetailResponse.model_validate(user)
    
    return user


@router.delete("/{user_id}", response_model=SuccessResponse)
async def delete_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Delete user account (soft delete).
    
    Users can delete their own account, or admins can delete any account.
    """
    # Check permission
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this user",
        )
    
    user = await crud_user.get(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Soft delete
    await crud_user.soft_delete(db, id=user_id)
    
    return {"success": True, "message": "User account deleted"}


@router.get("/{user_id}/followers", response_model=PaginatedResponse[UserPublicResponse])
async def get_user_followers(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    Get list of users who follow the specified user.
    
    - **user_id**: Target user's ID
    - **skip**: Number of followers to skip
    - **limit**: Maximum number of followers to return
    """
    # Verify user exists
    user = await crud_user.get(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Get followers
    followers = await crud_user.get_followers(db, user_id=user_id, skip=skip, limit=limit)
    total = await crud_user.get_followers_count(db, user_id=user_id)
    
    return {
        "items": followers,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/{user_id}/following", response_model=PaginatedResponse[UserPublicResponse])
async def get_user_following(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> Any:
    """
    Get list of users that the specified user follows.
    
    - **user_id**: Target user's ID
    - **skip**: Number of results to skip
    - **limit**: Maximum number of results to return
    """
    # Verify user exists
    user = await crud_user.get(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Get following
    following = await crud_user.get_following(db, user_id=user_id, skip=skip, limit=limit)
    total = await crud_user.get_following_count(db, user_id=user_id)
    
    return {
        "items": following,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.post("/{user_id}/follow", response_model=FollowResponse)
async def follow_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Follow a user.
    
    Creates a follow relationship from current user to target user.
    """
    # Can't follow yourself
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot follow yourself",
        )
    
    # Verify target user exists
    target_user = await crud_user.get(db, user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Check if already following
    existing_follow = await crud_follow.get_by_users(
        db, follower_id=current_user.id, following_id=user_id
    )
    if existing_follow:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already following this user",
        )
    
    # Create follow
    from app.schemas.social import FollowCreate
    follow = await crud_follow.create(
        db,
        obj_in=FollowCreate(following_id=user_id),
        follower_id=current_user.id,
    )
    
    return follow


@router.delete("/{user_id}/follow", response_model=SuccessResponse)
async def unfollow_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Unfollow a user.
    
    Removes follow relationship from current user to target user.
    """
    # Get follow relationship
    follow = await crud_follow.get_by_users(
        db, follower_id=current_user.id, following_id=user_id
    )
    
    if not follow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not following this user",
        )
    
    # Delete follow
    await crud_follow.delete(db, id=follow.id)
    
    return {"success": True, "message": "User unfollowed successfully"}


# Admin endpoints
@router.put("/{user_id}/admin", response_model=UserResponse)
async def admin_update_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    user_in: UserAdminUpdate,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Update any user's account.
    
    Admins can update role, status, and verification status.
    
    - **role**: Change user role (USER, CREATOR, MODERATOR, ADMIN)
    - **status**: Change account status (ACTIVE, INACTIVE, SUSPENDED, BANNED)
    - **email_verified**: Set email verification status
    """
    user = await crud_user.get(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Prevent self-demotion
    if user_id == admin.id and user_in.role and user_in.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own admin role",
        )
    
    user = await crud_user.update(db, db_obj=user, obj_in=user_in)
    return user


@router.post("/{user_id}/activate", response_model=SuccessResponse)
async def admin_activate_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Activate a user account.
    
    Sets user status to ACTIVE and marks email as verified.
    """
    user = await crud_user.activate_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return {"success": True, "message": f"User {user.username} activated"}


@router.post("/{user_id}/deactivate", response_model=SuccessResponse)
async def admin_deactivate_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Deactivate a user account.
    
    Sets user status to INACTIVE.
    """
    user = await crud_user.deactivate_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return {"success": True, "message": f"User {user.username} deactivated"}


@router.post("/{user_id}/suspend", response_model=SuccessResponse)
async def admin_suspend_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: UUID,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Suspend a user account.
    
    Sets user status to SUSPENDED. User cannot access their account.
    """
    # Prevent self-suspension
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot suspend your own account",
        )
    
    user = await crud_user.suspend_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return {"success": True, "message": f"User {user.username} suspended"}
