"""
Like endpoints for managing likes on posts, videos, and comments.

This module provides comprehensive API endpoints for like management.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.api.auth import get_current_active_user
from app.core.database import get_db
from app.core.exceptions import NotFoundError, ValidationError
from app.auth.models.user import User
from app.posts.schemas.like import (
    LikeCreate,
    LikeResponse,
    LikeUpdate,
    LikeListResponse,
    LikeCheckResponse,
)
from app.posts.services.like_service import LikeService

router = APIRouter()


@router.post("/", response_model=LikeResponse, status_code=status.HTTP_201_CREATED)
async def create_like(
    like_data: LikeCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new like.

    - **like_type**: Type of entity being liked (post, video, comment)
    - **is_like**: True for like, False for dislike (default: True)
    - **post_id**: Post ID being liked (required if like_type is 'post')
    - **video_id**: Video ID being liked (required if like_type is 'video')
    - **comment_id**: Comment ID being liked (required if like_type is 'comment')
    """
    service = LikeService(db)

    try:
        like = await service.create_like(current_user.id, like_data)
        return like
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.put("/", response_model=LikeResponse)
async def update_like(
    like_data: LikeUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update an existing like (change from like to dislike or vice versa).

    - **like_type**: Type of entity (post, video, comment)
    - **is_like**: New like status (True for like, False for dislike)
    - **post_id**: Post ID (required if like_type is 'post')
    - **video_id**: Video ID (required if like_type is 'video')
    - **comment_id**: Comment ID (required if like_type is 'comment')
    """
    service = LikeService(db)

    try:
        like = await service.update_like(current_user.id, like_data)
        return like
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_like(
    like_data: LikeUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a like.

    - **like_type**: Type of entity (post, video, comment)
    - **post_id**: Post ID (required if like_type is 'post')
    - **video_id**: Video ID (required if like_type is 'video')
    - **comment_id**: Comment ID (required if like_type is 'comment')
    """
    service = LikeService(db)

    try:
        await service.delete_like(current_user.id, like_data)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/user/{user_id}", response_model=LikeListResponse)
async def get_user_likes(
    user_id: UUID,
    like_type: str = Query(None, pattern="^(post|video|comment)$"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Get likes by a specific user.

    - **user_id**: UUID of the user
    - **like_type**: Optional filter by like type (post, video, comment)
    - **skip**: Number of likes to skip (for pagination)
    - **limit**: Maximum number of likes to return (1-100)
    """
    service = LikeService(db)
    likes = await service.get_user_likes(user_id, like_type, skip, limit)

    return LikeListResponse(
        likes=likes,
        total=len(likes),  # TODO: Add proper total count
        skip=skip,
        limit=limit
    )


@router.get("/entity/{entity_type}/{entity_id}", response_model=LikeListResponse)
async def get_entity_likes(
    entity_type: str,
    entity_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Get likes for a specific entity.

    - **entity_type**: Type of entity (post, video, comment)
    - **entity_id**: UUID of the entity
    - **skip**: Number of likes to skip (for pagination)
    - **limit**: Maximum number of likes to return (1-100)
    """
    if entity_type not in ["post", "video", "comment"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid entity type")

    service = LikeService(db)
    likes = await service.get_entity_likes(entity_id, entity_type, skip, limit)

    return LikeListResponse(
        likes=likes,
        total=len(likes),  # TODO: Add proper total count
        skip=skip,
        limit=limit
    )


@router.get("/check", response_model=LikeCheckResponse)
async def check_user_like(
    like_data: LikeUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Check if current user has liked a specific entity.

    - **like_type**: Type of entity (post, video, comment)
    - **post_id**: Post ID (required if like_type is 'post')
    - **video_id**: Video ID (required if like_type is 'video')
    - **comment_id**: Comment ID (required if like_type is 'comment')
    """
    service = LikeService(db)
    like = await service.check_user_like(current_user.id, like_data)

    if like:
        return LikeCheckResponse(
            is_liked=True,
            like_id=like.id,
            is_like=like.is_like
        )
    else:
        return LikeCheckResponse(
            is_liked=False,
            like_id=None,
            is_like=None
        )
