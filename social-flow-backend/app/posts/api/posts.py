"""
Post endpoints for CRUD operations and reposting.

This module provides comprehensive API endpoints for post management.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.api.auth import get_current_active_user
from app.core.database import get_db
from app.core.exceptions import NotFoundError, ValidationError
from app.models.user import User
from app.posts.schemas.post import (
    PostCreate,
    PostResponse,
    PostUpdate,
    RepostCreate,
)
from app.posts.services.post_service import PostService

router = APIRouter()


@router.post("/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    post_data: PostCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new post.
    
    - **content**: Post content (1-2000 characters)
    - **media_url**: Optional media URL
    - **media_type**: Optional media type (image, video, gif)
    """
    service = PostService(db)
    
    try:
        # Service expects (post_data, user_id)
        post = await service.create_post(post_data, current_user.id)
        return post
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{post_id}", response_model=PostResponse)
async def get_post(
    post_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific post by ID.
    
    - **post_id**: UUID of the post
    """
    service = PostService(db)
    post = await service.get_post(post_id)
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    
    return post


@router.put("/{post_id}", response_model=PostResponse)
async def update_post(
    post_id: UUID,
    post_data: PostUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a post.
    
    - **post_id**: UUID of the post to update
    - **content**: New content (optional)
    - **media_url**: New media URL (optional)
    """
    service = PostService(db)
    
    try:
        # Service expects (post_id, post_data, user_id)
        post = await service.update_post(post_id, post_data, current_user.id)
        return post
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a post.
    
    - **post_id**: UUID of the post to delete
    """
    service = PostService(db)
    
    try:
        await service.delete_post(post_id, current_user.id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.post("/repost", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def repost(
    repost_data: RepostCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Repost an existing post.
    
    - **original_post_id**: UUID of the post to repost
    - **reason**: Optional reason/comment for the repost
    """
    service = PostService(db)
    
    try:
        post = await service.repost(
            current_user.id,
            repost_data.original_post_id,
            repost_data.reason,
        )
        return post
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/user/{user_id}", response_model=List[PostResponse])
async def get_user_posts(
    user_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    include_reposts: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    """
    Get posts by a specific user.
    
    - **user_id**: UUID of the user
    - **skip**: Number of posts to skip (for pagination)
    - **limit**: Maximum number of posts to return (1-100)
    - **include_reposts**: Whether to include reposts
    """
    service = PostService(db)
    posts = await service.get_user_posts(user_id, skip, limit, include_reposts)
    return posts


@router.get("/feed/", response_model=List[PostResponse])
async def get_feed(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    algorithm: str = Query("ml_ranked", pattern="^(chronological|engagement|ml_ranked)$"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get personalized feed for the current user.
    
    - **skip**: Number of posts to skip (for pagination)
    - **limit**: Maximum number of posts to return (1-100)
    - **algorithm**: Feed algorithm to use
        - `chronological`: Posts sorted by time
        - `engagement`: Posts sorted by engagement metrics
        - `ml_ranked`: ML-ranked hybrid feed (default)
    """
    service = PostService(db)
    posts = await service.get_feed(current_user.id, skip, limit, algorithm)
    return posts


@router.post("/{post_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def like_post(
    post_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Like a post.
    
    - **post_id**: UUID of the post to like
    """
    service = PostService(db)
    
    try:
        await service.like_post(post_id, current_user.id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{post_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def unlike_post(
    post_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Unlike a post.
    
    - **post_id**: UUID of the post to unlike
    """
    service = PostService(db)
    
    try:
        await service.unlike_post(post_id, current_user.id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

