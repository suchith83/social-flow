"""
Comment endpoints for CRUD operations and replies.

This module provides comprehensive API endpoints for comment management.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.api.auth import get_current_active_user
from app.core.database import get_db
from app.core.exceptions import NotFoundError, ValidationError
from app.auth.models.user import User
from app.posts.schemas.comment import (
    CommentCreate,
    CommentResponse,
    CommentUpdate,
    CommentListResponse,
    CommentRepliesResponse,
)
from app.posts.services.comment_service import CommentService

router = APIRouter()


@router.post("/", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def create_comment(
    comment_data: CommentCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new comment or reply.

    - **content**: Comment content (1-1000 characters)
    - **post_id**: Post ID this comment belongs to (optional if video_id provided)
    - **video_id**: Video ID this comment belongs to (optional if post_id provided)
    - **parent_comment_id**: Parent comment ID for replies (optional)
    """
    service = CommentService(db)

    try:
        comment = await service.create_comment(current_user.id, comment_data)
        return comment
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{comment_id}", response_model=CommentResponse)
async def get_comment(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific comment by ID.

    - **comment_id**: UUID of the comment
    """
    service = CommentService(db)
    comment = await service.get_comment(comment_id)

    if not comment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found")

    return comment


@router.put("/{comment_id}", response_model=CommentResponse)
async def update_comment(
    comment_id: UUID,
    comment_data: CommentUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a comment.

    - **comment_id**: UUID of the comment to update
    - **content**: New content (optional)
    """
    service = CommentService(db)

    try:
        comment = await service.update_comment(comment_id, current_user.id, comment_data)
        return comment
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comment(
    comment_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a comment.

    - **comment_id**: UUID of the comment to delete
    """
    service = CommentService(db)

    try:
        await service.delete_comment(comment_id, current_user.id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.get("/post/{post_id}", response_model=CommentListResponse)
async def get_post_comments(
    post_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    include_replies: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    """
    Get comments for a specific post.

    - **post_id**: UUID of the post
    - **skip**: Number of comments to skip (for pagination)
    - **limit**: Maximum number of comments to return (1-100)
    - **include_replies**: Whether to include replies in the response
    """
    service = CommentService(db)
    comments = await service.get_post_comments(post_id, skip, limit, include_replies)

    return CommentListResponse(
        comments=comments,
        total=len(comments),  # TODO: Add proper total count
        skip=skip,
        limit=limit
    )


@router.get("/{comment_id}/replies", response_model=CommentRepliesResponse)
async def get_comment_replies(
    comment_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Get replies for a specific comment.

    - **comment_id**: UUID of the parent comment
    - **skip**: Number of replies to skip (for pagination)
    - **limit**: Maximum number of replies to return (1-100)
    """
    service = CommentService(db)
    replies = await service.get_comment_replies(comment_id, skip, limit)

    return CommentRepliesResponse(
        replies=replies,
        total=len(replies),  # TODO: Add proper total count
        skip=skip,
        limit=limit
    )


@router.post("/{comment_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def like_comment(
    comment_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Like a comment.

    - **comment_id**: UUID of the comment to like
    """
    service = CommentService(db)

    try:
        await service.like_comment(current_user.id, comment_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{comment_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def unlike_comment(
    comment_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Unlike a comment.

    - **comment_id**: UUID of the comment to unlike
    """
    service = CommentService(db)

    try:
        await service.unlike_comment(current_user.id, comment_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
