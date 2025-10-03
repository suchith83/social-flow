"""
Post API endpoints using clean architecture.

This module provides post management endpoints using PostApplicationService.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.application.services import PostApplicationService
from app.core.dependencies import get_post_service, get_current_active_user
from app.models.user import User as UserModel

router = APIRouter()


# Request/Response Models


class PostCreateRequest(BaseModel):
    """Post creation request."""
    content: str = Field(..., min_length=1, max_length=2000)
    media_url: str | None = Field(None, max_length=500)
    tags: List[str] | None = None


class ReplyCreateRequest(BaseModel):
    """Reply creation request."""
    content: str = Field(..., min_length=1, max_length=2000)
    media_url: str | None = Field(None, max_length=500)


class PostUpdateRequest(BaseModel):
    """Post update request."""
    content: str = Field(..., min_length=1, max_length=2000)


class PostResponse(BaseModel):
    """Post response model."""
    id: UUID
    user_id: UUID
    content: str
    media_url: str | None
    parent_post_id: UUID | None
    like_count: int
    comment_count: int
    share_count: int
    impression_count: int
    tags: List[str]
    is_flagged: bool
    created_at: str
    updated_at: str | None
    
    class Config:
        from_attributes = True


# Routes


@router.post("/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    data: PostCreateRequest,
    current_user: UserModel = Depends(get_current_active_user),
    service: PostApplicationService = Depends(get_post_service),
):
    """
    Create a new post.
    
    - **content**: Post content (1-2000 characters)
    - **media_url**: Optional media URL
    - **tags**: Optional list of tags
    """
    try:
        post = await service.create_post(
            user_id=current_user.id,
            content=data.content,
            media_url=data.media_url,
            tags=data.tags,
        )
        
        return PostResponse(
            id=post.id,
            user_id=post.user_id,
            content=post.content,
            media_url=post.media_url,
            parent_post_id=post.parent_post_id,
            like_count=post.engagement.likes,
            comment_count=post.engagement.comments,
            share_count=post.engagement.shares,
            impression_count=post.engagement.impressions,
            tags=post.tags,
            is_flagged=post.is_flagged,
            created_at=post.created_at.isoformat(),
            updated_at=post.updated_at.isoformat() if post.updated_at else None,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{post_id}/reply", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_reply(
    post_id: UUID,
    data: ReplyCreateRequest,
    current_user: UserModel = Depends(get_current_active_user),
    service: PostApplicationService = Depends(get_post_service),
):
    """
    Create a reply to a post.
    
    - **post_id**: Parent post ID
    - **content**: Reply content (1-2000 characters)
    - **media_url**: Optional media URL
    """
    try:
        reply = await service.create_reply(
            user_id=current_user.id,
            parent_post_id=post_id,
            content=data.content,
            media_url=data.media_url,
        )
        
        return PostResponse(
            id=reply.id,
            user_id=reply.user_id,
            content=reply.content,
            media_url=reply.media_url,
            parent_post_id=reply.parent_post_id,
            like_count=reply.engagement.likes,
            comment_count=reply.engagement.comments,
            share_count=reply.engagement.shares,
            impression_count=reply.engagement.impressions,
            tags=reply.tags,
            is_flagged=reply.is_flagged,
            created_at=reply.created_at.isoformat(),
            updated_at=reply.updated_at.isoformat() if reply.updated_at else None,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/{post_id}", response_model=PostResponse)
async def get_post(
    post_id: UUID,
    service: PostApplicationService = Depends(get_post_service),
):
    """Get post by ID."""
    post = await service.get_post_by_id(post_id)
    
    if post is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    return PostResponse(
        id=post.id,
        user_id=post.user_id,
        content=post.content,
        media_url=post.media_url,
        parent_post_id=post.parent_post_id,
        like_count=post.engagement.likes,
        comment_count=post.engagement.comments,
        share_count=post.engagement.shares,
        impression_count=post.engagement.impressions,
        tags=post.tags,
        is_flagged=post.is_flagged,
        created_at=post.created_at.isoformat(),
        updated_at=post.updated_at.isoformat() if post.updated_at else None,
    )


@router.put("/{post_id}", response_model=PostResponse)
async def update_post(
    post_id: UUID,
    data: PostUpdateRequest,
    current_user: UserModel = Depends(get_current_active_user),
    service: PostApplicationService = Depends(get_post_service),
):
    """
    Update post content.
    
    - **content**: New content
    """
    try:
        # Verify ownership
        post = await service.get_post_by_id(post_id)
        if post is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Post not found",
            )
        
        if post.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this post",
            )
        
        updated_post = await service.update_post_content(
            post_id=post_id,
            content=data.content,
        )
        
        return PostResponse(
            id=updated_post.id,
            user_id=updated_post.user_id,
            content=updated_post.content,
            media_url=updated_post.media_url,
            parent_post_id=updated_post.parent_post_id,
            like_count=updated_post.engagement.likes,
            comment_count=updated_post.engagement.comments,
            share_count=updated_post.engagement.shares,
            impression_count=updated_post.engagement.impressions,
            tags=updated_post.tags,
            is_flagged=updated_post.is_flagged,
            created_at=updated_post.created_at.isoformat(),
            updated_at=updated_post.updated_at.isoformat() if updated_post.updated_at else None,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: UUID,
    current_user: UserModel = Depends(get_current_active_user),
    service: PostApplicationService = Depends(get_post_service),
):
    """Delete post."""
    try:
        # Verify ownership
        post = await service.get_post_by_id(post_id)
        if post is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Post not found",
            )
        
        if post.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this post",
            )
        
        await service.delete_post(post_id)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{post_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def like_post(
    post_id: UUID,
    current_user: UserModel = Depends(get_current_active_user),
    service: PostApplicationService = Depends(get_post_service),
):
    """Like post."""
    try:
        await service.like_post(post_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete("/{post_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def unlike_post(
    post_id: UUID,
    current_user: UserModel = Depends(get_current_active_user),
    service: PostApplicationService = Depends(get_post_service),
):
    """Unlike post."""
    try:
        await service.unlike_post(post_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{post_id}/share", status_code=status.HTTP_204_NO_CONTENT)
async def share_post(
    post_id: UUID,
    current_user: UserModel = Depends(get_current_active_user),
    service: PostApplicationService = Depends(get_post_service),
):
    """Share post."""
    try:
        await service.share_post(post_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/{post_id}/replies", response_model=List[PostResponse])
async def get_replies(
    post_id: UUID,
    skip: int = 0,
    limit: int = 50,
    service: PostApplicationService = Depends(get_post_service),
):
    """Get replies to a post."""
    if limit > 100:
        limit = 100
    
    replies = await service.get_post_replies(post_id=post_id, skip=skip, limit=limit)
    
    return [
        PostResponse(
            id=reply.id,
            user_id=reply.user_id,
            content=reply.content,
            media_url=reply.media_url,
            parent_post_id=reply.parent_post_id,
            like_count=reply.engagement.likes,
            comment_count=reply.engagement.comments,
            share_count=reply.engagement.shares,
            impression_count=reply.engagement.impressions,
            tags=reply.tags,
            is_flagged=reply.is_flagged,
            created_at=reply.created_at.isoformat(),
            updated_at=reply.updated_at.isoformat() if reply.updated_at else None,
        )
        for reply in replies
    ]


@router.get("/feed/me", response_model=List[PostResponse])
async def get_my_feed(
    skip: int = 0,
    limit: int = 20,
    current_user: UserModel = Depends(get_current_active_user),
    service: PostApplicationService = Depends(get_post_service),
):
    """Get personalized feed for current user."""
    if limit > 100:
        limit = 100
    
    posts = await service.get_user_feed(user_id=current_user.id, skip=skip, limit=limit)
    
    return [
        PostResponse(
            id=post.id,
            user_id=post.user_id,
            content=post.content,
            media_url=post.media_url,
            parent_post_id=post.parent_post_id,
            like_count=post.engagement.likes,
            comment_count=post.engagement.comments,
            share_count=post.engagement.shares,
            impression_count=post.engagement.impressions,
            tags=post.tags,
            is_flagged=post.is_flagged,
            created_at=post.created_at.isoformat(),
            updated_at=post.updated_at.isoformat() if post.updated_at else None,
        )
        for post in posts
    ]


@router.get("/feed/global", response_model=List[PostResponse])
async def get_global_feed(
    skip: int = 0,
    limit: int = 20,
    service: PostApplicationService = Depends(get_post_service),
):
    """Get global feed (recent public posts)."""
    if limit > 100:
        limit = 100
    
    posts = await service.get_global_feed(skip=skip, limit=limit)
    
    return [
        PostResponse(
            id=post.id,
            user_id=post.user_id,
            content=post.content,
            media_url=post.media_url,
            parent_post_id=post.parent_post_id,
            like_count=post.engagement.likes,
            comment_count=post.engagement.comments,
            share_count=post.engagement.shares,
            impression_count=post.engagement.impressions,
            tags=post.tags,
            is_flagged=post.is_flagged,
            created_at=post.created_at.isoformat(),
            updated_at=post.updated_at.isoformat() if post.updated_at else None,
        )
        for post in posts
    ]


@router.get("/trending", response_model=List[PostResponse])
async def get_trending_posts(
    days: int = 7,
    skip: int = 0,
    limit: int = 20,
    service: PostApplicationService = Depends(get_post_service),
):
    """Get trending posts."""
    if limit > 100:
        limit = 100
    
    posts = await service.get_trending_posts(days=days, skip=skip, limit=limit)
    
    return [
        PostResponse(
            id=post.id,
            user_id=post.user_id,
            content=post.content,
            media_url=post.media_url,
            parent_post_id=post.parent_post_id,
            like_count=post.engagement.likes,
            comment_count=post.engagement.comments,
            share_count=post.engagement.shares,
            impression_count=post.engagement.impressions,
            tags=post.tags,
            is_flagged=post.is_flagged,
            created_at=post.created_at.isoformat(),
            updated_at=post.updated_at.isoformat() if post.updated_at else None,
        )
        for post in posts
    ]


@router.get("/search", response_model=List[PostResponse])
async def search_posts(
    query: str,
    skip: int = 0,
    limit: int = 20,
    service: PostApplicationService = Depends(get_post_service),
):
    """Search posts by content."""
    if limit > 100:
        limit = 100
    
    posts = await service.search_posts(query=query, skip=skip, limit=limit)
    
    return [
        PostResponse(
            id=post.id,
            user_id=post.user_id,
            content=post.content,
            media_url=post.media_url,
            parent_post_id=post.parent_post_id,
            like_count=post.engagement.likes,
            comment_count=post.engagement.comments,
            share_count=post.engagement.shares,
            impression_count=post.engagement.impressions,
            tags=post.tags,
            is_flagged=post.is_flagged,
            created_at=post.created_at.isoformat(),
            updated_at=post.updated_at.isoformat() if post.updated_at else None,
        )
        for post in posts
    ]


@router.get("/user/{user_id}", response_model=List[PostResponse])
async def get_user_posts(
    user_id: UUID,
    skip: int = 0,
    limit: int = 20,
    service: PostApplicationService = Depends(get_post_service),
):
    """Get posts by user."""
    if limit > 100:
        limit = 100
    
    posts = await service.get_user_posts(user_id=user_id, skip=skip, limit=limit)
    
    return [
        PostResponse(
            id=post.id,
            user_id=post.user_id,
            content=post.content,
            media_url=post.media_url,
            parent_post_id=post.parent_post_id,
            like_count=post.engagement.likes,
            comment_count=post.engagement.comments,
            share_count=post.engagement.shares,
            impression_count=post.engagement.impressions,
            tags=post.tags,
            is_flagged=post.is_flagged,
            created_at=post.created_at.isoformat(),
            updated_at=post.updated_at.isoformat() if post.updated_at else None,
        )
        for post in posts
    ]

