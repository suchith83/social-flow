"""
Video API endpoints using clean architecture.

This module provides video management endpoints using VideoApplicationService.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.application.services import VideoApplicationService
from app.core.dependencies import get_video_service, get_current_active_user
from app.models.user import User as UserModel

router = APIRouter()


# Request/Response Models


class VideoUploadRequest(BaseModel):
    """Video upload metadata."""
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=5000)
    tags: List[str] | None = None


class VideoUpdateRequest(BaseModel):
    """Video update request."""
    title: str | None = Field(None, max_length=200)
    description: str | None = Field(None, max_length=5000)


class VideoResponse(BaseModel):
    """Video response model."""
    id: UUID
    user_id: UUID
    title: str
    description: str | None
    thumbnail_url: str | None
    stream_url: str | None
    duration: int
    resolution: str
    format: str
    file_size: int
    state: str
    is_public: bool
    view_count: int
    like_count: int
    comment_count: int
    share_count: int
    tags: List[str]
    created_at: str
    
    class Config:
        from_attributes = True


# Routes


@router.post("/upload", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str | None = Form(None),
    tags: str | None = Form(None),  # Comma-separated tags
    current_user: UserModel = Depends(get_current_active_user),
    service: VideoApplicationService = Depends(get_video_service),
):
    """
    Upload a new video (initiates upload workflow).
    
    - **file**: Video file
    - **title**: Video title
    - **description**: Video description (optional)
    - **tags**: Comma-separated tags (optional)
    
    Returns video ID and upload status. Video will be processed asynchronously.
    """
    try:
        # TODO: Implement actual file upload to storage
        # For now, simulate metadata
        duration = 120  # Mock duration
        file_size = file.size or 0
        format_type = file.filename.split('.')[-1] if file.filename else 'mp4'
        resolution = "1080p"  # Mock resolution
        storage_path = f"videos/{current_user.id}/{file.filename}"
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else None
        
        # Initiate upload
        video = await service.initiate_video_upload(
            user_id=current_user.id,
            title=title,
            description=description,
            duration=duration,
            file_size=file_size,
            format_type=format_type,
            resolution=resolution,
            storage_path=storage_path,
            tags=tag_list,
        )
        
        return {
            "video_id": str(video.id),
            "status": "processing",
            "message": "Video uploaded successfully and is being processed",
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: UUID,
    service: VideoApplicationService = Depends(get_video_service),
):
    """Get video by ID."""
    video = await service.get_video_by_id(video_id)
    
    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    return VideoResponse(
        id=video.id,
        user_id=video.user_id,
        title=video.title,
        description=video.description,
        thumbnail_url=video.thumbnail_url,
        stream_url=video.stream_url,
        duration=video.metadata.duration,
        resolution=video.metadata.resolution,
        format=video.metadata.format,
        file_size=video.metadata.file_size,
        state=video.state.value,
        is_public=video.is_public,
        view_count=video.engagement.views,
        like_count=video.engagement.likes,
        comment_count=video.engagement.comments,
        share_count=video.engagement.shares,
        tags=video.tags,
        created_at=video.created_at.isoformat(),
    )


@router.put("/{video_id}", response_model=VideoResponse)
async def update_video(
    video_id: UUID,
    data: VideoUpdateRequest,
    current_user: UserModel = Depends(get_current_active_user),
    service: VideoApplicationService = Depends(get_video_service),
):
    """
    Update video details.
    
    - **title**: New title (optional)
    - **description**: New description (optional)
    """
    try:
        # Verify ownership (should be added to service layer)
        video = await service.get_video_by_id(video_id)
        if video is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found",
            )
        
        if video.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this video",
            )
        
        updated_video = await service.update_video_details(
            video_id=video_id,
            title=data.title,
            description=data.description,
        )
        
        return VideoResponse(
            id=updated_video.id,
            user_id=updated_video.user_id,
            title=updated_video.title,
            description=updated_video.description,
            thumbnail_url=updated_video.thumbnail_url,
            stream_url=updated_video.stream_url,
            duration=updated_video.metadata.duration,
            resolution=updated_video.metadata.resolution,
            format=updated_video.metadata.format,
            file_size=updated_video.metadata.file_size,
            state=updated_video.state.value,
            is_public=updated_video.is_public,
            view_count=updated_video.engagement.views,
            like_count=updated_video.engagement.likes,
            comment_count=updated_video.engagement.comments,
            share_count=updated_video.engagement.shares,
            tags=updated_video.tags,
            created_at=updated_video.created_at.isoformat(),
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: UUID,
    current_user: UserModel = Depends(get_current_active_user),
    service: VideoApplicationService = Depends(get_video_service),
):
    """Delete video."""
    try:
        # Verify ownership
        video = await service.get_video_by_id(video_id)
        if video is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found",
            )
        
        if video.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this video",
            )
        
        await service.delete_video(video_id)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{video_id}/view", status_code=status.HTTP_204_NO_CONTENT)
async def record_view(
    video_id: UUID,
    service: VideoApplicationService = Depends(get_video_service),
):
    """Record video view."""
    try:
        await service.record_video_view(video_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{video_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def like_video(
    video_id: UUID,
    current_user: UserModel = Depends(get_current_active_user),
    service: VideoApplicationService = Depends(get_video_service),
):
    """Like video."""
    try:
        await service.like_video(video_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete("/{video_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def unlike_video(
    video_id: UUID,
    current_user: UserModel = Depends(get_current_active_user),
    service: VideoApplicationService = Depends(get_video_service),
):
    """Unlike video."""
    try:
        await service.unlike_video(video_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/", response_model=List[VideoResponse])
async def discover_videos(
    skip: int = 0,
    limit: int = 20,
    service: VideoApplicationService = Depends(get_video_service),
):
    """Get videos for discovery feed."""
    if limit > 100:
        limit = 100
    
    videos = await service.discover_videos(skip=skip, limit=limit)
    
    return [
        VideoResponse(
            id=video.id,
            user_id=video.user_id,
            title=video.title,
            description=video.description,
            thumbnail_url=video.thumbnail_url,
            stream_url=video.stream_url,
            duration=video.metadata.duration,
            resolution=video.metadata.resolution,
            format=video.metadata.format,
            file_size=video.metadata.file_size,
            state=video.state.value,
            is_public=video.is_public,
            view_count=video.engagement.views,
            like_count=video.engagement.likes,
            comment_count=video.engagement.comments,
            share_count=video.engagement.shares,
            tags=video.tags,
            created_at=video.created_at.isoformat(),
        )
        for video in videos
    ]


@router.get("/trending", response_model=List[VideoResponse])
async def get_trending_videos(
    days: int = 7,
    skip: int = 0,
    limit: int = 20,
    service: VideoApplicationService = Depends(get_video_service),
):
    """Get trending videos."""
    if limit > 100:
        limit = 100
    
    videos = await service.get_trending_videos(days=days, skip=skip, limit=limit)
    
    return [
        VideoResponse(
            id=video.id,
            user_id=video.user_id,
            title=video.title,
            description=video.description,
            thumbnail_url=video.thumbnail_url,
            stream_url=video.stream_url,
            duration=video.metadata.duration,
            resolution=video.metadata.resolution,
            format=video.metadata.format,
            file_size=video.metadata.file_size,
            state=video.state.value,
            is_public=video.is_public,
            view_count=video.engagement.views,
            like_count=video.engagement.likes,
            comment_count=video.engagement.comments,
            share_count=video.engagement.shares,
            tags=video.tags,
            created_at=video.created_at.isoformat(),
        )
        for video in videos
    ]


@router.get("/search", response_model=List[VideoResponse])
async def search_videos(
    query: str,
    skip: int = 0,
    limit: int = 20,
    service: VideoApplicationService = Depends(get_video_service),
):
    """Search videos by title."""
    if limit > 100:
        limit = 100
    
    videos = await service.search_videos(query=query, skip=skip, limit=limit)
    
    return [
        VideoResponse(
            id=video.id,
            user_id=video.user_id,
            title=video.title,
            description=video.description,
            thumbnail_url=video.thumbnail_url,
            stream_url=video.stream_url,
            duration=video.metadata.duration,
            resolution=video.metadata.resolution,
            format=video.metadata.format,
            file_size=video.metadata.file_size,
            state=video.state.value,
            is_public=video.is_public,
            view_count=video.engagement.views,
            like_count=video.engagement.likes,
            comment_count=video.engagement.comments,
            share_count=video.engagement.shares,
            tags=video.tags,
            created_at=video.created_at.isoformat(),
        )
        for video in videos
    ]


@router.get("/user/{user_id}", response_model=List[VideoResponse])
async def get_user_videos(
    user_id: UUID,
    skip: int = 0,
    limit: int = 20,
    service: VideoApplicationService = Depends(get_video_service),
):
    """Get videos by user."""
    if limit > 100:
        limit = 100
    
    videos = await service.get_user_videos(user_id=user_id, skip=skip, limit=limit)
    
    return [
        VideoResponse(
            id=video.id,
            user_id=video.user_id,
            title=video.title,
            description=video.description,
            thumbnail_url=video.thumbnail_url,
            stream_url=video.stream_url,
            duration=video.metadata.duration,
            resolution=video.metadata.resolution,
            format=video.metadata.format,
            file_size=video.metadata.file_size,
            state=video.state.value,
            is_public=video.is_public,
            view_count=video.engagement.views,
            like_count=video.engagement.likes,
            comment_count=video.engagement.comments,
            share_count=video.engagement.shares,
            tags=video.tags,
            created_at=video.created_at.isoformat(),
        )
        for video in videos
    ]

