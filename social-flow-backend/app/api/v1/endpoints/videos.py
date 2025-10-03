"""
Video management endpoints for upload, streaming, and video operations.
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
    require_creator,
)
from app.infrastructure.crud import video as crud_video, like as crud_like
from app.models.user import User
from app.models.video import VideoStatus, VideoVisibility
from app.schemas.base import PaginatedResponse, SuccessResponse
from app.schemas.video import (
    VideoCreate,
    VideoUpdate,
    VideoResponse,
    VideoDetailResponse,
    VideoPublicResponse,
    VideoUploadInit,
    VideoUploadURL,
    VideoStreamingURLs,
    VideoAnalytics,
)

router = APIRouter()


@router.post("", response_model=VideoUploadURL, status_code=status.HTTP_201_CREATED)
async def initiate_video_upload(
    *,
    db: AsyncSession = Depends(get_db),
    upload_init: VideoUploadInit,
    current_user: User = Depends(require_creator),
) -> Any:
    """
    Initiate video upload process.
    
    Creates a video record and returns a pre-signed upload URL.
    Only creators can upload videos.
    
    - **filename**: Original filename
    - **file_size**: File size in bytes (max 10GB)
    - **content_type**: Video MIME type (mp4, mpeg, quicktime, webm, avi)
    
    Returns pre-signed S3 URL for direct upload.
    """
    # Create video record
    video_data = VideoCreate(
        title=upload_init.filename,  # Temporary title, will be updated later
        description=None,
        tags=[],
        visibility="private",  # Start as private
        original_filename=upload_init.filename,
        file_size=upload_init.file_size,
    )
    
    video = await crud_video.create_with_owner(
        db,
        obj_in=video_data,
        owner_id=current_user.id,
    )
    
    # Generate pre-signed upload URL (placeholder - integrate with S3)
    # In production, use boto3 to generate pre-signed URL
    upload_url = f"https://s3.amazonaws.com/social-flow-videos/{video.id}/{upload_init.filename}"
    
    return {
        "upload_url": upload_url,
        "video_id": video.id,
        "expires_in": 3600,  # 1 hour
    }


@router.post("/{video_id}/complete", response_model=VideoResponse)
async def complete_video_upload(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    video_update: VideoUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Complete video upload and update metadata.
    
    After uploading to S3, call this endpoint to:
    - Update video title, description, tags
    - Set visibility
    - Trigger transcoding
    
    - **title**: Video title (required)
    - **description**: Video description
    - **tags**: List of tags
    - **visibility**: public, unlisted, private, followers_only
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Check ownership
    if video.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this video",
        )
    
    # Update video metadata
    video = await crud_video.update(db, db_obj=video, obj_in=video_update)
    
    # Update status to processing (placeholder for transcoding trigger)
    video = await crud_video.update_status(
        db,
        video_id=video_id,
        status=VideoStatus.PROCESSING,
    )
    
    return video


@router.get("", response_model=PaginatedResponse[VideoPublicResponse])
async def list_videos(
    *,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    sort: str = Query("published_at", pattern="^(published_at|views_count|likes_count|created_at)$"),
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    List all public videos.
    
    - **skip**: Number of videos to skip (pagination)
    - **limit**: Maximum number of videos to return (1-100)
    - **sort**: Sort by: published_at, views_count, likes_count, created_at
    
    Returns paginated list of public videos.
    """
    # Get public videos
    videos = await crud_video.get_public_videos(db, skip=skip, limit=limit)
    
    # Count total
    total = await crud_video.count(
        db,
        filters={"visibility": VideoVisibility.PUBLIC, "status": VideoStatus.PROCESSED}
    )
    
    return {
        "items": videos,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/trending", response_model=PaginatedResponse[VideoPublicResponse])
async def get_trending_videos(
    *,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    days: int = Query(7, ge=1, le=30),
) -> Any:
    """
    Get trending videos based on view count.
    
    - **skip**: Number of videos to skip
    - **limit**: Maximum number of videos to return (1-100)
    - **days**: Number of days to look back (1-30)
    
    Returns videos sorted by view count in the specified time period.
    """
    videos = await crud_video.get_trending(db, skip=skip, limit=limit, days=days)
    
    # Count total trending videos
    total = len(videos)  # Simplified - could be a separate count query
    
    return {
        "items": videos,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/search", response_model=PaginatedResponse[VideoPublicResponse])
async def search_videos(
    *,
    db: AsyncSession = Depends(get_db),
    q: str = Query(..., min_length=1, max_length=100),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> Any:
    """
    Search videos by title or description.
    
    - **q**: Search query (min 1 char)
    - **skip**: Number of results to skip
    - **limit**: Maximum number of results (1-100)
    
    Returns videos matching the search query.
    """
    videos = await crud_video.search(db, query_text=q, skip=skip, limit=limit)
    
    # Simplified count
    total = len(videos)
    
    return {
        "items": videos,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/my", response_model=PaginatedResponse[VideoResponse])
async def get_my_videos(
    *,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[VideoStatus] = None,
    visibility: Optional[VideoVisibility] = None,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get current user's videos.
    
    - **skip**: Number of videos to skip
    - **limit**: Maximum number of videos to return (1-100)
    - **status**: Filter by status (uploading, processing, ready, failed, archived)
    - **visibility**: Filter by visibility (public, unlisted, private, followers_only)
    
    Returns user's videos with all metadata.
    """
    videos = await crud_video.get_by_user(
        db,
        user_id=current_user.id,
        skip=skip,
        limit=limit,
        status=status,
        visibility=visibility,
    )
    
    total = await crud_video.get_user_video_count(
        db,
        user_id=current_user.id,
        status=status,
    )
    
    return {
        "items": videos,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/{video_id}", response_model=VideoDetailResponse)
async def get_video(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    Get video details by ID.
    
    Returns detailed video information.
    Public videos: Anyone can view
    Private videos: Only owner can view
    Followers-only: Owner and followers can view
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Check visibility permissions
    if video.visibility == VideoVisibility.PRIVATE:
        if not current_user or current_user.id != video.owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Video is private",
            )
    
    elif video.visibility == VideoVisibility.FOLLOWERS_ONLY:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        
        if current_user.id != video.owner_id:
            # Check if current user follows video owner
            from app.infrastructure.crud import follow as crud_follow
            is_following = await crud_follow.is_following(
                db,
                follower_id=current_user.id,
                followed_id=video.owner_id,
            )
            if not is_following:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Video is only visible to followers",
                )
    
    return video


@router.put("/{video_id}", response_model=VideoResponse)
async def update_video(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    video_in: VideoUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update video metadata.
    
    - **title**: Video title
    - **description**: Video description
    - **tags**: List of tags
    - **visibility**: public, unlisted, private, followers_only
    - **thumbnail_url**: Custom thumbnail URL
    - **allow_comments**: Enable/disable comments
    - **allow_likes**: Enable/disable likes
    - **is_monetized**: Enable/disable monetization
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Check ownership
    if video.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this video",
        )
    
    video = await crud_video.update(db, db_obj=video, obj_in=video_in)
    return video


@router.delete("/{video_id}", response_model=SuccessResponse)
async def delete_video(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Delete video (soft delete).
    
    Only video owner or admin can delete.
    Video status is set to 'archived' for data preservation.
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Check ownership or admin
    from app.models.user import UserRole
    if video.owner_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this video",
        )
    
    # Soft delete by setting status to deleted
    await crud_video.update_status(db, video_id=video_id, status=VideoStatus.DELETED)
    
    return {"success": True, "message": "Video deleted successfully"}


@router.get("/{video_id}/stream", response_model=VideoStreamingURLs)
async def get_streaming_urls(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    Get streaming URLs for video playback.
    
    Returns HLS and DASH manifest URLs for adaptive streaming.
    Includes available quality levels and direct URLs.
    
    Checks visibility permissions before returning URLs.
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Check if video is ready for streaming
    if video.status != VideoStatus.PROCESSED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video is not ready for streaming (status: {video.status})",
        )
    
    # Check visibility permissions (same as get_video)
    if video.visibility == VideoVisibility.PRIVATE:
        if not current_user or current_user.id != video.owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Video is private",
            )
    
    elif video.visibility == VideoVisibility.FOLLOWERS_ONLY:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        
        if current_user.id != video.owner_id:
            from app.infrastructure.crud import follow as crud_follow
            is_following = await crud_follow.is_following(
                db,
                follower_id=current_user.id,
                followed_id=video.owner_id,
            )
            if not is_following:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Video is only visible to followers",
                )
    
    # Build qualities dict, excluding None values
    qualities = {}
    if video.hls_master_url:
        qualities["720p"] = f"{video.hls_master_url}/720p.m3u8"
        qualities["480p"] = f"{video.hls_master_url}/480p.m3u8"
        qualities["360p"] = f"{video.hls_master_url}/360p.m3u8"
    
    # Return streaming URLs
    return {
        "hls_url": video.hls_master_url,
        "dash_url": video.dash_manifest_url,
        "thumbnail_url": video.thumbnail_url,
        "poster_url": video.thumbnail_url,  # Using thumbnail as poster
        "qualities": qualities,
    }


@router.post("/{video_id}/view", response_model=SuccessResponse)
async def increment_video_view(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    Increment video view count.
    
    Call this endpoint when a user watches a video.
    Can be called by authenticated or anonymous users.
    
    In production, implement view tracking with:
    - Unique view detection (IP, user ID, session)
    - Watch time tracking
    - Analytics integration
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Increment view count
    await crud_video.increment_view_count(db, video_id=video_id)
    
    # TODO: In production, also create VideoView record for analytics
    # This would include: user_id, ip_address, watch_duration, etc.
    
    return {"success": True, "message": "View recorded"}


@router.post("/{video_id}/like", response_model=SuccessResponse)
async def like_video(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Like a video.
    
    Creates a like relationship for the video.
    If already liked, returns success without creating duplicate.
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Note: allow_likes field doesn't exist in Video model
    # if not video.allow_likes:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Likes are disabled for this video",
    #     )
    
    # Check if already liked
    existing_like = await crud_like.get_by_user_and_video(
        db,
        user_id=current_user.id,
        video_id=video_id,
    )
    
    if existing_like:
        return {"success": True, "message": "Video already liked"}
    
    # Create like
    from app.schemas.social import LikeCreate
    await crud_like.create(
        db,
        obj_in=LikeCreate(
            user_id=current_user.id,
            video_id=video_id,
        ),
    )
    
    return {"success": True, "message": "Video liked successfully"}


@router.delete("/{video_id}/like", response_model=SuccessResponse)
async def unlike_video(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Unlike a video.
    
    Removes the like relationship for the video.
    """
    like = await crud_like.get_by_user_and_video(
        db,
        user_id=current_user.id,
        video_id=video_id,
    )
    
    if not like:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not liked",
        )
    
    await crud_like.delete(db, id=like.id)
    
    return {"success": True, "message": "Video unliked successfully"}


@router.get("/{video_id}/analytics", response_model=VideoAnalytics)
async def get_video_analytics(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get video analytics data.
    
    Only video owner can access analytics.
    
    Returns:
    - View statistics (total, today, week, month)
    - Engagement metrics (likes, comments, shares)
    - Performance metrics (watch time, completion rate)
    - Geographic distribution
    - Traffic sources
    - Device types
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Check ownership
    if video.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view analytics for this video",
        )
    
    # TODO: In production, query analytics database/service
    # For now, return basic stats from video model
    return {
        "video_id": video_id,
        "views_total": video.view_count,
        "views_today": 0,  # Placeholder
        "views_week": 0,   # Placeholder
        "views_month": 0,  # Placeholder
        "likes_count": video.like_count,
        "comments_count": video.comment_count,
        "shares_count": video.share_count,
        "average_watch_time": 0.0,  # Placeholder
        "completion_rate": 0.0,     # Placeholder
        "engagement_rate": 0.0,     # Placeholder
        "top_countries": [],        # Placeholder
        "traffic_sources": {},      # Placeholder
        "device_types": {},         # Placeholder
    }


# Admin endpoints
@router.post("/{video_id}/admin/approve", response_model=SuccessResponse)
async def admin_approve_video(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Approve a video for publication.
    
    Sets video status to PROCESSED and makes it available for viewing.
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    await crud_video.update_status(db, video_id=video_id, status=VideoStatus.PROCESSED)
    
    return {"success": True, "message": f"Video '{video.title}' approved"}


@router.post("/{video_id}/admin/reject", response_model=SuccessResponse)
async def admin_reject_video(
    *,
    db: AsyncSession = Depends(get_db),
    video_id: UUID,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Reject a video.
    
    Sets video status to FAILED and prevents publication.
    """
    video = await crud_video.get(db, video_id)
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    await crud_video.update_status(db, video_id=video_id, status=VideoStatus.FAILED)
    
    return {"success": True, "message": f"Video '{video.title}' rejected"}
