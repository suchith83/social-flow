"""
Video endpoints.

This module contains all video-related API endpoints.
"""

from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import VideoServiceError, ValidationError
from app.models.user import User
from app.auth.api.auth import get_current_active_user
from app.videos.services.video_service import video_service
from app.analytics.services.analytics_service import analytics_service

router = APIRouter()


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Upload a new video."""
    try:
        # Validate file
        if not file.filename:
            raise ValidationError("No file provided")
        
        # Prepare metadata
        metadata = {
            "title": title,
            "description": description or "",
            "tags": tags or "",
        }
        
        # Upload video
        result = await video_service.upload_video(file, current_user, metadata, db)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="video_upload",
            user_id=str(current_user.id),
            data={
                "video_id": result["video_id"],
                "file_size": file.size,
                "filename": file.filename,
            }
        )
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Video upload failed")


@router.get("/{video_id}")
async def get_video(
    video_id: str,
    current_user: Optional[User] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get video by ID."""
    try:
        # Get video from database
        video = await video_service.get_video_by_id(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Track view event
        if current_user:
            await analytics_service.track_event(
                event_type="video_view",
                user_id=str(current_user.id),
                data={"video_id": video_id}
            )
        
        return video.to_dict()
        
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get video")


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: str,
    quality: str = "auto",
    current_user: Optional[User] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Stream video content."""
    try:
        # Get streaming URL
        result = await video_service.get_video_stream_url(video_id, quality)
        
        # Track streaming event
        if current_user:
            await analytics_service.track_event(
                event_type="video_stream",
                user_id=str(current_user.id),
                data={"video_id": video_id, "quality": quality}
            )
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Video streaming failed")


@router.post("/{video_id}/like")
async def like_video(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Like a video."""
    try:
        # Like video
        result = await video_service.like_video(video_id, current_user.id)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="video_like",
            user_id=str(current_user.id),
            data={"video_id": video_id}
        )
        
        return result
        
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to like video")


@router.delete("/{video_id}/like")
async def unlike_video(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Unlike a video."""
    try:
        # Unlike video
        result = await video_service.unlike_video(video_id, current_user.id)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="video_unlike",
            user_id=str(current_user.id),
            data={"video_id": video_id}
        )
        
        return result
        
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to unlike video")


@router.post("/{video_id}/view")
async def record_view(
    video_id: str,
    current_user: Optional[User] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Record video view."""
    try:
        # Increment view count
        result = await video_service.increment_view_count(
            video_id, 
            str(current_user.id) if current_user else None
        )
        
        # Track analytics event
        if current_user:
            await analytics_service.track_event(
                event_type="video_view",
                user_id=str(current_user.id),
                data={"video_id": video_id}
            )
        
        return result
        
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to record view")


@router.post("/live/create")
async def create_live_stream(
    title: str = Form(...),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create a live streaming session."""
    try:
        # Create live stream
        result = await video_service.create_live_stream(current_user, title, description)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="live_stream_create",
            user_id=str(current_user.id),
            data={"stream_id": result["stream_id"]}
        )
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Live stream creation failed")


@router.post("/live/{stream_id}/end")
async def end_live_stream(
    stream_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """End a live streaming session."""
    try:
        # End live stream
        result = await video_service.end_live_stream(stream_id)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="live_stream_end",
            user_id=str(current_user.id),
            data={"stream_id": stream_id}
        )
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Live stream ending failed")


@router.get("/")
async def get_videos(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get list of videos."""
    try:
        # Get videos from database
        videos = await video_service.get_videos(skip=skip, limit=limit, category=category)

        # Track analytics event
        if current_user:
            await analytics_service.track_event(
                event_type="video_list_view",
                user_id=str(current_user.id),
                data={"skip": skip, "limit": limit, "category": category}
            )

        return videos

    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get videos")


# Enhanced upload endpoints from Node.js service

@router.post("/upload/initiate")
async def initiate_upload(
    filename: str = Form(...),
    file_size: int = Form(...),
    chunk_size: Optional[int] = Form(5 * 1024 * 1024),  # 5MB default
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Initiate a chunked upload session."""
    try:
        metadata = {
            "filename": filename,
            "file_size": file_size,
            "chunk_size": chunk_size
        }
        
        result = await video_service.initiate_upload_session(current_user, metadata)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="video_upload_initiated",
            user_id=str(current_user.id),
            data={
                "upload_id": result["upload_id"],
                "file_size": file_size,
                "filename": filename
            }
        )
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to initiate upload")


@router.put("/upload/{upload_id}/chunk/{chunk_number}")
async def upload_chunk(
    upload_id: str,
    chunk_number: int,
    chunk_data: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Upload a chunk of video data."""
    try:
        # Read chunk data
        chunk_bytes = await chunk_data.read()
        
        result = await video_service.upload_chunk(upload_id, chunk_number, chunk_bytes)
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to upload chunk")


@router.post("/upload/{upload_id}/complete")
async def complete_upload(
    upload_id: str,
    title: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Complete the chunked upload and start processing."""
    try:
        metadata = {
            "title": title,
            "description": description or "",
            "tags": tags or ""
        }
        
        result = await video_service.complete_upload(upload_id, metadata)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="video_upload_completed",
            user_id=str(current_user.id),
            data={
                "video_id": result["video_id"],
                "upload_id": upload_id
            }
        )
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to complete upload")


@router.delete("/upload/{upload_id}")
async def cancel_upload(
    upload_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Cancel an ongoing upload."""
    try:
        result = await video_service.cancel_upload(upload_id)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="video_upload_cancelled",
            user_id=str(current_user.id),
            data={"upload_id": upload_id}
        )
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to cancel upload")


@router.get("/upload/{upload_id}/progress")
async def get_upload_progress(
    upload_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get upload progress."""
    try:
        result = await video_service.get_upload_progress(upload_id)
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get upload progress")


# Enhanced transcoding endpoints

@router.post("/{video_id}/transcode")
async def transcode_video(
    video_id: str,
    settings: Optional[dict] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Transcode video to multiple formats and resolutions."""
    try:
        result = await video_service.transcode_video(video_id, settings)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="video_transcode_started",
            user_id=str(current_user.id),
            data={"video_id": video_id, "settings": settings}
        )
        
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to transcode video")


@router.post("/{video_id}/thumbnails")
async def generate_thumbnails(
    video_id: str,
    count: int = 5,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Generate video thumbnails."""
    try:
        result = await video_service.generate_thumbnails(video_id, count)
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate thumbnails")


@router.post("/{video_id}/manifest")
async def create_streaming_manifest(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create HLS/DASH streaming manifest."""
    try:
        result = await video_service.create_streaming_manifest(video_id)
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to create streaming manifest")


@router.post("/{video_id}/optimize-mobile")
async def optimize_for_mobile(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create mobile-optimized video versions."""
    try:
        result = await video_service.optimize_for_mobile(video_id)
        return result
        
    except VideoServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to optimize for mobile")

