"""
Live Streaming endpoints.

This module contains all live streaming-related API endpoints.
"""

from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import LiveStreamingServiceError
from app.auth.models.user import User
from app.auth.api.auth import get_current_active_user
from app.live.services.live_streaming_service import live_streaming_service
from app.live.schemas import (
    LiveStreamCreate, LiveStreamUpdate, LiveStreamResponse,
    LiveStreamListResponse, StreamViewerResponse, StreamAnalyticsResponse
)

router = APIRouter()


@router.post("/create", response_model=LiveStreamResponse)
async def create_live_stream(
    stream_data: LiveStreamCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create a new live stream."""
    try:
        result = await live_streaming_service.create_live_stream(
            str(current_user.id), 
            stream_data.title, 
            stream_data.description, 
            stream_data.tags, 
            stream_data.thumbnail_url,
            db
        )
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to create live stream")


@router.post("/{stream_id}/start")
async def start_live_stream(
    stream_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Start a live stream."""
    try:
        # Check if user owns the stream
        stream_info = await live_streaming_service.get_live_stream(stream_id, db)
        if stream_info['user_id'] != str(current_user.id):
            raise HTTPException(status_code=403, detail="You don't have permission to start this stream")
        
        result = await live_streaming_service.start_live_stream(stream_id)
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to start live stream")


@router.post("/{stream_id}/stop")
async def stop_live_stream(
    stream_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Stop a live stream."""
    try:
        # Check if user owns the stream
        stream_info = await live_streaming_service.get_live_stream(stream_id, db)
        if stream_info['user_id'] != str(current_user.id):
            raise HTTPException(status_code=403, detail="You don't have permission to stop this stream")
        
        result = await live_streaming_service.stop_live_stream(stream_id)
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to stop live stream")


@router.get("/{stream_id}")
async def get_live_stream(
    stream_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get live stream information."""
    try:
        result = await live_streaming_service.get_live_stream(stream_id, db)
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get live stream")


@router.get("/")
async def get_live_streams(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str = Query("live"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get all live streams."""
    try:
        streams = await live_streaming_service.get_live_streams(limit, offset, status, db)
        return {"streams": streams}
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get live streams")


@router.get("/user/{user_id}")
async def get_user_live_streams(
    user_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get live streams for a user."""
    try:
        streams = await live_streaming_service.get_user_live_streams(user_id, limit, offset, db)
        return {"streams": streams}
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get user live streams")


@router.put("/{stream_id}")
async def update_live_stream(
    stream_id: str,
    updates: LiveStreamUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update live stream information."""
    try:
        # Check if user owns the stream
        stream_info = await live_streaming_service.get_live_stream(stream_id, db)
        if stream_info['user_id'] != str(current_user.id):
            raise HTTPException(status_code=403, detail="You don't have permission to update this stream")
        
        result = await live_streaming_service.update_live_stream(stream_id, updates.dict(exclude_unset=True), db)
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to update live stream")


@router.delete("/{stream_id}")
async def delete_live_stream(
    stream_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Delete a live stream."""
    try:
        # Check if user owns the stream
        stream_info = await live_streaming_service.get_live_stream(stream_id, db)
        if stream_info['user_id'] != str(current_user.id):
            raise HTTPException(status_code=403, detail="You don't have permission to delete this stream")
        
        result = await live_streaming_service.delete_live_stream(stream_id, db)
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete live stream")


@router.get("/{stream_id}/analytics")
async def get_stream_analytics(
    stream_id: str,
    time_range: str = Query("1h"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get analytics for a live stream."""
    try:
        result = await live_streaming_service.get_stream_analytics(stream_id, time_range)
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get stream analytics")


@router.get("/{stream_id}/viewers")
async def get_stream_viewers(
    stream_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get current viewers of a stream."""
    try:
        viewers = await live_streaming_service.get_stream_viewers(stream_id, db)
        return {"viewers": viewers}
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get stream viewers")


@router.post("/{stream_id}/viewer/join")
async def record_viewer_join(
    stream_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Record a viewer joining the stream."""
    try:
        result = await live_streaming_service.record_viewer_join(stream_id, str(current_user.id), db)
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to record viewer join")


@router.post("/{stream_id}/viewer/leave")
async def record_viewer_leave(
    stream_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Record a viewer leaving the stream."""
    try:
        result = await live_streaming_service.record_viewer_leave(stream_id, str(current_user.id), db)
        return result
    except LiveStreamingServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to record viewer leave")
