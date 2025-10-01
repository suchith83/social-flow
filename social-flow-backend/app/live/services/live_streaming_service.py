"""
Live Streaming Service for handling live video streaming operations.

This service integrates all existing live streaming modules from
the storage/video-storage/live-streaming directory into the FastAPI application.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import json
import boto3
from botocore.exceptions import ClientError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func

from app.core.config import settings
from app.core.exceptions import LiveStreamingServiceError
from app.core.redis import get_cache
from app.live.models.live_stream import LiveStream, LiveStreamViewer, LiveStreamStatus
from app.auth.models.user import User

logger = logging.getLogger(__name__)


class LiveStreamingService:
    """Main live streaming service integrating all live streaming capabilities."""

    def __init__(self):
        self.ivs_client = None
        self.media_live_client = None
        self.cache = None
        self._initialize_services()

    async def _get_cache(self):
        """Get Redis cache instance."""
        if self.cache is None:
            self.cache = await get_cache()
        return self.cache

    def _initialize_services(self):
        """Initialize live streaming services."""
        try:
            # Initialize AWS IVS client
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY and settings.AWS_REGION:
                self.ivs_client = boto3.client(
                    'ivs',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                
                self.media_live_client = boto3.client(
                    'medialive',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                
                logger.info("AWS IVS and MediaLive clients initialized successfully")
            else:
                logger.warning("AWS credentials not fully configured. Live streaming clients not initialized.")
            
            logger.info("Live Streaming Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Live Streaming Service: {e}")
            raise LiveStreamingServiceError(f"Failed to initialize Live Streaming Service: {e}")

    async def create_live_stream(self, user_id: str, title: str, description: str = None, 
                               tags: List[str] = None, thumbnail_url: str = None, db: AsyncSession = None) -> Dict[str, Any]:
        """Create a new live stream."""
        try:
            if not self.ivs_client:
                raise LiveStreamingServiceError("IVS client not initialized")
            
            stream_id = str(uuid.uuid4())
            
            # Create IVS channel
            response = self.ivs_client.create_channel(
                name=f"{title}_{stream_id}",
                latencyMode='LOW',
                type='STANDARD',
                authorized=False,
                tags={
                    'user_id': user_id,
                    'stream_id': stream_id,
                    'title': title
                }
            )
            
            channel_arn = response['channel']['arn']
            ingest_endpoint = response['channel']['ingestEndpoint']
            stream_key = response['streamKey']['value']
            playback_url = response['channel']['playbackUrl']
            
            # Create database record if db session provided
            if db:
                live_stream = LiveStream(
                    id=uuid.UUID(stream_id),
                    owner_id=uuid.UUID(user_id),
                    title=title,
                    description=description,
                    stream_key=stream_key,
                    rtmp_url=ingest_endpoint,
                    hls_url=playback_url,
                    status=LiveStreamStatus.STARTING,
                    is_private=False,
                    viewer_count=0,
                    chat_enabled=True,
                    recording_enabled=False,
                    tags=",".join(tags) if tags else None,
                    thumbnail_url=thumbnail_url,
                    started_at=None,
                    ended_at=None,
                    duration=None,
                    total_views=0,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(live_stream)
                await db.commit()
                await db.refresh(live_stream)
            
            # Store stream info in cache for quick access
            cache = await self._get_cache()
            stream_data = {
                "stream_id": stream_id,
                "user_id": user_id,
                "title": title,
                "description": description,
                "tags": tags or [],
                "thumbnail_url": thumbnail_url,
                "channel_arn": channel_arn,
                "ingest_endpoint": ingest_endpoint,
                "stream_key": stream_key,
                "playback_url": playback_url,
                "status": "created",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await cache.setex(f"live_stream:{stream_id}", 3600, json.dumps(stream_data))
            
            return stream_data
        except ClientError as e:
            raise LiveStreamingServiceError(f"AWS IVS channel creation failed: {str(e)}")
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to create live stream: {str(e)}")

    async def start_live_stream(self, stream_id: str) -> Dict[str, Any]:
        """Start a live stream."""
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if not stream_data:
                raise LiveStreamingServiceError("Stream not found")
            
            stream_info = json.loads(stream_data)
            
            # Update stream status
            stream_info["status"] = "live"
            stream_info["started_at"] = datetime.utcnow().isoformat()
            
            await cache.setex(f"live_stream:{stream_id}", 3600, json.dumps(stream_info))
            
            return stream_info
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to start live stream: {str(e)}")

    async def stop_live_stream(self, stream_id: str) -> Dict[str, Any]:
        """Stop a live stream."""
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if not stream_data:
                raise LiveStreamingServiceError("Stream not found")
            
            stream_info = json.loads(stream_data)
            
            # Update stream status
            stream_info["status"] = "ended"
            stream_info["ended_at"] = datetime.utcnow().isoformat()
            
            await cache.setex(f"live_stream:{stream_id}", 3600, json.dumps(stream_info))
            
            return stream_info
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to stop live stream: {str(e)}")

    async def get_live_stream(self, stream_id: str, db: AsyncSession = None) -> Dict[str, Any]:
        """Get live stream information."""
        try:
            # Try cache first for performance
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if stream_data:
                return json.loads(stream_data)
            
            # Fallback to database if cache miss and db session provided
            if db:
                stmt = select(LiveStream).where(LiveStream.id == uuid.UUID(stream_id))
                result = await db.execute(stmt)
                live_stream = result.scalar_one_or_none()
                
                if live_stream:
                    return {
                        "stream_id": str(live_stream.id),
                        "user_id": str(live_stream.owner_id),
                        "title": live_stream.title,
                        "description": live_stream.description,
                        "tags": live_stream.tags.split(",") if live_stream.tags else [],
                        "thumbnail_url": live_stream.thumbnail_url,
                        "rtmp_url": live_stream.rtmp_url,
                        "hls_url": live_stream.hls_url,
                        "status": live_stream.status.value,
                        "viewer_count": live_stream.viewer_count,
                        "is_private": live_stream.is_private,
                        "chat_enabled": live_stream.chat_enabled,
                        "recording_enabled": live_stream.recording_enabled,
                        "started_at": live_stream.started_at.isoformat() if live_stream.started_at else None,
                        "ended_at": live_stream.ended_at.isoformat() if live_stream.ended_at else None,
                        "created_at": live_stream.created_at.isoformat(),
                        "updated_at": live_stream.updated_at.isoformat()
                    }
            
            raise LiveStreamingServiceError("Stream not found")
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get live stream: {str(e)}")

    async def get_user_live_streams(self, user_id: str, limit: int = 50, 
                                  offset: int = 0, db: AsyncSession = None) -> List[Dict[str, Any]]:
        """Get live streams for a user."""
        try:
            if db:
                # Use database query
                stmt = select(LiveStream).where(LiveStream.owner_id == uuid.UUID(user_id))
                stmt = stmt.order_by(LiveStream.created_at.desc()).limit(limit).offset(offset)
                result = await db.execute(stmt)
                live_streams = result.scalars().all()
                
                return [{
                    "stream_id": str(ls.id),
                    "user_id": str(ls.owner_id),
                    "title": ls.title,
                    "description": ls.description,
                    "status": ls.status.value,
                    "viewer_count": ls.viewer_count,
                    "thumbnail_url": ls.thumbnail_url,
                    "hls_url": ls.hls_url,
                    "is_private": ls.is_private,
                    "created_at": ls.created_at.isoformat(),
                    "updated_at": ls.updated_at.isoformat()
                } for ls in live_streams]
            else:
                # Fallback to placeholder data
                streams = []
                
                # Placeholder data
                for i in range(min(limit, 10)):
                    streams.append({
                        "stream_id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "title": f"Live Stream {i+1}",
                        "status": "live" if i < 5 else "ended",
                        "created_at": datetime.utcnow().isoformat()
                    })
                
                return streams
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get user live streams: {str(e)}")

    async def get_live_streams(self, limit: int = 50, offset: int = 0, 
                             status: str = "live", db: AsyncSession = None) -> List[Dict[str, Any]]:
        """Get all live streams."""
        try:
            if db:
                # Use database query
                status_enum = LiveStreamStatus(status.upper()) if status.upper() in LiveStreamStatus.__members__ else None
                stmt = select(LiveStream)
                
                if status_enum:
                    stmt = stmt.where(LiveStream.status == status_enum)
                
                stmt = stmt.order_by(LiveStream.created_at.desc()).limit(limit).offset(offset)
                result = await db.execute(stmt)
                live_streams = result.scalars().all()
                
                return [{
                    "stream_id": str(ls.id),
                    "user_id": str(ls.owner_id),
                    "title": ls.title,
                    "description": ls.description,
                    "status": ls.status.value,
                    "viewer_count": ls.viewer_count,
                    "thumbnail_url": ls.thumbnail_url,
                    "hls_url": ls.hls_url,
                    "is_private": ls.is_private,
                    "created_at": ls.created_at.isoformat(),
                    "updated_at": ls.updated_at.isoformat()
                } for ls in live_streams]
            else:
                # Fallback to placeholder data
                streams = []
                
                # Placeholder data
                for i in range(min(limit, 20)):
                    streams.append({
                        "stream_id": str(uuid.uuid4()),
                        "user_id": str(uuid.uuid4()),
                        "title": f"Live Stream {i+1}",
                        "status": status,
                        "viewer_count": i * 10,
                        "created_at": datetime.utcnow().isoformat()
                    })
                
                return streams
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get live streams: {str(e)}")

    async def update_live_stream(self, stream_id: str, updates: Dict[str, Any], db: AsyncSession = None) -> Dict[str, Any]:
        """Update live stream information."""
        try:
            # Update cache
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if stream_data:
                stream_info = json.loads(stream_data)
                
                # Update stream info
                for key, value in updates.items():
                    if key in stream_info:
                        stream_info[key] = value
                
                stream_info["updated_at"] = datetime.utcnow().isoformat()
                
                await cache.setex(f"live_stream:{stream_id}", 3600, json.dumps(stream_info))
            
            # Update database if session provided
            if db:
                stmt = select(LiveStream).where(LiveStream.id == uuid.UUID(stream_id))
                result = await db.execute(stmt)
                live_stream = result.scalar_one_or_none()
                
                if live_stream:
                    for key, value in updates.items():
                        if hasattr(live_stream, key):
                            if key == "status" and isinstance(value, str):
                                setattr(live_stream, key, LiveStreamStatus(value.upper()))
                            else:
                                setattr(live_stream, key, value)
                    
                    live_stream.updated_at = datetime.utcnow()
                    await db.commit()
                    await db.refresh(live_stream)
                    
                    return {
                        "stream_id": str(live_stream.id),
                        "user_id": str(live_stream.owner_id),
                        "title": live_stream.title,
                        "description": live_stream.description,
                        "status": live_stream.status.value,
                        "viewer_count": live_stream.viewer_count,
                        "thumbnail_url": live_stream.thumbnail_url,
                        "updated_at": live_stream.updated_at.isoformat()
                    }
            
            # Return cache data if no database update
            if stream_data:
                return json.loads(stream_data)
            
            raise LiveStreamingServiceError("Stream not found")
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to update live stream: {str(e)}")

    async def delete_live_stream(self, stream_id: str, db: AsyncSession = None) -> Dict[str, Any]:
        """Delete a live stream."""
        try:
            # Delete from cache
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if stream_data:
                await cache.delete(f"live_stream:{stream_id}")
            
            # Delete from database if session provided
            if db:
                stmt = select(LiveStream).where(LiveStream.id == uuid.UUID(stream_id))
                result = await db.execute(stmt)
                live_stream = result.scalar_one_or_none()
                
                if live_stream:
                    await db.delete(live_stream)
                    await db.commit()
                    
                    return {
                        "stream_id": stream_id,
                        "status": "deleted",
                        "deleted_at": datetime.utcnow().isoformat()
                    }
            
            # Return success even if not in database
            return {
                "stream_id": stream_id,
                "status": "deleted",
                "deleted_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to delete live stream: {str(e)}")

    async def get_stream_analytics(self, stream_id: str, time_range: str = "1h") -> Dict[str, Any]:
        """Get analytics for a live stream."""
        try:
            # TODO: Retrieve analytics from database
            return {
                "stream_id": stream_id,
                "time_range": time_range,
                "viewer_count": 0,
                "peak_viewers": 0,
                "total_views": 0,
                "average_watch_time": 0,
                "engagement_rate": 0.0,
                "generated_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get stream analytics: {str(e)}")

    async def record_viewer_join(self, stream_id: str, viewer_id: str, db: AsyncSession = None) -> Dict[str, Any]:
        """Record a viewer joining the stream."""
        try:
            if db:
                # Check if viewer is already watching
                stmt = select(LiveStreamViewer).where(
                    and_(
                        LiveStreamViewer.live_stream_id == uuid.UUID(stream_id),
                        LiveStreamViewer.user_id == uuid.UUID(viewer_id),
                        LiveStreamViewer.is_active
                    )
                )
                result = await db.execute(stmt)
                existing_viewer = result.scalar_one_or_none()
                
                if existing_viewer:
                    # Update existing record
                    existing_viewer.joined_at = datetime.utcnow()
                    existing_viewer.left_at = None
                    existing_viewer.is_active = True
                    existing_viewer.updated_at = datetime.utcnow()
                else:
                    # Create new viewer record
                    viewer = LiveStreamViewer(
                        id=uuid.uuid4(),
                        live_stream_id=uuid.UUID(stream_id),
                        user_id=uuid.UUID(viewer_id),
                        joined_at=datetime.utcnow(),
                        left_at=None,
                        watch_duration=None,
                        is_active=True,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    db.add(viewer)
                
                # Update stream viewer count
                stmt = select(LiveStream).where(LiveStream.id == uuid.UUID(stream_id))
                result = await db.execute(stmt)
                live_stream = result.scalar_one_or_none()
                
                if live_stream:
                    # Count active viewers
                    stmt = select(func.count()).select_from(LiveStreamViewer).where(
                        and_(
                            LiveStreamViewer.live_stream_id == uuid.UUID(stream_id),
                            LiveStreamViewer.is_active
                        )
                    )
                    result = await db.execute(stmt)
                    active_count = result.scalar()
                    live_stream.viewer_count = active_count or 0
                    live_stream.updated_at = datetime.utcnow()
                
                await db.commit()
            
            return {
                "stream_id": stream_id,
                "viewer_id": viewer_id,
                "joined_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to record viewer join: {str(e)}")

    async def record_viewer_leave(self, stream_id: str, viewer_id: str, db: AsyncSession = None) -> Dict[str, Any]:
        """Record a viewer leaving the stream."""
        try:
            if db:
                # Find active viewer record
                stmt = select(LiveStreamViewer).where(
                    and_(
                        LiveStreamViewer.live_stream_id == uuid.UUID(stream_id),
                        LiveStreamViewer.user_id == uuid.UUID(viewer_id),
                        LiveStreamViewer.is_active
                    )
                )
                result = await db.execute(stmt)
                viewer = result.scalar_one_or_none()
                
                if viewer:
                    # Calculate watch duration
                    watch_duration = int((datetime.utcnow() - viewer.joined_at).total_seconds())
                    
                    # Update viewer record
                    viewer.left_at = datetime.utcnow()
                    viewer.watch_duration = watch_duration
                    viewer.is_active = False
                    viewer.updated_at = datetime.utcnow()
                    
                    # Update stream viewer count
                    stmt = select(LiveStream).where(LiveStream.id == uuid.UUID(stream_id))
                    result = await db.execute(stmt)
                    live_stream = result.scalar_one_or_none()
                    
                    if live_stream:
                        # Count active viewers
                        stmt = select(func.count()).select_from(LiveStreamViewer).where(
                            and_(
                                LiveStreamViewer.live_stream_id == uuid.UUID(stream_id),
                                LiveStreamViewer.is_active
                            )
                        )
                        result = await db.execute(stmt)
                        active_count = result.scalar()
                        live_stream.viewer_count = max(0, (active_count or 0) - 1)  # Subtract 1 for leaving viewer
                        live_stream.updated_at = datetime.utcnow()
                    
                    await db.commit()
            
            return {
                "stream_id": stream_id,
                "viewer_id": viewer_id,
                "left_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to record viewer leave: {str(e)}")

    async def get_stream_viewers(self, stream_id: str, db: AsyncSession = None) -> List[Dict[str, Any]]:
        """Get current viewers of a stream."""
        try:
            if db:
                # Get active viewers from database
                stmt = select(LiveStreamViewer).where(
                    and_(
                        LiveStreamViewer.live_stream_id == uuid.UUID(stream_id),
                        LiveStreamViewer.is_active
                    )
                ).order_by(LiveStreamViewer.joined_at.desc())
                
                result = await db.execute(stmt)
                viewers = result.scalars().all()
                
                return [{
                    "viewer_id": str(v.user_id),
                    "joined_at": v.joined_at.isoformat(),
                    "watch_duration": v.watch_duration,
                    "is_active": v.is_active
                } for v in viewers]
            else:
                # Fallback to placeholder data
                viewers = []
                
                # Placeholder data
                for i in range(5):
                    viewers.append({
                        "viewer_id": str(uuid.uuid4()),
                        "username": f"viewer{i+1}",
                        "joined_at": datetime.utcnow().isoformat()
                    })
                
                return viewers
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get stream viewers: {str(e)}")


live_streaming_service = LiveStreamingService()
