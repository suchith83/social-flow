"""
Live Streaming Service for RTMP/WebRTC streaming.

Handles stream key generation, RTMP ingest, WebRTC signaling, and real-time chat.
"""

import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.redis import get_cache
from app.models.live_stream import LiveStream, StreamStatus

logger = logging.getLogger(__name__)


class LiveStreamingService:
    """
    Service for live streaming operations.
    
    Features:
    - Stream key generation and validation
    - RTMP ingest URL generation
    - WebRTC signaling via WebSocket
    - Real-time chat with Redis pub/sub
    - Viewer count tracking
    - Stream recording to S3
    """
    
    def __init__(self):
        """Initialize live streaming service."""
        self.cache = None
    
    async def _get_cache(self):
        """Get Redis cache instance."""
        if self.cache is None:
            self.cache = await get_cache()
        return self.cache
    
    def _generate_stream_key(self, user_id: str) -> str:
        """
        Generate a unique stream key for user.
        
        Args:
            user_id: User ID
        
        Returns:
            Stream key string
        """
        # Generate secure random token
        random_part = secrets.token_urlsafe(32)
        
        # Combine with user ID and hash
        combined = f"{user_id}:{random_part}:{datetime.utcnow().isoformat()}"
        stream_key = hashlib.sha256(combined.encode()).hexdigest()[:32]
        
        return stream_key
    
    async def create_stream(
        self,
        db: AsyncSession,
        user_id: str,
        title: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        is_private: bool = False,
        record_stream: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new live stream session.
        
        Args:
            db: Database session
            user_id: User ID
            title: Stream title
            description: Stream description
            category: Stream category
            is_private: Whether stream is private
            record_stream: Whether to record stream to S3
        
        Returns:
            Dict with stream details including RTMP URL and stream key
        """
        try:
            # Check if user already has an active stream
            result = await db.execute(
                select(LiveStream)
                .where(
                    LiveStream.user_id == user_id,
                    LiveStream.status.in_([StreamStatus.ACTIVE, StreamStatus.STARTING])
                )
            )
            existing_stream = result.scalar_one_or_none()
            
            if existing_stream:
                raise ValueError("User already has an active stream")
            
            # Generate stream key
            stream_key = self._generate_stream_key(user_id)
            stream_id = str(uuid.uuid4())
            
            # Create stream record
            live_stream = LiveStream(
                id=stream_id,
                user_id=user_id,
                title=title,
                description=description,
                category=category,
                stream_key=stream_key,
                is_private=is_private,
                record_stream=record_stream,
                status=StreamStatus.STARTING,
                created_at=datetime.utcnow()
            )
            
            db.add(live_stream)
            await db.commit()
            await db.refresh(live_stream)
            
            # Store stream info in Redis for quick access
            cache = await self._get_cache()
            await cache.setex(
                f"stream:{stream_key}",
                3600 * 6,  # 6 hours expiry
                json.dumps({
                    'stream_id': stream_id,
                    'user_id': user_id,
                    'title': title,
                    'status': StreamStatus.STARTING.value,
                    'created_at': live_stream.created_at.isoformat()
                })
            )
            
            # Generate URLs
            rtmp_url = f"{settings.RTMP_INGEST_URL}/{stream_key}"
            rtmps_url = f"{settings.RTMPS_INGEST_URL}/{stream_key}" if settings.RTMPS_INGEST_URL else None
            playback_hls_url = f"{settings.HLS_PLAYBACK_URL}/{stream_key}/index.m3u8"
            playback_dash_url = f"{settings.DASH_PLAYBACK_URL}/{stream_key}/manifest.mpd"
            webrtc_url = f"{settings.WEBRTC_SIGNALING_URL}/stream/{stream_id}"
            
            logger.info(f"Live stream {stream_id} created for user {user_id}")
            
            return {
                'stream_id': stream_id,
                'stream_key': stream_key,
                'title': title,
                'status': StreamStatus.STARTING.value,
                'ingest_urls': {
                    'rtmp': rtmp_url,
                    'rtmps': rtmps_url
                },
                'playback_urls': {
                    'hls': playback_hls_url,
                    'dash': playback_dash_url,
                    'webrtc': webrtc_url
                },
                'chat_channel': f"chat:{stream_id}",
                'record_stream': record_stream,
                'created_at': live_stream.created_at.isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to create live stream: {e}")
            raise
    
    async def start_stream(
        self,
        db: AsyncSession,
        stream_id: str
    ) -> Dict[str, Any]:
        """
        Mark stream as active when streaming starts.
        
        Args:
            db: Database session
            stream_id: Stream ID
        
        Returns:
            Dict with updated stream status
        """
        try:
            # Update stream status
            result = await db.execute(
                select(LiveStream).where(LiveStream.id == stream_id)
            )
            stream = result.scalar_one_or_none()
            
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")
            
            stream.status = StreamStatus.ACTIVE
            stream.started_at = datetime.utcnow()
            await db.commit()
            
            # Update Redis cache
            cache = await self._get_cache()
            stream_data = await cache.get(f"stream:{stream.stream_key}")
            if stream_data:
                data = json.loads(stream_data)
                data['status'] = StreamStatus.ACTIVE.value
                data['started_at'] = stream.started_at.isoformat()
                await cache.setex(
                    f"stream:{stream.stream_key}",
                    3600 * 6,
                    json.dumps(data)
                )
            
            # Initialize viewer count
            await cache.set(f"stream_viewers:{stream_id}", 0)
            
            # Publish stream started event
            await cache.publish(
                f"stream_events:{stream_id}",
                json.dumps({
                    'event': 'stream_started',
                    'stream_id': stream_id,
                    'timestamp': datetime.utcnow().isoformat()
                })
            )
            
            logger.info(f"Live stream {stream_id} started")
            
            return {
                'stream_id': stream_id,
                'status': StreamStatus.ACTIVE.value,
                'started_at': stream.started_at.isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to start stream {stream_id}: {e}")
            raise
    
    async def end_stream(
        self,
        db: AsyncSession,
        stream_id: str
    ) -> Dict[str, Any]:
        """
        End a live stream session.
        
        Args:
            db: Database session
            stream_id: Stream ID
        
        Returns:
            Dict with stream statistics
        """
        try:
            # Get stream
            result = await db.execute(
                select(LiveStream).where(LiveStream.id == stream_id)
            )
            stream = result.scalar_one_or_none()
            
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")
            
            # Update stream status
            stream.status = StreamStatus.ENDED
            stream.ended_at = datetime.utcnow()
            
            # Calculate duration
            if stream.started_at:
                stream.duration = int((stream.ended_at - stream.started_at).total_seconds())
            
            # Get final viewer stats from Redis
            cache = await self._get_cache()
            peak_viewers = await cache.get(f"stream_peak_viewers:{stream_id}")
            total_viewers = await cache.get(f"stream_total_viewers:{stream_id}")
            
            stream.peak_viewers = int(peak_viewers) if peak_viewers else 0
            stream.total_viewers = int(total_viewers) if total_viewers else 0
            
            await db.commit()
            
            # Clean up Redis data
            await cache.delete(f"stream:{stream.stream_key}")
            await cache.delete(f"stream_viewers:{stream_id}")
            await cache.delete(f"stream_peak_viewers:{stream_id}")
            await cache.delete(f"stream_total_viewers:{stream_id}")
            
            # Publish stream ended event
            await cache.publish(
                f"stream_events:{stream_id}",
                json.dumps({
                    'event': 'stream_ended',
                    'stream_id': stream_id,
                    'duration': stream.duration,
                    'peak_viewers': stream.peak_viewers,
                    'timestamp': datetime.utcnow().isoformat()
                })
            )
            
            logger.info(f"Live stream {stream_id} ended. Duration: {stream.duration}s, Peak viewers: {stream.peak_viewers}")
            
            return {
                'stream_id': stream_id,
                'status': StreamStatus.ENDED.value,
                'duration': stream.duration,
                'peak_viewers': stream.peak_viewers,
                'total_viewers': stream.total_viewers,
                'ended_at': stream.ended_at.isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to end stream {stream_id}: {e}")
            raise
    
    async def validate_stream_key(
        self,
        stream_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate stream key and return stream info.
        
        Args:
            stream_key: Stream key to validate
        
        Returns:
            Dict with stream info if valid, None otherwise
        """
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"stream:{stream_key}")
            
            if not stream_data:
                logger.warning(f"Invalid stream key: {stream_key}")
                return None
            
            return json.loads(stream_data)
        
        except Exception as e:
            logger.error(f"Failed to validate stream key: {e}")
            return None
    
    async def get_stream_info(
        self,
        db: AsyncSession,
        stream_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed stream information.
        
        Args:
            db: Database session
            stream_id: Stream ID
        
        Returns:
            Dict with stream details
        """
        try:
            result = await db.execute(
                select(LiveStream).where(LiveStream.id == stream_id)
            )
            stream = result.scalar_one_or_none()
            
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")
            
            # Get current viewer count from Redis if stream is active
            current_viewers = 0
            if stream.status == StreamStatus.ACTIVE:
                cache = await self._get_cache()
                viewers = await cache.get(f"stream_viewers:{stream_id}")
                current_viewers = int(viewers) if viewers else 0
            
            return {
                'stream_id': stream_id,
                'user_id': str(stream.user_id),
                'title': stream.title,
                'description': stream.description,
                'category': stream.category,
                'status': stream.status.value,
                'is_private': stream.is_private,
                'current_viewers': current_viewers,
                'peak_viewers': stream.peak_viewers,
                'total_viewers': stream.total_viewers,
                'duration': stream.duration,
                'created_at': stream.created_at.isoformat(),
                'started_at': stream.started_at.isoformat() if stream.started_at else None,
                'ended_at': stream.ended_at.isoformat() if stream.ended_at else None
            }
        
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            raise
    
    async def join_stream(
        self,
        stream_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        User joins a stream as viewer.
        
        Args:
            stream_id: Stream ID
            user_id: User ID
        
        Returns:
            Dict with join status
        """
        try:
            cache = await self._get_cache()
            
            # Increment viewer count
            current_viewers = await cache.incr(f"stream_viewers:{stream_id}")
            
            # Update peak viewers if necessary
            peak_viewers = await cache.get(f"stream_peak_viewers:{stream_id}")
            if not peak_viewers or current_viewers > int(peak_viewers):
                await cache.set(f"stream_peak_viewers:{stream_id}", current_viewers)
            
            # Increment total unique viewers
            await cache.sadd(f"stream_unique_viewers:{stream_id}", user_id)
            total_viewers = await cache.scard(f"stream_unique_viewers:{stream_id}")
            await cache.set(f"stream_total_viewers:{stream_id}", total_viewers)
            
            # Publish viewer joined event
            await cache.publish(
                f"stream_events:{stream_id}",
                json.dumps({
                    'event': 'viewer_joined',
                    'stream_id': stream_id,
                    'user_id': user_id,
                    'current_viewers': current_viewers,
                    'timestamp': datetime.utcnow().isoformat()
                })
            )
            
            logger.info(f"User {user_id} joined stream {stream_id}. Current viewers: {current_viewers}")
            
            return {
                'stream_id': stream_id,
                'current_viewers': current_viewers,
                'joined': True
            }
        
        except Exception as e:
            logger.error(f"Failed to join stream: {e}")
            raise
    
    async def leave_stream(
        self,
        stream_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        User leaves a stream.
        
        Args:
            stream_id: Stream ID
            user_id: User ID
        
        Returns:
            Dict with leave status
        """
        try:
            cache = await self._get_cache()
            
            # Decrement viewer count
            current_viewers = await cache.decr(f"stream_viewers:{stream_id}")
            if current_viewers < 0:
                await cache.set(f"stream_viewers:{stream_id}", 0)
                current_viewers = 0
            
            # Publish viewer left event
            await cache.publish(
                f"stream_events:{stream_id}",
                json.dumps({
                    'event': 'viewer_left',
                    'stream_id': stream_id,
                    'user_id': user_id,
                    'current_viewers': current_viewers,
                    'timestamp': datetime.utcnow().isoformat()
                })
            )
            
            logger.info(f"User {user_id} left stream {stream_id}. Current viewers: {current_viewers}")
            
            return {
                'stream_id': stream_id,
                'current_viewers': current_viewers,
                'left': True
            }
        
        except Exception as e:
            logger.error(f"Failed to leave stream: {e}")
            raise
    
    async def send_chat_message(
        self,
        stream_id: str,
        user_id: str,
        username: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Send chat message to stream.
        
        Args:
            stream_id: Stream ID
            user_id: User ID
            username: Username
            message: Chat message
        
        Returns:
            Dict with message details
        """
        try:
            cache = await self._get_cache()
            
            message_id = str(uuid.uuid4())
            chat_data = {
                'message_id': message_id,
                'stream_id': stream_id,
                'user_id': user_id,
                'username': username,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Publish to chat channel
            await cache.publish(
                f"chat:{stream_id}",
                json.dumps(chat_data)
            )
            
            # Store in chat history (limited to last 100 messages)
            await cache.lpush(f"chat_history:{stream_id}", json.dumps(chat_data))
            await cache.ltrim(f"chat_history:{stream_id}", 0, 99)
            
            logger.info(f"Chat message sent to stream {stream_id} by {username}")
            
            return chat_data
        
        except Exception as e:
            logger.error(f"Failed to send chat message: {e}")
            raise
    
    async def get_chat_history(
        self,
        stream_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for stream.
        
        Args:
            stream_id: Stream ID
            limit: Maximum messages to return
        
        Returns:
            List of chat messages
        """
        try:
            cache = await self._get_cache()
            
            # Get messages from Redis
            messages = await cache.lrange(f"chat_history:{stream_id}", 0, limit - 1)
            
            chat_history = []
            for msg in messages:
                if msg:
                    chat_history.append(json.loads(msg))
            
            return chat_history
        
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            raise
    
    async def list_active_streams(
        self,
        db: AsyncSession,
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List active live streams.
        
        Args:
            db: Database session
            category: Filter by category
            limit: Maximum streams to return
            offset: Offset for pagination
        
        Returns:
            List of active streams
        """
        try:
            query = select(LiveStream).where(
                LiveStream.status == StreamStatus.ACTIVE,
                LiveStream.is_private == False
            )
            
            if category:
                query = query.where(LiveStream.category == category)
            
            query = query.order_by(LiveStream.current_viewers.desc())
            query = query.limit(limit).offset(offset)
            
            result = await db.execute(query)
            streams = result.scalars().all()
            
            # Get current viewer counts from Redis
            cache = await self._get_cache()
            stream_list = []
            
            for stream in streams:
                viewers = await cache.get(f"stream_viewers:{stream.id}")
                current_viewers = int(viewers) if viewers else 0
                
                stream_list.append({
                    'stream_id': str(stream.id),
                    'user_id': str(stream.user_id),
                    'title': stream.title,
                    'category': stream.category,
                    'current_viewers': current_viewers,
                    'started_at': stream.started_at.isoformat() if stream.started_at else None,
                    'thumbnail_url': stream.thumbnail_url
                })
            
            return stream_list
        
        except Exception as e:
            logger.error(f"Failed to list active streams: {e}")
            raise


# Singleton instance
_streaming_service = None


def get_streaming_service() -> LiveStreamingService:
    """Get singleton streaming service instance."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = LiveStreamingService()
    return _streaming_service
