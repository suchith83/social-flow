"""
WebSocket Chat Handler

Real-time chat functionality for live streams using WebSockets.
Handles message broadcasting, viewer presence, and chat moderation.
"""

import json
from datetime import datetime
from typing import Dict, Set, Optional, List
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.livestream.services.stream_service import LiveStreamService
from app.models.livestream import ChatMessageType
from app.core.logging import get_logger
from app.core.redis import get_redis

logger = get_logger(__name__)


class ConnectionManager:
    """
    WebSocket connection manager for live stream chat
    
    Features:
    - Real-time message broadcasting
    - Viewer presence tracking
    - Chat room management
    - Redis pub/sub for horizontal scaling
    """
    
    def __init__(self):
        # Active connections: stream_id -> {session_id: WebSocket}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        
        # Redis client for pub/sub (async; may be None in tests)
        self.redis_client = None
    
    async def connect(
        self,
        websocket: WebSocket,
        stream_id: str,
        session_id: str
    ):
        """
        Connect a client to a stream chat room
        
        Args:
            websocket: WebSocket connection
            stream_id: Stream ID
            session_id: Unique session identifier
        """
        await websocket.accept()
        
        # Add to connections
        if stream_id not in self.active_connections:
            self.active_connections[stream_id] = {}
        
        self.active_connections[stream_id][session_id] = websocket
        
        # Publish presence event to Redis
        await self._publish_event(stream_id, {
            "type": "viewer_joined",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Client {session_id} connected to stream {stream_id}")
    
    async def disconnect(self, stream_id: str, session_id: str):
        """
        Disconnect a client from a stream chat room
        
        Args:
            stream_id: Stream ID
            session_id: Session identifier
        """
        if stream_id in self.active_connections:
            self.active_connections[stream_id].pop(session_id, None)
            
            # Clean up empty rooms
            if not self.active_connections[stream_id]:
                del self.active_connections[stream_id]
        
        # Publish presence event to Redis
        await self._publish_event(stream_id, {
            "type": "viewer_left",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Client {session_id} disconnected from stream {stream_id}")
    
    async def send_personal_message(
        self,
        message: str,
        stream_id: str,
        session_id: str
    ):
        """Send a message to a specific client"""
        if stream_id in self.active_connections:
            if session_id in self.active_connections[stream_id]:
                websocket = self.active_connections[stream_id][session_id]
                await websocket.send_text(message)
    
    async def broadcast_to_stream(
        self,
        message: str,
        stream_id: str,
        exclude_session: Optional[str] = None
    ):
        """
        Broadcast a message to all clients in a stream
        
        Args:
            message: Message to broadcast
            stream_id: Stream ID
            exclude_session: Optional session ID to exclude
        """
        if stream_id not in self.active_connections:
            return
        
        # Send to all connected clients
        disconnected = []
        for session_id, websocket in self.active_connections[stream_id].items():
            if session_id == exclude_session:
                continue
            
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending to {session_id}: {e}")
                disconnected.append(session_id)
        
        # Clean up disconnected clients
        for session_id in disconnected:
            await self.disconnect(stream_id, session_id)
    
    async def broadcast_chat_message(
        self,
        stream_id: str,
        user_id: str,
        username: str,
        content: str,
        message_type: str = "message",
        metadata: Optional[Dict] = None
    ):
        """
        Broadcast a chat message to all viewers
        
        Args:
            stream_id: Stream ID
            user_id: User ID
            username: Username
            content: Message content
            message_type: Type of message
            metadata: Additional metadata
        """
        message_data = {
            "type": "chat_message",
            "stream_id": stream_id,
            "user_id": user_id,
            "username": username,
            "content": content,
            "message_type": message_type,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        message_json = json.dumps(message_data)
        await self.broadcast_to_stream(message_json, stream_id)
        
        # Publish to Redis for other servers
        await self._publish_event(stream_id, message_data)
    
    async def broadcast_system_message(
        self,
        stream_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """Broadcast a system message"""
        message_data = {
            "type": "system_message",
            "stream_id": stream_id,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        message_json = json.dumps(message_data)
        await self.broadcast_to_stream(message_json, stream_id)
    
    async def broadcast_viewer_count(
        self,
        stream_id: str,
        viewer_count: int
    ):
        """Broadcast updated viewer count"""
        message_data = {
            "type": "viewer_count",
            "stream_id": stream_id,
            "count": viewer_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_json = json.dumps(message_data)
        await self.broadcast_to_stream(message_json, stream_id)
    
    async def send_moderation_action(
        self,
        stream_id: str,
        action: str,
        target_user_id: str,
        reason: Optional[str] = None
    ):
        """Broadcast a moderation action"""
        message_data = {
            "type": "moderation_action",
            "stream_id": stream_id,
            "action": action,
            "target_user_id": target_user_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_json = json.dumps(message_data)
        await self.broadcast_to_stream(message_json, stream_id)
    
    def get_stream_viewer_count(self, stream_id: str) -> int:
        """Get current viewer count for a stream"""
        if stream_id not in self.active_connections:
            return 0
        return len(self.active_connections[stream_id])
    
    def get_all_streams(self) -> List[str]:
        """Get all active stream IDs"""
        return list(self.active_connections.keys())
    
    async def _publish_event(self, stream_id: str, event: Dict):
        """Publish event to Redis pub/sub for horizontal scaling"""
        try:
            channel = f"stream:{stream_id}:events"
            # Lazily acquire redis client
            if self.redis_client is None:
                self.redis_client = await get_redis()
            # In test mode, redis client may be None
            if self.redis_client is not None:
                await self.redis_client.publish(channel, json.dumps(event))
        except Exception as e:
            logger.error(f"Error publishing to Redis: {e}")


# Global connection manager instance
manager = ConnectionManager()


async def handle_websocket_chat(
    websocket: WebSocket,
    stream_id: UUID,
    session_id: str,
    user_id: Optional[UUID],
    db: AsyncSession
):
    """
    Handle WebSocket connection for stream chat
    
    Args:
        websocket: WebSocket connection
        stream_id: Stream ID
        session_id: Unique session identifier
        user_id: User ID (None for anonymous)
        db: Database session
    """
    stream_id_str = str(stream_id)
    stream_service = LiveStreamService(db)
    
    # Connect to chat room
    await manager.connect(websocket, stream_id_str, session_id)
    
    try:
        # Add viewer to stream
        if user_id:
            await stream_service.add_viewer(
                stream_id=stream_id,
                user_id=user_id,
                session_id=session_id
            )
        
        # Send welcome message
        await manager.send_personal_message(
            json.dumps({
                "type": "connected",
                "stream_id": stream_id_str,
                "message": "Connected to live chat"
            }),
            stream_id_str,
            session_id
        )
        
        # Broadcast viewer count update
        viewer_count = manager.get_stream_viewer_count(stream_id_str)
        await manager.broadcast_viewer_count(stream_id_str, viewer_count)
        
        # Message loop
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type")
            
            if message_type == "chat_message":
                # Handle chat message
                content = message_data.get("content", "")
                
                if not content.strip():
                    continue
                
                if not user_id:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": "Must be logged in to send messages"
                        }),
                        stream_id_str,
                        session_id
                    )
                    continue
                
                # Save message to database
                chat_message = await stream_service.send_chat_message(
                    stream_id=stream_id,
                    user_id=user_id,
                    content=content,
                    message_type=ChatMessageType.MESSAGE
                )
                
                # Broadcast to all viewers
                username = message_data.get("username", "User")
                await manager.broadcast_chat_message(
                    stream_id=stream_id_str,
                    user_id=str(user_id),
                    username=username,
                    content=content,
                    metadata={"message_id": str(chat_message.id)}
                )
            
            elif message_type == "heartbeat":
                # Update viewer heartbeat
                if user_id:
                    await stream_service.update_viewer_heartbeat(
                        stream_id=stream_id,
                        session_id=session_id
                    )
                
                # Send heartbeat acknowledgment
                await manager.send_personal_message(
                    json.dumps({
                        "type": "heartbeat_ack",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    stream_id_str,
                    session_id
                )
            
            elif message_type == "typing":
                # Broadcast typing indicator
                username = message_data.get("username", "User")
                await manager.broadcast_to_stream(
                    json.dumps({
                        "type": "typing",
                        "user_id": str(user_id) if user_id else None,
                        "username": username
                    }),
                    stream_id_str,
                    exclude_session=session_id
                )
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
    
    finally:
        # Disconnect from chat room
        await manager.disconnect(stream_id_str, session_id)
        
        # Remove viewer from stream
        if user_id:
            await stream_service.remove_viewer(
                stream_id=stream_id,
                session_id=session_id
            )
        
        # Broadcast viewer count update
        viewer_count = manager.get_stream_viewer_count(stream_id_str)
        await manager.broadcast_viewer_count(stream_id_str, viewer_count)
