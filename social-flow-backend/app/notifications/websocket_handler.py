"""
WebSocket Notification Handler

Real-time notification delivery via WebSocket connections.
Manages user connections and broadcasts notifications.
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set, Optional
import json
import logging
from datetime import datetime

from app.auth.dependencies import get_current_user_ws
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    WebSocket connection manager.
    
    Manages active WebSocket connections for real-time notifications.
    """
    
    def __init__(self):
        # user_id -> Set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Add new WebSocket connection"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections[user_id])}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove WebSocket connection"""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            # Clean up if no connections remain
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
            
            logger.info(f"User {user_id} disconnected")
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send message to all user's connections"""
        if user_id not in self.active_connections:
            logger.debug(f"User {user_id} not connected")
            return
        
        # Send to all user connections
        disconnected = set()
        for connection in self.active_connections[user_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to connection: {e}")
                disconnected.add(connection)
        
        # Clean up failed connections
        for connection in disconnected:
            self.active_connections[user_id].discard(connection)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected users"""
        for user_id in list(self.active_connections.keys()):
            await self.send_to_user(user_id, message)
    
    def get_connected_users(self) -> Set[str]:
        """Get set of connected user IDs"""
        return set(self.active_connections.keys())
    
    def is_user_connected(self, user_id: str) -> bool:
        """Check if user has active connections"""
        return user_id in self.active_connections and len(self.active_connections[user_id]) > 0


# Global connection manager instance
manager = ConnectionManager()


async def handle_notification_websocket(
    websocket: WebSocket,
    current_user = Depends(get_current_user_ws),
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time notifications.
    
    Maintains persistent connection and delivers notifications in real-time.
    """
    
    user_id = str(current_user.id)
    
    try:
        # Accept connection
        await manager.connect(websocket, user_id)
        
        # Send initial connection success message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif message.get("type") == "mark_read":
                    # Mark notification as read
                    notification_id = message.get("notification_id")
                    if notification_id:
                        # TODO: Mark as read in database
                        await websocket.send_json({
                            "type": "notification_read",
                            "notification_id": notification_id,
                            "status": "success"
                        })
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Internal server error"
                })
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Clean up connection
        manager.disconnect(websocket, user_id)


async def broadcast_notification(user_id: str, notification: dict):
    """
    Broadcast notification to user's WebSocket connections.
    
    Args:
        user_id: Target user ID
        notification: Notification data to send
    """
    
    message = {
        "type": "notification",
        "data": notification,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.send_to_user(user_id, message)


async def broadcast_to_all(notification: dict):
    """
    Broadcast notification to all connected users.
    
    Args:
        notification: Notification data to send
    """
    
    message = {
        "type": "announcement",
        "data": notification,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.broadcast(message)


def get_connection_manager() -> ConnectionManager:
    """Get global connection manager instance"""
    return manager
