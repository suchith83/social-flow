"""
RTMP/WebRTC stream ingestion service.
"""

import asyncio
import websockets
from .models import LiveStreamSession
from .utils import logger
from .config import config
import datetime


class StreamIngest:
    def __init__(self):
        self.active_streams = {}

    async def handle_rtmp(self, stream_id: str, user_id: str, title: str):
        """Register new RTMP live session"""
        session = LiveStreamSession(
            stream_id=stream_id,
            user_id=user_id,
            title=title,
            started_at=datetime.datetime.utcnow(),
            status="active",
        )
        self.active_streams[stream_id] = session
        logger.info(f"RTMP stream started: {stream_id} by {user_id}")

    async def handle_webrtc(self, websocket, path):
        """WebRTC signaling channel"""
        async for msg in websocket:
            logger.info(f"Received signaling: {msg}")
            await websocket.send("ACK: " + msg)


async def start_signaling_server():
    ingest = StreamIngest()
    server = await websockets.serve(
        ingest.handle_webrtc, config.WS_HOST, config.WS_PORT
    )
    logger.info(f"WebRTC signaling server running at ws://{config.WS_HOST}:{config.WS_PORT}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(start_signaling_server())
