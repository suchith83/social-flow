"""
WebRTC signaling handler for peer connections
"""

import json
import asyncio
import websockets
from .config import config
from .utils import logger


class WebRTCSignaling:
    def __init__(self):
        self.peers = {}

    async def handler(self, websocket, path):
        async for msg in websocket:
            data = json.loads(msg)
            if data["type"] == "offer":
                logger.info("Received WebRTC offer")
                await websocket.send(json.dumps({"type": "answer", "sdp": "fake-sdp"}))
            elif data["type"] == "candidate":
                logger.info("Received ICE candidate")


async def run_signaling():
    ws = WebRTCSignaling()
    server = await websockets.serve(ws.handler, config.WS_HOST, config.WS_PORT)
    logger.info(f"Signaling running on ws://{config.WS_HOST}:{config.WS_PORT}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(run_signaling())
