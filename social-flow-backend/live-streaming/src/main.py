from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import sys
import asyncio

# make repo importable when running in-place
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Defensive logger fallback
try:
    from common.libraries.python.monitoring.logger import get_logger
except Exception:
    import logging

    def get_logger(name: str):
        l = logging.getLogger(name)
        if not l.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            l.addHandler(h)
        l.setLevel(logging.INFO)
        return l

from live_streaming.src.manager import StreamManager  # type: ignore
from live_streaming.src.worker.encoder_worker import start_encoder_worker  # type: ignore

logger = get_logger("live-streaming-service")
app = FastAPI(title="Live Streaming Service", version="0.1.0")
router = APIRouter(prefix="/streams")

# instantiate a manager with defaults (override via env vars)
manager = StreamManager(
    host=os.environ.get("LIVE_RTMP_HOST", "localhost"),
    rtmp_port=int(os.environ.get("LIVE_RTMP_PORT", "1935")),
    hls_base=os.environ.get("LIVE_HLS_BASE", "https://cdn.example.com"),
)

# basic in-memory pubsub for websocket chat per stream
_chat_connections: Dict[str, List[WebSocket]] = {}


class CreateStreamPayload(BaseModel):
    title: str = Field(..., description="Stream title")
    uploader_id: str = Field(..., description="Uploader user id")


class StreamOut(BaseModel):
    id: str
    title: str
    uploader_id: str
    status: str
    ingest_url: str
    playback_url: str
    started_at: float = None
    created_at: float


@router.post("/", response_model=StreamOut, status_code=201)
def create_stream(payload: CreateStreamPayload):
    s = manager.create_stream(payload.title, payload.uploader_id)
    logger.info("Created stream %s by %s", s.id, s.uploader_id)
    return s


@router.post("/{stream_id}/start", response_model=StreamOut)
def start_stream(stream_id: str):
    s = manager.start_stream(stream_id)
    if not s:
        raise HTTPException(status_code=404, detail="stream not found")
    logger.info("Started stream %s", stream_id)
    return s


@router.post("/{stream_id}/stop", response_model=StreamOut)
def stop_stream(stream_id: str):
    s = manager.stop_stream(stream_id)
    if not s:
        raise HTTPException(status_code=404, detail="stream not found")
    logger.info("Stopped stream %s", stream_id)
    return s


@router.get("/{stream_id}", response_model=StreamOut)
def get_stream(stream_id: str):
    s = manager.get_stream(stream_id)
    if not s:
        raise HTTPException(status_code=404, detail="stream not found")
    return s


@router.get("/", response_model=List[Dict[str, Any]])
def list_streams(status: str = None):
    return manager.list_streams(status)


app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/{stream_id}")
async def websocket_chat(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    conns = _chat_connections.setdefault(stream_id, [])
    conns.append(websocket)
    logger.info("WebSocket connected for stream=%s, connections=%d", stream_id, len(conns))
    try:
        while True:
            data = await websocket.receive_text()
            # broadcast to all connections for this stream
            for ws in list(conns):
                try:
                    await ws.send_text(data)
                except Exception:
                    # ignore individual send errors
                    pass
    except WebSocketDisconnect:
        conns.remove(websocket)
        logger.info("WebSocket disconnected for stream=%s, connections=%d", stream_id, len(conns))


@app.on_event("startup")
def startup_event():
    logger.info("Starting encoder worker for live-streaming (background)")
    try:
        start_encoder_worker(manager=manager, logger=logger)
    except Exception:
        logger.exception("Failed to start encoder worker")
