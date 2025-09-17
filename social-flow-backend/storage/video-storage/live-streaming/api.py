"""
REST API for managing live streaming
"""

from fastapi import FastAPI
import datetime
from typing import Dict

from .ingest import StreamIngest
from .transcoder import Transcoder
from .packager import Packager
from .distributor import Distributor

app = FastAPI(title="Live Streaming API")

ingest = StreamIngest()
transcoder = Transcoder()
packager = Packager()
distributor = Distributor()


@app.post("/start-stream/{stream_id}")
async def start_stream(stream_id: str, user_id: str, title: str):
    await ingest.handle_rtmp(stream_id, user_id, title)
    return {"status": "started", "stream_id": stream_id}


@app.post("/stop-stream/{stream_id}")
async def stop_stream(stream_id: str):
    if stream_id in ingest.active_streams:
        ingest.active_streams[stream_id].status = "ended"
        ingest.active_streams[stream_id].ended_at = datetime.datetime.utcnow()
        return {"status": "stopped", "stream_id": stream_id}
    return {"error": "Stream not found"}


@app.get("/stream/{stream_id}/playlist")
def get_hls_playlist(stream_id: str) -> Dict:
    return {"hls": packager.get_hls_playlist(stream_id)}


@app.get("/stream/{stream_id}/manifest")
def get_dash_manifest(stream_id: str) -> Dict:
    return {"dash": packager.get_dash_manifest(stream_id)}


@app.get("/stream/{stream_id}/url")
def get_stream_url(stream_id: str, protocol: str = "hls") -> Dict:
    return {"url": distributor.get_stream_url(stream_id, protocol)}
