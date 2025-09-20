import time
import uuid
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class Stream:
    id: str
    title: str
    uploader_id: str
    status: str  # "idle", "live", "stopped"
    ingest_url: str
    playback_url: str
    started_at: Optional[float]
    created_at: float


class StreamManager:
    """Simple in-memory manager for live streams. Replace with DB + external control in prod."""

    def __init__(self, host: str = "localhost", rtmp_port: int = 1935, hls_base: str = "https://cdn.example.com"):
        self._store: Dict[str, Stream] = {}
        self.host = host
        self.rtmp_port = rtmp_port
        self.hls_base = hls_base

    def create_stream(self, title: str, uploader_id: str) -> Stream:
        sid = f"stream_{uuid.uuid4().hex[:12]}"
        ingest_url = f"rtmp://{self.host}:{self.rtmp_port}/{uploader_id}/{sid}"
        playback_url = f"{self.hls_base}/{sid}/playlist.m3u8"
        now = time.time()
        s = Stream(
            id=sid,
            title=title,
            uploader_id=uploader_id,
            status="idle",
            ingest_url=ingest_url,
            playback_url=playback_url,
            started_at=None,
            created_at=now,
        )
        self._store[sid] = s
        return s

    def start_stream(self, stream_id: str) -> Optional[Stream]:
        s = self._store.get(stream_id)
        if not s:
            return None
        s.status = "live"
        s.started_at = time.time()
        return s

    def stop_stream(self, stream_id: str) -> Optional[Stream]:
        s = self._store.get(stream_id)
        if not s:
            return None
        s.status = "stopped"
        return s

    def get_stream(self, stream_id: str) -> Optional[Stream]:
        return self._store.get(stream_id)

    def list_streams(self, status: Optional[str] = None) -> List[Dict]:
        streams = list(self._store.values())
        if status:
            streams = [s for s in streams if s.status == status]
        return [asdict(s) for s in streams]
