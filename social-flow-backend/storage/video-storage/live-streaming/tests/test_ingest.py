import pytest
import asyncio
from live_streaming.ingest import StreamIngest


@pytest.mark.asyncio
async def test_ingest():
    ingest = StreamIngest()
    await ingest.handle_rtmp("s1", "u1", "test stream")
    assert "s1" in ingest.active_streams
