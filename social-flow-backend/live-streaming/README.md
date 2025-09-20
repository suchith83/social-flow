# Live Streaming (Lightweight)

This package provides a small control API and WebSocket chat for live streaming used for local dev and prototypes.

Features
- REST endpoints to create/start/stop streams and query stream state.
- WebSocket endpoint for simple live chat per stream.
- In-memory StreamManager suitable for local dev; replace with persistent store and real ingest/playback URLs in production.
- encoder_worker skeleton to simulate encoding / state transitions.

Run locally:
- Create venv, install FastAPI & Uvicorn:
  python -m venv .venv
  . .venv/bin/activate
  pip install fastapi uvicorn

- Start the service:
  python -m uvicorn live_streaming.src.main:app --port 8200

Notes:
- This implementation produces placeholder RTMP ingest and HLS playback URLs. Integrate with AWS MediaLive/IVS or ffmpeg pipelines in production.
