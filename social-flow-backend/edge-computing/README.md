# Edge Computing (Lightweight)

This folder contains a minimal Edge service useful for low-latency reads and caching at the edge.

What it provides
- FastAPI edge service exposing:
  - GET /health
  - GET /edge/recommendations/{user_id}?limit=
  - GET /edge/trending?limit=
  - POST /edge/feedback
- Thread-safe in-memory cache with TTL and background cleanup.
- Sync worker that warms the cache from central services (Redis subscribe or HTTP polling).

Quick start (local)
- Create venv, install FastAPI + Uvicorn:
  python -m venv .venv && ./.venv/Scripts/Activate.ps1
  pip install fastapi uvicorn requests

- Run:
  python -m uvicorn edge-computing.edge_service:app --reload --port 8100

Notes
- The code is defensive: it will run without the full monorepo installed and falls back to simple implementations.
- For production, replace the in-memory cache with a real edge cache (Redis/Edge CDN), and run the sync worker as a separate process that subscribes to your messaging system.
