# Analytics Service (Lightweight)

This directory contains a minimal analytics microservice useful for development and testing.

Run locally:
- Create a virtualenv and install FastAPI + Uvicorn:
  python -m venv .venv && .\.venv\Scripts\Activate.ps1
  pip install fastapi uvicorn

- Start service:
  python -m uvicorn src.main:app --port 8010

Endpoints:
- GET /health
- POST /ingest (body: {name, value, tags, timestamp})
- GET /metrics?name=...&start_ts=...&end_ts=...&limit=...

Notes:
- The service uses a simple in-memory store. For production replace with InfluxDB/TimescaleDB/Prometheus remote_write ingestion.
