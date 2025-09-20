from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import threading
import time

# Make monorepo importable when running in place
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Defensive imports for shared helpers
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

# Use ai-models recommendation stub if available
try:
    from ai_models.recommendation_engine.collaborative_filtering import load_model as load_cf  # type: ignore
except Exception:
    # try alternative path used earlier in this repo
    try:
        from ai_models.recommendation_engine.collaborative_filtering import load_model as load_cf  # type: ignore
    except Exception:
        load_cf = None  # type: ignore

from edge-computing.cache import InMemoryCache  # type: ignore
from edge-computing.sync_worker import start_sync_worker  # type: ignore

logger = get_logger("edge-service")
app = FastAPI(title="Edge Service", version="0.1.0")

# instantiate cache used by the edge service
cache = InMemoryCache(default_ttl=int(os.environ.get("EDGE_CACHE_TTL", "60")))


class RecommendationOut(BaseModel):
    item_id: str
    score: float


class FeedbackPayload(BaseModel):
    user_id: str
    item_id: str
    action: str
    timestamp: int


# Load a local model if available; otherwise use simple generator
if load_cf:
    try:
        cf_model = load_cf()
    except Exception:
        cf_model = None
else:
    cf_model = None


def _generate_recs(user_id: str, limit: int = 20):
    # deterministic fallback
    return [{"item_id": f"video_edge_{i}", "score": 1.0 / (i + 1)} for i in range(limit)]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/edge/recommendations/{user_id}")
def edge_recommendations(user_id: str, limit: int = 20):
    cache_key = f"recs:{user_id}:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("edge cache hit for %s", cache_key)
        return {"user_id": user_id, "recommendations": cached, "cached": True}
    # generate recommendations via model or fallback
    try:
        if cf_model:
            recs = cf_model.predict(user_id, limit)
        else:
            recs = _generate_recs(user_id, limit)
        cache.set(cache_key, recs)
        return {"user_id": user_id, "recommendations": recs, "cached": False}
    except Exception as exc:
        logger.exception("Failed to generate recommendations")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/edge/trending")
def edge_trending(limit: int = 20):
    cache_key = f"trending:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return {"trending": cached, "cached": True}
    # fallback trending generator
    trending = [{"item_id": f"trending_edge_{i}", "views": 1000 - i * 10} for i in range(limit)]
    cache.set(cache_key, trending)
    return {"trending": trending, "cached": False}


@app.post("/edge/feedback")
def edge_feedback(payload: FeedbackPayload):
    # Best-effort: try to publish to central broker; if not available, accept and cache a short record
    try:
        # lazy import to avoid hard dependency
        try:
            from common.libraries.python.messaging.redis_broker import RedisBroker  # type: ignore
            broker = RedisBroker()
            broker.publish("recommendation.feedback", payload.dict())
            return {"status": "forwarded"}
        except Exception:
            # fallback: store feedback in a short-lived cache for worker pickup
            local_key = f"fb_pending:{payload.user_id}:{int(time.time()*1000)}"
            cache.set(local_key, payload.dict(), ttl=30)
            return {"status": "accepted_local"}
    except Exception as exc:
        logger.exception("Failed to handle feedback")
        raise HTTPException(status_code=500, detail=str(exc))


# Start sync worker on import/startup to warm the cache; worker runs in background thread.
@app.on_event("startup")
def startup_event():
    logger.info("Starting edge sync worker")
    # start_sync_worker will start threads and return immediately
    try:
        start_sync_worker(cache=cache, logger=logger)
    except Exception:
        logger.exception("Failed to start sync worker")


@app.on_event("shutdown")
def shutdown_event():
    try:
        cache.stop()
    except Exception:
        pass
