"""
Recommendation Service

This service provides recommendations for users based on their preferences and feedback.

Routes:
  - /health:        Health check endpoint
  - /recommendations/{user_id}: Get recommendations for a user
  - /trending:      Get trending items
  - /feedback:      Record user feedback on recommendations
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os

# Add repository root to sys.path so `common` package can be imported when
# running inside the monorepo without installing the package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try to import shared helpers from the monorepo; provide lightweight fallbacks
# when running standalone (tests / CI) so the service is runnable without the
# full monorepo installed.
try:
    from common.libraries.python.monitoring.logger import get_logger
except Exception:
    import logging

    def get_logger(name: str):
        l = logging.getLogger(name)
        if not l.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            l.addHandler(handler)
        l.setLevel(logging.INFO)
        return l

try:
    from common.libraries.python.utils.config import Config
except Exception:
    # Minimal fallback Config that reads from environment vars.
    class Config:
        def get(self, key: str, default=None):
            return os.environ.get(key, default)


# Import service implementation
from src.services.inference_service import InferenceService, FeedbackMessage  # type: ignore

config = Config()
logger = get_logger("recommendation-service")

app = FastAPI(title="Recommendation Service", version="0.1.0")


class RecommendationRequest(BaseModel):
    user_id: str
    limit: int = 20


class FeedbackPayload(BaseModel):
    user_id: str
    item_id: str
    action: str  # view, like, share, dislike
    timestamp: int


# Initialize inference service (it uses guarded imports internally)
inference = InferenceService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, limit: int = 20):
    try:
        recs = inference.get_recommendations(user_id, limit)
        return {"user_id": user_id, "recommendations": recs}
    except Exception as e:
        logger.exception("Failed to get recommendations")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trending")
def get_trending(limit: int = 20):
    try:
        return {"trending": inference.get_trending(limit)}
    except Exception as e:
        logger.exception("Failed to get trending")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def record_feedback(payload: FeedbackPayload):
    try:
        # Convert the pydantic model to the internal FeedbackMessage dataclass so
        # the InferenceService receives a consistent shape.
        fb = FeedbackMessage(
            user_id=payload.user_id,
            item_id=payload.item_id,
            action=payload.action,
            timestamp=payload.timestamp,
        )
        inference.record_feedback(fb)
        return {"status": "recorded"}
    except Exception as e:
        logger.exception("Failed to record feedback")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=int(config.get("RECOMMENDER_PORT", 8003)), reload=False)
