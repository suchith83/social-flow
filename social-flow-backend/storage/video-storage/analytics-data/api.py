"""
API layer for analytics-data
Provides endpoints for querying video analytics.
"""

from fastapi import FastAPI
from typing import Dict
from .storage_backend import AnalyticsStorage
from .utils import logger

app = FastAPI(title="Analytics Data API")
storage = AnalyticsStorage()


@app.get("/metrics/{video_id}")
def get_metrics(video_id: str) -> Dict:
    try:
        with storage.Session() as s:
            res = s.get(storage.Session().query(storage.engine.table_names()), video_id)
            if not res:
                return {"error": "Video not found"}
            return dict(res)
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        return {"error": "Internal Server Error"}
