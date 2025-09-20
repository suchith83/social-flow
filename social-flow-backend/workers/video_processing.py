"""Video processing worker: listens for uploaded videos and runs encoding simulation."""

from typing import Dict, Any
import time
import threading

from workers.common import get_broker, get_db, logger

BROKER = get_broker()
DB = get_db()


def handle_video_upload(message: Dict[str, Any]) -> None:
    logger.info("Received video.uploaded message: %s", message)
    try:
        video_id = message.get("video_id") or message.get("key")
        # Simulate encoding work
        logger.info("Start encoding video=%s", video_id)
        time.sleep(1)  # simulate work; replace with ffmpeg invocation
        # Persist minimal status update if DB available
        if DB is not None:
            DB.insert("videos", {"id": video_id, "status": "ready", "meta": message})
            logger.info("Persisted video status for %s", video_id)
        else:
            logger.info("DB not available; skipping persist for %s", video_id)
    except Exception:
        logger.exception("Error handling video upload")


def start_worker() -> threading.Thread:
    # subscribe and return thread that keeps process alive (subscribe is non-blocking)
    try:
        BROKER.subscribe("video.uploaded", handle_video_upload)
    except Exception:
        logger.exception("Failed to subscribe to video.uploaded")
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    start_worker()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("video_processing exiting")
