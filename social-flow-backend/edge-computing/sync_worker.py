import threading
import time
from typing import Any, Dict, Optional
import os

# Defensive imports
try:
    from common.libraries.python.messaging.redis_broker import RedisBroker  # type: ignore
except Exception:
    RedisBroker = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


def _poll_trending(cache, logger, interval: int = 30, limit: int = 20):
    """Poll central recommendation/trending endpoints and warm edge cache."""
    central_url = os.environ.get("CENTRAL_RECOMMENDER_URL", "http://localhost:8003")
    while True:
        try:
            if requests:
                r = requests.get(f"{central_url}/trending?limit={limit}", timeout=5)
                if r.status_code == 200:
                    data = r.json().get("trending", [])
                    cache.set(f"trending:{limit}", data)
                    logger.info("Warmed trending cache with %d items", len(data))
        except Exception:
            logger.exception("Polling trending failed")
        time.sleep(interval)


def _subscribe_feedback(cache, logger):
    """Subscribe to recommendation.feedback and optionally persist locally."""
    if RedisBroker is None:
        logger.warning("RedisBroker not available for subscription; subscribe skipped")
        return
    try:
        broker = RedisBroker()
    except Exception:
        logger.exception("Failed to initialize RedisBroker for subscribe")
        return

    def _on_msg(msg: Dict[str, Any]):
        try:
            # msg expected to be dict with keys user_id/item_id/action/timestamp
            key = f"fb_recent:{msg.get('user_id')}:{int(time.time()*1000)}"
            cache.set(key, msg, ttl=60)
            logger.info("Stored recent feedback in edge cache key=%s", key)
        except Exception:
            logger.exception("Failed to handle message in subscribe callback")

    try:
        broker.subscribe("recommendation.feedback", _on_msg)
        logger.info("Subscribed to recommendation.feedback")
    except Exception:
        logger.exception("Failed to subscribe to recommendation.feedback")


def start_sync_worker(cache, logger, poll_interval: int = 30):
    """Start background threads: subscribe to Redis topic (if available) and poll trending."""
    # subscribe thread (best-effort)
    t1 = threading.Thread(target=_subscribe_feedback, args=(cache, logger), daemon=True)
    t1.start()

    # polling thread to warm trending
    t2 = threading.Thread(target=_poll_trending, args=(cache, logger, poll_interval), daemon=True)
    t2.start()

    # return immediately; threads are daemonic
    return (t1, t2)
