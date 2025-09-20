"""Analytics worker: subscribes to analytics.events and persists to DB (or logs)."""

from typing import Dict, Any
import threading
import time

from workers.common import get_broker, get_db, logger

BROKER = get_broker()
DB = get_db()

def handle_event(message: Dict[str, Any]) -> None:
    logger.info("Analytics event: %s", message)
    try:
        if DB is not None:
            DB.insert("analytics_events", message)
        else:
            logger.debug("No DB available; event skipped")
    except Exception:
        logger.exception("Failed to persist analytics event")


def start_worker() -> threading.Thread:
    try:
        BROKER.subscribe("analytics.events", handle_event)
    except Exception:
        logger.exception("Failed to subscribe to analytics.events")
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    start_worker()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("analytics_worker exiting")
