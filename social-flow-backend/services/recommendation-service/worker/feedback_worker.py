"""Feedback worker for recommendation-service.

Subscribes to Redis topic `recommendation.feedback` and writes entries to the
database using the common database service. This is intentionally minimal and
meant as a skeleton to extend (add batching, DLQ, retries, metrics).
"""

import time
from typing import Dict, Any, Callable, Optional

# Guard imports so the worker can be inspected/run in minimal environments.
try:
    from common.libraries.python.messaging.redis_broker import RedisBroker
except Exception:
    RedisBroker = None

try:
    from common.libraries.python.database.database_service import database_service
except Exception:
    database_service = None

try:
    from common.libraries.python.monitoring.logger import get_logger
    logger = get_logger("feedback-worker")
except Exception:
    import logging
    logger = logging.getLogger("feedback-worker")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


def handle_message(message: Dict[str, Any]) -> None:
    logger.info(f"Handling feedback message: {message}")
    try:
        if database_service is None:
            logger.warning("database_service not available; skipping persist")
            return
        # Persist minimal record into a feedback table.
        database_service.insert("recommendation_feedback", message)
    except Exception:
        logger.exception("Failed to persist feedback")


def run() -> None:
    if RedisBroker is None:
        logger.error("RedisBroker not available; feedback worker will not subscribe. Exiting.")
        return

    try:
        broker = RedisBroker()
    except Exception:
        logger.exception("Failed to initialize RedisBroker; exiting.")
        return

    # subscribe(topic, callback)
    broker.subscribe("recommendation.feedback", handle_message)
    logger.info("Feedback worker started and subscribed to recommendation.feedback")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Feedback worker shutting down")


if __name__ == "__main__":
    run()
