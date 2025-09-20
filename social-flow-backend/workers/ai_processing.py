"""AI processing worker: subscribe to ai.tasks and run lightweight inference/feature extraction."""

from typing import Dict, Any
import threading
import time

from workers.common import get_broker, get_db, logger

# best-effort import of ai-models stubs
try:
    from ai_models.recommendation_engine.trending import load_model as load_trending  # type: ignore
except Exception:
    load_trending = None  # type: ignore

BROKER = get_broker()
DB = get_db()

def handle_ai_task(message: Dict[str, Any]) -> None:
    logger.info("AI task received: %s", message)
    try:
        task_type = message.get("type", "infer")
        payload = message.get("payload", {})
        if task_type == "trending" and load_trending:
            model = load_trending()
            result = model.predict(payload.get("limit", 10))
            logger.info("AI trending result len=%d", len(result) if hasattr(result, "__len__") else 0)
            if DB is not None:
                DB.insert("kv_store", {"task": "trending", "result": result})
        else:
            # fallback: simulate light compute
            time.sleep(0.5)
            if DB is not None:
                DB.insert("kv_store", {"task": task_type, "payload": payload})
    except Exception:
        logger.exception("Error processing AI task")


def start_worker() -> threading.Thread:
    try:
        BROKER.subscribe("ai.tasks", handle_ai_task)
    except Exception:
        logger.exception("Failed to subscribe to ai.tasks")
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    start_worker()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("ai_processing exiting")
