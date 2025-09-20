"""Common helpers for workers: logger, broker client and database service fallbacks."""

from typing import Callable, Dict, Any, Optional
import threading
import time

# Lightweight logger fallback (try common package first)
try:
    from common.libraries.python.monitoring.logger import get_logger  # type: ignore
except Exception:
    try:
        from src.utils import get_logger  # type: ignore
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

logger = get_logger("workers.common")

# Broker: prefer RedisBroker from common; otherwise provide in-process DummyBroker
try:
    from common.libraries.python.messaging.redis_broker import RedisBroker  # type: ignore
except Exception:
    RedisBroker = None  # type: ignore

class _InMemoryBroker:
    def __init__(self):
        self._subs = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            cbs = list(self._subs.get(topic, []))
        for cb in cbs:
            try:
                threading.Thread(target=cb, args=(payload,), daemon=True).start()
            except Exception:
                logger.exception("in-memory broker callback failed")

    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subs.setdefault(topic, []).append(callback)
        logger.info("In-memory broker subscribed to %s", topic)

# Database service fallback: try common.database.database_service
try:
    from common.libraries.python.database.database_service import database_service  # type: ignore
except Exception:
    database_service = None  # type: ignore

# Expose factory functions
def get_broker() -> Any:
    if RedisBroker is not None:
        try:
            return RedisBroker()
        except Exception:
            logger.exception("Failed to init RedisBroker; falling back to in-memory broker")
    return _InMemoryBroker()

def get_db() -> Optional[Any]:
    return database_service
