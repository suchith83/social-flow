# common/libraries/python/messaging/redis_broker.py
"""
Redis Pub/Sub broker implementation.
"""

import threading
import json
import logging
import redis
from threading import Thread
from typing import Dict, Any, Callable, Optional
from .base_broker import BaseBroker
from .config import MessagingConfig
from .serializers import SERIALIZERS

logger = logging.getLogger("common.messaging.redis_broker")

class DummyBroker:
    """No-op broker used in tests/local when redis is not available."""

    def publish(self, topic: str, payload: dict) -> None:
        logger.debug("DummyBroker.publish topic=%s payload=%s", topic, payload)
        return

    def subscribe(self, topic: str, callback: Callable[[dict], None]) -> None:
        logger.debug("DummyBroker.subscribe topic=%s (noop)", topic)
        return


class RedisBroker(BaseBroker):
    """Lightweight Redis pub/sub wrapper.

    publish(topic, payload) -> publishes JSON (utf-8)
    subscribe(topic, callback) -> starts a background thread consuming messages
    """

    def __init__(self, url: Optional[str] = None):
        url = url or "redis://localhost:6379/0"
        try:
            self._client = redis.Redis.from_url(url)
            self._pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        except Exception as exc:
            logger.exception("Failed to initialize redis client, falling back to DummyBroker: %s", exc)
            raise

    def publish(self, topic: str, payload: dict) -> None:
        try:
            self._client.publish(topic, json.dumps(payload))
        except Exception:
            logger.exception("Failed to publish to redis topic=%s", topic)

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        def _run():
            try:
                self._pubsub.subscribe(topic)
                for msg in self._pubsub.listen():
                    if msg is None:
                        continue
                    data = msg.get("data")
                    if isinstance(data, bytes):
                        try:
                            payload = json.loads(data.decode("utf-8"))
                        except Exception:
                            payload = {"raw": data.decode("utf-8", errors="ignore")}
                    else:
                        payload = data
                    try:
                        handler(payload)
                    except Exception:
                        logger.exception("Callback raised while handling message for topic=%s", topic)
            except Exception:
                logger.exception("Redis subscribe loop exited for topic=%s", topic)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

RedisBroker = RedisBroker if 'redis' in locals() and redis else DummyBroker  # type: ignore
