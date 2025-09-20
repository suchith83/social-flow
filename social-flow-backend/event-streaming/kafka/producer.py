"""Kafka producer wrapper with safe fallback to an in-memory broker."""

import json
import threading
import time
from typing import Any, Dict, Optional

try:
    # prefer kafka-python if available
    from kafka import KafkaProducer as _KafkaProducer  # type: ignore
except Exception:
    _KafkaProducer = None  # type: ignore

# Simple in-memory broker used when Kafka is not available
class _InMemoryBroker:
    def __init__(self):
        self._subs = {}  # topic -> list of callbacks
        self._lock = threading.Lock()

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            cbs = list(self._subs.get(topic, []))
        for cb in cbs:
            try:
                # deliver in background thread to mimic async delivery
                threading.Thread(target=cb, args=(payload,), daemon=True).start()
            except Exception:
                pass

    def subscribe(self, topic: str, callback):
        with self._lock:
            self._subs.setdefault(topic, []).append(callback)


# singleton for in-memory broker
_INMEMORY_BROKER = _InMemoryBroker()


class KafkaProducer:
    """
    Wrapper to publish JSON payloads to Kafka or to an in-memory broker if Kafka is not installed.

    Params:
      bootstrap_servers: comma-separated host:port or list (passed to kafka client)
    """

    def __init__(self, bootstrap_servers: Optional[str] = None):
        self._producer = None
        if _KafkaProducer is not None and bootstrap_servers:
            try:
                self._producer = _KafkaProducer(
                    bootstrap_servers=bootstrap_servers.split(",") if isinstance(bootstrap_servers, str) else bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                )
            except Exception:
                self._producer = None
        # fall back to in-memory broker
        self._broker = _INMEMORY_BROKER

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish payload to topic. Non-blocking for in-memory; synchronous send for Kafka client."""
        if self._producer:
            try:
                self._producer.send(topic, payload)
                # optionally flush for reliability in simple cases
                self._producer.flush(timeout=1)
                return
            except Exception:
                # on failure, fall through to in-memory publish
                pass
        # in-memory publish
        self._broker.publish(topic, payload)
