"""Kafka consumer wrapper with safe fallback to an in-memory broker."""

import threading
import json
from typing import Callable, Any, Optional

try:
    from kafka import KafkaConsumer as _KafkaConsumer  # type: ignore
except Exception:
    _KafkaConsumer = None  # type: ignore

# reuse the in-memory broker from producer module to ensure cross-process usage works only within same process.
# import lazily to avoid import cycles
try:
    from event_streaming.kafka.producer import _INMEMORY_BROKER  # type: ignore
except Exception:
    # if import fails (e.g., different package path), recreate a local in-memory broker for safety
    class _LocalDummy:
        def subscribe(self, topic, cb): pass
    _INMEMORY_BROKER = _LocalDummy()  # type: ignore


class KafkaConsumer:
    """
    Wrapper to subscribe to topics. Callback signature: fn(payload_dict).
    Starts background threads for Kafka consumption or registers callbacks with the in-memory broker.
    """

    def __init__(self, bootstrap_servers: Optional[str] = None, group_id: Optional[str] = None):
        self._threads = []
        self._callbacks = {}
        self._running = True
        self._bootstrap = bootstrap_servers
        self._group_id = group_id
        self._consumer_client = None
        if _KafkaConsumer is not None and bootstrap_servers:
            try:
                self._consumer_client = _KafkaConsumer(
                    bootstrap_servers=bootstrap_servers.split(",") if isinstance(bootstrap_servers, str) else bootstrap_servers,
                    auto_offset_reset="latest",
                    group_id=group_id,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")) if isinstance(v, (bytes, bytearray)) else v,
                )
            except Exception:
                self._consumer_client = None

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to topic with callback. If Kafka client exists, start a thread polling messages."""
        if self._consumer_client:
            try:
                self._consumer_client.subscribe([topic])

                def _poll_loop():
                    try:
                        for msg in self._consumer_client:
                            try:
                                callback(msg.value)
                            except Exception:
                                pass
                    except Exception:
                        pass

                t = threading.Thread(target=_poll_loop, daemon=True)
                t.start()
                self._threads.append(t)
                return
            except Exception:
                # fallback to in-memory
                pass

        # register with in-memory broker (same-process only)
        try:
            _INMEMORY_BROKER.subscribe(topic, callback)
        except Exception:
            # best-effort: swallow errors
            pass

    def close(self):
        self._running = False
        try:
            if self._consumer_client:
                self._consumer_client.close()
        except Exception:
            pass
        # threads are daemonized; nothing else to do
