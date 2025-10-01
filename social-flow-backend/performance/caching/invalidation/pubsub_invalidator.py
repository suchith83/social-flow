# pubsub_invalidator.py
# Created by Create-Invalidation.ps1
"""
pubsub_invalidator.py
---------------------
Publish invalidation messages to a message bus so that distributed cache nodes
and edge invalidation workers can respond. Works with any simple `publish(topic, message)` client.

Design notes:
- messages are JSON with structure { "cmd": "invalidate", "keys": [...], "origin": "...", "meta": {...} }
- idempotent consumers are expected
"""

from __future__ import annotations
import json
import logging
from typing import Iterable, Mapping, Optional

from .exceptions import RemoteInvalidateError

logger = logging.getLogger(__name__)


class PubSubInvalidator:
    """
    Lightweight adapter over a pub/sub client (Redis pubsub, Kafka producer, SNS, etc.).

    The client must expose a `publish(topic: str, payload: str)` method. We keep the adapter generic.
    """

    def __init__(self, client, topic: str = "cache-invalidation"):
        self.client = client
        self.topic = topic

    def publish(self, message: Mapping) -> None:
        """Publish a dict message as JSON."""
        try:
            payload = json.dumps(message, separators=(",", ":"), sort_keys=True)
            # Some clients return metadata; we ignore it but surface exceptions
            self.client.publish(self.topic, payload)
            logger.debug("Published invalidation message to %s", self.topic)
        except Exception as e:
            logger.exception("Failed to publish invalidation message")
            raise RemoteInvalidateError(str(e))

    def invalidate(self, keys: Iterable[str], origin: Optional[str] = None, meta: Optional[Mapping] = None) -> None:
        msg = {"cmd": "invalidate", "keys": list(keys), "origin": origin, "meta": meta or {}}
        self.publish(msg)

    def schedule_invalidate(self, keys: Iterable[str], when: Optional[float] = None, origin: Optional[str] = None) -> None:
        msg = {"cmd": "invalidate_at", "keys": list(keys), "when": when, "origin": origin}
        self.publish(msg)
