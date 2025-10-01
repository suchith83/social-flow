"""Pub/Sub abstraction for Redis messaging."""
"""
Redis Pub/Sub Abstraction
- Wraps publish/subscribe in Redis.
"""

import threading
from typing import Callable
from .redis_connection import get_redis


class RedisPubSub:
    def __init__(self, channel_prefix: str = "app:events"):
        self.client = get_redis()
        self.pubsub = self.client.pubsub()
        self.channel_prefix = channel_prefix

    def _channel(self, name: str) -> str:
        return f"{self.channel_prefix}:{name}"

    def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        return self.client.publish(self._channel(channel), message)

    def subscribe(self, channel: str, handler: Callable[[str], None]) -> None:
        """Subscribe to a channel with callback."""
        full_channel = self._channel(channel)
        self.pubsub.subscribe(full_channel)

        def _listen():
            for msg in self.pubsub.listen():
                if msg["type"] == "message":
                    handler(msg["data"].decode("utf-8"))

        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()
