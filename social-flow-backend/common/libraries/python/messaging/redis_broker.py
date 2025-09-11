# common/libraries/python/messaging/redis_broker.py
"""
Redis Pub/Sub broker implementation.
"""

import redis
from threading import Thread
from typing import Dict, Any, Callable
from .base_broker import BaseBroker
from .config import MessagingConfig
from .serializers import SERIALIZERS

class RedisBroker(BaseBroker):
    def __init__(self):
        self.serializer = SERIALIZERS[MessagingConfig.SERIALIZER]
        self.client = redis.StrictRedis.from_url(MessagingConfig.BROKER_URL, decode_responses=False)

    def publish(self, topic: str, message: Dict[str, Any], headers: Dict[str, str] = None):
        payload = self.serializer.dumps({"message": message, "headers": headers or {}})
        self.client.publish(topic, payload)

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        pubsub = self.client.pubsub()
        pubsub.subscribe(topic)

        def run():
            for msg in pubsub.listen():
                if msg["type"] == "message":
                    data = self.serializer.loads(msg["data"])
                    handler(data["message"])

        Thread(target=run, daemon=True).start()
