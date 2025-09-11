# common/libraries/python/messaging/kafka_broker.py
"""
Kafka broker implementation.
"""

from kafka import KafkaProducer, KafkaConsumer
from threading import Thread
from typing import Dict, Any, Callable
from .base_broker import BaseBroker
from .config import MessagingConfig
from .serializers import SERIALIZERS

class KafkaBroker(BaseBroker):
    def __init__(self):
        self.serializer = SERIALIZERS[MessagingConfig.SERIALIZER]
        self.producer = KafkaProducer(
            bootstrap_servers=MessagingConfig.BROKER_URL,
            value_serializer=self.serializer.dumps,
        )

    def publish(self, topic: str, message: Dict[str, Any], headers: Dict[str, str] = None):
        self.producer.send(topic, value=message, headers=[(k, v.encode()) for k, v in (headers or {}).items()])
        self.producer.flush()

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=MessagingConfig.BROKER_URL,
            value_deserializer=self.serializer.loads,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )

        def run():
            for msg in consumer:
                handler(msg.value)

        Thread(target=run, daemon=True).start()
