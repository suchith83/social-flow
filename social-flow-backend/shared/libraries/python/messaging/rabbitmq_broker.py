# common/libraries/python/messaging/rabbitmq_broker.py
"""
RabbitMQ broker implementation.
"""

import pika
from threading import Thread
from typing import Dict, Any, Callable
from .base_broker import BaseBroker
from .config import MessagingConfig
from .serializers import SERIALIZERS

class RabbitMQBroker(BaseBroker):
    def __init__(self):
        self.serializer = SERIALIZERS[MessagingConfig.SERIALIZER]
        params = pika.URLParameters(MessagingConfig.BROKER_URL)
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()

    def publish(self, topic: str, message: Dict[str, Any], headers: Dict[str, str] = None):
        self.channel.queue_declare(queue=topic, durable=True)
        self.channel.basic_publish(
            exchange="",
            routing_key=topic,
            body=self.serializer.dumps(message),
            properties=pika.BasicProperties(headers=headers or {}),
        )

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        self.channel.queue_declare(queue=topic, durable=True)

        def callback(ch, method, properties, body):
            msg = self.serializer.loads(body)
            handler(msg)

        Thread(target=self.channel.basic_consume, args=(topic, callback, False), daemon=True).start()
        Thread(target=self.channel.start_consuming, daemon=True).start()
