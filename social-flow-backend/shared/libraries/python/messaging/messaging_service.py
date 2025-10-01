# common/libraries/python/messaging/messaging_service.py
"""
Messaging service abstraction combining broker, DLQ, middleware, retries.
"""

from typing import Dict, Any, Callable
from .config import MessagingConfig
from .kafka_broker import KafkaBroker
from .rabbitmq_broker import RabbitMQBroker
from .redis_broker import RedisBroker
from .dlq import DeadLetterQueue

class MessagingService:
    def __init__(self):
        if MessagingConfig.BROKER_TYPE == "kafka":
            self.broker = KafkaBroker()
        elif MessagingConfig.BROKER_TYPE == "rabbitmq":
            self.broker = RabbitMQBroker()
        else:
            self.broker = RedisBroker()

        self.dlq = DeadLetterQueue(self.broker)

    def publish(self, topic: str, message: Dict[str, Any], headers: Dict[str, str] = None):
        self.broker.publish(topic, message, headers)

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        def wrapped_handler(message: Dict[str, Any]):
            retries = 0
            while retries < MessagingConfig.MAX_RETRIES:
                try:
                    return handler(message)
                except Exception as e:
                    retries += 1
                    print(f"[Messaging] Error handling message: {e}, retry {retries}")
            self.dlq.send_to_dlq(message, reason="Max retries exceeded")

        self.broker.subscribe(topic, wrapped_handler)

messaging_service = MessagingService()
