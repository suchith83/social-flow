# common/libraries/python/messaging/dlq.py
"""
Dead Letter Queue implementation.
"""

from typing import Dict, Any
from .config import MessagingConfig
from .serializers import SERIALIZERS

class DeadLetterQueue:
    def __init__(self, broker):
        self.broker = broker
        self.serializer = SERIALIZERS[MessagingConfig.SERIALIZER]

    def send_to_dlq(self, message: Dict[str, Any], reason: str):
        payload = {**message, "_dlq_reason": reason}
        self.broker.publish(MessagingConfig.DLQ_TOPIC, payload)
