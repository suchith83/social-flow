# common/libraries/python/messaging/base_broker.py
"""
Abstract base class for message brokers.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any

class BaseBroker(ABC):
    @abstractmethod
    def publish(self, topic: str, message: Dict[str, Any], headers: Dict[str, str] = None):
        pass

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        pass
