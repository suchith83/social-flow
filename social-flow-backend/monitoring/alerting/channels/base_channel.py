# Base class for all communication channels
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class BaseChannel(ABC):
    """
    Abstract base class for all alerting channels.
    Enforces implementation of `send_alert` method.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def send_alert(self, message: str, **kwargs) -> bool:
        """
        Send an alert message.
        Must be implemented by all subclasses.

        Args:
            message: The alert message to send.
            kwargs: Optional parameters specific to the channel.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement send_alert")

    def validate_message(self, message: str) -> str:
        """Ensure message is safe and within limits."""
        if not message or not isinstance(message, str):
            raise ValueError("Alert message must be a non-empty string.")
        return message.strip()
