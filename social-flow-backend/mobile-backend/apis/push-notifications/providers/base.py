# Base class/interface for notification providers
"""
Abstract provider interface for push services (FCM, APNs, WebPush).
Each provider must implement send_batch(tokens, title, body, payload) -> list[dict]
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod


class ProviderResponse(dict):
    """
    Standardized provider response dict:
    {
        "token": "<device token>",
        "status": "ok" | "failed",
        "provider_message": {...}
    }
    """


class PushProvider(ABC):
    @abstractmethod
    def send_batch(self, tokens: List[str], title: str, body: str, payload: Dict[str, Any]) -> List[ProviderResponse]:
        """Send a batch of notifications. Should be robust and non-blocking-friendly (can be used in Celery)."""
        raise NotImplementedError()
