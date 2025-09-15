from typing import Dict, Any
from .utils import get_logger

logger = get_logger(__name__)


class StateStore:
    """In-memory key-value state store for stream processing."""

    def __init__(self):
        self.store: Dict[str, Any] = {}

    def update(self, key: str, value: Any):
        self.store[key] = value
        logger.debug(f"Updated state: {key}={value}")

    def get(self, key: str, default=None) -> Any:
        return self.store.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Returns full state snapshot."""
        return self.store.copy()
