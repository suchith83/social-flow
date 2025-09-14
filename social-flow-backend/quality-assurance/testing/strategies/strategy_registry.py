"""
Registry for available strategies. Allows dynamic registration and discovery.
Useful for plugins or extending with organization-specific strategies.
"""

from typing import Dict, Type, Optional
from .exceptions import StrategyError


class StrategyRegistry:
    """Registry that maps a strategy name to a class implementing it."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}

    def register(self, name: str, cls: Type):
        if name in self._registry:
            raise StrategyError(f"Strategy '{name}' already registered")
        self._registry[name] = cls

    def get(self, name: str) -> Optional[Type]:
        return self._registry.get(name)

    def list(self):
        return list(self._registry.keys())

    def unregister(self, name: str):
        if name in self._registry:
            del self._registry[name]
        else:
            raise StrategyError(f"Strategy '{name}' is not registered")
