# Package initializer for suppression module
"""
Suppression package for alerting.

This package provides:
- SuppressionRule model (suppress matching events/alerts)
- Suppressor engine to decide whether to suppress an incoming event
- Windowed suppression policies (suppress after N events in T seconds)
- In-memory persistence for active suppression entries (SilenceStore)
- Utility helpers

Exports:
- SuppressionRule, Suppressor, WindowedSuppressionRule, InMemorySilenceStore
"""

from .suppression_rule import SuppressionRule, SuppressionScope
from .suppressor import Suppressor, SuppressionDecision
from .windowed_suppression import WindowedSuppressionRule
from .silence_store import InMemorySilenceStore
from .utils import extract_path, now_utc, seconds_to_iso

__all__ = [
    "SuppressionRule",
    "SuppressionScope",
    "Suppressor",
    "SuppressionDecision",
    "WindowedSuppressionRule",
    "InMemorySilenceStore",
    "extract_path",
    "now_utc",
    "seconds_to_iso",
]
