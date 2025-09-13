# Package initializer for bandwidth module
"""
Bandwidth Optimization Package

This package provides advanced bandwidth optimization techniques for mobile backends.
It includes modules for compression, adaptive streaming, batching, predictive caching,
and dynamic network throttling. All modules integrate with BandwidthManager, which
serves as the orchestrator for strategy execution.

Author: Enterprise Backend Team
"""

from .bandwidth_manager import BandwidthManager
from .compression import CompressionEngine
from .adaptive_streaming import AdaptiveStreamer
from .request_batching import RequestBatcher
from .predictive_caching import PredictiveCache
from .network_throttling import Throttler
from .metrics import BandwidthMetrics

__all__ = [
    "BandwidthManager",
    "CompressionEngine",
    "AdaptiveStreamer",
    "RequestBatcher",
    "PredictiveCache",
    "Throttler",
    "BandwidthMetrics",
]
