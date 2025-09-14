
"""
API Optimization Package

This package provides advanced performance optimization strategies for APIs:
- Rate limiting (token bucket, leaky bucket, sliding window)
- Caching (in-memory, distributed, async-aware)
- Compression (gzip, brotli, zstd)
- Load balancing (round-robin, least-connections, hash-based)
- Throttling & backpressure mechanisms
- Monitoring & observability
- Middleware integrations for frameworks (FastAPI, Flask, Django)

All components are designed to be production-ready, async-compatible, 
and highly configurable.
"""

from .rate_limiting import TokenBucketRateLimiter, SlidingWindowRateLimiter
from .caching import InMemoryCache, DistributedCache
from .compression import GzipCompressor, BrotliCompressor, ZstdCompressor
from .load_balancing import RoundRobinBalancer, LeastConnectionsBalancer, HashBalancer
from .throttling import RequestThrottler
from .monitoring import APIMetricsCollector
from .middleware import OptimizationMiddleware

__all__ = [
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "InMemoryCache",
    "DistributedCache",
    "GzipCompressor",
    "BrotliCompressor",
    "ZstdCompressor",
    "RoundRobinBalancer",
    "LeastConnectionsBalancer",
    "HashBalancer",
    "RequestThrottler",
    "APIMetricsCollector",
    "OptimizationMiddleware",
]
