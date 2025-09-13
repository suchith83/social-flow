# Orchestrates optimization strategies
"""
Bandwidth Manager
Coordinates all optimization strategies together.
"""

import threading
from .compression import CompressionEngine
from .adaptive_streaming import AdaptiveStreamer
from .request_batching import RequestBatcher
from .predictive_caching import PredictiveCache
from .network_throttling import Throttler
from .metrics import BandwidthMetrics


class BandwidthManager:
    def __init__(self):
        self.compressor = CompressionEngine()
        self.streamer = AdaptiveStreamer()
        self.batcher = RequestBatcher()
        self.cache = PredictiveCache()
        self.throttler = Throttler()
        self.metrics = BandwidthMetrics()

        # Start batching thread
        t = threading.Thread(target=self.batcher.periodic_flush, daemon=True)
        t.start()

    def optimize_request(self, user_id: str, request: dict) -> dict:
        """Optimize outgoing request payload with compression + batching."""
        compressed = self.compressor.compress(request["payload"])
        self.batcher.add_request({"user": user_id, "payload": compressed})
        self.metrics.log("request_size", len(compressed))
        return {"user": user_id, "payload": compressed}

    def optimize_stream(self, user_id: str, latency_ms: float, bandwidth_kbps: float):
        """Optimize streaming bitrate dynamically."""
        bitrate = self.streamer.adjust_bitrate(latency_ms, bandwidth_kbps)
        self.throttler.record_usage(user_id, bitrate / 8)
        self.metrics.log("bitrate", bitrate)
        return self.streamer.get_stream_chunk(user_id)

    def serve_resource(self, user_id: str, resource_id: str, fetch_fn):
        """Serve resource with predictive caching."""
        if self.throttler.is_throttled(user_id):
            return None  # Simulate denial
        data = self.cache.get(resource_id, fetch_fn)
        self.metrics.log("resource_size", len(data))
        return data

    def flush_metrics(self) -> dict:
        """Flush collected metrics."""
        return self.metrics.flush()
