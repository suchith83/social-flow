# Dynamically adjusts streaming quality
"""
Adaptive Streaming Module
Dynamically adjusts media bitrate based on network conditions.
"""

import random
from .config import CONFIG


class AdaptiveStreamer:
    def __init__(self):
        self.current_bitrate = CONFIG.default_bitrate

    def adjust_bitrate(self, latency_ms: float, bandwidth_kbps: float) -> int:
        """Adjust stream bitrate based on network conditions."""
        if latency_ms > CONFIG.network_latency_threshold:
            self.current_bitrate = max(
                CONFIG.min_bitrate, int(self.current_bitrate * 0.8)
            )
        elif bandwidth_kbps > self.current_bitrate * 1.5:
            self.current_bitrate = min(
                CONFIG.max_bitrate, int(self.current_bitrate * 1.2)
            )
        return self.current_bitrate

    def get_stream_chunk(self, content_id: str) -> bytes:
        """Fetch a stream chunk at the current bitrate (simulated)."""
        return bytes(random.getrandbits(8) for _ in range(self.current_bitrate * 128))
