# Manages playback buffers
"""
buffer_management.py

Implement jitter buffer and playback smoothing strategies.

- JitterBuffer: maintains a buffer of segments/frames, drops stale packets, allows adaptive size.
- SmoothingPolicy: basic pacing / rebuffer avoidance heuristics.
"""

import time
import asyncio
from collections import deque
from typing import Any, Deque, Optional


class JitterBuffer:
    """
    JitterBuffer keeps a time-ordered queue of packets/segments and exposes pop/peek functions.

    - target_size: desired buffer occupancy in seconds/units
    - max_size: maximum occupancy allowed
    - drop_policy: callable deciding how to drop when full (default: drop oldest)
    """

    def __init__(self, target_size: float = 3.0, max_size: float = 10.0, drop_policy=None):
        self.target_size = target_size
        self.max_size = max_size
        self.buffer: Deque = deque()
        self._size_seconds = 0.0  # approximate occupancy in seconds
        self.lock = asyncio.Lock()
        self.drop_policy = drop_policy or (lambda buf: buf.popleft())

    async def push(self, packet: Any, duration: float):
        """
        Push a packet with an associated play-duration (seconds).
        """
        async with self.lock:
            # Evict oldest until there's room under max_size
            while self._size_seconds + duration > self.max_size and self.buffer:
                evicted = self.drop_policy(self.buffer)
                # If the evicted item included duration tracking, adjust accordingly.
                # Here we assume items are tuples (packet, duration)
                if isinstance(evicted, tuple) and len(evicted) >= 2:
                    self._size_seconds -= float(evicted[1])
                else:
                    # unknown size; we conservatively decrement small amount
                    self._size_seconds -= 0.02
            self.buffer.append((packet, duration))
            self._size_seconds += duration

    async def pop(self) -> Optional[Any]:
        async with self.lock:
            if not self.buffer:
                return None
            packet, duration = self.buffer.popleft()
            self._size_seconds -= duration
            return packet

    async def peek(self) -> Optional[Any]:
        async with self.lock:
            if not self.buffer:
                return None
            return self.buffer[0][0]

    async def occupancy(self) -> float:
        async with self.lock:
            return max(0.0, self._size_seconds)

    async def clear(self):
        async with self.lock:
            self.buffer.clear()
            self._size_seconds = 0.0


class SmoothingPolicy:
    """
    Simple smoothing / rebuffering avoidance heuristics.

    - If buffer occupancy < low_watermark -> slow playback / prefetch
    - If occupancy > high_watermark -> allow playback or increase bitrate
    """

    def __init__(self, low_watermark: float = 1.0, high_watermark: float = 5.0):
        self.low = low_watermark
        self.high = high_watermark

    def decide(self, occupancy_seconds: float) -> str:
        if occupancy_seconds < self.low:
            return "prefetch"
        if occupancy_seconds > self.high:
            return "allow_play"
        return "steady"
