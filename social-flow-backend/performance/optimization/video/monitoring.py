# Monitors video pipeline health and metrics
"""
monitoring.py

Collect video-centric metrics:
- Transcoding job durations / failure counts
- Segment delivery latencies / cache hit rates
- Playback latency / rebuffer events
- ABR decisions and health indicators

This collector is in-memory and meant to be a local aggregation point; in production
push these metrics to Prometheus, Datadog, or a metrics pipeline.
"""

import time
import statistics
from collections import defaultdict
from typing import Dict, List


class VideoMetricsCollector:
    def __init__(self):
        # Transcoding
        self.transcode_durations: List[float] = []
        self.transcode_failures = 0

        # Segment delivery
        self.segment_latencies: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0

        # Playback
        self.rebuffer_events: List[float] = []  # timestamps
        self.playback_latencies: List[float] = []

        # ABR decisions
        self.abr_decisions: List[Dict] = []

        # Per-variant
        self.variant_stats: Dict[str, List[float]] = defaultdict(list)

    # Transcoding
    def record_transcode(self, duration: float, success: bool = True):
        self.transcode_durations.append(duration)
        if not success:
            self.transcode_failures += 1

    # Segment delivery
    def record_segment_latency(self, latency: float):
        self.segment_latencies.append(latency)

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    # Playback
    def record_rebuffer(self, timestamp: float = None):
        self.rebuffer_events.append(timestamp or time.time())

    def record_playback_latency(self, latency: float):
        self.playback_latencies.append(latency)

    # ABR
    def record_abr_decision(self, decision: Dict):
        # Decision example: {"from": "720p", "to": "480p", "reason": "throughput"}
        self.abr_decisions.append(decision)

    def summary(self) -> Dict:
        def pctile(data: List[float], p: float) -> Optional[float]:
            if not data:
                return None
            k = max(0, min(len(data)-1, int(len(data) * p)))
            return sorted(data)[k]

        total_segments = len(self.segment_latencies)
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)) if (self.cache_hits + self.cache_misses) else None

        return {
            "transcode_count": len(self.transcode_durations),
            "transcode_failures": self.transcode_failures,
            "avg_transcode_duration": statistics.mean(self.transcode_durations) if self.transcode_durations else None,
            "segment_count": total_segments,
            "avg_segment_latency": statistics.mean(self.segment_latencies) if self.segment_latencies else None,
            "p95_segment_latency": pctile(self.segment_latencies, 0.95),
            "cache_hit_rate": cache_hit_rate,
            "rebuffer_count": len(self.rebuffer_events),
            "avg_playback_latency": statistics.mean(self.playback_latencies) if self.playback_latencies else None,
            "abr_decisions": len(self.abr_decisions),
        }
