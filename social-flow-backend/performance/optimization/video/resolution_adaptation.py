# Adjusts resolution dynamically
"""
resolution_adaptation.py

Adaptive Bitrate (ABR) controller to pick next-bitrate/resolution based on buffer,
throughput estimation, and simple reinforcement heuristics.

Provides:
- Simple throughput estimator
- ABRController that implements a utility-based selection (throughput / buffer / penalty)
"""

import time
import statistics
from typing import List, Dict, Tuple, Optional


class ThroughputEstimator:
    """
    Maintain a sliding window of measured download-throughputs (kbps) and produce
    conservative estimates (e.g., harmonic mean / weighted median).
    """

    def __init__(self, window: int = 8):
        self.window = window
        self.samples: List[float] = []

    def add_sample(self, kbps: float):
        if kbps <= 0:
            return
        self.samples.append(kbps)
        if len(self.samples) > self.window:
            self.samples.pop(0)

    def estimate(self) -> Optional[float]:
        if not self.samples:
            return None
        # Use harmonic mean to be conservative for skewed distributions
        invs = [1.0 / s for s in self.samples if s > 0]
        if not invs:
            return None
        return len(invs) / sum(invs)


class ABRController:
    """
    ABR controller decides which bitrate/resolution to select for next segment.

    - renditions: list of tuples (name, bandwidth_kbps, resolution)
    - conservative selection: pick highest rendition with bandwidth < alpha * estimated_throughput
    - consider buffer occupancy: if buffer low, step down more aggressively
    """

    def __init__(self, renditions: List[Tuple[str, int, str]], alpha: float = 0.85):
        self.renditions = sorted(renditions, key=lambda r: r[1])  # sort by bandwidth
        self.alpha = alpha
        self.estimator = ThroughputEstimator()

    def on_download_sample(self, kbps: float):
        self.estimator.add_sample(kbps)

    def select_rendition(self, buffer_seconds: float) -> Dict:
        est = self.estimator.estimate()
        # If no estimate, choose lowest or medium depending on buffer
        if est is None:
            if buffer_seconds < 2.0:
                return {"name": self.renditions[0][0], "bandwidth": self.renditions[0][1], "resolution": self.renditions[0][2]}
            return {"name": self.renditions[len(self.renditions) // 2][0], "bandwidth": self.renditions[len(self.renditions) // 2][1], "resolution": self.renditions[len(self.renditions) // 2][2]}

        target = est * self.alpha
        # Buffer sensitivity: if buffer low, reduce alpha effectively
        if buffer_seconds < 1.5:
            target *= 0.6
        elif buffer_seconds < 3.0:
            target *= 0.8

        chosen = self.renditions[0]
        for r in self.renditions:
            if r[1] <= target:
                chosen = r
            else:
                break
        return {"name": chosen[0], "bandwidth": chosen[1], "resolution": chosen[2]}

    def renditions_summary(self) -> List[Dict]:
        return [{"name": r[0], "bandwidth": r[1], "resolution": r[2]} for r in self.renditions]
