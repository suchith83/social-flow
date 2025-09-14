# Correlates logs across services for root cause analysis
# monitoring/logging/analysis/correlation_engine.py
"""
Correlation engine for logs.
Finds temporal correlations and assigns causal scores between services/events.
"""

import pandas as pd
import itertools
from .config import CONFIG


class CorrelationEngine:
    def __init__(self):
        self.tolerance = CONFIG["CORRELATION"]["time_tolerance_sec"]
        self.threshold = CONFIG["CORRELATION"]["causal_score_threshold"]

    def correlate(self, logs: list, service_field="host"):
        """Find correlations between events across services."""
        df = pd.DataFrame(logs)
        if "timestamp" not in df or service_field not in df:
            return []

        correlations = []
        services = df[service_field].unique()

        for s1, s2 in itertools.combinations(services, 2):
            events1 = df[df[service_field] == s1]
            events2 = df[df[service_field] == s2]
            overlap = self._temporal_overlap(events1, events2)
            causal_score = overlap / max(len(events1), 1)
            if causal_score >= self.threshold:
                correlations.append({
                    "service_a": s1,
                    "service_b": s2,
                    "causal_score": causal_score
                })

        return correlations

    def _temporal_overlap(self, df1, df2):
        """Count overlapping events within tolerance window."""
        count = 0
        for t1 in df1["timestamp"]:
            close_events = df2[
                (df2["timestamp"] >= t1 - pd.Timedelta(seconds=self.tolerance)) &
                (df2["timestamp"] <= t1 + pd.Timedelta(seconds=self.tolerance))
            ]
            count += len(close_events)
        return count
