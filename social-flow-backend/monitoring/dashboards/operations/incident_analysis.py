# Analyze incidents and generate reports
"""
Incident Analysis & Correlation

- Correlates spikes across metrics to infer root-candidate components
- Provides short summaries of impacted resource groups (node, region, service)
- Uses simple heuristics: temporal overlap + relative spike magnitude
"""

from typing import Dict, List, Any
import statistics


class IncidentAnalyzer:
    def __init__(self, spike_z_threshold: float = 2.0):
        """
        spike_z_threshold: z-score threshold to consider a value a 'spike'
        """
        self.z_threshold = spike_z_threshold

    def correlate_recent_incidents(self, data_bundle: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Input:
          data_bundle: metric_name -> sorted list of values (older -> newer)
        Output:
          dict with simple correlation summary like counts and top suspected metrics
        """
        # Detect spikes
        spikes = {}
        for metric, values in data_bundle.items():
            if not values or len(values) < 3:
                continue
            mean = statistics.mean(values)
            pstdev = statistics.pstdev(values)
            latest = values[-1]
            if pstdev == 0:
                continue
            z = (latest - mean) / pstdev
            if z >= self.z_threshold:
                spikes[metric] = {"latest": latest, "mean": round(mean, 2), "z": round(z, 2)}

        # Correlate by heuristic: spike count, relative z-scores
        if not spikes:
            return {"spike_count": 0, "top_suspects": []}

        # Rank by z-score and magnitude
        suspects = sorted(spikes.items(), key=lambda kv: (kv[1]["z"], kv[1]["latest"]), reverse=True)
        top_suspects = [{"metric": m, **meta} for m, meta in suspects[:5]]

        # Simple bucketization to infer resource groups by metric name tokens
        buckets = {}
        for m, meta in spikes.items():
            key = self._bucket_by_metric(m)
            buckets.setdefault(key, []).append((m, meta))

        bucket_summary = {k: len(v) for k, v in buckets.items()}

        return {
            "spike_count": len(spikes),
            "top_suspects": top_suspects,
            "buckets": bucket_summary
        }

    def _bucket_by_metric(self, metric_name: str) -> str:
        """
        Very light-weight bucketization: map metric naming patterns to buckets.
        Customize for your metric naming scheme (prometheus, datadog tags).
        """
        lower = metric_name.lower()
        if "cpu" in lower:
            return "compute/cpu"
        if "memory" in lower:
            return "compute/memory"
        if "disk" in lower or "iops" in lower:
            return "storage/disk"
        if "latency" in lower or "network" in lower:
            return "network"
        if "incidents" in lower or "alerts" in lower:
            return "incident"
        return "other"
