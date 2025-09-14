# Monitors training and inference performance
import time
import statistics


class MLMetricsCollector:
    """
    Collects ML metrics:
    - Inference latency
    - Throughput
    - GPU usage placeholder (extendable with NVML)
    """

    def __init__(self):
        self.latencies = []
        self.count = 0

    def record_inference(self, start_time: float, end_time: float):
        latency = end_time - start_time
        self.latencies.append(latency)
        self.count += 1

    def summary(self):
        if not self.latencies:
            return {"inferences": 0}
        return {
            "inferences": self.count,
            "avg_latency": statistics.mean(self.latencies),
            "p95_latency": statistics.quantiles(self.latencies, n=100)[94],
            "throughput": self.count / sum(self.latencies),
        }
