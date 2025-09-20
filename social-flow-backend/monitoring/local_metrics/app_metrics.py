"""
Simple Prometheus metrics endpoint for local dev.

Run:
    python monitoring/local_metrics/app_metrics.py
Then point Prometheus to http://localhost:9100/metrics
"""
from prometheus_client import start_http_server, Counter, Gauge
import time
import random

REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["job", "status"])
IN_PROGRESS = Gauge("in_progress_requests", "In progress requests", ["job"])
JOB_NAME = "local_dummy_service"

def simulate():
    while True:
        status = "200" if random.random() > 0.05 else "500"
        IN_PROGRESS.labels(job=JOB_NAME).set(random.randint(0, 5))
        REQUESTS.labels(job=JOB_NAME, status=status).inc(random.randint(1, 3))
        time.sleep(2)

if __name__ == "__main__":
    start_http_server(9100)
    print("Started local metrics on :9100/metrics")
    try:
        simulate()
    except KeyboardInterrupt:
        print("Stopping")
