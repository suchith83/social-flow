# Exposes application metrics (e.g., via Prometheus)
"""
metrics_server.py
Prometheus-compatible metrics HTTP endpoint using prometheus_client.
Provides:
 - /metrics endpoint
 - optional pushgateway push
 - health & readiness endpoints
 - middleware to record request latencies
"""

from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, Counter, Gauge, Summary
from prometheus_client import start_http_server
from prometheus_client.core import REGISTRY
from wsgiref.simple_server import make_server, WSGIRequestHandler
from typing import Optional
import threading
import time
import logging
from utils import setup_logger

logger = setup_logger("MetricsServer")


class MetricsServer:
    """
    Start a simple metrics endpoint in a background thread.
    You can also embed the WSGI app in an existing HTTP server.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8001, registry: Optional[CollectorRegistry] = None):
        self.host = host
        self.port = port
        self.registry = registry or REGISTRY
        # Core model metrics for standardization
        self.model_inference_count = Counter("model_inference_total", "Total model inference requests", ["model", "status"], registry=self.registry)
        self.model_inference_latency = Summary("model_inference_latency_seconds", "Inference latency seconds", ["model"], registry=self.registry)
        self.model_input_size = Gauge("model_input_size_bytes", "Size of input payload to model", ["model"], registry=self.registry)
        self.model_last_success = Gauge("model_last_success_timestamp", "Last successful inference timestamp", ["model"], registry=self.registry)

        self._server_thread = None
        self._running = False

    def _wsgi_app(self, environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path == "/metrics":
            output = generate_latest(self.registry)
            start_response("200 OK", [("Content-Type", CONTENT_TYPE_LATEST)])
            return [output]
        elif path == "/health":
            start_response("200 OK", [("Content-Type", "application/json")])
            return [b'{"status": "ok"}']
        else:
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found"]

    def start(self):
        if self._running:
            return
        self._running = True

        def _run():
            logger.info(f"Starting metrics server on {self.host}:{self.port}")
            httpd = make_server(self.host, self.port, self._wsgi_app, handler_class=WSGIRequestHandler)
            httpd.serve_forever()

        self._server_thread = threading.Thread(target=_run, daemon=True, name="metrics-server")
        self._server_thread.start()

    def stop(self):
        # we rely on process exit to stop; implement graceful shutdown if embedding in real server
        self._running = False

    # convenience wrappers for instrumentation
    def observe_inference(self, model_name: str, latency_seconds: float, input_size_bytes: int, success: bool = True):
        status = "success" if success else "failure"
        self.model_inference_count.labels(model=model_name, status=status).inc()
        self.model_inference_latency.labels(model=model_name).observe(latency_seconds)
        self.model_input_size.labels(model=model_name).set(input_size_bytes)
        if success:
            self.model_last_success.labels(model=model_name).set(time.time())
