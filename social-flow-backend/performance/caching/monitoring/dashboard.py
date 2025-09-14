from flask import Flask, jsonify
from threading import Thread
from typing import Callable

class CacheDashboard:
    """
    Simple Flask-based dashboard for cache monitoring.
    Displays metrics in JSON format (integrates with Grafana).
    """

    def __init__(self, metrics_provider: Callable):
        self.app = Flask(__name__)
        self.metrics_provider = metrics_provider

        @self.app.route("/metrics")
        def metrics():
            return jsonify(self.metrics_provider())

    def run(self, host="0.0.0.0", port=5001):
        thread = Thread(target=lambda: self.app.run(host=host, port=port), daemon=True)
        thread.start()
