# scripts/deployment/health_checker.py
import logging
import requests
from typing import Dict, Any


class HealthChecker:
    """
    Performs health checks after deployment to ensure services are running correctly.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoints = config.get("health", {}).get("endpoints", [])

    def check(self):
        logging.info("🩺 Performing health checks...")
        for url in self.endpoints:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logging.info(f"✅ Healthy: {url}")
                else:
                    logging.warning(f"⚠️ Unhealthy: {url} - Status {response.status_code}")
            except Exception as e:
                logging.error(f"❌ Failed to check {url}: {e}")
