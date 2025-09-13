# Interact with Grafana HTTP API
"""
grafana_api.py
Small helper to push dashboards to Grafana using its HTTP API.
Note: in production use the grafana_client library or grafana-http-api wrapper.
"""

import requests
from utils import setup_logger
import json
from typing import Dict

logger = setup_logger("GrafanaAPI")


class GrafanaAPI:
    def __init__(self, base_url: str, api_key: str, verify: bool = True):
        self.base = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.verify = verify

    def create_or_update_dashboard(self, dashboard_json: Dict, folder_id: int = 0, overwrite: bool = True):
        payload = {
            "dashboard": dashboard_json,
            "folderId": folder_id,
            "overwrite": overwrite
        }
        url = f"{self.base}/api/dashboards/db"
        logger.info(f"Pushing dashboard to Grafana: {url}")
        r = requests.post(url, headers=self.headers, json=payload, verify=self.verify, timeout=20)
        r.raise_for_status()
        logger.info("Grafana dashboard uploaded successfully")
        return r.json()

    def create_folder(self, title: str):
        url = f"{self.base}/api/folders"
        r = requests.post(url, headers=self.headers, json={"title": title}, verify=self.verify, timeout=10)
        r.raise_for_status()
        return r.json()
