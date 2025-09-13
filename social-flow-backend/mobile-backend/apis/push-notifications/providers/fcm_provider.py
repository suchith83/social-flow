# Firebase Cloud Messaging provider
"""
Simplified FCM provider. In production use google-cloud-messaging or pyfcm.
This implementation uses an HTTP POST to FCM legacy API (server key) as an illustration.
"""

import os
from typing import List, Dict, Any
import requests
from .base import PushProvider, ProviderResponse
from ..config import get_config
import logging

logger = logging.getLogger("push.providers.fcm")
config = get_config()


class FCMProvider(PushProvider):
    FCM_ENDPOINT = "https://fcm.googleapis.com/fcm/send"

    def __init__(self, server_key: str = None):
        self.server_key = server_key or config.FCM_SERVER_KEY

    def send_batch(self, tokens: List[str], title: str, body: str, payload: Dict[str, Any]) -> List[ProviderResponse]:
        if not self.server_key:
            logger.error("FCM server key not configured.")
            return [ProviderResponse(token=t, status="failed", provider_message={"error": "no_server_key"}) for t in tokens]

        # Build payload - batch via registration_ids
        message = {
            "registration_ids": tokens,
            "notification": {"title": title, "body": body},
            "data": payload or {}
        }
        headers = {
            "Authorization": f"key={self.server_key}",
            "Content-Type": "application/json"
        }
        try:
            resp = requests.post(self.FCM_ENDPOINT, json=message, headers=headers, timeout=10)
            resp.raise_for_status()
            body_json = resp.json()
            # The FCM legacy response contains per-token results; map simply here
            results = []
            # attempt to map results to tokens; fallback to single status
            if "results" in body_json and isinstance(body_json["results"], list):
                for tok, res in zip(tokens, body_json["results"]):
                    status = "ok" if "message_id" in res else "failed"
                    results.append(ProviderResponse(token=tok, status=status, provider_message=res))
            else:
                # fallback
                for tok in tokens:
                    results.append(ProviderResponse(token=tok, status="ok", provider_message=body_json))
            return results
        except Exception as e:
            logger.exception("FCM send batch error")
            return [ProviderResponse(token=t, status="failed", provider_message={"error": str(e)}) for t in tokens]
