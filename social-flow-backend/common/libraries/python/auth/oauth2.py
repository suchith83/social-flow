# common/libraries/python/auth/oauth2.py
"""
OAuth2 client helper for token retrieval and validation.
Framework-agnostic, uses requests.
"""

import requests
from typing import Dict, Any

class OAuth2Client:
    def __init__(self, client_id: str, client_secret: str, token_url: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url

    def fetch_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for token."""
        response = requests.post(
            self.token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        response.raise_for_status()
        return response.json()
