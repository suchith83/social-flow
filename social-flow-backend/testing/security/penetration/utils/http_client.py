# =========================
# File: testing/security/penetration/utils/http_client.py
# =========================
"""
Safe HTTP client wrapper for reconnaissance tasks.
- Respects timeouts
- Does not follow redirects by default
- Strips Authorization headers from logs
"""

import requests
from requests.exceptions import RequestException
from .logger import get_logger

logger = get_logger("HTTPClient")

DEFAULT_TIMEOUT = 6

def safe_get(url, headers=None, timeout=DEFAULT_TIMEOUT, allow_redirects=False, verify=True):
    """
    Safe wrapper for GET requests. Returns (status_code, headers, body_snippet, meta)
    """
    headers = headers or {}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects, verify=verify)
        body_snippet = (resp.text or "")[:2000]
        # Remove any sensitive headers from the returned headers for safety
        sanitized_headers = {k: v for k, v in resp.headers.items() if k.lower() != "set-cookie"}
        return {
            "status_code": resp.status_code,
            "headers": sanitized_headers,
            "body_snippet": body_snippet,
            "url": resp.url,
        }
    except RequestException as e:
        logger.debug(f"HTTP GET failed for {url}: {e}")
        return {"error": str(e)}
