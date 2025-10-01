# cdn_invalidator.py
# Created by Create-Invalidation.ps1
"""
cdn_invalidator.py
------------------
Utilities to invalidate cached assets at CDN edge.

Supports:
- HTTP invalidation API (e.g., Fastly, CloudFront, Cloudflare)
- batch invalidation with exponential backoff
- signed request support (API key/secret)
- path-based and tag-based invalidation (provider dependent)
"""

from __future__ import annotations
import time
import logging
from typing import Iterable, Dict, Optional
import requests

from .exceptions import RemoteInvalidateError

logger = logging.getLogger(__name__)


class CDNInvalidator:
    """
    Generic CDN invalidator built on top of HTTP APIs.
    Provider-specific implementations can subclass or configure endpoint/payload format.
    """

    def __init__(
        self,
        api_endpoint: str,
        auth: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        max_retries: int = 3,
    ):
        """
        api_endpoint: full URL to POST invalidation requests
        auth: optional dict like {"Authorization": "Bearer ..."} or {"api_key": "..."}
        """
        self.api_endpoint = api_endpoint
        self.auth = auth or {}
        self.timeout = timeout
        self.max_retries = max_retries

    def _do_request(self, payload: Dict) -> Dict:
        headers = {"Content-Type": "application/json"}
        headers.update(self.auth)
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(self.api_endpoint, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code in (200, 201, 202):
                    logger.debug("CDN invalidation accepted: %s", r.text)
                    return r.json() if r.text else {}
                # treat 429/5xx as retryable
                if r.status_code >= 500 or r.status_code == 429:
                    logger.warning("CDN invalidation returned %d: %s", r.status_code, r.text)
                    time.sleep(0.5 * attempt)
                    continue
                # otherwise treat as fatal
                raise RemoteInvalidateError(f"CDN invalidation failed: {r.status_code} {r.text}")
            except requests.RequestException as e:
                last_exc = e
                time.sleep(0.5 * attempt)
        raise RemoteInvalidateError(str(last_exc))

    def invalidate_paths(self, paths: Iterable[str]) -> Dict:
        """
        Invalidate a list of paths. Returns provider response (parsed JSON if available).
        Payload varies by provider. The default sends {"paths": [...]}.
        """
        p = list(paths)
        if not p:
            return {}
        payload = {"paths": p}
        logger.info("Sending CDN invalidation for %d paths", len(p))
        return self._do_request(payload)

    def invalidate_by_tag(self, tags: Iterable[str]) -> Dict:
        """Some CDNs support tag-based invalidation. Provider-specific."""
        t = list(tags)
        if not t:
            return {}
        payload = {"tags": t}
        logger.info("Sending CDN invalidation for %d tags", len(t))
        return self._do_request(payload)
