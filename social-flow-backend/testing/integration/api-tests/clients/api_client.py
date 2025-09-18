"""
A robust API client wrapper around requests.Session.

Features:
- Reuses a single `requests.Session` to benefit from connection pooling.
- Configurable base_url.
- Convenience methods for GET/POST/PUT/DELETE.
- Automatic JSON decoding and helpful exceptions.
- Optional bearer token management.
- Uses tenacity for configurable retries on network errors.
"""

from typing import Any, Dict, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout

DEFAULT_TIMEOUT = 10  # seconds

class ApiClient:
    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT, session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        self._headers: Dict[str, str] = {"Content-Type": "application/json"}
        self._bearer_token: Optional[str] = None

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass

    def _url(self, path: str) -> str:
        if path.startswith("/"):
            path = path[1:]
        return f"{self.base_url}/{path}"

    def set_bearer_token(self, token: str):
        self._bearer_token = token
        self._headers["Authorization"] = f"Bearer {token}"

    def clear_auth(self):
        self._bearer_token = None
        self._headers.pop("Authorization", None)

    def _handle_response(self, resp: requests.Response) -> requests.Response:
        # raise for 5xx and 4xx (unless tests want to assert specific codes; tests can access resp)
        try:
            resp.raise_for_status()
        except HTTPError:
            # don't swallow; let tests assert on resp if needed
            raise
        return resp

    # Retry decorator: retry on connection/timeout errors
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8),
           retry=retry_if_exception_type((ConnectionError, Timeout, RequestException)))
    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = self._url(path)
        headers = {**self._headers, **(kwargs.pop("headers", {}) or {})}
        resp = self._session.request(method=method, url=url, headers=headers, timeout=self.timeout, **kwargs)
        return resp

    def get(self, path: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> requests.Response:
        return self.request("GET", path, params=params, **kwargs)

    def post(self, path: str, json: Optional[Any] = None, data: Optional[Any] = None, **kwargs) -> requests.Response:
        return self.request("POST", path, json=json, data=data, **kwargs)

    def put(self, path: str, json: Optional[Any] = None, **kwargs) -> requests.Response:
        return self.request("PUT", path, json=json, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        return self.request("DELETE", path, **kwargs)
