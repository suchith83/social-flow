# =========================
# File: testing/security/penetration/scanners/web_scanner.py
# =========================
"""
Web reconnaissance:
- Retrieve safe HTTP headers and TLS info
- Check robots.txt, security headers
- Basic directory discovery using a small, configurable, safe wordlist and rate limiting
"""

import os
import ssl
import socket
from ..utils.http_client import safe_get
from ..utils.logger import get_logger
from ..utils.network import RateLimiter
import time

logger = get_logger("WebScanner")

SECURITY_HEADERS = [
    "content-security-policy", "x-frame-options", "x-xss-protection",
    "strict-transport-security", "x-content-type-options", "referrer-policy"
]

def tls_info(hostname, port=443, timeout=5):
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                protocol = ssock.version()
                return {"tls": True, "protocol": protocol, "cert": cert}
    except Exception as e:
        logger.debug(f"TLS check failed for {hostname}:{port} - {e}")
        return {"tls": False, "error": str(e)}

def headers_and_body(url):
    return safe_get(url)

def check_robots(base_url):
    url = base_url.rstrip("/") + "/robots.txt"
    return safe_get(url)

def directory_discovery(base_url, wordlist=None, max_checks=50, rate_limit_per_minute=60):
    """
    Non-destructive directory discovery:
    - Uses a small wordlist (default embedded or provided)
    - Respects rate limiting and small max checks
    - DOES NOT POST or attempt known vulnerable payloads
    """
    if not wordlist:
        # small default list of template directories typically safe to probe
        wordlist = ["admin", "login", "dashboard", "api", "robots.txt", "health", ".git"]
    rate_limiter = RateLimiter(rate_limit_per_minute, 60.0)
    results = {}
    checks = 0
    for entry in wordlist:
        if checks >= max_checks:
            break
        rate_limiter.wait()
        url = base_url.rstrip("/") + f"/{entry}"
        resp = safe_get(url)
        results[entry] = {"url": url, "response": resp}
        checks += 1
    return results
