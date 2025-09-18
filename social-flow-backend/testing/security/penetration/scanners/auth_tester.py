# =========================
# File: testing/security/penetration/scanners/auth_tester.py
# =========================
"""
Auth tester: strictly a simulator by default.
- Offers a 'simulate' mode where attempts are not sent but counted.
- If 'execute' is explicitly enabled (and user confirms), actual attempts occur and are heavily rate-limited.
- Ensure operational safety and responsibility.
"""

import time
from ..utils.logger import get_logger
from ..utils.network import RateLimiter
from ..utils.http_client import safe_get, DEFAULT_TIMEOUT

logger = get_logger("AuthTester")

def simulate_bruteforce(username, wordlist, max_attempts=10):
    """
    Simulates attempts by iterating over wordlist and returning what WOULD be attempted.
    Non-destructive and safe.
    """
    attempts = []
    for i, pw in enumerate(wordlist):
        if i >= max_attempts:
            break
        attempts.append({"attempt_no": i + 1, "username": username, "password": pw})
    return {"simulated": True, "attempts": attempts}

def execute_bruteforce(url, username, wordlist, rate_limit_per_minute=10, max_attempts=50, verify_ssl=True):
    """
    Execute attempts against a login endpoint.
    WARNING: This WILL send requests. Requires explicit confirmation before using.
    Uses RateLimiter to throttle attempts.
    Returns list of responses (without recording full response bodies to avoid logging sensitive content)
    """
    rl = RateLimiter(rate_limit_per_minute, 60.0)
    results = []
    attempts = 0
    for pw in wordlist:
        if attempts >= max_attempts:
            break
        rl.wait()
        # Example: POST to url with username/password. Real use must adapt to real form fields.
        try:
            # For safety this uses GET to the URL with query params -- in practice you'd use POST and proper headers.
            # Using safe_get to capture response metadata only.
            resp = safe_get(f"{url}?user={username}&password={pw}", timeout=DEFAULT_TIMEOUT, verify=verify_ssl)
            results.append({"attempt": attempts + 1, "password": pw, "status": resp.get("status_code"), "url_checked": resp.get("url")})
        except Exception as e:
            results.append({"attempt": attempts + 1, "password": pw, "error": str(e)})
        attempts += 1
    return {"simulated": False, "results": results}
