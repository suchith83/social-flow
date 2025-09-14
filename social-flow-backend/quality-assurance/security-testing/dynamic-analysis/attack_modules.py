"""
Attack modules for simulating common vulnerabilities.
"""

import requests
from .config import PAYLOADS, SCAN_CONFIG
from .utils import logger


class AttackModules:
    def __init__(self, base_url: str = SCAN_CONFIG["base_url"]):
        self.base_url = base_url

    def test_xss(self, endpoint: str):
        """Inject XSS payloads."""
        vulns = []
        for payload in PAYLOADS["xss"]:
            try:
                resp = requests.get(f"{self.base_url}{endpoint}", params={"q": payload}, timeout=SCAN_CONFIG["timeout"])
                if payload in resp.text:
                    logger.warning(f"[XSS] Reflected payload detected at {endpoint}")
                    vulns.append({"type": "XSS", "payload": payload, "endpoint": endpoint})
            except Exception as e:
                logger.error(f"XSS test failed at {endpoint}: {e}")
        return vulns

    def test_sqli(self, endpoint: str):
        """Inject SQLi payloads."""
        vulns = []
        for payload in PAYLOADS["sqli"]:
            try:
                resp = requests.get(f"{self.base_url}{endpoint}", params={"id": payload}, timeout=SCAN_CONFIG["timeout"])
                if "syntax error" in resp.text.lower() or "sql" in resp.text.lower():
                    logger.warning(f"[SQLi] Injection vulnerability detected at {endpoint}")
                    vulns.append({"type": "SQLi", "payload": payload, "endpoint": endpoint})
            except Exception as e:
                logger.error(f"SQLi test failed at {endpoint}: {e}")
        return vulns

    def test_ssrf(self, endpoint: str):
        """Test for SSRF injection."""
        vulns = []
        for payload in PAYLOADS["ssrf"]:
            try:
                resp = requests.get(f"{self.base_url}{endpoint}", params={"url": payload}, timeout=SCAN_CONFIG["timeout"])
                if "127.0.0.1" in resp.text or "root:" in resp.text:
                    logger.warning(f"[SSRF] Possible SSRF vulnerability at {endpoint}")
                    vulns.append({"type": "SSRF", "payload": payload, "endpoint": endpoint})
            except Exception as e:
                logger.error(f"SSRF test failed at {endpoint}: {e}")
        return vulns
