# =========================
# File: testing/security/penetration/scanners/vuln_scanner.py
# =========================
"""
Passive vulnerability scanner and fingerprinting:
- Collects banners, headers, versions (passive)
- Checks for default docs, exposed .git, .env, etc. (non-destructive)
- DOES NOT attempt exploits or payload injections here
"""

import re
from ..utils.logger import get_logger
from .web_scanner import headers_and_body

logger = get_logger("VulnScanner")

def fingerprint_web(url):
    """
    Return simple fingerprinting: server banner, powered-by, known plugins
    """
    resp = headers_and_body(url)
    fingerprint = {}
    if "headers" in resp:
        hdrs = resp["headers"]
        fingerprint["server"] = hdrs.get("Server")
        fingerprint["x_powered_by"] = hdrs.get("X-Powered-By")
        # detect common frameworks
        body = resp.get("body_snippet", "")
        if re.search(r"wp-content|wordpress", body, re.IGNORECASE):
            fingerprint["framework"] = "WordPress"
        elif re.search(r"django", body, re.IGNORECASE):
            fingerprint["framework"] = "Django"
        elif re.search(r"express|node", body, re.IGNORECASE):
            fingerprint["framework"] = "Express/Node"
    return fingerprint

def check_exposed_files(base_url, files=None):
    if files is None:
        files = [".env", ".git/config", "config.yml", "secrets.yml", ".DS_Store"]
    findings = {}
    for f in files:
        url = base_url.rstrip("/") + "/" + f
        resp = headers_and_body(url)
        status = resp.get("status_code")
        # treat common non-404 responses as worth surfacing
        findings[f] = {"url": url, "status_code": status, "snippet": resp.get("body_snippet")}
    return findings
