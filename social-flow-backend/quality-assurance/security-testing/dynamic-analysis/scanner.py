"""
Dynamic Scanner
Executes attack modules against application endpoints.
"""

from .attack_modules import AttackModules
from .vulnerability_database import VULNERABILITY_DB
from .utils import logger


class DynamicScanner:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.attacks = AttackModules(base_url)

    def scan(self, endpoints: list) -> dict:
        results = {"target": self.base_url, "vulnerabilities": []}

        for ep in endpoints:
            logger.info(f"Scanning endpoint {ep}")
            results["vulnerabilities"].extend(self.attacks.test_xss(ep))
            results["vulnerabilities"].extend(self.attacks.test_sqli(ep))
            results["vulnerabilities"].extend(self.attacks.test_ssrf(ep))

        # Enrich with severity & mitigation
        for vuln in results["vulnerabilities"]:
            details = VULNERABILITY_DB.get(vuln["type"], {})
            vuln.update(details)

        return results
