"""
Dependency Scanner
"""

from typing import List, Dict
from .vulnerability_database import DependencyVulnerabilityDatabase
from .utils import logger


class DependencyScanner:
    def __init__(self):
        self.vuln_db = DependencyVulnerabilityDatabase()

    def scan(self, ecosystem: str, dependencies: List[tuple]) -> Dict:
        """Scan dependencies against OSV DB."""
        findings = {"ecosystem": ecosystem, "dependencies": []}
        for pkg, ver in dependencies:
            vulns = self.vuln_db.query(ecosystem, pkg, ver)
            findings["dependencies"].append({
                "package": pkg,
                "version": ver,
                "vulnerabilities": vulns
            })
            if vulns:
                logger.warning(f"{pkg}@{ver} has {len(vulns)} vulnerabilities")
        return findings
