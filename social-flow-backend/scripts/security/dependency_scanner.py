# scripts/security/dependency_scanner.py
import logging
import json
from typing import Dict, Any, List, Tuple, Optional

from .utils import which, run_command, write_json

logger = logging.getLogger("security.deps")


class DependencyScanner:
    """
    Run dependency scanners for multiple ecosystems. Prefer local tools if present:
      - Python: pip-audit (or safety)
      - Node: npm audit / yarn audit
      - Java: mvn dependency:check (if owasp plugin installed)
    The implementation wraps available tools and emits normalized JSON reports.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("security", {}).get("dependency", {})
        self.output_dir = config.get("security", {}).get("reporter", {}).get("output_dir", "./security-reports")
        self.enabled = bool(self.cfg.get("enabled", True))
        self.types = self.cfg.get("types", ["python"])

    def scan_python(self) -> Dict[str, Any]:
        """
        Use pip-audit if available, otherwise fall back to safety if present.
        """
        if not which("pip-audit") and not which("safety"):
            logger.warning("No python dependency scanner (pip-audit or safety) found")
            return {"tool": None, "ok": True, "note": "no-scanner-available"}

        if which("pip-audit"):
            rc, out, err = run_command(["pip-audit", "-f", "json"])
            if rc != 0:
                logger.warning("pip-audit returned non-zero rc: %s", rc)
            try:
                data = json.loads(out or "null")
            except Exception as e:
                logger.exception("Failed to parse pip-audit output")
                data = {"raw": out, "err": err}
            report = {"tool": "pip-audit", "data": data}
            write_json(f"{self.output_dir}/dependency-python-pip-audit.json", report)
            return report

        # safety fallback
        rc, out, err = run_command(["safety", "check", "--json"])
        try:
            data = json.loads(out or "null")
        except Exception:
            data = {"raw": out, "err": err}
        report = {"tool": "safety", "data": data}
        write_json(f"{self.output_dir}/dependency-python-safety.json", report)
        return report

    def scan_node(self) -> Dict[str, Any]:
        if which("npm"):
            # npm audit --json
            rc, out, err = run_command(["npm", "audit", "--json"])
            try:
                data = json.loads(out or "{}")
            except Exception:
                data = {"raw": out, "err": err}
            report = {"tool": "npm-audit", "data": data}
            write_json(f"{self.output_dir}/dependency-node-npm-audit.json", report)
            return report
        logger.warning("npm not present; skipping node dependency scan")
        return {"tool": None, "ok": True, "note": "npm-not-found"}

    def run(self) -> Dict[str, Any]:
        if not self.enabled:
            logger.info("Dependency scanning disabled in config")
            return {}
        reports = {}
        for t in self.types:
            if t == "python":
                reports["python"] = self.scan_python()
            elif t == "node":
                reports["node"] = self.scan_node()
            else:
                logger.warning("Unsupported dependency type: %s", t)
        return reports
