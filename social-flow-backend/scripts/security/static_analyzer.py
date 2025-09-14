# scripts/security/static_analyzer.py
import logging
import json
from typing import Dict, Any, List

from .utils import which, run_command, write_json

logger = logging.getLogger("security.static")


class StaticAnalyzer:
    """
    Orchestrate static analysis tools. Examples:
     - Python: bandit
     - JavaScript/TypeScript: eslint (with security plugin)
     - Go: gosec
    The class detects available tools and runs them with JSON outputs when possible.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("security", {}).get("static", {})
        self.tools = self.cfg.get("tools", ["bandit"])
        self.output_dir = config.get("security", {}).get("reporter", {}).get("output_dir", "./security-reports")
        self.enabled = bool(self.cfg.get("enabled", True))

    def run_bandit(self) -> Dict[str, Any]:
        if not which("bandit"):
            logger.warning("bandit not installed")
            return {"tool": None, "note": "bandit-missing"}
        rc, out, err = run_command(["bandit", "-r", ".", "-f", "json"])
        try:
            parsed = json.loads(out or "{}")
        except Exception:
            parsed = {"raw": out, "err": err}
        write_json(f"{self.output_dir}/static-bandit.json", parsed)
        return {"tool": "bandit", "data": parsed}

    def run_eslint(self) -> Dict[str, Any]:
        if not which("eslint"):
            logger.warning("eslint not found")
            return {"tool": None, "note": "eslint-missing"}
        rc, out, err = run_command(["eslint", ".", "-f", "json"])
        try:
            parsed = json.loads(out or "[]")
        except Exception:
            parsed = {"raw": out, "err": err}
        write_json(f"{self.output_dir}/static-eslint.json", parsed)
        return {"tool": "eslint", "data": parsed}

    def run_gosec(self) -> Dict[str, Any]:
        if not which("gosec"):
            logger.warning("gosec not found")
            return {"tool": None, "note": "gosec-missing"}
        rc, out, err = run_command(["gosec", "./...", "-fmt", "json"])
        try:
            parsed = json.loads(out or "{}")
        except Exception:
            parsed = {"raw": out, "err": err}
        write_json(f"{self.output_dir}/static-gosec.json", parsed)
        return {"tool": "gosec", "data": parsed}

    def run(self) -> Dict[str, Any]:
        if not self.enabled:
            logger.info("Static analysis disabled")
            return {}
        reports = {}
        for tool in self.tools:
            try:
                if tool == "bandit":
                    reports["bandit"] = self.run_bandit()
                elif tool == "eslint":
                    reports["eslint"] = self.run_eslint()
                elif tool == "gosec":
                    reports["gosec"] = self.run_gosec()
                else:
                    logger.warning("Unknown static tool requested: %s", tool)
            except Exception:
                logger.exception("Static analyzer failed for tool: %s", tool)
        return reports
