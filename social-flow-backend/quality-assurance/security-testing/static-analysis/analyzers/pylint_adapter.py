"""
Adapter to run pylint programmatically and convert output to normalized findings.
Uses pylint's JSON reporter when available. If pylint is not installed, this adapter will
fall back to a subprocess invocation (requires pylint in PATH).
"""

import json
import subprocess
import shutil
from typing import List, Dict, Any
from ..utils import logger
from ..config import TOOLS

class PylintAdapter:
    def __init__(self, pylint_cmd: str = None):
        self.pylint_cmd = pylint_cmd or TOOLS.get("pylint_cmd", "pylint")
        self._has_pylint = shutil.which(self.pylint_cmd) is not None

    def run(self, paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run pylint on a list of file paths and return normalized findings.
        Findings format:
        {
            "file": str,
            "line": int,
            "message_id": str,
            "message": str,
            "symbol": str,
            "severity": "LOW"/"MEDIUM"/"HIGH"
        }
        """
        if not self._has_pylint:
            logger.warning("pylint not found on PATH; skipping pylint analysis")
            return []

        cmd = [self.pylint_cmd, "--output-format=json"] + paths
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True, timeout=300)
            if proc.stderr:
                logger.debug("pylint stderr: " + proc.stderr.strip())
            raw = proc.stdout.strip()
            if not raw:
                return []

            issues = json.loads(raw)
            findings = []
            for i in issues:
                severity = self._map_pylint_category_to_severity(i.get("type"))
                findings.append({
                    "file": i.get("path") or i.get("module"),
                    "line": i.get("line"),
                    "message_id": i.get("symbol") or i.get("message-id"),
                    "message": i.get("message"),
                    "severity": severity,
                    "tool": "pylint"
                })
            return findings
        except Exception as e:
            logger.exception("Failed to run pylint: %s", e)
            return []

    @staticmethod
    def _map_pylint_category_to_severity(cat: str) -> str:
        # pylint "convention", "refactor", "warning", "error", "fatal"
        if not cat:
            return "LOW"
        cat = cat.lower()
        if cat in ("error", "fatal"):
            return "HIGH"
        if cat == "warning":
            return "MEDIUM"
        return "LOW"
