"""
Adapter to run eslint and normalize findings.
Relies on eslint being installed and available on PATH. We request JSON output.
If eslint is not available the adapter will no-op (but the custom rule engine may still find issues).
"""

import json
import subprocess
import shutil
from typing import List, Dict, Any
from ..utils import logger
from ..config import TOOLS

class ESLintAdapter:
    def __init__(self, eslint_cmd: str = None):
        self.eslint_cmd = eslint_cmd or TOOLS.get("eslint_cmd", "eslint")
        self._has_eslint = shutil.which(self.eslint_cmd) is not None

    def run(self, paths: List[str]) -> List[Dict[str, Any]]:
        if not self._has_eslint:
            logger.warning("eslint not found on PATH; skipping eslint analysis")
            return []
        cmd = [self.eslint_cmd, "-f", "json"] + paths
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True, timeout=300)
            if proc.stderr:
                logger.debug("eslint stderr: " + proc.stderr.strip())
            raw = proc.stdout.strip()
            if not raw:
                return []
            results = json.loads(raw)
            findings = []
            for file_report in results:
                file_path = file_report.get("filePath")
                for msg in file_report.get("messages", []):
                    findings.append({
                        "file": file_path,
                        "line": msg.get("line"),
                        "message_id": msg.get("ruleId"),
                        "message": msg.get("message"),
                        "severity": self._map_eslint_severity(msg.get("severity")),
                        "tool": "eslint"
                    })
            return findings
        except Exception as e:
            logger.exception("Failed to run eslint: %s", e)
            return []

    @staticmethod
    def _map_eslint_severity(num: int) -> str:
        # eslint severity: 1 = warning, 2 = error
        if num == 2:
            return "HIGH"
        if num == 1:
            return "MEDIUM"
        return "LOW"
