# scripts/security/dynamic_scanner.py
import logging
from typing import Dict, Any, List
from .utils import which, run_command, write_json

logger = logging.getLogger("security.dynamic")


class DynamicScanner:
    """
    Lightweight dynamic scanner orchestration. Use OWASP ZAP if present.
    For higher trust environments, use ZAP API/docker image to run active scans.
    This file intentionally keeps integration local and non-invasive by default.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("security", {}).get("dynamic", {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.output_dir = config.get("security", {}).get("reporter", {}).get("output_dir", "./security-reports")

    def zap_baseline(self, target: str) -> Dict[str, Any]:
        """
        Attempt to run 'zap-baseline.py' from ZAP if available in PATH (as provided by ZAP package).
        This is a conservative, read-only scan.
        """
        if not which("zap-baseline.py"):
            logger.warning("ZAP baseline script not found")
            return {"tool": None, "note": "zap-not-found"}
        cmd = ["zap-baseline.py", "-t", target, "-r", "zap-baseline-report.html", "-j"]
        rc, out, err = run_command(cmd, timeout=1800)
        # ZAP's -j option may produce JSON on stdout or outputs a file; handle gracefully
        try:
            # attempt to parse stdout
            import json as _json
            data = _json.loads(out) if out else {"stdout": out}
        except Exception:
            data = {"stdout": out, "stderr": err}
        write_json(f"{self.output_dir}/dynamic-zap-baseline-{target.replace('/', '_')}.json", data)
        return {"tool": "zap-baseline", "target": target, "data": data}

    def run(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            logger.info("Dynamic scanning disabled")
            return []
        targets = self.cfg.get("targets", [])
        results = []
        for t in targets:
            try:
                results.append(self.zap_baseline(t))
            except Exception:
                logger.exception("Dynamic scan failed for %s", t)
        return results
