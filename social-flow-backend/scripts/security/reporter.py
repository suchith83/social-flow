# scripts/security/reporter.py
import logging
import os
import json
from typing import Dict, Any, List, Optional

from .utils import write_json, safe_mkdir

logger = logging.getLogger("security.reporter")


class Reporter:
    """
    Aggregate scan results and write reports in JSON and (optionally) notify via Slack.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("security", {})
        self.output_dir = config.get("security", {}).get("reporter", {}).get("output_dir", "./security-reports")
        safe_mkdir(self.output_dir)
        self.slack_webhook = config.get("security", {}).get("reporter", {}).get("slack_webhook")

    def aggregate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and write a top-level report file.
        """
        report = {
            "summary": {
                "total_findings": 0,
                "by_component": {}
            },
            "details": results
        }
        # naive summary counting
        total = 0
        by_comp = {}
        for k, v in results.items():
            # attempt to count issues in common formats
            count = 0
            if isinstance(v, list):
                count = len(v)
            elif isinstance(v, dict):
                # look for known keys
                if "data" in v and isinstance(v["data"], list):
                    count = len(v["data"])
                elif "data" in v and isinstance(v["data"], dict):
                    # try to find vulnerabilities list
                    # best-effort parse
                    count = sum(len(vv) if isinstance(vv, list) else 0 for vv in v["data"].values())
                else:
                    count = 1
            else:
                count = 0
            total += count
            by_comp[k] = count
        report["summary"]["total_findings"] = total
        report["summary"]["by_component"] = by_comp

        # write report
        path = os.path.join(self.output_dir, "security-scan-report.json")
        write_json(path, report)
        logger.info("Wrote aggregated security report to %s", path)

        # optional slack notify (short message)
        if self.slack_webhook:
            try:
                import requests
                text = f"Security scan completed. Total findings: {total}. See report at {path}"
                requests.post(self.slack_webhook, json={"text": text}, timeout=5)
            except Exception:
                logger.exception("Failed to send slack notification")

        return report
