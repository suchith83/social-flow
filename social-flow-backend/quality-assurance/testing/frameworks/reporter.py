"""
Reporters transform raw adapter output into consumable formats:
- JSON aggregated report suitable for CI storage
- Human-friendly summary for console / logs
- Optionally integrate with coverage / badges (left to other packages)
"""

from typing import Dict, Any, List
import datetime
import os
import logging

from .config import FRAMEWORKS_CONFIG
from .utils import write_json_atomic

logger = logging.getLogger("qa-testing-frameworks.reporter")


class BaseReporter:
    def __init__(self, config=FRAMEWORKS_CONFIG):
        self.config = config

    def collect(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError()

    def persist(self, aggregated: Dict[str, Any]):
        raise NotImplementedError()


class JSONReporter(BaseReporter):
    """Aggregate multiple framework outputs into a single JSON file."""

    def collect(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregated = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "frameworks": results,
            "summary": self._build_summary(results),
        }
        return aggregated

    def _build_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = passed = failed = skipped = 0
        for r in results:
            raw = r.get("raw", {})
            counts = {}
            # try to extract counts from common shapes
            if isinstance(raw, dict) and "counts" in raw:
                counts = raw["counts"]
                total += counts.get("total", 0)
                passed += counts.get("passed", 0)
                failed += counts.get("failed", 0)
                skipped += counts.get("skipped", 0)
            elif isinstance(raw, dict) and "total" in raw:
                total += raw.get("total", 0)
                failed += raw.get("failures", 0) + raw.get("errors", 0)
                skipped += raw.get("skipped", 0)
            else:
                # best effort: if rc non-zero assume some failures
                total += 0
                if r.get("rc", 0) != 0:
                    failed += 1
        return {"total": total, "passed": passed, "failed": failed, "skipped": skipped}

    def persist(self, aggregated: Dict[str, Any]):
        out = self.config.json_report_path
        os.makedirs(os.path.dirname(out), exist_ok=True)
        write_json_atomic(out, aggregated)
        logger.info("Wrote aggregated test report to %s", out)
        return out


class SummaryReporter(BaseReporter):
    """Human-readable summary to console/logs."""

    def collect(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Build a minimalist summary
        lines = []
        for r in results:
            fw = r.get("framework")
            rc = r.get("rc")
            raw = r.get("raw", {})
            counts = {}
            if isinstance(raw, dict):
                if "counts" in raw:
                    counts = raw["counts"]
                elif "total" in raw:
                    counts = {"total": raw["total"], "failed": raw.get("failures", 0) + raw.get("errors", 0)}
            lines.append({"framework": fw, "rc": rc, "counts": counts})
        summary = {"generated_at": datetime.datetime.utcnow().isoformat() + "Z", "results": lines}
        return summary

    def persist(self, aggregated: Dict[str, Any]):
        # Print nicely to console
        for r in aggregated.get("results", []):
            framework = r["framework"]
            counts = r["counts"]
            if counts:
                total = counts.get("total", "?")
                failed = counts.get("failed", counts.get("failures", 0))
                skipped = counts.get("skipped", 0)
                logger.info("[%s] total=%s failed=%s skipped=%s", framework, total, failed, skipped)
            else:
                logger.info("[%s] rc=%s (no counts)", framework, r["rc"])
        return None
