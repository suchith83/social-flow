# Threat detection logic and alerts
"""
Threat Detection Engine

- Combines rule-based matching, anomaly scoring, and lightweight enrichment to surface threats.
- Exposes methods:
  - scan_bundle(telemetry) -> structured findings
  - score_anomaly(metric_series) -> float
"""

from typing import Dict, Any, List
import statistics


class ThreatDetector:
    def __init__(self, anomaly_z_threshold: float = 3.0):
        self.z_threshold = anomaly_z_threshold

    def scan_bundle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        findings: Dict[str, Any] = {}

        # 1) High severity alerts count
        high_count = telemetry.get("alerts.high", 0)
        findings["alerts_high_count"] = high_count

        # 2) Failed login anomalies
        failed_series = telemetry.get("auth.failed_logins", [])
        if isinstance(failed_series, list) and failed_series:
            anomaly_score = self.score_anomaly(failed_series)
            findings["failed_login_anomaly_score"] = anomaly_score
            if anomaly_score >= 1.0:
                # identify possible brute-force
                suspicious_ips = telemetry.get("suspicious_ips", [])
                findings["brute_force_candidates"] = suspicious_ips[:5]

        # 3) IDS signature hits
        ids_alerts = telemetry.get("ids_alerts", [])
        if ids_alerts:
            # prioritize by confidence
            high_conf = [a for a in ids_alerts if a.get("confidence", 0) >= 0.85]
            findings["ids_high_confidence_count"] = len(high_conf)
            findings.setdefault("suspicious_signatures", []).extend([a["signature"] for a in high_conf])

        # 4) Suspicious IP aggregation / scoring
        suspicious_ips = telemetry.get("suspicious_ips", [])
        if suspicious_ips:
            # score IPs by naive heuristics (hits, cross-backend presence)
            ip_scores = self._score_ips(telemetry)
            # select top suspects
            top = sorted(ip_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
            findings["suspicious_ips"] = [ip for ip, score in top]

        # 5) Vulnerability early-warning
        crit_vulns = telemetry.get("vulns.critical_open", telemetry.get("vulns_critical_open", 0))
        findings["critical_vulns_open"] = crit_vulns
        if crit_vulns > 25:
            findings["vuln_risk"] = "high"
        elif crit_vulns > 5:
            findings["vuln_risk"] = "medium"
        else:
            findings["vuln_risk"] = "low"

        return findings

    def score_anomaly(self, series: List[float]) -> float:
        """
        Return a normalized anomaly score:
         - compute z of latest point: (latest - mean)/stdev
         - returns 0.0 for no anomaly, >1.0 indicates notable anomaly, >3 extreme
        """
        if not series or len(series) < 3:
            return 0.0
        mean = statistics.mean(series[:-1])  # exclude latest to avoid self-bias
        stdev = statistics.pstdev(series[:-1]) or 0.0
        latest = series[-1]
        if stdev == 0:
            return 0.0
        z = (latest - mean) / stdev
        # normalize to 0..+inf; divide by configured threshold to yield a relative score
        return max(0.0, z / self.z_threshold)

    def _score_ips(self, telemetry: Dict[str, Any]) -> Dict[str, float]:
        """
        Simple IP scoring:
         - base weight from observed hits (if details exist)
         - add bonus if present in IDS alerts
         - add small randomness for tie-breaking
        """
        scores: Dict[str, float] = {}
        ip_details = {d["ip"]: d for d in telemetry.get("suspicious_ip_details", [])} if telemetry.get("suspicious_ip_details") else {}
        ids_alerts = telemetry.get("ids_alerts", [])
        ids_ips = {a["src_ip"] for a in ids_alerts}

        for ip in telemetry.get("suspicious_ips", []):
            base = float(ip_details.get(ip, {}).get("hits", 1))
            score = base
            if ip in ids_ips:
                score += 50.0
            # small tie-breaker: presence in multiple sources
            score += 1.0 if ip in ip_details else 0.0
            scores[ip] = round(score, 2)

        return scores
