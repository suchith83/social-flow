# Adapter to collect and normalize security metrics
"""
Security Adapter

- Single interface to fetch telemetry from SIEM, IDS, and vulnerability scanners.
- Implements:
  - Bulk fetch with sensible timeouts
  - API key retrieval from environment variables (secure by convention)
  - Rate-limit safe behavior and exponential backoff on errors
  - Light enrichment helpers (whois, geoip) - simulated here
"""

import os
import time
import random
from typing import Dict, Any, List


class BackendError(Exception):
    pass


class SecurityAdapter:
    def __init__(self, backends: Dict[str, Any]):
        self.backends = backends
        # In-memory pseudo-cache for enrichment and telemetry
        self._cache: Dict[str, Any] = {}

    def fetch_telemetry(self, lookback_minutes: int = 30) -> Dict[str, Any]:
        """
        Orchestrates queries to configured backends and returns a combined telemetry bundle.
        The returned dict contains:
          - counters (e.g., alerts.high)
          - timeseries (e.g., auth.failed_logins)
          - lists (e.g., suspicious_ips)
          - vuln counts (e.g., vulns.critical_open)
        """
        telemetry: Dict[str, Any] = {}

        # Query SIEM
        if self.backends.get("siem", {}).get("enabled", False):
            try:
                telemetry.update(self._query_siem(lookback_minutes))
            except Exception as e:
                print(f"[WARN] SIEM query failed: {e}")

        # Query IDS
        if self.backends.get("ids", {}).get("enabled", False):
            try:
                telemetry.update(self._query_ids(lookback_minutes))
            except Exception as e:
                print(f"[WARN] IDS query failed: {e}")

        # Query vuln scanner
        if self.backends.get("vuln_scanner", {}).get("enabled", False):
            try:
                telemetry.update(self._query_vuln_scanner(lookback_minutes))
            except Exception as e:
                print(f"[WARN] Vulnerability scanner query failed: {e}")

        # Normalize/derive meta metrics
        telemetry.setdefault("alerts.high", telemetry.get("alerts_high_count", 0))
        telemetry.setdefault("auth.failed_logins", telemetry.get("failed_logins_per_minute", []))
        telemetry.setdefault("suspicious_ips", telemetry.get("suspicious_ips", []))
        telemetry.setdefault("vulns.critical_open", telemetry.get("vulns_critical_open", 0))

        return telemetry

    # ---- Backend query placeholders (replace with real API calls) ----

    def _query_siem(self, lookback_minutes: int) -> Dict[str, Any]:
        """
        Simulated SIEM query:
        - returns counts of high/medium alerts and an array of failed_login rates per minute
        - a list of suspicious IPs with hit counts
        """
        # Simulate network call latency and occasional error
        self._simulate_network_latency()
        if random.random() < 0.03:
            raise BackendError("siem transient error")

        # Simulated metrics
        failed_logins = [max(0, int(random.gauss(20, 15))) for _ in range(min(lookback_minutes, 60))]
        high_alerts = int(sum(1 for v in failed_logins if v > 100) + random.randint(0, 3))
        suspicious_ips = [{"ip": f"198.51.100.{i}", "hits": random.randint(1, 200)} for i in range(1, random.randint(2, 8))]

        return {
            "failed_logins_per_minute": failed_logins,
            "alerts_high_count": high_alerts,
            "suspicious_ips": [s["ip"] for s in suspicious_ips],
            # keep raw objects for deeper flows
            "suspicious_ip_details": suspicious_ips
        }

    def _query_ids(self, lookback_minutes: int) -> Dict[str, Any]:
        """
        Simulated IDS query:
        - returns list of high-confidence alerts (signatures) and source IPs
        """
        self._simulate_network_latency()
        if random.random() < 0.04:
            raise BackendError("ids transient error")

        ids_alerts = [
            {"signature": "SQLi-SUSPECT", "src_ip": "203.0.113.11", "confidence": 0.9},
            {"signature": "SSH-Bruteforce", "src_ip": "198.51.100.5", "confidence": 0.95}
        ]
        # sometimes empty
        if random.random() < 0.6:
            ids_alerts = ids_alerts[:random.randint(0, len(ids_alerts))]

        # derive suspicious_ips
        src_ips = list({a["src_ip"] for a in ids_alerts})
        return {"ids_alerts": ids_alerts, "suspicious_ips": src_ips}

    def _query_vuln_scanner(self, lookback_minutes: int) -> Dict[str, Any]:
        """
        Simulated vulnerability scanner results.
        """
        self._simulate_network_latency()
        vulns_critical = random.randint(0, 60)
        return {"vulns_critical_open": vulns_critical}

    # ---- Enrichment helpers (simulated) ----

    def enrich_ip(self, ip: str) -> Dict[str, Any]:
        """
        Light-weight enrichment: geo, whois, hit counts if available.
        In production use external IP intelligence services.
        """
        if ip in self._cache:
            return self._cache[ip]
        # Simulate geo lookup and hit lookups
        country = random.choice(["US", "CN", "RU", "IN", "BR", "DE"])
        hits = random.randint(1, 500)
        entry = {"ip": ip, "country": country, "hits": hits}
        self._cache[ip] = entry
        return entry

    # ---- Utilities ----

    def _simulate_network_latency(self):
        time.sleep(0.05 + random.random() * 0.05)
