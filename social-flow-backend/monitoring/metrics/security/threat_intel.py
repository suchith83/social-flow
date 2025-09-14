# Integrates external threat intelligence feeds and enriches events
"""
Pluggable threat intelligence client.

Provides wrappers to check IP reputation / domain reputation with providers like AbuseIPDB,
VirusTotal, or other enterprise feeds. Implementations are intentionally minimal and pluggable.
"""

import logging
from typing import Dict, Any, Optional
from .config import SecurityMetricsConfig

logger = logging.getLogger("security_metrics.threatintel")

class ThreatIntelClient:
    """
    Simple facade for threat intelligence lookups.

    Usage:
       tic = ThreatIntelClient(api_keys={"abuseipdb": "...", "virustotal": "..."})
       verdict = tic.lookup_ip("1.2.3.4")
    """
    def __init__(self, api_keys: Optional[Dict[str, str]] = None, providers: Optional[list] = None):
        self.api_keys = api_keys or {}
        self.providers = providers or SecurityMetricsConfig.THREAT_INTEL_PROVIDERS

    def lookup_ip(self, ip: str) -> Dict[str, Any]:
        """
        Lookup an IP across configured providers. Returns aggregated verdict:
          {
            "ip": ip,
            "malicious": bool,
            "scores": {"abuseipdb": 0.8, "vt": 2},
            "raw": {...}
          }
        This implementation only contains placeholders to avoid shipping provider credentials in code.
        """
        verdict = {"ip": ip, "malicious": False, "scores": {}, "raw": {}}
        for p in self.providers:
            try:
                if p == "abuseipdb":
                    # placeholder: call AbuseIPDB API using self.api_keys.get("abuseipdb")
                    verdict["scores"]["abuseipdb"] = 0.0
                elif p == "virustotal":
                    verdict["scores"]["virustotal"] = 0
                else:
                    verdict["scores"][p] = None
            except Exception:
                logger.exception("Threat intel lookup failed for provider=%s", p)
        # simple aggregation rule: if any provider shows high risk, mark malicious
        for sc in verdict["scores"].values():
            try:
                if sc and float(sc) > 0.7:
                    verdict["malicious"] = True
            except Exception:
                continue
        return verdict
