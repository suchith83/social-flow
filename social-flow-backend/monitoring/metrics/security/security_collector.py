# Collects security events from logs, endpoints, and cloud providers
"""
Collect security-related telemetry and export to prometheus metrics.
Designed to be used by application services, log processors, or SIEM-forwarding agents.

Provides:
 - thread-safe recorder methods (record_auth_failure, record_suspicious_ip, record_privilege_escalation, ...)
 - lightweight in-memory counters for short-lived correlation
 - optional push to SIEM via configured hook (pluggable)
"""

import threading
import time
import logging
from typing import Optional, Dict, Any
from prometheus_client import Counter, Gauge
from .config import SecurityMetricsConfig
from .utils import parse_ip, hash_event_id

logger = logging.getLogger("security_metrics.collector")

class SecurityCollector:
    """
    Collects security events and updates metrics. Methods are safe to call from multiple threads.
    """

    def __init__(self, siem_push_callable: Optional[callable] = None):
        self._lock = threading.Lock()
        self._last_seen: Dict[str, float] = {}  # event_id -> ts for basic dedupe/TTL

        # Prometheus metrics
        self.auth_failures = Counter(
            "security_auth_failures_total",
            "Total authentication failures observed",
            ["auth_type", "reason", "realm"]
        )
        self.suspicious_ip_count = Counter(
            "security_suspicious_ip_total",
            "Suspicious IP addresses observed",
            ["source", "threat_level"]
        )
        self.privilege_escalations = Counter(
            "security_privilege_escalations_total",
            "Detected privilege escalation attempts",
            ["user", "method"]
        )
        self.misconfigurations = Counter(
            "security_misconfigurations_total",
            "Detected security misconfigurations (e.g., open S3 buckets)",
            ["resource_type", "severity"]
        )
        self.active_incidents = Gauge(
            "security_active_incidents",
            "Number of active security incidents being tracked"
        )

        # Counters for suspicious behaviour windows
        self.brute_force_window = {}  # username -> [timestamps]

        # SIEM push hook (optional)
        self.siem_push = siem_push_callable if siem_push_callable is not None else None

        # Ingestion rate limiting
        self._ingest_tokens = SecurityMetricsConfig.INGEST_RATE_PER_SEC
        self._last_token_ts = time.time()
        self._token_lock = threading.Lock()

    def _consume_token(self) -> bool:
        """Token bucket style rate limiting for ingestion (simple)."""
        with self._token_lock:
            now = time.time()
            elapsed = now - self._last_token_ts
            # refill tokens
            refill = int(elapsed) * SecurityMetricsConfig.INGEST_RATE_PER_SEC
            if refill > 0:
                self._ingest_tokens = min(SecurityMetricsConfig.INGEST_RATE_PER_SEC, self._ingest_tokens + refill)
                self._last_token_ts = now
            if self._ingest_tokens <= 0:
                return False
            self._ingest_tokens -= 1
            return True

    def _maybe_push_siem(self, event: Dict[str, Any]):
        """Push event to SIEM if configured; swallow errors (do not break flow)."""
        if SecurityMetricsConfig.ENABLE_SIEM_PUSH and self.siem_push:
            try:
                self.siem_push(event)
            except Exception:
                logger.exception("SIEM push failed")

    def _should_record(self, event_id: str, ttl_seconds: int = 300) -> bool:
        """Basic dedupe: ignore events with same id within ttl_seconds."""
        with self._lock:
            last = self._last_seen.get(event_id)
            now = time.time()
            if last and (now - last) < ttl_seconds:
                logger.debug("Event deduped: %s", event_id)
                return False
            self._last_seen[event_id] = now
            return True

    # Public recording APIs -------------------------------------------------
    def record_auth_failure(self, username: str, ip: str, auth_type: str = "password", reason: str = "invalid_credentials", realm: str = "app"):
        """
        Record an authentication failure. This increments Prometheus counters and optionally pushes to SIEM.
        """
        if not self._consume_token():
            logger.debug("Ingest rate limited: auth_failure")
            return

        try:
            ip_obj = parse_ip(ip)
            ip_str = str(ip_obj)
        except Exception:
            ip_str = ip or "unknown"

        event_id = hash_event_id("auth_failure", username, ip_str, reason)
        if not self._should_record(event_id):
            return

        with self._lock:
            self.auth_failures.labels(auth_type, reason, realm).inc()

        # Optional SIEM push
        event = {
            "type": "auth_failure",
            "username": username,
            "ip": ip_str,
            "auth_type": auth_type,
            "reason": reason,
            "realm": realm,
            "ts": int(time.time())
        }
        self._maybe_push_siem(event)

        # simple brute-force detection heuristic
        self._record_bruteforce_candidate(username, ip_str)

    def _record_bruteforce_candidate(self, username: str, ip_str: str, window_seconds: int = 300, thresh: int = 10):
        """Track recent failures per username and raise incident if threshold exceeded."""
        now = time.time()
        with self._lock:
            timestamps = self.brute_force_window.get(username, [])
            timestamps = [ts for ts in timestamps if now - ts < window_seconds]
            timestamps.append(now)
            self.brute_force_window[username] = timestamps

            if len(timestamps) >= thresh:
                # record a suspicious IP event and an incident gauge bump
                self.suspicious_ip_count.labels(source="bruteforce", threat_level="high").inc()
                self.active_incidents.inc()
                # Reset window to avoid repeated hits
                self.brute_force_window[username] = []
                logger.warning("Brute-force detected for username=%s count=%s", username, len(timestamps))

    def record_suspicious_ip(self, ip: str, source: str = "external", threat_level: str = "medium", ctx: Optional[dict] = None):
        """Record a suspicious IP sighting; ctx can contain raw evidence."""
        if not self._consume_token():
            logger.debug("Ingest rate limited: suspicious_ip")
            return

        try:
            ip_obj = parse_ip(ip)
            ip_str = str(ip_obj)
        except Exception:
            ip_str = ip or "unknown"

        event_id = hash_event_id("suspicious_ip", ip_str, source, threat_level)
        if not self._should_record(event_id):
            return

        with self._lock:
            self.suspicious_ip_count.labels(source, threat_level).inc()

        event = {
            "type": "suspicious_ip",
            "ip": ip_str,
            "source": source,
            "threat_level": threat_level,
            "ctx": ctx or {},
            "ts": int(time.time())
        }
        self._maybe_push_siem(event)

    def record_privilege_escalation(self, user: str, method: str, evidence: Optional[dict] = None):
        """Record suspected privilege escalation attempts or successful events."""
        if not self._consume_token():
            logger.debug("Ingest rate limited: privilege_escalation")
            return

        event_id = hash_event_id("priv_escalation", user, method)
        if not self._should_record(event_id):
            return

        with self._lock:
            self.privilege_escalations.labels(user=user, method=method).inc()
            self.active_incidents.inc()

        event = {
            "type": "privilege_escalation",
            "user": user,
            "method": method,
            "evidence": evidence or {},
            "ts": int(time.time())
        }
        self._maybe_push_siem(event)

    def record_misconfiguration(self, resource_type: str, severity: str = "medium", description: Optional[str] = None):
        """Record detected misconfiguration (open buckets, weak ACLs, etc.)"""
        if not self._consume_token():
            logger.debug("Ingest rate limited: misconfiguration")
            return

        event_id = hash_event_id("misconfig", resource_type, severity, description)
        if not self._should_record(event_id):
            return

        with self._lock:
            self.misconfigurations.labels(resource_type, severity).inc()
            self.active_incidents.inc()

        event = {
            "type": "misconfiguration",
            "resource_type": resource_type,
            "severity": severity,
            "description": description,
            "ts": int(time.time())
        }
        self._maybe_push_siem(event)

    def resolve_incident(self):
        """Decrement active_incidents gauge (call when incident is resolved)."""
        with self._lock:
            try:
                self.active_incidents.dec()
            except Exception:
                # gauge can't go below 0; ignore
                pass
