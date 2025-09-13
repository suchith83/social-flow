# Helpers for security incident response
"""
Incident Response Helper

- Maps findings to playbooks (non-destructive by default)
- Provides safe helper functions to produce suggested actions and notifications
- Playbooks are represented as simple tokenized strings in config and executed
  by composing pre-defined action snippets. Actual execution should be gated
  behind manual approval and proper auth checks.
"""

from typing import Dict, Any, List


class IncidentResponseHelper:
    def __init__(self, playbooks: Dict[str, str]):
        # playbooks: intent -> tokenized action string (e.g., "isolate_host && enrich_whois && notify_team")
        self.playbooks = playbooks
        self.action_map = {
            "isolate_host": self._act_isolate_host,
            "enrich_whois": self._act_enrich_whois,
            "notify_security_team": self._act_notify_team,
            "block_ip": self._act_block_ip,
            "notify_oncall": self._act_notify_oncall,
            "create_incident": self._act_create_incident
        }

    def suggest_playbook_for_ip(self, ip: str) -> Dict[str, Any]:
        """
        Suggest a playbook for a suspicious IP based on simple heuristics.
        Returns a dict with: playbook_tokens, simulated_actions (strings), rationale
        """
        # naive heuristics: block if IP in private threatlist (simulated)
        tokens = []
        if ip.startswith("198.51.100") or ip.startswith("203.0.113"):
            tokens = self._get_playbook_tokens("suspicious_ip")
        else:
            tokens = self._get_playbook_tokens("brute_force")

        simulated = [self.action_map.get(t, self._act_unknown)(ip) for t in tokens]
        return {"ip": ip, "tokens": tokens, "simulated_actions": simulated, "rationale": "heuristic match"}

    def _get_playbook_tokens(self, intent: str) -> List[str]:
        raw = self.playbooks.get(intent, "")
        return [t.strip() for t in raw.split("&&") if t.strip()]

    # ---- simulated action implementations ----
    def _act_isolate_host(self, ip: str) -> str:
        return f"[SIM] isolate host with IP {ip} (network policy suggestion)"

    def _act_enrich_whois(self, ip: str) -> str:
        return f"[SIM] whois/enrich for {ip} (org: ExampleCorp, abuse: abuse@example.com)"

    def _act_notify_team(self, ip: str) -> str:
        return f"[SIM] notify security team about {ip} via slack #sec-ops"

    def _act_block_ip(self, ip: str) -> str:
        return f"[SIM] block ip {ip} in firewall (proposed rule id: sim-123)"

    def _act_notify_oncall(self, ip: str) -> str:
        return f"[SIM] page oncall for ip {ip}"

    def _act_create_incident(self, ip: str) -> str:
        return f"[SIM] create incident ticket for {ip} in tracker (priority: P1)"

    def _act_unknown(self, ip: str) -> str:
        return f"[SIM] unknown action for {ip}"
