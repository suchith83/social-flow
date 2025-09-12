# incident_response.py
import uuid
import datetime
from typing import List, Dict, Any, Optional
from .utils import log_event, secure_hash

class IncidentResponseEngine:
    """
    Orchestrates incident lifecycle:
    - Triage (severity classification)
    - Containment recommendations
    - Evidence collection plan
    - Playbook execution (idempotent steps)
    """

    SEVERITY_MAP = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4
    }

    def __init__(self, playbooks: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """
        playbooks: mapping severity -> list of steps
        Step example: {"id":"step1","action":"isolate","target":"asset_id","cmd":"iptables -A ...", "idempotent": True}
        """
        self.playbooks = playbooks or self._default_playbooks()
        self.incidents = {}  # id -> incident record

    def _default_playbooks(self):
        # Default minimal playbooks per severity
        return {
            "low": [
                {"id": "p_low_1", "action": "notify", "message": "Log and monitor", "idempotent": True}
            ],
            "medium": [
                {"id": "p_med_1", "action": "snapshot", "target": "<asset>", "idempotent": True},
                {"id": "p_med_2", "action": "notify", "message": "Escalate to SOC", "idempotent": True}
            ],
            "high": [
                {"id": "p_high_1", "action": "isolate", "target": "<asset>", "idempotent": True},
                {"id": "p_high_2", "action": "collect_evidence", "target": "<asset>", "idempotent": False},
                {"id": "p_high_3", "action": "notify", "message": "Open incident ticket", "idempotent": True}
            ],
            "critical": [
                {"id": "p_crit_1", "action": "isolate_network", "target": "subnet", "idempotent": True},
                {"id": "p_crit_2", "action": "disable_accounts", "target": "compromised_users", "idempotent": False},
                {"id": "p_crit_3", "action": "forensic_snapshot", "target": "<asset>", "idempotent": False}
            ]
        }

    def create_incident(self, title: str, description: str, observed_evidence: List[Dict[str, Any]], severity: str = "medium") -> str:
        incident_id = str(uuid.uuid4())
        severity = severity if severity in self.SEVERITY_MAP else "medium"
        record = {
            "id": incident_id,
            "title": title,
            "description": description,
            "severity": severity,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "evidence": observed_evidence,
            "status": "open",
            "actions": []
        }
        # compute confidentiality fingerprint to deduplicate similar incidents
        fingerprint = secure_hash(title + description + str(len(observed_evidence)))
        record["fingerprint"] = fingerprint
        self.incidents[incident_id] = record
        log_event(f"Incident created: {incident_id}", level="WARNING", incident_id=incident_id, severity=severity)
        return incident_id

    def triage(self, incident_id: str) -> Dict[str, Any]:
        """
        Attach triage metadata: assign priority, recommended playbook.
        """
        rec = self.incidents.get(incident_id)
        if not rec:
            raise KeyError("Incident not found")
        severity = rec["severity"]
        rec["priority"] = self.SEVERITY_MAP.get(severity, 2)
        rec["recommended_playbook"] = self.playbooks.get(severity, [])
        log_event(f"Incident triaged: {incident_id}", level="INFO", incident_id=incident_id, severity=severity)
        return rec

    def execute_playbook(self, incident_id: str, executor_callback=None) -> Dict[str, Any]:
        """
        Execute the recommended playbook steps. executor_callback is a callable that
        receives step dict and must implement the action (or be mocked during tests).
        The engine records step outcomes and enforces idempotency for steps that are marked idempotent.
        """
        rec = self.incidents.get(incident_id)
        if not rec:
            raise KeyError("Incident not found")
        playbook = rec.get("recommended_playbook") or self.playbooks.get(rec["severity"], [])
        executed = []
        for step in playbook:
            step_id = step.get("id")
            # Skip if already executed and idempotent
            previously = any(a["step_id"] == step_id for a in rec["actions"])
            if previously and step.get("idempotent", False):
                log_event(f"Skipping already-executed idempotent step: {step_id}", level="INFO", incident_id=incident_id, step=step_id)
                executed.append({"step_id": step_id, "status": "skipped_idempotent"})
                continue
            # Execute (via callback) if provided, else mock success
            try:
                result = executor_callback(step) if executor_callback else {"status": "ok", "detail": "executed_mock"}
                rec["actions"].append({"step_id": step_id, "executed_at": datetime.datetime.utcnow().isoformat(), "result": result})
                log_event(f"Executed step {step_id} for incident {incident_id}", level="INFO", incident_id=incident_id, step=step_id)
                executed.append({"step_id": step_id, "status": "executed", "result": result})
            except Exception as e:
                rec["actions"].append({"step_id": step_id, "executed_at": datetime.datetime.utcnow().isoformat(), "result": {"status": "failed", "error": str(e)}})
                log_event(f"Playbook step failed: {step_id} error={e}", level="ERROR", incident_id=incident_id, step=step_id)
                executed.append({"step_id": step_id, "status": "failed", "error": str(e)})
                # on failure, stop further execution to preserve state (common practice)
                break
        # update status if critical steps succeeded (simple policy)
        if any(a["status"] == "executed" for a in executed):
            rec["status"] = "in_progress"
        log_event(f"Playbook execution complete for {incident_id}", level="INFO", incident_id=incident_id)
        return {"incident_id": incident_id, "execution": executed}

    def close_incident(self, incident_id: str, resolution_notes: str):
        rec = self.incidents.get(incident_id)
        if not rec:
            raise KeyError("Incident not found")
        rec["status"] = "closed"
        rec["closed_at"] = datetime.datetime.utcnow().isoformat()
        rec["resolution_notes"] = resolution_notes
        log_event(f"Incident {incident_id} closed", level="INFO", incident_id=incident_id)
        return rec
