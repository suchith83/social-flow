# tests/test_security.py
import unittest
import datetime
from compliance.audit_compliance.security.access_control import AccessControlEvaluator
from compliance.audit_compliance.security.vulnerability_scanner import VulnerabilityScanner
from compliance.audit_compliance.security.incident_response import IncidentResponseEngine
from compliance.audit_compliance.security.secrets_management import FileSecretsAdapter, SecretsManager
from compliance.audit_compliance.security.encryption_policy import EncryptionPolicy

class TestSecurityCompliance(unittest.TestCase):

    def test_access_control_basic_allow(self):
        policies = [
            {"id":"p1","effect":"allow","subjects":["role:admin"],"actions":["*"],"resources":["*"], "conditions":{}}
        ]
        ace = AccessControlEvaluator(policies)
        self.assertTrue(ace.is_allowed("role:admin", "delete", "s3:bucket/private"))

    def test_access_control_deny_overrides(self):
        policies = [
            {"id":"p1","effect":"allow","subjects":["role:*"],"actions":["*"],"resources":["*"]},
            {"id":"p2","effect":"deny","subjects":["user:evil"],"actions":["*"],"resources":["*"]}
        ]
        ace = AccessControlEvaluator(policies)
        self.assertFalse(ace.is_allowed("user:evil", "read", "any"))

    def test_vuln_scanner_reports(self):
        today = datetime.date(2025, 9, 13)
        feed = [
            {"id":"CVE-2025-0001","package":"openssl","version_range":"<=1.1.1","cvss":9.8,"description":"Critical bug","disclosed_on": datetime.date(2025,9,1)}
        ]
        inventory = [{"package":"openssl","version":"1.1.0","asset_id":"srv1","exposed":True}]
        vs = VulnerabilityScanner(feed)
        findings = vs.scan_inventory(inventory, today=today)
        self.assertTrue(len(findings) >= 1)
        self.assertEqual(findings[0]["vuln_id"], "CVE-2025-0001")

    def test_incident_workflow(self):
        eng = IncidentResponseEngine()
        inc_id = eng.create_incident("test", "desc", [{"e":"evidence"}], severity="high")
        triaged = eng.triage(inc_id)
        self.assertIn("recommended_playbook", triaged)
        # Execute with a mock executor that returns success
        def executor(step):
            return {"status":"ok","detail":f"executed {step['id']}"}
        res = eng.execute_playbook(inc_id, executor_callback=executor)
        self.assertTrue(any(r["status"] == "executed" for r in res["execution"]))
        closed = eng.close_incident(inc_id, "resolved")
        self.assertEqual(closed["status"], "closed")

    def test_secrets_adapter_and_discovery(self):
        # write a temp secrets store
        adapter = FileSecretsAdapter(path="test_secrets.json")
        manager = SecretsManager(adapter)
        adapter.set_secret("api_key", "supersecretvalue")
        val = adapter.get_secret("api_key")
        self.assertEqual(val, "supersecretvalue")
        # create a temporary file containing a fake secret for discovery
        with open("temp.env", "w", encoding="utf-8") as f:
            f.write("API_KEY=supersecretvalue\n")
        findings = manager.discover_in_files(".", patterns=("temp.env",))
        # cleanup
        import os
        os.remove("temp.env")
        os.remove("test_secrets.json")
        self.assertTrue(any("temp.env" in k for k in findings.keys()))

    def test_encryption_policy(self):
        policy = EncryptionPolicy()
        metadata = {
            "asset_id": "db-1",
            "disk_encrypted": True,
            "cipher": "AES-GCM",
            "key_length": 256,
            "key_last_rotation": "2025-09-01T00:00:00",
            "tls_min_version": "1.2"
        }
        res = policy.evaluate_asset(metadata)
        self.assertTrue(res["compliant"])

if __name__ == "__main__":
    unittest.main()
