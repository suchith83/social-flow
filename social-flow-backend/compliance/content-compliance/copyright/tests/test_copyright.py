"""
test_copyright.py

Unit + simple integration tests for copyright compliance module.
Uses filesystem-backed evidence store; tests clean up after themselves.
"""

import unittest
import os
import shutil
from pathlib import Path
from datetime import datetime

from compliance.content_compliance.copyright.notices import validate_notice_payload
from compliance.content_compliance.copyright.evidence_store import store_evidence, retrieve_evidence, delete_evidence
from compliance.content_compliance.copyright.takedown_manager import TakedownManager, _TAKEDOWN_DB
from compliance.content_compliance.copyright.config import CopyrightConfig
from compliance.content_compliance.copyright.content_fingerprinter import sha256_bytes

TEST_EVIDENCE_BYTES = b"this is a test evidence file content for demonstration"
TEST_FILENAME = "evidence_test.txt"

class TestCopyrightModule(unittest.TestCase):
    def setUp(self):
        # Ensure empty DB
        _TAKEDOWN_DB.clear()
        # Ensure evidence base exists
        self.evidence_base = Path(CopyrightConfig.EVIDENCE_BASE)
        self.evidence_base.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Remove any created evidence directories
        shutil.rmtree(self.evidence_base, ignore_errors=True)
        _TAKEDOWN_DB.clear()

    def test_validate_notice_payload_ok(self):
        payload = {
            "complainant_id": "user-123",
            "claimed_owner": "Owner Corp",
            "infringing_content_id": "content-abc",
            "description": "This is our content that was reposted without authorization."
        }
        notice = validate_notice_payload(payload)
        self.assertTrue(notice.id)
        self.assertEqual(notice.complainant_id, "user-123")

    def test_store_and_retrieve_evidence(self):
        payload = {
            "complainant_id": "user-123",
            "claimed_owner": "Owner Corp",
            "infringing_content_id": "content-abc",
            "description": "Unauthorized copy"
        }
        result = TakedownManager.submit_notice(payload, TEST_EVIDENCE_BYTES, TEST_FILENAME)
        notice_id = result["notice_id"]
        # retrieve evidence using store API
        evidence = retrieve_evidence(notice_id)
        self.assertIn("metadata", evidence)
        self.assertIn(TEST_FILENAME, evidence["files"])

    def test_auto_remove_flow(self):
        # Set config to ensure auto takedown works (default may already be True)
        payload = {
            "complainant_id": "user-xyz",
            "claimed_owner": "Owner",
            "infringing_content_id": "content-xyz",
            "description": "This is stolen copyrighted content"
        }
        res = TakedownManager.submit_notice(payload, TEST_EVIDENCE_BYTES, TEST_FILENAME)
        nid = res["notice_id"]
        # After submission, triage may have auto-removed depending on heuristics
        entry = TakedownManager.get_notice(nid)
        self.assertIn("triage", entry)
        # status should be either submitted/review/removed based on heuristics
        self.assertIn(entry["status"], ("submitted", "removed", "restored"))

if __name__ == "__main__":
    unittest.main()
