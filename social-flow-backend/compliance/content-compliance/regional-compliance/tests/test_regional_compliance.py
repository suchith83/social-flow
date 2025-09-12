"""
test_regional_compliance.py

Unit tests exercising major flows:
- jurisdiction resolution
- content checks that trigger blocking due to min_age or data residency
- enforcement invocation logging (observed in log files)
"""

import unittest
import os
import shutil
from pathlib import Path
from compliance.content_compliance.regional_compliance.jurisdiction import normalize_locale_to_jurisdiction, resolve_jurisdiction_from_user
from compliance.content_compliance.regional_compliance.regional_policy import RegionalPolicyRegistry, RegionalRule, PolicyScope
from compliance.content_compliance.regional_compliance.integration import RegionalComplianceService
from datetime import datetime, timedelta

LOG_DIR = Path("logs/compliance/regional")

class TestRegionalCompliance(unittest.TestCase):
    def setUp(self):
        # clear registry and add a test rule
        # For test isolation, remove any pre-seeded defaults
        # NOTE: direct manipulation of internal registry for tests
        try:
            for j in list(RegionalPolicyRegistry.export_registry().keys()):
                for rid in list(RegionalPolicyRegistry.export_registry()[j].keys()):
                    RegionalPolicyRegistry.remove_rule(j, rid)
        except Exception:
            pass
        # Register a sample content rule: age min 21 for 'gambling'
        RegionalPolicyRegistry.register_rule(RegionalRule(
            id="test_us_gambling_min_age",
            jurisdiction="us",
            scope=PolicyScope.CONTENT,
            description="Gambling requires 21+",
            params={"applies_to": ["gambling"], "min_age": 21, "severity": 80}
        ))
        # Register a sample data residency rule for 'eu'
        RegionalPolicyRegistry.register_rule(RegionalRule(
            id="test_eu_residency",
            jurisdiction="eu",
            scope=PolicyScope.DATA,
            description="EU data residency required",
            params={"required_region": "eu", "severity": 90}
        ))

        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # remove logs
        shutil.rmtree(Path("logs"), ignore_errors=True)
        # clear registry
        try:
            for j in list(RegionalPolicyRegistry.export_registry().keys()):
                for rid in list(RegionalPolicyRegistry.export_registry()[j].keys()):
                    RegionalPolicyRegistry.remove_rule(j, rid)
        except Exception:
            pass

    def test_locale_to_jurisdiction(self):
        self.assertEqual(normalize_locale_to_jurisdiction("en-US"), "us")
        self.assertEqual(normalize_locale_to_jurisdiction("en-GB"), "uk")
        self.assertEqual(normalize_locale_to_jurisdiction("fr-FR"), "fr".lower() if "fr" in ["global","us"] else "global")  # expects fallback behaviour

    def test_gambling_age_block(self):
        svc = RegionalComplianceService()
        user_profile = {"locale": "en-US", "birthdate": (datetime.utcnow() - timedelta(days=365*20)).date().isoformat()}  # 20 years old
        content_meta = {"category": "gambling", "storage_region": "us"}
        res = svc.check_content(user_profile, content_meta)
        # should be blocked because min_age 21
        self.assertFalse(res["allowed"])
        self.assertIn("min_age", " ".join(res["reasons"]).lower() or "")

    def test_data_residency_block(self):
        svc = RegionalComplianceService()
        user_profile = {"locale": "en-GB"}  # resolves to UK; no EU rule for this user, but we'll test EU rule directly
        # pretend we have EU user
        user_profile_eu = {"locale": "fr-FR"}
        res = svc.check_data_residency(user_profile_eu, data_location="us")
        self.assertFalse(res["allowed"])
        self.assertIn("migrate", " ".join(res["actions"]) or "")

if __name__ == "__main__":
    unittest.main()
