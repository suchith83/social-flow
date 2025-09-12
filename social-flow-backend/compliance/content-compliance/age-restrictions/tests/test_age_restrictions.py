"""
test_age_restrictions.py

Comprehensive tests for age restriction compliance module.
"""

import unittest
from datetime import date, timedelta

from compliance.content_compliance.age_restrictions.age_policy import (
    ContentCategory,
    Jurisdiction,
    AgePolicy,
)
from compliance.content_compliance.age_restrictions.validator import AgeValidator
from compliance.content_compliance.age_restrictions.exceptions import (
    AgeRestrictionViolation,
)


class TestAgeRestrictions(unittest.TestCase):
    def setUp(self):
        self.today = date.today()

    def test_age_policy_export(self):
        policies = AgePolicy.export_policies()
        self.assertIn("us", policies)
        self.assertIn("teen", policies["us"])

    def test_valid_access(self):
        birthdate = self.today.replace(year=self.today.year - 20)
        AgeValidator.validate_access(birthdate, ContentCategory.ADULT, Jurisdiction.US)

    def test_invalid_access(self):
        birthdate = self.today.replace(year=self.today.year - 10)
        with self.assertRaises(AgeRestrictionViolation):
            AgeValidator.validate_access(
                birthdate, ContentCategory.ADULT, Jurisdiction.US
            )


if __name__ == "__main__":
    unittest.main()
