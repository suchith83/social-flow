# =========================
# File: testing/security/penetration/payloads/injection_tests.py
# =========================
"""
Non-destructive input tests. These are safe patterns used for detecting handling of special chars,
but do NOT perform exploitation. They should be used only in authorized contexts and typically against
test/staging environments. Examples:
  - SQL-like markers such as "' OR '1'='1' -- " should be used in simulation or low-impact scan only.
  - XSS markers are short and non-persistent.
"""

SAFE_TESTS = [
    {"id": "XSS-TEST-01", "payload": "<script>alert(1)</script>", "description": "Basic XSS marker (non-persistent)"},
    {"id": "SQL-TEST-01", "payload": "' OR '1'='1' -- ", "description": "SQL boolean marker for detection only"},
    {"id": "CMD-TEST-01", "payload": "`whoami`", "description": "Command marker for detecting naive command eval"}
]

def list_payloads():
    return SAFE_TESTS
