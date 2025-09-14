# Tests for security
# monitoring/logging/centralized/tests/test_security.py
from monitoring.logging.centralized.security import mask_sensitive, RBAC

def test_masking():
    log = {"password": "12345", "message": "ok"}
    masked = mask_sensitive(log)
    assert masked["password"] == "***REDACTED***"

def test_rbac():
    rbac = RBAC()
    assert rbac.can_read("admin")
    assert not rbac.can_write("guest")
