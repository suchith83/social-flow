# Tests for policy module
# monitoring/logging/retention/tests/test_policy.py
import datetime
from monitoring.logging.retention.policy import RetentionPolicy

def test_policy_classification():
    policy = RetentionPolicy()
    recent = datetime.datetime.utcnow()
    old = recent - datetime.timedelta(days=100)
    assert policy.classify(recent) == "hot"
    assert policy.classify(old) == "archive"
