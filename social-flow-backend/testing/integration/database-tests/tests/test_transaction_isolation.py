"""
Test to ensure transaction isolation and proper rollback behavior.

- Start a transaction in one session and ensure another session cannot see uncommitted changes.
"""

import pytest
from db.sync_client import engine, SessionLocal
from db.models import User
from sqlalchemy import text

@pytest.mark.integration
def test_transaction_isolation():
    conn1 = engine.connect()
    trans1 = conn1.begin()
    s1 = SessionLocal(bind=conn1)

    # create a user inside trans1 but do not commit
    u = User(username="isolation_test_user", hashed_password="x")
    s1.add(u)
    s1.flush()  # ensure INSERT executed

    # Open a second connection and try to read user
    conn2 = engine.connect()
    s2 = SessionLocal(bind=conn2)
    res = s2.query(User).filter_by(username="isolation_test_user").all()
    # In properly isolated DB, uncommitted row shouldn't be visible (depends on isolation level)
    assert len(res) == 0, "Uncommitted data should not be visible to other sessions"

    # Cleanup: rollback trans1
    trans1.rollback()
    s1.close()
    s2.close()
    conn1.close()
    conn2.close()
