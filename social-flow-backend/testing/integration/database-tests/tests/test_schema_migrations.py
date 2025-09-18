"""
Schema & migrations tests.

- Ensure tables exist after migrations step (create_all_tables)
- Check critical columns exist and types are plausible
"""

import pytest
from db.migrations import table_exists
from db.models import User, Item
from utils.sql_utils import scalar_query

@pytest.mark.integration
def test_tables_exist():
    assert table_exists(User.__tablename__), f"Users table {User.__tablename__} should exist"
    assert table_exists(Item.__tablename__), f"Items table {Item.__tablename__} should exist"

@pytest.mark.integration
def test_users_table_has_expected_columns():
    # Check for column presence via information_schema
    table = User.__tablename__
    # count username column
    val = scalar_query(
        "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = :t AND column_name = 'username'",
        t=table
    )
    assert val == 1
