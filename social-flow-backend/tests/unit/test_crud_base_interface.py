import inspect
from app.infrastructure.crud.base import CRUDBase

def test_crud_base_expected_methods():
    expected = {"get", "get_by_field", "get_multi", "create", "update", "delete", "soft_delete", "count", "exists"}
    actual = {name for name, member in inspect.getmembers(CRUDBase, predicate=inspect.isfunction)}
    missing = expected - actual
    assert not missing, f"CRUDBase missing methods: {missing}"