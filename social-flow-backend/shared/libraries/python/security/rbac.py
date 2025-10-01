# rbac.py
from typing import Dict, List


class RBAC:
    """
    Role-Based Access Control.
    """

    def __init__(self):
        self.roles: Dict[str, List[str]] = {}

    def add_role(self, role: str, permissions: List[str]):
        self.roles[role] = permissions

    def check_permission(self, role: str, permission: str) -> bool:
        return permission in self.roles.get(role, [])
