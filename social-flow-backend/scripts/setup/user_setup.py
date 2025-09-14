# scripts/setup/user_setup.py
import logging
import os
from typing import Dict, Any
from .utils import run, which

logger = logging.getLogger("setup.users")

class UserSetup:
    """
    Create system users and groups required by services.
    Idempotent: only creates if missing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.users = config.get("setup", {}).get("users", [])

    def user_exists(self, username: str) -> bool:
        try:
            import pwd
            pwd.getpwnam(username)
            return True
        except KeyError:
            return False

    def group_exists(self, groupname: str) -> bool:
        try:
            import grp
            grp.getgrnam(groupname)
            return True
        except KeyError:
            return False

    def create_group(self, groupname: str, gid: int = None):
        if self.group_exists(groupname):
            logging.info("Group %s already exists", groupname)
            return
        cmd = ["sudo", "groupadd"]
        if gid:
            cmd += ["-g", str(gid)]
        cmd.append(groupname)
        run(cmd, check=False)
        logging.info("Created group %s", groupname)

    def create_user(self, username: str, uid: int = None, groups: list = None, home: str = None, shell: str = "/bin/bash"):
        if self.user_exists(username):
            logging.info("User %s already exists", username)
            return
        cmd = ["sudo", "useradd", "-m"]
        if uid:
            cmd += ["-u", str(uid)]
        if home:
            cmd += ["-d", home]
        if shell:
            cmd += ["-s", shell]
        if groups:
            cmd += ["-G", ",".join(groups)]
        cmd.append(username)
        run(cmd, check=True)
        logging.info("Created user %s", username)

    def run(self):
        for u in self.users:
            name = u.get("name")
            if not name:
                continue
            gid = u.get("gid")
            groups = u.get("groups", [])
            # ensure groups
            for g in groups:
                if not self.group_exists(g):
                    self.create_group(g)
            self.create_user(name, uid=u.get("uid"), groups=groups, home=u.get("home"), shell=u.get("shell"))
