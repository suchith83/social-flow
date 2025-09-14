# Handles database replication and failover
from typing import Dict, List
import random


class ReplicationManager:
    """
    Simple master-replica manager.
    """

    def __init__(self, master: str, replicas: List[str]):
        self.master = master
        self.replicas = replicas

    def get_master(self) -> str:
        return self.master

    def get_replica(self) -> str:
        return random.choice(self.replicas) if self.replicas else self.master
