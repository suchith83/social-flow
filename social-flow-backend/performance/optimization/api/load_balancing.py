# Implements load balancing strategies
import itertools
import random
from typing import List


class RoundRobinBalancer:
    """Round Robin Load Balancer."""

    def __init__(self, servers: List[str]):
        self.servers = servers
        self.iterator = itertools.cycle(servers)

    def get_server(self) -> str:
        return next(self.iterator)


class LeastConnectionsBalancer:
    """Least Connections Load Balancer."""

    def __init__(self, servers: List[str]):
        self.servers = {server: 0 for server in servers}

    def get_server(self) -> str:
        server = min(self.servers, key=self.servers.get)
        self.servers[server] += 1
        return server

    def release(self, server: str):
        if server in self.servers and self.servers[server] > 0:
            self.servers[server] -= 1


class HashBalancer:
    """Consistent Hashing Load Balancer."""

    def __init__(self, servers: List[str]):
        self.servers = servers

    def get_server(self, key: str) -> str:
        return self.servers[hash(key) % len(self.servers)]
