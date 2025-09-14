# memcached_config.py
# Created by Create-Configurations.ps1
"""
memcached_config.py
-------------------
Defines configuration for Memcached including slab allocation, 
object compression, and client-side connection pooling.
"""

import os
import pylibmc


class MemcachedConfig:
    """Configuration handler for Memcached."""

    SERVERS = os.getenv("MEMCACHED_SERVERS", "127.0.0.1").split(",")
    BINARY = os.getenv("MEMCACHED_BINARY", "true").lower() == "true"
    COMPRESSION = os.getenv("MEMCACHED_COMPRESSION", "true").lower() == "true"
    BEHAVIOR = {
        "tcp_nodelay": True,
        "ketama": True,
        "remove_failed": 1,
        "retry_timeout": 2,
        "dead_timeout": 10,
    }

    @classmethod
    def get_client(cls) -> pylibmc.Client:
        """Return a configured Memcached client."""
        client = pylibmc.Client(cls.SERVERS, binary=cls.BINARY)
        client.behaviors = cls.BEHAVIOR
        return client

    @classmethod
    def summary(cls) -> dict:
        return {
            "servers": cls.SERVERS,
            "binary": cls.BINARY,
            "compression": cls.COMPRESSION,
            "behaviors": cls.BEHAVIOR,
        }


if __name__ == "__main__":
    print("🔧 Memcached Config:", MemcachedConfig.summary())
