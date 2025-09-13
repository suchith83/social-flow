"""Connection pool, TLS, retries, replica-set awareness, async & sync clients."""
"""
connection.py
-------------
Advanced MongoDB connection manager supporting:
- PyMongo synchronous client
- Motor async client
- TLS/SSL and replica-set awareness
- Connection retries with exponential backoff
- Health-check utilities
- Centralized place to get DB and client objects for app code
"""

from typing import Optional, Dict, Any
import yaml
from pathlib import Path
import logging
import time
from pymongo import MongoClient, read_preferences, WriteConcern
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from motor.motor_asyncio import AsyncIOMotorClient
import threading

logger = logging.getLogger("MongoDBConnection")
logger.setLevel(logging.INFO)


def _load_config(path: str = "config/databases/mongodb/config.yaml") -> Dict[str, Any]:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)["mongodb"]


class MongoConnectionManager:
    """
    Singleton-style connection manager exposing:
      - get_sync_client() -> pymongo.MongoClient
      - get_async_client() -> motor.motor_asyncio.AsyncIOMotorClient
      - get_database() -> pymongo.database.Database
    """

    _instance_lock = threading.Lock()
    _instance: Optional["MongoConnectionManager"] = None

    def __init__(self, config_path: str = "config/databases/mongodb/config.yaml"):
        self.conf = _load_config(config_path)
        self.sync_client: Optional[MongoClient] = None
        self.async_client: Optional[AsyncIOMotorClient] = None
        self._connect_with_retry()

    def __new__(cls, *args, **kwargs):
        # Ensure singleton across threads
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _build_uri(self) -> str:
        # Build replica-set aware connection string
        nodes = ",".join(f"{n['host']}:{n['port']}" for n in self.conf["nodes"])
        user = self.conf.get("user")
        password = self.conf.get("password")
        auth = f"{user}:{password}@" if user and password else ""
        replica_set = self.conf.get("replica_set")
        auth_source = "admin"  # assume admin; change if needed
        uri = f"mongodb://{auth}{nodes}/{self.conf['database']}?authSource={auth_source}"
        if replica_set:
            uri += f"&replicaSet={replica_set}"
        # retryWrites/read settings are applied via options below
        return uri

    def _connect_with_retry(self, retries: int = 5, initial_delay: float = 1.0):
        uri = self._build_uri()
        pool = self.conf.get("pool", {})
        delay = initial_delay
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Attempt {attempt} connecting to MongoDB: {uri}")
                self.sync_client = MongoClient(
                    uri,
                    maxPoolSize=pool.get("maxPoolSize", 100),
                    minPoolSize=pool.get("minPoolSize", 0),
                    maxIdleTimeMS=pool.get("maxIdleTimeMS"),
                    connectTimeoutMS=pool.get("connectTimeoutMS"),
                    serverSelectionTimeoutMS=pool.get("serverSelectionTimeoutMS"),
                    socketTimeoutMS=pool.get("socketTimeoutMS"),
                    tls=self.conf.get("tls", False),
                    tlsCAFile=self.conf.get("tlsCAFile"),
                    tlsCertificateKeyFile=self.conf.get("tlsCertificateKeyFile"),
                    retryWrites=pool.get("retryWrites", True),
                    retryReads=pool.get("retryReads", True)
                )

                # Force server selection to detect connectivity early
                self.sync_client.admin.command("ping")
                logger.info("✅ Synchronous MongoDB client connected successfully")

                # Build async client with the same uri and options
                self.async_client = AsyncIOMotorClient(
                    uri,
                    maxPoolSize=pool.get("maxPoolSize", 100),
                    minPoolSize=pool.get("minPoolSize", 0),
                    socketTimeoutMS=pool.get("socketTimeoutMS"),
                    tls=self.conf.get("tls", False)
                )
                logger.info("✅ Async Motor client initialized")
                return
            except ServerSelectionTimeoutError as e:
                logger.warning(f"Server selection timeout on attempt {attempt}: {e}")
            except PyMongoError as e:
                logger.warning(f"PyMongo error on attempt {attempt}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error connecting to MongoDB on attempt {attempt}: {e}")

            time.sleep(delay)
            delay *= 2

        raise RuntimeError("❌ Unable to connect to MongoDB cluster after retries")

    def get_sync_client(self) -> MongoClient:
        if not self.sync_client:
            self._connect_with_retry()
        return self.sync_client

    def get_async_client(self) -> AsyncIOMotorClient:
        if not self.async_client:
            self._connect_with_retry()
        return self.async_client

    def get_database(self):
        client = self.get_sync_client()
        return client[self.conf["database"]]

    def get_async_database(self):
        client = self.get_async_client()
        return client[self.conf["database"]]

    # Utility: close clients
    def close(self):
        try:
            if self.sync_client:
                self.sync_client.close()
            if self.async_client:
                self.async_client.close()
        except Exception:
            logger.exception("Error closing MongoDB clients")
