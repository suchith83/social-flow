"""Optional lightweight ODM setup (Motor + Beanie example) and PyMongo wrappers."""
"""
orm.py
------
Provides two patterns:
1) Lightweight PyMongo wrapper helpers for traditional sync code (transactions, sessions).
2) Motor + Beanie example for document-model ODM (async). Beanie is included as an example
   — you can swap for odmantic or mongoengine based on preference.

This file aims to show:
  - transaction/session usage (required for multi-doc transactions in replica sets/replicated clusters)
  - model example with Beanie (async)
"""

from typing import Optional, Dict, Any
import logging
from pymongo import ReturnDocument
from pymongo.client_session import ClientSession
from pymongo.errors import PyMongoError
from .connection import MongoConnectionManager

logger = logging.getLogger("MongoORM")
logger.setLevel(logging.INFO)

# -------------------------
# Sync helper wrappers
# -------------------------
def with_session(fn):
    """
    Decorator to open a session (and optionally a transaction) for the wrapped function.
    Usage:
      @with_session
      def my_op(db, session=None, ...):
          ...
    The decorator will pass session arg to the function.
    """
    def wrapper(*args, **kwargs):
        conn = MongoConnectionManager()
        client = conn.get_sync_client()
        # Start a session and pass into the function
        with client.start_session() as session:
            try:
                # Optionally start a transaction; by default don't unless requested
                if kwargs.pop("_use_transaction", False):
                    with session.start_transaction():
                        return fn(*args, session=session, **kwargs)
                else:
                    return fn(*args, session=session, **kwargs)
            except PyMongoError as e:
                logger.exception("MongoDB operation failed in session")
                raise
    return wrapper

# Example: atomic find-and-update pattern using session
@with_session
def find_and_update(collection, filter_doc: Dict[str, Any], update_doc: Dict[str, Any],
                    upsert: bool = False, session: Optional[ClientSession] = None):
    """
    Performs a findOneAndUpdate with appropriate write concern and returns the updated doc.
    """
    return collection.find_one_and_update(
        filter_doc,
        {"$set": update_doc},
        upsert=upsert,
        return_document=ReturnDocument.AFTER,
        session=session
    )

# -------------------------
# Async ODM example using Beanie
# -------------------------
# NOTE: Beanie requires motor and pydantic. This block leaves the import gated so that code
# won't break if Beanie isn't installed. Remove gating if using Beanie in production.
try:
    from beanie import Document, init_beanie
    from pydantic import BaseModel
    from motor.motor_asyncio import AsyncIOMotorClient
    BEANIE_AVAILABLE = True
except Exception:
    BEANIE_AVAILABLE = False

if BEANIE_AVAILABLE:
    class UserDoc(Document):
        username: str
        email: str
        created_at: Optional[str]

        class Collection:
            name = "users"

    async def init_odm():
        from .connection import MongoConnectionManager
        conn = MongoConnectionManager()
        client: AsyncIOMotorClient = conn.get_async_client()
        db = client[conn.conf["database"]]
        await init_beanie(database=db, document_models=[UserDoc])
else:
    logger.info("Beanie not installed — skipping ODM initialization block. Install beanie+motor+pydantic to use.")
