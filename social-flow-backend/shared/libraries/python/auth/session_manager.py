# common/libraries/python/auth/session_manager.py
"""
Session management with in-memory and pluggable backend support.
"""

import time
import uuid
from typing import Dict, Optional
from .models import Session
from .config import AuthConfig

class InMemorySessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self, user_id: str, metadata: Optional[dict] = None) -> Session:
        sid = str(uuid.uuid4())
        now = int(time.time())
        session = Session(
            session_id=sid,
            user_id=user_id,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            expires_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now + AuthConfig.SESSION_TTL)),
            metadata=metadata or {},
        )
        self._sessions[sid] = session
        return session

    def get_session(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    def delete_session(self, sid: str) -> None:
        self._sessions.pop(sid, None)
