from fastapi import Depends, HTTPException, status, Header
from typing import Optional, Dict, Any
from app.core.config import settings
from app.core.auth import decode_access_token
from app.core.db import get_user_by_id

def get_db_path():
    return settings.DB_PATH

def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1]
    else:
        token = authorization
    try:
        sub = decode_access_token(token)
        user = get_user_by_id(settings.DB_PATH, sub.get("id"))
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        # Avoid returning sensitive fields
        user.pop("salt", None)
        user.pop("pwd_hash", None)
        return user
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
