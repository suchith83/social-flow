from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict
import uuid
from app.core.db import create_user, get_user_by_email
from app.core.auth import hash_password, verify_password, create_access_token
from app.api.deps import get_db_path, get_current_user
from app.core.config import settings

router = APIRouter()

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: Optional[str] = None

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class UserOut(BaseModel):
    id: str
    email: EmailStr
    display_name: Optional[str]
    role: str

@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(payload: UserCreate, db_path: str = Depends(get_db_path)):
    existing = get_user_by_email(db_path, payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="email already registered")
    uid = f"user_{uuid.uuid4().hex[:12]}"
    hp = hash_password(payload.password)
    create_user(db_path, uid, payload.email, payload.display_name or payload.email.split("@")[0], hp["salt"], hp["hash"])
    user = get_user_by_email(db_path, payload.email)
    return {"id": user["id"], "email": user["email"], "display_name": user.get("display_name"), "role": user.get("role", "user")}

class LoginPayload(BaseModel):
    email: EmailStr
    password: str

@router.post("/login", response_model=TokenOut)
def login(payload: LoginPayload, db_path: str = Depends(get_db_path)):
    user = get_user_by_email(db_path, payload.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    if not verify_password(payload.password, user["salt"], user["pwd_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    token = create_access_token({"id": user["id"], "email": user["email"]})
    return {"access_token": token, "expires_in": settings.ACCESS_TOKEN_EXPIRE_SECONDS}

@router.get("/me", response_model=UserOut)
def me(current_user: Dict = Depends(get_current_user)):
    # current_user already sanitized in deps
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "display_name": current_user.get("display_name"),
        "role": current_user.get("role", "user"),
    }
