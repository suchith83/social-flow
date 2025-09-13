# FastAPI routes for push notifications
"""
FastAPI routes for push notifications.
Provides endpoints for device registration, sending notifications, and audit.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from .models import DeviceRegisterRequest, DeviceRead, SendNotificationRequest, NotificationRead
from .database import get_db, init_db
from .service import PushService
from .repository import DeviceRepository, NotificationRepository

router = APIRouter(prefix="/push", tags=["Push Notifications"])


@router.on_event("startup")
def _startup():
    # ensure DB tables exist
    init_db()


@router.post("/devices", response_model=DeviceRead, status_code=status.HTTP_201_CREATED)
def register_device(req: DeviceRegisterRequest, db: Session = Depends(get_db)):
    service = PushService(db)
    device = service.register_device(req)
    return device


@router.delete("/devices/{token}", status_code=status.HTTP_204_NO_CONTENT)
def unregister_device(token: str, db: Session = Depends(get_db)):
    service = PushService(db)
    ok = service.remove_token(token)
    if not ok:
        raise HTTPException(status_code=404, detail="Device token not found")
    return None


@router.post("/send", response_model=NotificationRead)
def send_notification(req: SendNotificationRequest, db: Session = Depends(get_db)):
    service = PushService(db)
    try:
        notif = service.send_notification(req)
        return notif
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.get("/notifications/{notification_id}", response_model=NotificationRead)
def get_notification(notification_id: int, db: Session = Depends(get_db)):
    notif = db.query(NotificationRepository.__annotations__.get("Notification", None))  # not used; use direct query
    # simpler direct retrieval
    from .models import Notification
    n = db.query(Notification).filter(Notification.id == notification_id).first()
    if not n:
        raise HTTPException(status_code=404, detail="Notification not found")
    return n
