# Repository layer for storing notification logs
"""
Repository layer hiding SQLAlchemy details from service.
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from .models import Device, Notification, NotificationAttempt, DevicePlatform, NotificationStatus
from .models import DeviceRegisterRequest, SendNotificationRequest


class DeviceRepository:
    @staticmethod
    def register_or_update(db: Session, data: DeviceRegisterRequest) -> Device:
        # Upsert device by token
        device = db.query(Device).filter(Device.token == data.token).first()
        if device:
            device.user_id = data.user_id
            device.platform = data.platform
            device.app_version = data.app_version
            device.is_active = True
        else:
            device = Device(
                user_id=data.user_id,
                token=data.token,
                platform=data.platform,
                app_version=data.app_version,
                is_active=True,
            )
            db.add(device)
        db.commit()
        db.refresh(device)
        return device

    @staticmethod
    def deactivate_token(db: Session, token: str) -> bool:
        device = db.query(Device).filter(Device.token == token).first()
        if device:
            device.is_active = False
            db.commit()
            return True
        return False

    @staticmethod
    def tokens_for_user_ids(db: Session, user_ids: List[str]) -> List[str]:
        results = db.query(Device.token).filter(Device.user_id.in_(user_ids), Device.is_active == True).all()
        return [r[0] for r in results]


class NotificationRepository:
    @staticmethod
    def create_notification(db: Session, title: str, body: str, payload: dict) -> Notification:
        notif = Notification(title=title, body=body, payload=payload)
        db.add(notif)
        db.commit()
        db.refresh(notif)
        return notif

    @staticmethod
    def update_status(db: Session, notification_id: int, status: NotificationStatus, provider_response: Optional[dict] = None):
        notif = db.query(Notification).filter(Notification.id == notification_id).first()
        if notif:
            notif.status = status
            if provider_response is not None:
                notif.provider_response = provider_response
            db.commit()
            db.refresh(notif)
        return notif

    @staticmethod
    def log_attempt(db: Session, notification_id: int, device_token: str, status: str, response: dict = None):
        attempt = NotificationAttempt(notification_id=notification_id, device_token=device_token, status=status, response=response)
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
        return attempt
