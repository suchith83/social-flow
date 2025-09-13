# Service layer for sending notifications
"""
Service layer that coordinates device management, templating, and dispatch.
It is provider-agnostic and uses repository + providers + tasks.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from .repository import DeviceRepository, NotificationRepository
from .templates import render_template
from .providers.fcm_provider import FCMProvider
from .providers.apns_provider import APNsProvider
from .tasks import send_notification_task
from .models import SendNotificationRequest
from .config import get_config
from .utils import chunked, validate_tokens
import logging

logger = logging.getLogger("push.service")
config = get_config()


class PushService:

    def __init__(self, db: Session):
        self.db = db
        # provider instances could be injected
        self.fcm = FCMProvider()
        self.apns = APNsProvider()

    def register_device(self, device_data):
        return DeviceRepository.register_or_update(self.db, device_data)

    def remove_token(self, token: str):
        return DeviceRepository.deactivate_token(self.db, token)

    def send_notification(self, req: SendNotificationRequest):
        """
        Main entrypoint to create notification records and enqueue send tasks.
        Resolves user_ids -> tokens if necessary.
        """
        tokens = req.tokens or []
        if req.user_ids:
            tokens += DeviceRepository.tokens_for_user_ids(self.db, req.user_ids)

        tokens = validate_tokens(tokens)
        if not tokens:
            raise ValueError("No valid tokens found to send notification")

        # Create DB notification record
        notif = NotificationRepository.create_notification(self.db, req.title, req.body or "", req.payload or {})

        # enqueue tasks in batches
        batch_size = config.MAX_BATCH_SIZE
        for batch in chunked(tokens, batch_size):
            # enqueue Celery task per batch
            send_notification_task.delay(notif.id, batch, req.title, req.body or "", req.payload or {})

        return notif

    def send_template(self, template_name: str, context: Dict[str, Any], tokens: Optional[List[str]] = None, user_ids: Optional[List[str]] = None):
        rendered = render_template(template_name, context)
        req = SendNotificationRequest(title=rendered["title"], body=rendered["body"], payload=context, tokens=tokens, user_ids=user_ids)
        return self.send_notification(req)
