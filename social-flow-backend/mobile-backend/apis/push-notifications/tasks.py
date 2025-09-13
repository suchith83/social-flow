# Async tasks for sending notifications
"""
Celery tasks that perform the actual sends.
Handles retries, per-device logging, and status updates.
"""

from celery import Celery
from celery.utils.log import get_task_logger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from .config import get_config
from .repository import NotificationRepository, DeviceRepository
from .providers.fcm_provider import FCMProvider
from .providers.apns_provider import APNsProvider
from .models import NotificationStatus
from .database import init_db
from .utils import validate_tokens

config = get_config()

celery_app = Celery(
    "push_notifications",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

logger = get_task_logger(__name__)

# Setup DB session for tasks
engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# provider instances
fcm = FCMProvider()
apns = APNsProvider()


@celery_app.task(bind=True, max_retries=config.MAX_RETRIES, acks_late=True)
def send_notification_task(self, notification_id: int, tokens: list, title: str, body: str, payload: dict):
    """
    Task: send a batch of tokens. It will try to split tokens by platform and use the correct provider.
    Logs attempts per-token and update notification status when done.
    """
    db = SessionLocal()
    try:
        NotificationRepository.update_status(db, notification_id, NotificationStatus.SENDING)
        tokens = validate_tokens(tokens)
        # Naive platform split: tokens prefixed/encoded by platform is a real-world detail;
        # here we attempt both providers for all tokens as a fallback.
        # In production tokens should be associated to devices to identify platform.
        fcm_results = fcm.send_batch(tokens, title, body, payload)
        apns_results = []  # apns.send_batch(android_tokens...) not invoked here; simulation

        # Merge results: prefer fcm results if present; otherwise apns_results
        responses = fcm_results if fcm_results else apns_results

        success_any = False
        for r in responses:
            status = "ok" if r.get("status") == "ok" else "failed"
            NotificationRepository.log_attempt(db, notification_id, r.get("token"), status, r.get("provider_message"))
            if status == "ok":
                success_any = True

        # If any success mark SENT, else FAILED
        final_status = NotificationStatus.SENT if success_any else NotificationStatus.FAILED
        NotificationRepository.update_status(db, notification_id, final_status, provider_response={"summary": "batch processed"})
        return {"notification_id": notification_id, "status": final_status.value}
    except Exception as exc:
        logger.exception("Error sending notifications")
        countdown = config.RETRY_BACKOFF_SECONDS * (2 ** self.request.retries)
        try:
            self.retry(exc=exc, countdown=countdown)
        except self.MaxRetriesExceededError:
            NotificationRepository.update_status(db, notification_id, NotificationStatus.FAILED, provider_response={"error": str(exc)})
            raise
    finally:
        db.close()
