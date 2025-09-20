from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("mobile.notifications.push")

try:
    import firebase_admin  # type: ignore
    from firebase_admin import messaging, credentials  # type: ignore
except Exception:
    firebase_admin = None
    messaging = None
    credentials = None


class PushSender:
    """
    Simple push sender with optional Firebase Admin support.
    - provider: "fcm" or "noop"
    - creds: path to credentials or JSON string (optional)
    """

    def __init__(self, provider: str = "fcm", creds: Optional[str] = None):
        self.provider = provider or "noop"
        self.creds = creds
        self._init_provider()

    def _init_provider(self):
        if self.provider != "fcm":
            logger.info("Push provider set to noop")
            return
        if firebase_admin is None:
            logger.warning("firebase_admin not installed; push will be logged")
            return
        try:
            if self.creds:
                # accept path to service account JSON
                try:
                    cred = credentials.Certificate(self.creds)
                except Exception:
                    cred = credentials.ApplicationDefault()
            else:
                cred = credentials.ApplicationDefault()
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            logger.info("Initialized firebase_admin for push")
        except Exception:
            logger.exception("Failed to init firebase_admin; falling back to logging")

    def send(self, token: Optional[str], title: str = "", body: str = "", data: Optional[Dict[str, Any]] = None) -> None:
        if not token:
            logger.warning("No token provided for push; skipping")
            return
        if self.provider == "fcm" and messaging is not None:
            try:
                message = messaging.Message(
                    notification=messaging.Notification(title=title, body=body),
                    data={k: str(v) for k, v in (data or {}).items()},
                    token=token,
                )
                resp = messaging.send(message)
                logger.debug("FCM send response: %s", resp)
                return
            except Exception:
                logger.exception("FCM send failed; falling back to log")
        # fallback: log the push
        logger.info("Push (logged) to token=%s title=%s body=%s data=%s", token, title, body, data or {})
