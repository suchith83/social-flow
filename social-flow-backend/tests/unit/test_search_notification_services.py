from unittest.mock import Mock, AsyncMock
from app.services.search_service import SearchService
from app.notifications.services.notification_service import NotificationService

def test_search_service_init():
    # SearchService requires db parameter
    mock_db = AsyncMock()
    svc = SearchService(db=mock_db)
    assert svc is not None
    assert svc.db == mock_db

def test_notification_service_init():
    svc = NotificationService()
    assert svc is not None