# Unit tests for push-notifications
"""
Unit tests for Push Notifications API.
These are simple functional tests using TestClient and the in-memory SQLite DB.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from mobile_backend.apis.push_notifications.routes import router
from mobile_backend.apis.push_notifications.database import init_db
from mobile_backend.apis.push_notifications.models import DevicePlatform

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()


def test_register_device():
    payload = {
        "user_id": "user_123",
        "token": "token_abcdef1234567890",
        "platform": "android",
        "app_version": "1.0.0"
    }
    r = client.post("/push/devices", json=payload)
    assert r.status_code == 201
    body = r.json()
    assert body["token"] == payload["token"]
    assert body["platform"] == payload["platform"]


def test_send_notification_no_tokens():
    payload = {
        "title": "Hello",
        "body": "No tokens",
        "tokens": []
    }
    r = client.post("/push/send", json=payload)
    assert r.status_code == 400  # expected because no tokens and no user_ids


def test_send_notification_with_token():
    # register token
    token = "token_send_1234567890"
    client.post("/push/devices", json={
        "user_id": "user_abc",
        "token": token,
        "platform": "android"
    })
    payload = {
        "title": "Hello",
        "body": "You have a message",
        "tokens": [token]
    }
    r = client.post("/push/send", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "Hello"
