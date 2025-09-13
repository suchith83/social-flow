"""
Unit tests for Background Sync API.
"""

import pytest
from fastapi.testclient import TestClient
from mobile_backend.apis.background_sync.routes import router
from mobile_backend.apis.background_sync.database import init_db
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()


def test_create_job():
    response = client.post("/background-sync/", json={"job_name": "test_job", "payload": {"data": 123}})
    assert response.status_code == 200
    data = response.json()
    assert data["job_name"] == "test_job"


def test_list_jobs():
    response = client.get("/background-sync/")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
