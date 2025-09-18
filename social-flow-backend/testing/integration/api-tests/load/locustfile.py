"""
Simple Locust file to stress a few endpoints; adapt to your API.

Run with:
    locust -f locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between

class ApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health(self):
        self.client.get("/health")

    @task(2)
    def list_buckets(self):
        self.client.get("/storage/buckets")

    @task(1)
    def login(self):
        self.client.post("/auth/login", json={"username": "admin", "password": "password123"})
