"""
Performance tests for the Social Flow backend.

This module contains performance tests to ensure the backend
can handle expected load and meets performance requirements.
"""

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestPerformance:
    """Performance tests for the backend."""

    def test_concurrent_user_registration(self, client: TestClient):
        """Test concurrent user registration performance."""
        def register_user(user_id):
            user_data = {
                "username": f"testuser{user_id}",
                "email": f"test{user_id}@example.com",
                "password": "testpassword123",
                "display_name": f"Test User {user_id}",
            }
            return client.post("/api/v1/auth/register", json=user_data)
        
        # Test with 10 concurrent registrations
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_user, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All requests should succeed
        assert all(result.status_code == 201 for result in results)
        
        # Should complete within reasonable time (5 seconds)
        assert duration < 5.0
        
        print(f"Concurrent user registration completed in {duration:.2f} seconds")

    def test_concurrent_video_views(self, client: TestClient, test_video):
        """Test concurrent video view performance."""
        def view_video():
            return client.post(f"/api/v1/videos/{test_video.id}/view")
        
        # Test with 50 concurrent views
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(view_video) for _ in range(50)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All requests should succeed
        assert all(result.status_code == 200 for result in results)
        
        # Should complete within reasonable time (3 seconds)
        assert duration < 3.0
        
        print(f"Concurrent video views completed in {duration:.2f} seconds")

    def test_concurrent_likes(self, client: TestClient, auth_headers, test_video):
        """Test concurrent video like performance."""
        def like_video():
            return client.post(f"/api/v1/videos/{test_video.id}/like", headers=auth_headers)
        
        # Test with 20 concurrent likes
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(like_video) for _ in range(20)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All requests should succeed
        assert all(result.status_code == 200 for result in results)
        
        # Should complete within reasonable time (2 seconds)
        assert duration < 2.0
        
        print(f"Concurrent video likes completed in {duration:.2f} seconds")

    def test_video_feed_performance(self, client: TestClient, auth_headers):
        """Test video feed retrieval performance."""
        # Test multiple feed requests
        start_time = time.time()
        
        for _ in range(10):
            response = client.get("/api/v1/videos/feed", headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (2 seconds)
        assert duration < 2.0
        
        print(f"Video feed requests completed in {duration:.2f} seconds")

    def test_search_performance(self, client: TestClient):
        """Test search performance with various queries."""
        queries = ["test", "gaming", "music", "sports", "tech"]
        
        start_time = time.time()
        
        for query in queries:
            response = client.get(f"/api/v1/videos/search?q={query}")
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (3 seconds)
        assert duration < 3.0
        
        print(f"Search queries completed in {duration:.2f} seconds")

    def test_analytics_performance(self, client: TestClient, auth_headers, test_video):
        """Test analytics retrieval performance."""
        # Test multiple analytics requests
        start_time = time.time()
        
        for _ in range(5):
            response = client.get(f"/api/v1/videos/{test_video.id}/analytics", headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (2 seconds)
        assert duration < 2.0
        
        print(f"Analytics requests completed in {duration:.2f} seconds")

    def test_notification_performance(self, client: TestClient, auth_headers):
        """Test notification retrieval performance."""
        # Test multiple notification requests
        start_time = time.time()
        
        for _ in range(10):
            response = client.get("/api/v1/notifications/", headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (2 seconds)
        assert duration < 2.0
        
        print(f"Notification requests completed in {duration:.2f} seconds")

    def test_live_stream_performance(self, client: TestClient, auth_headers):
        """Test live streaming performance."""
        # Test multiple live stream requests
        start_time = time.time()
        
        for _ in range(5):
            response = client.get("/api/v1/live/active", headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (2 seconds)
        assert duration < 2.0
        
        print(f"Live stream requests completed in {duration:.2f} seconds")

    def test_ml_recommendations_performance(self, client: TestClient, auth_headers):
        """Test ML recommendations performance."""
        # Test multiple recommendation requests
        start_time = time.time()
        
        for _ in range(5):
            response = client.get("/api/v1/ml/recommendations/user123", headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (3 seconds)
        assert duration < 3.0
        
        print(f"ML recommendation requests completed in {duration:.2f} seconds")

    def test_payment_processing_performance(self, client: TestClient, auth_headers):
        """Test payment processing performance."""
        payment_data = {
            "amount": 1000,
            "currency": "USD",
            "payment_method": "stripe",
        }
        
        # Test multiple payment requests
        start_time = time.time()
        
        for _ in range(5):
            response = client.post("/api/v1/payments/process", json=payment_data, headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (3 seconds)
        assert duration < 3.0
        
        print(f"Payment processing requests completed in {duration:.2f} seconds")

    def test_ads_serving_performance(self, client: TestClient, auth_headers, test_video):
        """Test ad serving performance."""
        # Test multiple ad requests
        start_time = time.time()
        
        for _ in range(10):
            response = client.get(f"/api/v1/ads/video/{test_video.id}", headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (2 seconds)
        assert duration < 2.0
        
        print(f"Ad serving requests completed in {duration:.2f} seconds")

    def test_database_connection_pool(self, client: TestClient):
        """Test database connection pool performance."""
        # Test multiple database operations
        start_time = time.time()
        
        for i in range(20):
            user_data = {
                "username": f"pooltest{i}",
                "email": f"pooltest{i}@example.com",
                "password": "testpassword123",
                "display_name": f"Pool Test {i}",
            }
            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 201
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (5 seconds)
        assert duration < 5.0
        
        print(f"Database connection pool test completed in {duration:.2f} seconds")

    def test_memory_usage(self, client: TestClient):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        for i in range(100):
            user_data = {
                "username": f"memtest{i}",
                "email": f"memtest{i}@example.com",
                "password": "testpassword123",
                "display_name": f"Memory Test {i}",
            }
            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 201
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100
        
        print(f"Memory usage increased by {memory_increase:.2f} MB")

    def test_response_time_percentiles(self, client: TestClient, test_video):
        """Test response time percentiles."""
        response_times = []
        
        # Perform 100 requests
        for _ in range(100):
            start_time = time.time()
            response = client.get(f"/api/v1/videos/{test_video.id}")
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Calculate percentiles
        response_times.sort()
        p50 = response_times[49]  # 50th percentile
        p90 = response_times[89]  # 90th percentile
        p95 = response_times[94]  # 95th percentile
        p99 = response_times[98]  # 99th percentile
        
        # Response times should be reasonable
        assert p50 < 0.1  # 50% of requests under 100ms
        assert p90 < 0.2  # 90% of requests under 200ms
        assert p95 < 0.5  # 95% of requests under 500ms
        assert p99 < 1.0  # 99% of requests under 1s
        
        print(f"Response time percentiles - P50: {p50:.3f}s, P90: {p90:.3f}s, P95: {p95:.3f}s, P99: {p99:.3f}s")
