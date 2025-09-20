#!/usr/bin/env python3
"""
Integration test script for Social Flow backend.

This script performs end-to-end integration testing to ensure
all components work together correctly.
"""

import asyncio
import json
import time
from typing import Dict, Any
import httpx
from pathlib import Path


class IntegrationTester:
    """Integration tester for Social Flow backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self.test_data = {}
        self.results = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            result = await test_func()
            duration = time.time() - start_time
            
            self.results.append({
                "test": test_name,
                "status": "PASS",
                "duration": duration,
                "result": result
            })
            print(f"✅ {test_name} - PASSED ({duration:.2f}s)")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.results.append({
                "test": test_name,
                "status": "FAIL",
                "duration": duration,
                "error": str(e)
            })
            print(f"❌ {test_name} - FAILED ({duration:.2f}s): {e}")
            return None
    
    async def test_health_check(self):
        """Test health check endpoint."""
        response = await self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        return data
    
    async def test_user_registration(self):
        """Test user registration flow."""
        user_data = {
            "username": "integration_test_user",
            "email": "integration@test.com",
            "password": "testpassword123",
            "display_name": "Integration Test User",
        }
        
        response = await self.client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert "verification_token" in data
        
        self.test_data["user"] = user_data
        self.test_data["verification_token"] = data["verification_token"]
        return data
    
    async def test_email_verification(self):
        """Test email verification flow."""
        verification_data = {
            "token": self.test_data["verification_token"],
        }
        
        response = await self.client.post("/api/v1/auth/verify-email", json=verification_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "verified"
        return data
    
    async def test_user_login(self):
        """Test user login flow."""
        login_data = {
            "email": self.test_data["user"]["email"],
            "password": self.test_data["user"]["password"],
        }
        
        response = await self.client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        
        self.test_data["access_token"] = data["access_token"]
        self.test_data["refresh_token"] = data["refresh_token"]
        self.test_data["user_id"] = data["user"]["id"]
        return data
    
    async def test_video_upload(self):
        """Test video upload flow."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        video_data = {
            "title": "Integration Test Video",
            "description": "This is a test video for integration testing",
            "filename": "test_video.mp4",
            "file_size": 1024000,
        }
        
        response = await self.client.post("/api/v1/videos/upload", json=video_data, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert "video_id" in data
        
        self.test_data["video_id"] = data["video_id"]
        return data
    
    async def test_video_retrieval(self):
        """Test video retrieval."""
        video_id = self.test_data["video_id"]
        
        response = await self.client.get(f"/api/v1/videos/{video_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == video_id
        assert data["title"] == "Integration Test Video"
        return data
    
    async def test_video_like(self):
        """Test video like functionality."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        video_id = self.test_data["video_id"]
        
        response = await self.client.post(f"/api/v1/videos/{video_id}/like", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "liked"
        return data
    
    async def test_video_view(self):
        """Test video view tracking."""
        video_id = self.test_data["video_id"]
        
        response = await self.client.post(f"/api/v1/videos/{video_id}/view")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "viewed"
        return data
    
    async def test_video_feed(self):
        """Test video feed retrieval."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        response = await self.client.get("/api/v1/videos/feed", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data
        return data
    
    async def test_video_search(self):
        """Test video search functionality."""
        response = await self.client.get("/api/v1/videos/search?q=integration")
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data
        return data
    
    async def test_video_analytics(self):
        """Test video analytics retrieval."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        video_id = self.test_data["video_id"]
        
        response = await self.client.get(f"/api/v1/videos/{video_id}/analytics", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "video_id" in data
        assert "views" in data
        return data
    
    async def test_ml_recommendations(self):
        """Test ML recommendations."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        user_id = self.test_data["user_id"]
        
        response = await self.client.get(f"/api/v1/ml/recommendations/{user_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        return data
    
    async def test_notifications(self):
        """Test notification functionality."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        response = await self.client.get("/api/v1/notifications/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "notifications" in data
        return data
    
    async def test_live_streaming(self):
        """Test live streaming functionality."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        # Test getting active live streams
        response = await self.client.get("/api/v1/live/active", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "active_streams" in data
        return data
    
    async def test_ads_serving(self):
        """Test ad serving functionality."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        video_id = self.test_data["video_id"]
        
        response = await self.client.get(f"/api/v1/ads/video/{video_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "ads" in data
        return data
    
    async def test_payments(self):
        """Test payment functionality."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        payment_data = {
            "amount": 1000,
            "currency": "USD",
            "payment_method": "stripe",
        }
        
        response = await self.client.post("/api/v1/payments/process", json=payment_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        return data
    
    async def test_analytics(self):
        """Test analytics functionality."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        response = await self.client.get("/api/v1/analytics/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "analytics" in data
        return data
    
    async def test_search(self):
        """Test search functionality."""
        response = await self.client.get("/api/v1/search?q=test")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        return data
    
    async def test_user_profile(self):
        """Test user profile functionality."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        response = await self.client.get("/api/v1/auth/profile", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        return data
    
    async def test_user_preferences(self):
        """Test user preferences functionality."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        preferences = {
            "notifications": {"email": True, "push": False},
            "privacy": {"profile_visibility": "public"},
        }
        
        response = await self.client.put("/api/v1/auth/preferences", json=preferences, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        return data
    
    async def test_logout(self):
        """Test user logout."""
        headers = {"Authorization": f"Bearer {self.test_data['access_token']}"}
        
        logout_data = {
            "refresh_token": self.test_data["refresh_token"],
        }
        
        response = await self.client.post("/api/v1/auth/logout", json=logout_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        return data
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Social Flow Backend Integration Tests")
        print(f"Target URL: {self.base_url}")
        
        # Core functionality tests
        await self.run_test("Health Check", self.test_health_check)
        await self.run_test("User Registration", self.test_user_registration)
        await self.run_test("Email Verification", self.test_email_verification)
        await self.run_test("User Login", self.test_user_login)
        await self.run_test("User Profile", self.test_user_profile)
        await self.run_test("User Preferences", self.test_user_preferences)
        
        # Video functionality tests
        await self.run_test("Video Upload", self.test_video_upload)
        await self.run_test("Video Retrieval", self.test_video_retrieval)
        await self.run_test("Video Like", self.test_video_like)
        await self.run_test("Video View", self.test_video_view)
        await self.run_test("Video Feed", self.test_video_feed)
        await self.run_test("Video Search", self.test_video_search)
        await self.run_test("Video Analytics", self.test_video_analytics)
        
        # ML/AI functionality tests
        await self.run_test("ML Recommendations", self.test_ml_recommendations)
        
        # Additional functionality tests
        await self.run_test("Notifications", self.test_notifications)
        await self.run_test("Live Streaming", self.test_live_streaming)
        await self.run_test("Ads Serving", self.test_ads_serving)
        await self.run_test("Payments", self.test_payments)
        await self.run_test("Analytics", self.test_analytics)
        await self.run_test("Search", self.test_search)
        
        # Cleanup
        await self.run_test("User Logout", self.test_logout)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("INTEGRATION TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['error']}")
        
        total_duration = sum(r["duration"] for r in self.results)
        print(f"\nTotal Duration: {total_duration:.2f}s")
        
        if failed_tests == 0:
            print("\n🎉 All integration tests passed!")
        else:
            print(f"\n⚠️  {failed_tests} integration tests failed!")
        
        # Save results to file
        results_file = Path("integration_test_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Social Flow backend integration tests")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for the backend")
    parser.add_argument("--test", help="Run specific test only")
    
    args = parser.parse_args()
    
    async with IntegrationTester(args.url) as tester:
        if args.test:
            # Run specific test
            test_func = getattr(tester, f"test_{args.test}")
            await tester.run_test(args.test, test_func)
        else:
            # Run all tests
            await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
