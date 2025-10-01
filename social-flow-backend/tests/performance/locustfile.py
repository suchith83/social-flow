"""
Performance tests for Social Flow API.

This module contains load testing scenarios using Locust
to test API performance under various load conditions.
"""

from locust import HttpUser, task, between, events
import random
import json


class BaseAPIUser(HttpUser):
    """Base class for API users with common setup."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Register and login to get authentication token
        self.register_and_login()
    
    def register_and_login(self):
        """Register a test user and obtain authentication token."""
        # Generate unique username and email
        username = f"testuser_{random.randint(1000, 9999)}"
        email = f"{username}@test.com"
        
        # Register user
        register_data = {
            "username": username,
            "email": email,
            "password": "testpass123",
            "display_name": f"Test User {username}",
        }
        
        response = self.client.post(
            "/api/v1/auth/register",
            json=register_data,
        )
        
        if response.status_code == 201:
            # Login to get token
            login_data = {
                "email": email,
                "password": "testpass123",
            }
            
            login_response = self.client.post(
                "/api/v1/auth/login",
                json=login_data,
            )
            
            if login_response.status_code == 200:
                data = login_response.json()
                self.token = data.get("access_token")
                self.headers = {"Authorization": f"Bearer {self.token}"}
            else:
                self.token = None
                self.headers = {}
        else:
            self.token = None
            self.headers = {}


class VideoStreamingUser(HttpUser):
    """Simulates users watching videos."""
    
    wait_time = between(2, 5)
    
    @task(5)
    def get_trending_videos(self):
        """Get trending videos list."""
        self.client.get(
            "/api/v1/videos/trending",
            params={"limit": 20},
            name="/api/v1/videos/trending"
        )
    
    @task(3)
    def get_video_details(self):
        """Get details of a specific video."""
        video_id = f"video_{random.randint(1, 1000)}"
        self.client.get(
            f"/api/v1/videos/{video_id}",
            name="/api/v1/videos/[id]"
        )
    
    @task(10)
    def stream_video(self):
        """Simulate video streaming."""
        video_id = f"video_{random.randint(1, 1000)}"
        self.client.get(
            f"/api/v1/videos/{video_id}/stream",
            name="/api/v1/videos/[id]/stream"
        )
    
    @task(2)
    def search_videos(self):
        """Search for videos."""
        search_queries = ["tutorial", "funny", "music", "gaming", "tech"]
        query = random.choice(search_queries)
        self.client.get(
            "/api/v1/videos/search",
            params={"q": query, "limit": 20},
            name="/api/v1/videos/search"
        )


class FeedUser(BaseAPIUser):
    """Simulates users browsing their feed."""
    
    wait_time = between(1, 4)
    
    @task(10)
    def get_user_feed(self):
        """Get personalized user feed."""
        algorithms = ["chronological", "engagement", "ml"]
        algorithm = random.choice(algorithms)
        
        self.client.get(
            "/api/v1/posts/feed",
            params={"algorithm": algorithm, "limit": 20},
            headers=self.headers,
            name="/api/v1/posts/feed"
        )
    
    @task(5)
    def get_trending_posts(self):
        """Get trending posts."""
        self.client.get(
            "/api/v1/posts/trending",
            params={"limit": 20},
            name="/api/v1/posts/trending"
        )
    
    @task(3)
    def get_post_details(self):
        """Get details of a specific post."""
        post_id = f"post_{random.randint(1, 1000)}"
        self.client.get(
            f"/api/v1/posts/{post_id}",
            name="/api/v1/posts/[id]"
        )
    
    @task(2)
    def like_post(self):
        """Like a post."""
        post_id = f"post_{random.randint(1, 1000)}"
        self.client.post(
            f"/api/v1/posts/{post_id}/like",
            headers=self.headers,
            name="/api/v1/posts/[id]/like"
        )
    
    @task(1)
    def create_post(self):
        """Create a new post."""
        post_data = {
            "content": f"Test post from load testing {random.randint(1000, 9999)}",
            "visibility": "public",
        }
        
        self.client.post(
            "/api/v1/posts/",
            json=post_data,
            headers=self.headers,
            name="/api/v1/posts/"
        )


class ContentCreatorUser(BaseAPIUser):
    """Simulates content creators uploading and managing content."""
    
    wait_time = between(5, 10)
    
    @task(1)
    def upload_video(self):
        """Initiate video upload."""
        video_data = {
            "title": f"Test Video {random.randint(1000, 9999)}",
            "description": "Load testing video upload",
            "tags": ["test", "performance"],
            "visibility": "public",
        }
        
        self.client.post(
            "/api/v1/videos/upload/initiate",
            json=video_data,
            headers=self.headers,
            name="/api/v1/videos/upload/initiate"
        )
    
    @task(3)
    def get_my_videos(self):
        """Get creator's videos."""
        self.client.get(
            "/api/v1/videos/me",
            params={"limit": 20},
            headers=self.headers,
            name="/api/v1/videos/me"
        )
    
    @task(2)
    def get_video_analytics(self):
        """Get analytics for a video."""
        video_id = f"video_{random.randint(1, 100)}"
        self.client.get(
            f"/api/v1/videos/{video_id}/analytics",
            headers=self.headers,
            name="/api/v1/videos/[id]/analytics"
        )


class AuthenticationUser(HttpUser):
    """Simulates authentication operations."""
    
    wait_time = between(1, 2)
    
    @task(5)
    def register_user(self):
        """Register a new user."""
        username = f"testuser_{random.randint(10000, 99999)}"
        register_data = {
            "username": username,
            "email": f"{username}@test.com",
            "password": "testpass123",
            "display_name": f"Test User {username}",
        }
        
        self.client.post(
            "/api/v1/auth/register",
            json=register_data,
            name="/api/v1/auth/register"
        )
    
    @task(10)
    def login_user(self):
        """Login existing user."""
        login_data = {
            "email": f"testuser_{random.randint(1000, 9999)}@test.com",
            "password": "testpass123",
        }
        
        self.client.post(
            "/api/v1/auth/login",
            json=login_data,
            name="/api/v1/auth/login"
        )
    
    @task(2)
    def refresh_token(self):
        """Refresh authentication token."""
        self.client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "test_refresh_token"},
            name="/api/v1/auth/refresh"
        )


class LiveStreamingUser(BaseAPIUser):
    """Simulates live streaming viewers."""
    
    wait_time = between(2, 5)
    
    @task(5)
    def get_live_streams(self):
        """Get list of active live streams."""
        self.client.get(
            "/api/v1/live/streams",
            params={"status": "active", "limit": 20},
            name="/api/v1/live/streams"
        )
    
    @task(10)
    def watch_live_stream(self):
        """Watch a live stream."""
        stream_id = f"stream_{random.randint(1, 100)}"
        self.client.get(
            f"/api/v1/live/streams/{stream_id}",
            headers=self.headers,
            name="/api/v1/live/streams/[id]"
        )
    
    @task(3)
    def send_chat_message(self):
        """Send a chat message during live stream."""
        stream_id = f"stream_{random.randint(1, 100)}"
        message_data = {
            "message": f"Test message {random.randint(1000, 9999)}",
        }
        
        self.client.post(
            f"/api/v1/live/streams/{stream_id}/chat",
            json=message_data,
            headers=self.headers,
            name="/api/v1/live/streams/[id]/chat"
        )
    
    @task(1)
    def react_to_stream(self):
        """React to live stream."""
        stream_id = f"stream_{random.randint(1, 100)}"
        reaction_data = {
            "reaction_type": random.choice(["like", "love", "wow"]),
        }
        
        self.client.post(
            f"/api/v1/live/streams/{stream_id}/react",
            json=reaction_data,
            headers=self.headers,
            name="/api/v1/live/streams/[id]/react"
        )


class SearchUser(HttpUser):
    """Simulates users performing searches."""
    
    wait_time = between(2, 4)
    
    @task(5)
    def search_videos(self):
        """Search for videos."""
        queries = ["tutorial", "funny", "music", "gaming", "tech", "cooking", "travel"]
        query = random.choice(queries)
        
        self.client.get(
            "/api/v1/search/videos",
            params={"q": query, "limit": 20},
            name="/api/v1/search/videos"
        )
    
    @task(3)
    def search_users(self):
        """Search for users."""
        query = f"user{random.randint(1, 100)}"
        
        self.client.get(
            "/api/v1/search/users",
            params={"q": query, "limit": 20},
            name="/api/v1/search/users"
        )
    
    @task(2)
    def search_posts(self):
        """Search for posts."""
        hashtag = random.choice(["tech", "coding", "python", "ai", "ml"])
        
        self.client.get(
            "/api/v1/search/posts",
            params={"hashtag": hashtag, "limit": 20},
            name="/api/v1/search/posts"
        )


class HealthCheckUser(HttpUser):
    """Simulates health check requests (monitoring systems)."""
    
    wait_time = between(10, 30)  # Health checks are less frequent
    
    @task(5)
    def health_check_basic(self):
        """Basic health check."""
        self.client.get(
            "/api/v1/health/",
            name="/api/v1/health/"
        )
    
    @task(3)
    def health_check_ready(self):
        """Readiness check."""
        self.client.get(
            "/api/v1/health/ready",
            name="/api/v1/health/ready"
        )
    
    @task(1)
    def health_check_detailed(self):
        """Detailed health check."""
        self.client.get(
            "/api/v1/health/detailed",
            name="/api/v1/health/detailed"
        )


# Event handlers for reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    print("Starting performance test...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    print("Performance test completed!")
    
    # Print summary statistics
    stats = environment.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {stats.total.min_response_time}ms")
    print(f"Max response time: {stats.total.max_response_time}ms")
    print(f"Requests per second: {stats.total.total_rps:.2f}")
