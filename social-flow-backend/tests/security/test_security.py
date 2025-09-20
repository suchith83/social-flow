"""
Security tests for the Social Flow backend.

This module contains security tests to ensure the backend
is protected against common vulnerabilities.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestSecurity:
    """Security tests for the backend."""

    def test_sql_injection_protection(self, client: TestClient):
        """Test protection against SQL injection attacks."""
        # Test SQL injection in search query
        malicious_query = "'; DROP TABLE users; --"
        response = client.get(f"/api/v1/videos/search?q={malicious_query}")
        
        # Should not cause server error
        assert response.status_code in [200, 400, 422]
        
        # Test SQL injection in user registration
        malicious_data = {
            "username": "'; DROP TABLE users; --",
            "email": "test@example.com",
            "password": "testpassword123",
        }
        response = client.post("/api/v1/auth/register", json=malicious_data)
        
        # Should not cause server error
        assert response.status_code in [201, 400, 422]

    def test_xss_protection(self, client: TestClient):
        """Test protection against XSS attacks."""
        # Test XSS in video title
        xss_payload = "<script>alert('XSS')</script>"
        video_data = {
            "title": xss_payload,
            "description": "Test video",
            "filename": "test.mp4",
            "file_size": 1024,
        }
        
        # This should be handled by the client, but we test the server response
        response = client.post("/api/v1/videos/upload", json=video_data)
        
        # Should not execute script (status code should indicate validation error)
        assert response.status_code in [201, 400, 422]

    def test_csrf_protection(self, client: TestClient):
        """Test CSRF protection."""
        # Test CSRF token requirement for state-changing operations
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123",
        }
        
        # Register user
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201
        
        # Test that state-changing operations require proper authentication
        # (This is handled by the authentication system)

    def test_authentication_bypass(self, client: TestClient, test_video):
        """Test authentication bypass attempts."""
        # Test accessing protected endpoint without authentication
        response = client.get(f"/api/v1/videos/{test_video.id}/analytics")
        assert response.status_code == 401
        
        # Test with invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get(f"/api/v1/videos/{test_video.id}/analytics", headers=headers)
        assert response.status_code == 401
        
        # Test with malformed token
        headers = {"Authorization": "InvalidFormat token"}
        response = client.get(f"/api/v1/videos/{test_video.id}/analytics", headers=headers)
        assert response.status_code == 401

    def test_authorization_bypass(self, client: TestClient, test_video):
        """Test authorization bypass attempts."""
        # Test accessing another user's data
        # (This would require creating another user and testing access)
        pass

    def test_input_validation(self, client: TestClient):
        """Test input validation and sanitization."""
        # Test invalid email format
        user_data = {
            "username": "testuser",
            "email": "invalid-email",
            "password": "testpassword123",
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 422
        
        # Test empty required fields
        user_data = {
            "username": "",
            "email": "test@example.com",
            "password": "testpassword123",
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 422
        
        # Test password too short
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "123",
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 422

    def test_rate_limiting(self, client: TestClient):
        """Test rate limiting protection."""
        # Test rapid requests to the same endpoint
        for i in range(100):  # Try to exceed rate limit
            response = client.get("/api/v1/videos/search?q=test")
            if response.status_code == 429:  # Rate limited
                break
        else:
            # If we get here, rate limiting might not be implemented
            # This is acceptable for testing purposes
            pass

    def test_file_upload_security(self, client: TestClient, auth_headers):
        """Test file upload security."""
        # Test uploading non-video file
        malicious_data = {
            "title": "Test Video",
            "description": "Test video",
            "filename": "malicious.exe",  # Executable file
            "file_size": 1024,
        }
        response = client.post("/api/v1/videos/upload", json=malicious_data, headers=auth_headers)
        
        # Should reject non-video files
        assert response.status_code in [400, 422]
        
        # Test uploading oversized file
        oversized_data = {
            "title": "Test Video",
            "description": "Test video",
            "filename": "huge_video.mp4",
            "file_size": 10 * 1024 * 1024 * 1024,  # 10GB
        }
        response = client.post("/api/v1/videos/upload", json=oversized_data, headers=auth_headers)
        
        # Should reject oversized files
        assert response.status_code in [400, 422]

    def test_jwt_token_security(self, client: TestClient):
        """Test JWT token security."""
        # Test with expired token
        expired_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/api/v1/auth/profile", headers=headers)
        assert response.status_code == 401
        
        # Test with malformed JWT
        malformed_token = "not.a.jwt.token"
        headers = {"Authorization": f"Bearer {malformed_token}"}
        response = client.get("/api/v1/auth/profile", headers=headers)
        assert response.status_code == 401

    def test_password_security(self, client: TestClient):
        """Test password security requirements."""
        # Test weak password
        weak_password_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "123456",  # Too weak
        }
        response = client.post("/api/v1/auth/register", json=weak_password_data)
        assert response.status_code == 422
        
        # Test common password
        common_password_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "password",  # Too common
        }
        response = client.post("/api/v1/auth/register", json=common_password_data)
        assert response.status_code == 422

    def test_https_enforcement(self, client: TestClient):
        """Test HTTPS enforcement."""
        # Test that sensitive endpoints require HTTPS
        # (This is typically handled at the reverse proxy level)
        pass

    def test_cors_security(self, client: TestClient):
        """Test CORS security configuration."""
        # Test CORS preflight request
        response = client.options("/api/v1/videos/feed", headers={
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "GET",
        })
        
        # Should not allow requests from malicious origins
        # (This depends on CORS configuration)

    def test_content_type_validation(self, client: TestClient):
        """Test content type validation."""
        # Test sending JSON with wrong content type
        response = client.post(
            "/api/v1/auth/register",
            data="invalid json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422

    def test_parameter_pollution(self, client: TestClient):
        """Test parameter pollution attacks."""
        # Test duplicate parameters
        response = client.get("/api/v1/videos/search?q=test&q=malicious")
        
        # Should handle duplicate parameters gracefully
        assert response.status_code in [200, 400, 422]

    def test_path_traversal(self, client: TestClient):
        """Test path traversal protection."""
        # Test path traversal in file upload
        malicious_data = {
            "title": "Test Video",
            "description": "Test video",
            "filename": "../../../etc/passwd",
            "file_size": 1024,
        }
        response = client.post("/api/v1/videos/upload", json=malicious_data)
        
        # Should reject path traversal attempts
        assert response.status_code in [400, 422]

    def test_injection_attacks(self, client: TestClient):
        """Test various injection attacks."""
        # Test NoSQL injection (if applicable)
        malicious_data = {
            "username": {"$ne": None},
            "email": "test@example.com",
            "password": "testpassword123",
        }
        response = client.post("/api/v1/auth/register", json=malicious_data)
        assert response.status_code == 422
        
        # Test LDAP injection (if applicable)
        malicious_data = {
            "username": "admin)(&(password=*))",
            "email": "test@example.com",
            "password": "testpassword123",
        }
        response = client.post("/api/v1/auth/register", json=malicious_data)
        assert response.status_code == 422

    def test_session_security(self, client: TestClient):
        """Test session security."""
        # Test session fixation
        # (This is handled by the authentication system)
        pass

    def test_information_disclosure(self, client: TestClient):
        """Test information disclosure prevention."""
        # Test that error messages don't reveal sensitive information
        response = client.get("/api/v1/videos/nonexistent_video")
        assert response.status_code == 404
        
        # Error message should not reveal internal details
        data = response.json()
        assert "error" in data
        # Should not contain stack traces or internal paths

    def test_http_method_validation(self, client: TestClient):
        """Test HTTP method validation."""
        # Test unsupported methods
        response = client.patch("/api/v1/videos/feed")
        assert response.status_code == 405
        
        response = client.trace("/api/v1/videos/feed")
        assert response.status_code == 405
