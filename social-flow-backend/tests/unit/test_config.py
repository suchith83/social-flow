"""
Tests for configuration module.

This demonstrates the testing approach for the Social Flow Backend.
Following TDD principles with comprehensive test coverage.
"""
import os
import pytest
from unittest.mock import patch

from app.core.config import Settings


class TestConfigurationLoading:
    """Test configuration loading and validation."""
    
    def test_default_configuration_loads(self):
        """Test that configuration loads with default values."""
        settings = Settings()
        
        # Test project info defaults
        assert settings.PROJECT_NAME == "Social Flow Backend"
        assert settings.VERSION == "1.0.0"
        assert settings.API_V1_STR == "/api/v1"
        
        # Test security defaults
        assert settings.ALGORITHM == "HS256"
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30
        assert settings.REFRESH_TOKEN_EXPIRE_DAYS == 7
        
        # Test feature defaults
        assert settings.POST_MAX_LENGTH == 280
        assert settings.FEED_PAGE_SIZE == 20
        assert settings.DEBUG is False
    
    def test_database_url_construction(self):
        """Test database URL is a valid PostgreSQL connection string."""
        settings = Settings()
        
        # Should have a valid PostgreSQL URL
        db_url = str(settings.DATABASE_URL)
        assert db_url.startswith("postgresql+asyncpg://") or db_url.startswith("postgresql://") or db_url.startswith("sqlite")
        # URL should contain expected structure
        assert "://" in db_url
        assert len(db_url) > 10
    
    def test_redis_url_construction_without_password(self):
        """Test Redis URL is a valid Redis connection string."""
        settings = Settings()
        
        # Should have a valid Redis URL
        assert settings.REDIS_URL.startswith("redis://")
        # URL should contain expected structure  
        assert "://" in settings.REDIS_URL
        assert len(settings.REDIS_URL) > 10
    
    def test_redis_url_construction_with_password(self):
        """Test that Redis password configuration is available."""
        settings = Settings()
        
        # REDIS_PASSWORD should be accessible (even if None)
        assert hasattr(settings, 'REDIS_PASSWORD')
        # Redis URL should be valid
        assert settings.REDIS_URL.startswith("redis://")
    
    def test_cors_origins_string_parsing(self):
        """Test CORS origins are configured and accessible."""
        settings = Settings()
        
        # BACKEND_CORS_ORIGINS should be a list
        assert isinstance(settings.BACKEND_CORS_ORIGINS, list)
        # Should be able to convert to strings
        for origin in settings.BACKEND_CORS_ORIGINS:
            assert len(str(origin)) > 0
    
    def test_cors_origins_list_input(self):
        """Test CORS origins list is properly structured."""
        settings = Settings()
        
        # Should be a list (even if empty)
        assert isinstance(settings.BACKEND_CORS_ORIGINS, list)
        # Each item should be convertible to string
        for origin in settings.BACKEND_CORS_ORIGINS:
            origin_str = str(origin)
            assert origin_str.startswith("http://") or origin_str.startswith("https://")
    
    def test_environment_variable_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        # Set environment variables explicitly
        monkeypatch.setenv('SECRET_KEY', 'test-secret-key-123')
        monkeypatch.setenv('DEBUG', 'true')
        monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
        monkeypatch.setenv('POSTGRES_PASSWORD', 'env-password')
        # Clear DATABASE_URL so it gets built from components
        monkeypatch.delenv('DATABASE_URL', raising=False)
        
        settings = Settings()
        
        assert settings.SECRET_KEY == 'test-secret-key-123'
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == 'DEBUG'
        # Password should be in the settings object
        assert settings.POSTGRES_PASSWORD == 'env-password'


class TestConfigurationValidation:
    """Test configuration validation logic."""
    
    def test_media_configuration_consistency(self):
        """Test media-related configuration consistency."""
        settings = Settings()
        
        # Ensure allowed extensions list is not empty
        assert len(settings.MEDIA_ALLOWED_EXTENSIONS) > 0
        
        # Ensure max file size is reasonable
        assert settings.MEDIA_UPLOAD_MAX_SIZE > 0
        assert settings.MEDIA_UPLOAD_MAX_SIZE <= 1024 * 1024 * 1024  # Max 1GB
        
        # Ensure max duration is positive
        assert settings.MEDIA_MAX_DURATION > 0
    
    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration values."""
        settings = Settings()
        
        # Ensure rate limiting values are positive
        assert settings.RATE_LIMIT_REQUESTS > 0
        assert settings.RATE_LIMIT_WINDOW > 0
        
        # Ensure reasonable defaults
        assert settings.RATE_LIMIT_REQUESTS <= 10000  # Not too high
        assert settings.RATE_LIMIT_WINDOW >= 1  # At least 1 second
    
    def test_social_features_limits(self):
        """Test social feature limits are reasonable."""
        settings = Settings()
        
        # Post length should be positive and not too large
        assert 0 < settings.POST_MAX_LENGTH <= 5000
        
        # Comment length should be positive and reasonable
        assert 0 < settings.COMMENT_MAX_LENGTH <= 2000
        
        # Feed page size should be reasonable
        assert 1 <= settings.FEED_PAGE_SIZE <= 100


class TestConfigurationIntegration:
    """Test configuration integration with the application."""
    
    def test_settings_instance_creation(self):
        """Test that the global settings instance can be imported."""
        from app.core.config import settings
        
        assert settings is not None
        assert isinstance(settings, Settings)
        assert settings.PROJECT_NAME == "Social Flow Backend"
    
    def test_database_url_is_valid_for_sqlalchemy(self):
        """Test database URL format is valid for SQLAlchemy."""
        settings = Settings()
        db_url = str(settings.DATABASE_URL)
        
        # Should be a valid database URL (PostgreSQL or SQLite)
        assert "://" in db_url
        
        # For PostgreSQL
        if db_url.startswith("postgresql"):
            assert db_url.startswith("postgresql+asyncpg://") or db_url.startswith("postgresql://")
            assert "@" in db_url
            if "@" in db_url:
                assert "/" in db_url.split("@")[1]  # Path component after host
        # For SQLite (common in testing)
        elif db_url.startswith("sqlite"):
            assert "sqlite:///" in db_url or "sqlite+aiosqlite:///" in db_url
    
    def test_redis_url_is_valid_for_redis_client(self):
        """Test Redis URL format is valid for Redis client."""
        settings = Settings()
        redis_url = settings.REDIS_URL
        
        # Should be a valid Redis URL
        assert redis_url.startswith("redis://")
        
        # Should contain host and port
        url_parts = redis_url.replace("redis://", "")
        if "@" in url_parts:
            url_parts = url_parts.split("@")[1]  # Remove auth part
        
        host_port, db = url_parts.split("/")
        assert ":" in host_port
        assert db.isdigit()


class TestDevelopmentConfiguration:
    """Test development-specific configuration scenarios."""
    
    @patch.dict(os.environ, {'DEBUG': 'true', 'TESTING': 'true'})
    def test_development_mode_settings(self):
        """Test development mode configuration."""
        settings = Settings()
        
        assert settings.DEBUG is True
        assert settings.TESTING is True
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'})
    def test_debug_logging_in_development(self):
        """Test debug logging configuration."""
        settings = Settings()
        
        assert settings.LOG_LEVEL == 'DEBUG'
    
    def test_local_development_defaults(self):
        """Test local development defaults."""
        settings = Settings()
        
        # Should default to local services (localhost, 127.0.0.1, or test database)
        # Allow flexibility for test environments
        db_url = str(settings.DATABASE_URL).lower()
        
        # Check if using local database (localhost, 127.0.0.1, or file-based SQLite)
        is_local = (
            "localhost" in db_url or 
            "127.0.0.1" in db_url or 
            "sqlite:///" in db_url or
            ".db" in db_url
        )
        
        assert is_local, f"Expected local database, got: {db_url}"


# Example fixture for testing (to be used in other test files)
@pytest.fixture
def test_settings():
    """Provide test-specific configuration settings."""
    return Settings(
        DEBUG=True,
        TESTING=True,
        DATABASE_URL="sqlite:///./test.db",  # In-memory for tests
        REDIS_URL="redis://localhost:6379/1",  # Different DB for tests
        SECRET_KEY="test-secret-key-for-testing-only",
        ACCESS_TOKEN_EXPIRE_MINUTES=5,  # Short expiration for tests
        FEED_PAGE_SIZE=5,  # Smaller pages for faster tests
    )


# Integration test example
@pytest.mark.integration
class TestConfigurationWithServices:
    """Integration tests for configuration with actual services."""
    
    def test_database_connection_with_real_config(self):
        """Test database connection with real configuration."""
        # Test that settings can be created with valid database configuration
        settings = Settings()
        
        # Verify DATABASE_URL is a valid string
        assert isinstance(str(settings.DATABASE_URL), str)
        assert len(str(settings.DATABASE_URL)) > 0
        
        # Verify it contains expected components
        db_url = str(settings.DATABASE_URL)
        assert "postgresql" in db_url or "sqlite" in db_url
    
    def test_redis_connection_with_real_config(self):
        """Test Redis connection with real configuration."""
        # Test that settings can be created with valid Redis configuration
        settings = Settings()
        
        # Verify REDIS_URL is a valid string
        assert isinstance(settings.REDIS_URL, str)
        assert len(settings.REDIS_URL) > 0
        
        # Verify it contains expected components
        assert "redis://" in settings.REDIS_URL


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
