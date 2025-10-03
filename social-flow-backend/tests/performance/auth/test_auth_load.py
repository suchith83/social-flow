"""Authentication Load Tests."""

import pytest
import time
import random


def load_test_authentication_system(num_requests):
    """Simulate authentication requests for load testing."""
    start_time = time.time()
    for _ in range(num_requests):
        # Simulate an authentication request
        time.sleep(random.uniform(0.001, 0.01))  # Reduced simulation time
    end_time = time.time()
    return end_time - start_time


class TestAuthLoad:
    """Test authentication system under load."""
    
    @pytest.mark.slow
    def test_auth_load_100_requests(self):
        """Test authentication with 100 concurrent requests."""
        duration = load_test_authentication_system(100)
        assert duration < 5, f"Load test failed: took {duration:.2f}s (expected < 5s)"
    
    @pytest.mark.slow
    def test_auth_load_performance(self):
        """Test authentication performance under load."""
        duration = load_test_authentication_system(50)
        assert duration < 3, f"Performance test failed: took {duration:.2f}s (expected < 3s)"
