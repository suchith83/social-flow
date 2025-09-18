"""
Authentication E2E tests using Page Objects.
"""

import pytest
from pages.login_page import LoginPage
from pages.dashboard_page import DashboardPage


@pytest.mark.e2e
def test_login_success(driver, base_url, users):
    login = LoginPage(driver, base_url=base_url)
    dashboard = DashboardPage(driver, base_url=base_url)

    login.open()
    assert login.login_and_wait_dashboard(users[0]['username'], users[0]['password']), "Login -> Dashboard not reached"
    dashboard.open()
    assert dashboard.widget_count() >= 0  # basic smoke assertion


@pytest.mark.e2e
def test_login_failure_shows_error(driver, base_url):
    login = LoginPage(driver, base_url=base_url)
    login.open()
    login.login("baduser", "badpass")
    err = login.get_error_text()
    assert "Invalid" in err or err != "", f"Expected login error message, got '{err}'"
