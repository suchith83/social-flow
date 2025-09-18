"""
Regression smoke tests that ensure basic app functionality.
"""

import pytest
from pages.login_page import LoginPage
from pages.dashboard_page import DashboardPage


@pytest.mark.smoke
def test_dashboard_widgets_load(driver, base_url, users):
    login = LoginPage(driver, base_url=base_url)
    dashboard = DashboardPage(driver, base_url=base_url)
    login.open()
    assert login.login_and_wait_dashboard(users[0]['username'], users[0]['password'])
    dashboard.open()
    assert dashboard.widget_count() > 1, "Expected more than 1 widget on dashboard"
