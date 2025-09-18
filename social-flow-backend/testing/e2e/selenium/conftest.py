"""
Pytest configuration and fixtures for Selenium E2E tests.

- Provides browser fixture (local or remote)
- Manages test-level logging, screenshot-on-failure, and report attachments
- Loads environment variables (.env)
"""

import os
import json
import pytest
from datetime import datetime
from pathlib import Path
from selenium.common.exceptions import WebDriverException
from helpers.browser_factory import BrowserFactory
from dotenv import load_dotenv

# Load .env from this folder by default
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Fixture scope: session-level browser factory configuration
@pytest.fixture(scope="session")
def browser_name():
    # Choose browser via env var, default to chrome
    return os.getenv("E2E_BROWSER", "chrome").lower()


@pytest.fixture(scope="session")
def remote_url():
    # If using Selenium Grid or remote webdriver, set REMOTE_URL in .env
    return os.getenv("REMOTE_URL")


@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:3000")


@pytest.fixture(scope="function")
def driver(request, browser_name, remote_url):
    """
    Instantiate a webdriver for each test function.
    Closes and quits after test. Captures screenshot on failure.
    """
    bf = BrowserFactory()
    driver = bf.get_driver(browser_name=browser_name, remote_url=remote_url)

    # maximize and set implicit wait modestly
    driver.maximize_window()
    driver.implicitly_wait(3)

    yield driver

    # If test failed, capture screenshot (best-effort)
    if request.node.rep_call.failed:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = request.node.name.replace("/", "_").replace(" ", "_")
        screenshot_path = REPORTS_DIR / f"{safe_name}_{ts}.png"
        try:
            driver.save_screenshot(str(screenshot_path))
            print(f"Saved screenshot to {screenshot_path}")
        except WebDriverException as e:
            print(f"Could not capture screenshot: {e}")

    driver.quit()


# Pytest hook to make request.node.rep_call available in fixtures
def pytest_runtest_makereport(item, call):
    # attach the report object to the item for use in fixtures
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)
    return rep


@pytest.fixture(scope="session")
def users():
    # Load test users from fixture file
    path = PROJECT_ROOT / "fixtures" / "users.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []
