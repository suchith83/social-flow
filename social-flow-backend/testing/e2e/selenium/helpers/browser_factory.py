"""
Browser factory: returns selenium WebDriver instances for Chrome/Firefox,
capable of local and remote (Selenium Grid) usage.

- Uses webdriver-manager for local driver binaries for ease of setup.
- Accepts 'headless' option via E2E_HEADLESS env var.
- Configures useful capabilities for E2E.
"""

import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


class BrowserFactory:
    def __init__(self):
        self.headless = os.getenv("E2E_HEADLESS", "true").lower() in ("1", "true", "yes")

    def get_driver(self, browser_name: str = "chrome", remote_url: str | None = None):
        browser = browser_name.lower()
        if remote_url:
            return self._remote_driver(browser, remote_url)

        if browser == "chrome":
            opts = ChromeOptions()
            if self.headless:
                opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-gpu")
            opts.add_argument("--window-size=1920,1080")
            service = ChromeService(ChromeDriverManager().install())
            return webdriver.Chrome(service=service, options=opts)

        elif browser == "firefox":
            opts = FirefoxOptions()
            if self.headless:
                opts.add_argument("-headless")
            opts.set_preference("browser.download.folderList", 2)
            service = FirefoxService(GeckoDriverManager().install())
            return webdriver.Firefox(service=service, options=opts)

        else:
            raise ValueError(f"Unsupported browser: {browser}")

    def _remote_driver(self, browser: str, remote_url: str):
        # Configure desired capabilities for remote (Selenium Grid)
        if browser == "chrome":
            capabilities = DesiredCapabilities.CHROME.copy()
        elif browser == "firefox":
            capabilities = DesiredCapabilities.FIREFOX.copy()
        else:
            capabilities = DesiredCapabilities.CHROME.copy()

        # Example capabilities: enable VNC for Selenoid, etc.
        capabilities.setdefault("goog:loggingPrefs", {"browser": "ALL"})
        return webdriver.Remote(command_executor=remote_url, desired_capabilities=capabilities)
