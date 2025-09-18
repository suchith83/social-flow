"""
BasePage: common functionality for all page objects.

- Holds driver reference
- Common helpers (navigate, find, click, type)
- Central place to adjust wait strategies or logging
"""

from selenium.webdriver.common.by import By
from helpers.wait_utils import wait_for_visible, wait_for_clickable


class BasePage:
    def __init__(self, driver, base_url=None):
        self.driver = driver
        self.base_url = base_url.rstrip("/") if base_url else None

    def url(self, path: str):
        if self.base_url:
            return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        return path

    def go(self, path: str):
        self.driver.get(self.url(path))

    def find(self, by_locator):
        return self.driver.find_element(*by_locator)

    def click(self, by_locator, timeout=8):
        if wait_for_clickable(self.driver, by_locator, timeout=timeout):
            self.find(by_locator).click()
        else:
            raise RuntimeError(f"Element not clickable: {by_locator}")

    def type(self, by_locator, text: str, timeout=8, clear_first=True):
        if wait_for_visible(self.driver, by_locator, timeout=timeout):
            el = self.find(by_locator)
            if clear_first:
                el.clear()
            el.send_keys(text)
        else:
            raise RuntimeError(f"Element not visible to type: {by_locator}")

    def is_visible(self, by_locator, timeout=5):
        return wait_for_visible(self.driver, by_locator, timeout=timeout)
