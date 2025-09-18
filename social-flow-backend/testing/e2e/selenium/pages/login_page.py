"""
Page object for the Login page.
Selectors are examples; adjust to match your app's DOM.
"""

from selenium.webdriver.common.by import By
from .base_page import BasePage


class LoginPage(BasePage):
    USERNAME = (By.NAME, "username")
    PASSWORD = (By.NAME, "password")
    SUBMIT = (By.CSS_SELECTOR, "button[type='submit']")
    ERROR = (By.CSS_SELECTOR, ".alert.alert-danger")

    def open(self):
        self.go("/login")

    def login(self, username: str, password: str):
        self.type(self.USERNAME, username)
        self.type(self.PASSWORD, password)
        self.click(self.SUBMIT)

    def login_and_wait_dashboard(self, username: str, password: str):
        self.login(username, password)
        # example: wait until dashboard element visible
        from selenium.webdriver.common.by import By
        return self.is_visible((By.CSS_SELECTOR, ".dashboard-container"), timeout=10)

    def get_error_text(self):
        if self.is_visible(self.ERROR, timeout=2):
            return self.find(self.ERROR).text
        return ""
