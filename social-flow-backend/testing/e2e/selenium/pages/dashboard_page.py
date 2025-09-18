"""
Page object for Dashboard interactions.
"""

from selenium.webdriver.common.by import By
from .base_page import BasePage


class DashboardPage(BasePage):
    WIDGETS = (By.CSS_SELECTOR, ".widget")
    SETTINGS_LINK = (By.LINK_TEXT, "Settings")

    def open(self):
        self.go("/dashboard")

    def widget_count(self):
        return len(self.driver.find_elements(*self.WIDGETS))

    def go_to_settings(self):
        self.click(self.SETTINGS_LINK)
