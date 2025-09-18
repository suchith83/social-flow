"""
Page object for Multi-Cloud manager UI.
"""

from selenium.webdriver.common.by import By
from .base_page import BasePage


class MultiCloudPage(BasePage):
    CURRENT_PROVIDER = (By.CSS_SELECTOR, "#currentProvider")
    SELECT = (By.CSS_SELECTOR, "#switchProvider")
    FAILOVER_MSG = (By.XPATH, "//*[contains(., 'Failover activated')]")

    def open(self):
        self.go("/multi-cloud")

    def get_current_provider_text(self):
        return self.find(self.CURRENT_PROVIDER).text

    def switch_provider(self, provider_label: str):
        select = self.find(self.SELECT)
        for option in select.find_elements(By.TAG_NAME, "option"):
            if option.text.strip() == provider_label:
                option.click()
                return True
        return False

    def is_failover_shown(self):
        return self.is_visible(self.FAILOVER_MSG, timeout=8)
