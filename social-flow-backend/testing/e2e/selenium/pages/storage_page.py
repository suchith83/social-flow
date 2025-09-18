"""
Page object for Storage pages (S3 / Azure / GCS).
Includes upload and download helpers that wrap UI and assertions.
"""

from selenium.webdriver.common.by import By
from .base_page import BasePage
from helpers.file_utils import sample_file_path
import os


class StoragePage(BasePage):
    UPLOAD_INPUT = (By.CSS_SELECTOR, "input[type='file']#uploadInput")
    UPLOAD_BTN = (By.CSS_SELECTOR, "button#uploadBtn")
    UPLOAD_SUCCESS = (By.XPATH, "//div[contains(., 'Upload successful')]")
    DOWNLOAD_LINK_TEMPLATE = "//a[contains(., '{}')]"  # format filename into inner text

    def open_provider(self, provider_slug: str):
        # provider_slug examples: 's3', 'azure', 'gcs'
        self.go(f"/storage/{provider_slug}")

    def upload_file(self, fixture_name: str):
        path = sample_file_path(fixture_name)
        if not path.exists():
            raise FileNotFoundError(f"Fixture not found: {path}")
        # set file input
        file_input = self.find(self.UPLOAD_INPUT)
        # Selenium expects absolute path
        file_input.send_keys(str(path.resolve()))
        # click upload
        self.click(self.UPLOAD_BTN)
        return self.is_visible(self.UPLOAD_SUCCESS, timeout=12)

    def download_file(self, filename: str):
        link_xpath = self.DOWNLOAD_LINK_TEMPLATE.format(filename)
        elem = self.driver.find_element(By.XPATH, link_xpath)
        elem.click()
        # downloading verification is platform/browser-specific; leave to tests
