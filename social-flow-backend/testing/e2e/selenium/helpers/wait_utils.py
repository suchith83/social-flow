"""
Wait utilities that centralize explicit waits and common conditions.
Improves readability and reusability in page objects.
"""

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By


def wait_for_visible(driver, by_locator, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(EC.visibility_of_element_located(by_locator))
        return True
    except TimeoutException:
        return False


def wait_for_clickable(driver, by_locator, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(EC.element_to_be_clickable(by_locator))
        return True
    except TimeoutException:
        return False
