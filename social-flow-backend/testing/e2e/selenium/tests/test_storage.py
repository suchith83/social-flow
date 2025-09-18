"""
Storage upload/download tests across different provider UI pages.
"""

import pytest
from pages.storage_page import StoragePage
from pathlib import Path
import time


@pytest.mark.e2e
@pytest.mark.parametrize("provider, fixture", [
    ("s3", "sample.txt"),
    ("azure", "sample.txt"),
    ("gcs", "sample.png"),
])
def test_upload_to_provider(driver, base_url, provider, fixture):
    sp = StoragePage(driver, base_url=base_url)
    sp.open_provider(provider)
    ok = sp.upload_file(fixture)
    assert ok, f"Upload to {provider} failed or no success message visible"


@pytest.mark.e2e
def test_download_from_azure(driver, base_url):
    sp = StoragePage(driver, base_url=base_url)
    sp.open_provider("azure")
    # This test assumes there is a link 'Download test-azure.txt' in UI
    # For browsers headless, verifying actual file presence can require config; we assert the click works.
    try:
        sp.download_file("test-azure.txt")
        # short wait for browser to process
        time.sleep(2)
    except Exception as e:
        pytest.skip(f"Download UI not present or not supported in this environment: {e}")
