"""
Multi-cloud manager tests: provider switching and simulated failover.
"""

import pytest
from pages.multi_cloud_page import MultiCloudPage


@pytest.mark.e2e
def test_switch_providers(driver, base_url):
    mc = MultiCloudPage(driver, base_url=base_url)
    mc.open()
    current = mc.get_current_provider_text()
    assert current, "No provider text shown"

    # attempt to switch providers; adjust labels to match UI
    assert mc.switch_provider("Azure Blob"), "Could not switch to Azure Blob"
    assert "Azure" in mc.get_current_provider_text()

    assert mc.switch_provider("Google Cloud Storage"), "Could not switch to GCS"
    assert "Google" in mc.get_current_provider_text()


@pytest.mark.e2e
def test_failover_on_api_failure(driver, base_url):
    mc = MultiCloudPage(driver, base_url=base_url)
    mc.open()
    # Simulate failure by executing a JS override that makes API fail (best-effort)
    driver = mc.driver
    try:
        driver.execute_script("""
            // intercept fetch to /api/storage/status to always return 500 (works if app uses fetch)
            const orig = window.fetch;
            window.fetch = function(url, opts) {
                if (url.includes('/api/storage/status')) {
                    return Promise.resolve(new Response('error', {status:500}));
                }
                return orig.apply(this, arguments);
            };
        """)
    except Exception:
        pytest.skip("Could not inject fetch override in this env")

    assert mc.is_failover_shown(), "Failover message not displayed after simulated API failure"
