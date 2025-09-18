import { test, expect } from '@playwright/test';

test.describe('Multi-cloud manager tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.locator('input[name="username"]').fill('admin');
    await page.locator('input[name="password"]').fill('password123');
    await page.locator('button[type="submit"]').click();
    await expect(page).toHaveURL(/\/dashboard/);
  });

  test('switch providers in UI', async ({ page }) => {
    await page.goto('/multi-cloud');
    await expect(page.locator('text=Current Provider')).toBeVisible();

    await page.locator('#switchProvider').selectOption({ label: 'Azure Blob' });
    await expect(page.locator('text=Current Provider: Azure Blob')).toBeVisible();

    await page.locator('#switchProvider').selectOption({ label: 'Google Cloud Storage' });
    await expect(page.locator('text=Current Provider: Google Cloud Storage')).toBeVisible();
  });

  test('simulate provider failure and failover', async ({ page, context }) => {
    // Intercept the status API to simulate failure for primary provider
    await page.route('**/api/storage/status', (route) => {
      route.fulfill({ status: 500, body: 'error' });
    });
    await page.goto('/multi-cloud');
    await expect(page.locator('text=Failover activated')).toBeVisible();
  });
});
