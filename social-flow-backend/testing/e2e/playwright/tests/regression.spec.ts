import { test, expect } from '@playwright/test';

test.describe('Regression smoke tests', () => {
  test('dashboard loads and has widgets', async ({ page }) => {
    await page.goto('/login');
    await page.locator('input[name="username"]').fill('admin');
    await page.locator('input[name="password"]').fill('password123');
    await page.locator('button[type="submit"]').click();
    await expect(page).toHaveURL(/\/dashboard/);
    const widgets = await page.locator('.widget').count();
    expect(widgets).toBeGreaterThan(2);
  });

  test('user can update settings', async ({ page }) => {
    await page.goto('/login');
    await page.locator('input[name="username"]').fill('user');
    await page.locator('input[name="password"]').fill('userpass');
    await page.locator('button[type="submit"]').click();
    await page.goto('/settings');
    await page.locator('input[name="email"]').fill('newemail@example.com');
    await page.locator('button[type="submit"]').click();
    await expect(page.locator('text=Settings updated')).toBeVisible();
  });
});
