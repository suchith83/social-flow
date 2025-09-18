import { test, expect } from '@playwright/test';
import users from '../fixtures/users.json';
import { apiLogin } from '../utils/api';

test.describe('Authentication tests', () => {
  test('login via UI with valid credentials', async ({ page, baseURL }) => {
    const user = users[0];
    await page.goto(`${baseURL}/login`);
    await page.locator('input[name="username"]').fill(user.username);
    await page.locator('input[name="password"]').fill(user.password);
    await page.locator('button[type="submit"]').click();
    await expect(page).toHaveURL(/\/dashboard/);
    await expect(page.locator('text=Welcome')).toBeVisible();
  });

  test('rejects invalid login', async ({ page, baseURL }) => {
    await page.goto(`${baseURL}/login`);
    await page.locator('input[name="username"]').fill('no-user');
    await page.locator('input[name="password"]').fill('badpass');
    await page.locator('button[type="submit"]').click();
    await expect(page.locator('text=Invalid credentials')).toBeVisible();
  });

  test('login via API and set cookie for faster auth', async ({ page, context, baseURL }) => {
    const res = await apiLogin(users[0].username, users[0].password);
    test.skip(!res || !(res.cookie || res.token), 'API login not available in this env');
    if (res.cookie) {
      // set returned cookie(s)
      for (const c of res.cookie) {
        await context.addCookies([{ name: c.split('=')[0], value: c.split('=')[1].split(';')[0], domain: new URL(baseURL || 'http://localhost').hostname, path: '/' }]);
      }
    } else if (res.token) {
      await context.addCookies([{ name: 'auth_token', value: res.token, domain: new URL(baseURL || 'http://localhost').hostname, path: '/' }]);
    }
    await page.goto(`${baseURL}/dashboard`);
    await expect(page.locator('text=Welcome')).toBeVisible();
  });
});
