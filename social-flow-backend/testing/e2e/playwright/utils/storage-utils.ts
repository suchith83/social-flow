/**
 * Helpers for interacting with storage pages/components in the UI.
 * Used by storage.spec.ts and multi-cloud.spec.ts.
 *
 * These helpers are intentionally UI-agnostic: they accept selectors
 * and rely on the app's DOM structure. Adjust selectors to your app.
 */

import { Page } from '@playwright/test';
import path from 'path';

export async function uploadFileToPage(page: Page, inputSelector: string, fixturePath: string) {
  // Convert fixture path relative to project root
  const absolute = path.join(process.cwd(), 'testing', 'e2e', 'playwright', fixturePath);
  await page.setInputFiles(inputSelector, absolute);
}

export async function assertUploadSucceeded(page: Page, successSelector = 'text=Upload successful') {
  await page.waitForSelector(successSelector, { timeout: 8000 });
}
