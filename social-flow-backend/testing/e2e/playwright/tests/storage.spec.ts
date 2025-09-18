import { test, expect } from '@playwright/test';
import testdata from '../fixtures/testdata.json';
import { uploadFileToPage, assertUploadSucceeded } from '../utils/storage-utils';
import path from 'path';

test.describe('Storage E2E tests', () => {
  test.beforeEach(async ({ page }) => {
    // ensure logged in before each storage test (UI login)
    await page.goto('/login');
    await page.locator('input[name="username"]').fill('admin');
    await page.locator('input[name="password"]').fill('password123');
    await page.locator('button[type="submit"]').click();
    await expect(page).toHaveURL(/\/dashboard/);
  });

  test('upload file to AWS S3 page', async ({ page, baseURL }) => {
    await page.goto(`${baseURL}/storage/s3`);
    await uploadFileToPage(page, 'input[type=file]#uploadInput', testdata.files.sampleTxt);
    await assertUploadSucceeded(page);
  });

  test('download file from Azure Blob', async ({ page, baseURL }) => {
    await page.goto(`${baseURL}/storage/azure`);
    // assume a button/link triggers download; click it and assert the download artifact
    const [ download ] = await Promise.all([
      page.waitForEvent('download'),
      page.locator('text=Download test-azure.txt').click()
    ]);
    const suggested = await download.suggestedFilename();
    expect(suggested).toContain('test-azure.txt');
    const pathSaved = await download.path();
    expect(pathSaved).not.toBeNull();
  });

  test('upload image to GCS page', async ({ page, baseURL }) => {
    await page.goto(`${baseURL}/storage/gcs`);
    await uploadFileToPage(page, 'input[type=file]#uploadInput', testdata.files.sampleImage);
    await assertUploadSucceeded(page);
  });
});
