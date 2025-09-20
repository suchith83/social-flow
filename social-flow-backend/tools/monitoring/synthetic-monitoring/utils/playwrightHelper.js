import { chromium } from "@playwright/test";

export async function withBrowser(fn) {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    return await fn(page);
  } finally {
    await browser.close();
  }
}
