import { chromium } from "@playwright/test";

export default async function run() {
  let browser;
  const start = Date.now();
  try {
    browser = await chromium.launch();
    const page = await browser.newPage();
    await page.goto(`${process.env.TARGET_URL}/login`);

    await page.fill("input[name='username']", "testuser");
    await page.fill("input[name='password']", "testpass");
    await page.click("button[type='submit']");

    await page.waitForURL("**/dashboard", { timeout: 5000 });

    const latency = Date.now() - start;
    await browser.close();
    return {
      name: "Login Flow",
      status: "PASS",
      latency
    };
  } catch (err) {
    if (browser) await browser.close();
    return {
      name: "Login Flow",
      status: "FAIL",
      error: err.message
    };
  }
}
