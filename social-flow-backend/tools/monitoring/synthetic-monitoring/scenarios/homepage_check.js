import { chromium } from "@playwright/test";

export default async function run() {
  const start = Date.now();
  let browser;
  try {
    browser = await chromium.launch();
    const page = await browser.newPage();
    await page.goto(process.env.TARGET_URL, { waitUntil: "domcontentloaded" });
    const title = await page.title();
    const latency = Date.now() - start;
    await browser.close();
    return {
      name: "Homepage Load",
      status: title ? "PASS" : "FAIL",
      latency,
      title
    };
  } catch (err) {
    if (browser) await browser.close();
    return {
      name: "Homepage Load",
      status: "FAIL",
      error: err.message
    };
  }
}
