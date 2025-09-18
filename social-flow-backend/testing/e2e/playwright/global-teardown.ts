/**
 * Global teardown runs after all tests.
 * - Clean up seeded fixtures or uploaded test artifacts if possible.
 */

import { FullConfig } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

const OUT_FILE = path.join(__dirname, 'global-setup-state.json');

export default async function globalTeardown(config: FullConfig) {
  console.log('Playwright global teardown starting...');
  if (!fs.existsSync(OUT_FILE)) {
    console.log('No global state file found; skipping teardown.');
    return;
  }
  const state = JSON.parse(fs.readFileSync(OUT_FILE, 'utf-8'));
  const apiBase = process.env.API_BASE_URL || 'http://localhost:3000/api';

  // Example: attempt to remove seeded users (best-effort)
  if (Array.isArray(state.seededUsers)) {
    for (const u of state.seededUsers) {
      try {
        await axios.post(`${apiBase}/testing/delete-user`, { username: u.username }, { timeout: 5000 });
      } catch (err) {
        console.log(`Could not delete user ${u.username}: ${err?.message || err}`);
      }
    }
  }

  // remove file
  try { fs.unlinkSync(OUT_FILE); } catch (e) {}
  console.log('Global teardown complete.');
}
