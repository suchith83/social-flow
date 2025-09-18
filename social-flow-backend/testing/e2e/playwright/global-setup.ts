/**
 * Global setup runs once before all tests.
 * - Can seed fixtures, create test users, or sign tokens for fast auth.
 * - Should return a JSON-serializable state to be consumed by global-teardown or tests.
 */

import { FullConfig } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

const OUT_FILE = path.join(__dirname, 'global-setup-state.json');

export default async function globalSetup(config: FullConfig) {
  console.log('Playwright global setup starting...');

  // Example: create test users via internal API (if available)
  const apiBase = process.env.API_BASE_URL || 'http://localhost:3000/api';

  const users = [
    { username: 'admin', password: 'password123', role: 'admin' },
    { username: 'user', password: 'userpass', role: 'user' }
  ];

  const created: any[] = [];
  for (const u of users) {
    try {
      // best-effort: create user if API exists; ignore failures
      const res = await axios.post(`${apiBase}/testing/create-user`, u, { timeout: 5000 });
      created.push(res.data);
    } catch (err) {
      console.log(`Could not create user ${u.username}: ${err?.message || err}`);
    }
  }

  // Persist some metadata for tests
  const state = {
    seededUsers: created,
    timestamp: new Date().toISOString()
  };

  fs.writeFileSync(OUT_FILE, JSON.stringify(state), 'utf-8');
  console.log('Global setup complete.');
  return;
}
