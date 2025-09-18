/**
 * API helper utilities for Playwright tests.
 * Wraps axios and exposes commonly used test actions.
 */

import axios, { AxiosInstance } from 'axios';
import dotenv from 'dotenv';
dotenv.config();

const API_BASE = process.env.API_BASE_URL || 'http://localhost:3000/api';

export const api: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 10_000,
  validateStatus: (s) => s < 500
});

/**
 * Authenticate a user via API and return a session cookie or token (best-effort).
 * Tests can use this to set authenticated state quickly.
 */
export async function apiLogin(username: string, password: string) {
  const resp = await api.post('/auth/login', { username, password }).catch((e) => e.response || e);
  if (resp?.status === 200 && resp.data?.token) {
    return { token: resp.data.token, cookie: resp.headers['set-cookie'] };
  }
  return resp.data ?? null;
}
