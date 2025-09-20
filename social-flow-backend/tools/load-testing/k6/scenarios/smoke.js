import http from 'k6/http';
import { check, sleep } from 'k6';
import { getAuthHeaders } from '../utils/auth.js';

export const options = {
  vus: 1,
  duration: '30s'
};

export default function () {
  const headers = getAuthHeaders();

  let res = http.get(`${__ENV.K6_TARGET}/health`, { headers });
  check(res, { 'health is 200': (r) => r.status === 200 });

  res = http.get(`${__ENV.K6_TARGET}/v1/explore`, { headers });
  check(res, { 'explore is 200': (r) => r.status === 200 });

  sleep(1);
}
