import http from 'k6/http';
import { check, sleep } from 'k6';
import { getAuthHeaders } from '../utils/auth.js';
import { randomComment } from '../utils/random.js';

export const options = {
  vus: 50,
  duration: '2m',
  thresholds: {
    http_req_duration: ['p(95)<800']
  }
};

export default function () {
  const headers = getAuthHeaders();

  let res = http.get(`${__ENV.K6_TARGET}/v1/explore`, { headers });
  check(res, { 'explore 200': (r) => r.status === 200 });

  const firstItem = res.json('items[0].id');

  res = http.post(`${__ENV.K6_TARGET}/v1/items/${firstItem}/comments`, JSON.stringify({ text: randomComment() }), { headers });
  check(res, { 'comment success': (r) => r.status === 201 });

  sleep(0.8);

  res = http.post(`${__ENV.K6_TARGET}/v1/items/${firstItem}/like`, null, { headers });
  check(res, { 'like success': (r) => r.status === 200 });

  sleep(1);
}
