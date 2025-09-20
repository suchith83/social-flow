import http from 'k6/http';
import { sleep } from 'k6';
import { getAuthHeaders } from '../utils/auth.js';

export const options = {
  vus: 100,
  duration: '1h',
  thresholds: {
    http_req_failed: ['rate<0.01']
  }
};

export default function () {
  const headers = getAuthHeaders();

  http.get(`${__ENV.K6_TARGET}/health`, { headers });

  http.get(`${__ENV.K6_TARGET}/v1/explore?page=${Math.floor(Math.random()*50)+1}&limit=20`, { headers });

  sleep(0.2);
}
