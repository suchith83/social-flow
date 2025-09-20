import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { randomIntBetween } from 'k6';

export let options = {
  vus: __ENV.K6_VUS ? parseInt(__ENV.K6_VUS) : 20,
  duration: __ENV.K6_DURATION || '30s',
  thresholds: {
    http_req_duration: ['p(95)<500'],
    'http_req_failed': ['rate<0.01'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8003';
const ACTIONS = ['view', 'like', 'share', 'dislike'];

function randomUser() {
  return `test_user_${Math.floor(Math.random() * 1000)}`;
}

export default function () {
  group('health', function () {
    let r = http.get(`${BASE_URL}/health`);
    check(r, { 'health ok': (res) => res.status === 200 });
  });

  group('recommendations', function () {
    const user = randomUser();
    let r = http.get(`${BASE_URL}/recommendations/${user}?limit=5`);
    check(r, {
      'recs 200': (res) => res.status === 200,
      'recs body has recommendations': (res) => {
        try {
          const j = res.json();
          return Array.isArray(j.recommendations);
        } catch (e) { return false; }
      },
    });
    sleep(0.1);
  });

  group('trending', function () {
    let r = http.get(`${BASE_URL}/trending?limit=5`);
    check(r, {
      'trending 200': (res) => res.status === 200,
    });
    sleep(0.1);
  });

  group('feedback', function () {
    const payload = {
      user_id: randomUser(),
      item_id: `video_${randomIntBetween(1,100)}`,
      action: ACTIONS[randomIntBetween(0, ACTIONS.length-1)],
      timestamp: Math.floor(Date.now() / 1000),
    };
    const headers = { 'Content-Type': 'application/json' };
    let r = http.post(`${BASE_URL}/feedback`, JSON.stringify(payload), { headers });
    check(r, {
      'feedback recorded': (res) => res.status === 200,
    });
    sleep(0.5);
  });

  // small pause between iterations
  sleep(1);
}
