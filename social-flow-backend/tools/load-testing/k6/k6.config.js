export const options = {
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests < 500ms
    http_req_failed: ['rate<0.01']    // < 1% errors allowed
  }
};
