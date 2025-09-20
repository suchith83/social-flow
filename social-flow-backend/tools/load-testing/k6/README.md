# K6 Load Testing Suite

Advanced load-testing setup using [k6](https://k6.io).

### Features
- Modular scenarios (smoke, user-journey, soak).
- JWT authentication.
- Custom utility functions for random data & reporting.
- Dockerized execution.
- CI/CD integration with GitHub Actions.

### Usage

```bash
# Run smoke test
k6 run scenarios/smoke.js

# Run user journey
k6 run scenarios/user_journey.js

# Run soak test
k6 run scenarios/soak_test.js
