
// ---

// ## `artillery.config.js`

// ```js
/**
 * Central Artillery JS config loader.
 * Loads environment variables and exports helper methods used by YAML scenarios.
 *
 * We export an object that scenarios can require as a JS hook via "processor" key.
 * This allows sharing variables like baseUrl, tokens, timeouts, etc.
 */

const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '.env') });

const env = {
  baseUrl: process.env.BASE_URL || 'http://localhost:3000/api',
  username: process.env.TEST_USER_USERNAME || 'admin',
  password: process.env.TEST_USER_PASSWORD || 'password123',
  storageBucket: process.env.STORAGE_BUCKET || 'test-bucket',
  storageSampleFile: process.env.STORAGE_SAMPLE_FILE_PATH || path.resolve(__dirname, 'fixtures', 'sample.txt'),
  vus: parseInt(process.env.VUS_DEFAULT || '50', 10),
  duration: parseInt(process.env.DURATION_DEFAULT || '60', 10),
  rampUp: parseInt(process.env.RAMP_UP_SECONDS || '30', 10),
  reportDir: process.env.REPORT_DIR || path.resolve(__dirname, 'reports'),
  httpProxy: process.env.HTTP_PROXY || undefined
};

module.exports = {
  env,
  // Small helper to ensure report dir exists
  ensureReportDir() {
    if (!fs.existsSync(env.reportDir)) {
      fs.mkdirSync(env.reportDir, { recursive: true });
    }
  }
};
