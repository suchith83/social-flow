/**
 * result-processor.js
 *
 * Post-processing helpers. Artillery writes JSON to reports; this module can:
 *  - normalize events (emit counters)
 *  - run custom assertions at runtime (e.g., percent failed > threshold)
 *  - aggregate metrics and optionally push to an external monitoring ingest
 *
 * This file demonstrates how to subscribe to Artillery events when run with `processor`.
 */

module.exports = {
  init: function (script, util, env, next) {
    // subscribe to counters
    this.counters = {};
    next();
  },

  /**
   * Example: a hook that is called after each response (via beforeRequest/afterResponse)
   * You can attach functions to record custom counters via events.emit('counter', key, 1).
   */
  afterResponse: function (requestParams, response, context, ee, next) {
    // track HTTP 5xx codes
    if (response.statusCode && response.statusCode >= 500) {
      ee.emit('counter', 'http_5xx');
    }
    next();
  },

  /**
   * Optional: called when the script finishes — can be used to compute derived metrics and fail the run by throwing.
   * But Artillery typically writes reports; CI should assert thresholds using report JSON.
   */
  done: function (report, next) {
    // you can inspect report.summary for custom checks
    const total5xx = (report && report.counters && report.counters.http_5xx) || 0;
    if (total5xx > 0) {
      console.warn('Run had 5xx responses:', total5xx);
    }
    next();
  }
};
