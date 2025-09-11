/**
 * Alerts helpers: send notifications to Slack & PagerDuty.
 * - Wraps outgoing HTTP calls and exposes triggerAlert() for programmatic alerts
 * - Also provides a convenience wrapper for capturing unhandled exceptions and unhandled rejections
 *
 * Be careful: do not spam alerting endpoints; ideally integrate with deduping/aggregation.
 */

const config = require('./config');
const fetch = require('node-fetch');
const axios = require('axios');
const logger = require('./logger').base;

/**
 * Send a message payload to Slack incoming webhook.
 */
async function sendSlack(text, options = {}) {
  if (!config.ALERTS_ENABLED) return;
  if (!config.SLACK_WEBHOOK) {
    logger.warn('Slack webhook not configured');
    return;
  }
  const body = {
    text,
    channel: options.channel || config.ALERTS_DEFAULT_CHANNEL,
    attachments: options.attachments || [],
    blocks: options.blocks || undefined,
  };
  try {
    const r = await fetch(config.SLACK_WEBHOOK, {
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'Content-Type': 'application/json' },
      timeout: 5000,
    });
    if (!r.ok) logger.warn({ status: r.status }, 'Slack webhook returned non-OK');
  } catch (err) {
    logger.warn({ err }, 'Failed to send Slack alert');
  }
}

/**
 * Trigger an incident in PagerDuty using Events v2 API (simple integration)
 * Requires PAGERDUTY_INTEGRATION_KEY
 */
async function sendPagerDuty(summary, severity = 'error', source = config.SERVICE_NAME) {
  if (!config.ALERTS_ENABLED) return;
  if (!config.PAGERDUTY_INTEGRATION_KEY) {
    logger.warn('PagerDuty integration key not configured');
    return;
  }
  const event = {
    routing_key: config.PAGERDUTY_INTEGRATION_KEY,
    event_action: 'trigger',
    payload: {
      summary,
      severity,
      source,
      timestamp: new Date().toISOString(),
    },
  };
  try {
    const r = await fetch('https://events.pagerduty.com/v2/enqueue', {
      method: 'POST',
      body: JSON.stringify(event),
      headers: { 'Content-Type': 'application/json' },
      timeout: 5000,
    });
    if (!r.ok) logger.warn({ status: r.status }, 'PagerDuty returned non-OK');
  } catch (err) {
    logger.warn({ err }, 'Failed to send PagerDuty event');
  }
}

/**
 * triggerAlert: convenience function to call both Slack & PagerDuty (with backoff and dedupe considerations)
 */
async function triggerAlert({ title, message, severity = 'error', extra = {} } = {}) {
  const text = `*${title || 'Alert'}* \n${message}\n\`\`\`${JSON.stringify(extra)}\`\`\``;
  try {
    await Promise.allSettled([sendSlack(text, extra), sendPagerDuty(`${title} - ${message}`, severity)]);
  } catch (err) {
    logger.warn({ err }, 'Alert delivery failed');
  }
}

/**
 * Hook global exceptions to send alerts (best-effort)
 */
function hookGlobalHandlers() {
  process.on('uncaughtException', async (err) => {
    logger.error({ err }, 'uncaughtException');
    try {
      await triggerAlert({ title: 'uncaughtException', message: err.stack || String(err), severity: 'critical' });
    } catch (e) {}
    // recommended to exit in production after alert
    if (config.ENV === 'production') {
      setTimeout(() => process.exit(1), 1000);
    }
  });

  process.on('unhandledRejection', async (reason) => {
    logger.error({ reason }, 'unhandledRejection');
    try {
      await triggerAlert({ title: 'unhandledRejection', message: String(reason), severity: 'critical' });
    } catch (e) {}
  });
}

module.exports = {
  sendSlack,
  sendPagerDuty,
  triggerAlert,
  hookGlobalHandlers,
};
