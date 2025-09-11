/**
 * System and process collectors beyond prom-client default
 * - Event loop lag (via perf_hooks)
 * - GC metrics (if available)
 * - Custom CPU/memory/disk sampling using os module
 *
 * These collectors push metrics into the metrics module's registry.
 */

const os = require('os');
const { performance, monitorEventLoopDelay } = require('perf_hooks');
const metrics = require('./metrics');
const config = require('./config');
const logger = require('./logger').base;

let eventLoopHistogram;
let lastCpuInfo = os.cpus();

function initEventLoopCollector() {
  try {
    const h = monitorEventLoopDelay({ resolution: 10 });
    h.enable();
    eventLoopHistogram = metrics.histogram('node_event_loop_delay_seconds', 'Event loop delay in seconds', [], [0.001, 0.005, 0.01, 0.05, 0.1]);
    // schedule periodic sampling
    setInterval(() => {
      const delayMs = h.mean / 1e6; // nanoseconds to ms
      eventLoopHistogram.observe({ service: config.SERVICE_NAME, env: config.ENV }, delayMs / 1000);
      // reset summary counters (monitorEventLoopDelay auto-maintains)
    }, 5000).unref();
    logger.debug('Event loop collector initialized');
  } catch (e) {
    logger.warn('Event loop collector not available', e);
  }
}

function initCpuMemoryCollector() {
  const cpuGauge = metrics.gauge('node_process_cpu_user_seconds_total', 'Process CPU user seconds total', []);
  const memGauge = metrics.gauge('node_process_resident_memory_bytes', 'Resident memory size in bytes', []);
  setInterval(() => {
    try {
      // memory
      const mem = process.memoryUsage();
      memGauge.set({ service: config.SERVICE_NAME, env: config.ENV, metric: 'rss' }, mem.rss);
      memGauge.set({ service: config.SERVICE_NAME, env: config.ENV, metric: 'heapTotal' }, mem.heapTotal);
      memGauge.set({ service: config.SERVICE_NAME, env: config.ENV, metric: 'heapUsed' }, mem.heapUsed);
      memGauge.set({ service: config.SERVICE_NAME, env: config.ENV, metric: 'external' }, mem.external);

      // CPU sampling (user+system delta)
      const cpus = os.cpus();
      let totalDiff = 0;
      let userDiff = 0;
      for (let i = 0; i < cpus.length; i++) {
        const prev = lastCpuInfo[i].times;
        const cur = cpus[i].times;
        const prevTotal = Object.values(prev).reduce((a, b) => a + b, 0);
        const curTotal = Object.values(cur).reduce((a, b) => a + b, 0);
        const total = curTotal - prevTotal;
        const user = cur.user - prev.user;
        totalDiff += total;
        userDiff += user;
      }
      const cpuUsage = totalDiff > 0 ? userDiff / totalDiff : 0;
      cpuGauge.set({ service: config.SERVICE_NAME, env: config.ENV, metric: 'user_fraction' }, cpuUsage);
      lastCpuInfo = cpus;
    } catch (err) {
      logger.warn({ err }, 'cpu/mem collector error');
    }
  }, 5000).unref();
}

function initGcCollector() {
  // Node exposes GC metrics via perf_hooks only when --expose-gc is enabled and Node's native hooks
  try {
    const gcHistogram = metrics.histogram('node_gc_duration_seconds', 'Duration of GC in seconds', [], [0.001, 0.01, 0.1, 1, 2, 5]);
    const { performance, PerformanceObserver } = require('perf_hooks');
    const obs = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      for (const entry of entries) {
        if (entry.entryType === 'gc') {
          gcHistogram.observe({ service: config.SERVICE_NAME, env: config.ENV, kind: entry.kind }, entry.duration / 1000);
        }
      }
    });
    obs.observe({ entryTypes: ['gc'], buffered: false });
    logger.debug('GC collector initialized');
  } catch (e) {
    logger.warn('GC collector unavailable', e);
  }
}

/**
 * Initialize all configured collectors
 */
function initAll() {
  if (config.COLLECT_PROCESS_METRICS) {
    initCpuMemoryCollector();
    initEventLoopCollector();
  }
  if (config.COLLECT_GC_METRICS) {
    initGcCollector();
  }
}

module.exports = {
  initAll,
  initEventLoopCollector,
  initCpuMemoryCollector,
  initGcCollector,
};
