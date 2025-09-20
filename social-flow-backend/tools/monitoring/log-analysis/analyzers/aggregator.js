/**
 * Aggregates logs into metrics
 */
export function aggregateLogs(logs) {
  const metrics = { total: logs.length, errors: 0, warnings: 0, byService: {} };

  logs.forEach(log => {
    if (log.level === "ERROR") metrics.errors++;
    if (log.level === "WARN") metrics.warnings++;
    if (!metrics.byService[log.service]) metrics.byService[log.service] = 0;
    metrics.byService[log.service]++;
  });

  metrics.errorRate = metrics.total > 0 ? metrics.errors / metrics.total : 0;
  return metrics;
}
