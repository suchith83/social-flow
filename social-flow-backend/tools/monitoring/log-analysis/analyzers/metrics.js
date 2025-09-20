/**
 * Extracts time-series metrics from logs.
 */
export function computeMetrics(logs) {
  const timeline = {};
  logs.forEach(log => {
    const minute = log.timestamp.substring(0, 16);
    if (!timeline[minute]) timeline[minute] = { count: 0, errors: 0 };
    timeline[minute].count++;
    if (log.level === "ERROR") timeline[minute].errors++;
  });
  return timeline;
}
