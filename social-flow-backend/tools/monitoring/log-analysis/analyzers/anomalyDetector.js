/**
 * Rule-based anomaly detection
 * Could be extended with ML (Isolation Forest, clustering).
 */
export function detectAnomalies(logs, threshold = 0.05) {
  const anomalies = [];
  const errorRate = logs.filter(l => l.level === "ERROR").length / logs.length;

  if (errorRate > threshold) {
    anomalies.push({
      type: "HighErrorRate",
      details: `Error rate ${errorRate} exceeds threshold ${threshold}`
    });
  }

  return anomalies;
}
