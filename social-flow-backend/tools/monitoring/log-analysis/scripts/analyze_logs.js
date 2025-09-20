import fs from "fs";
import { parseJSONLog } from "../parsers/jsonParser.js";
import { parseTextLog } from "../parsers/textParser.js";
import { aggregateLogs } from "../analyzers/aggregator.js";
import { detectAnomalies } from "../analyzers/anomalyDetector.js";
import { computeMetrics } from "../analyzers/metrics.js";

const file = process.argv[2];
if (!file) {
  console.error("❌ Provide a log file path");
  process.exit(1);
}

const lines = fs.readFileSync(file, "utf-8").split("\n").filter(Boolean);

const parsed = lines.map(line => parseJSONLog(line) || parseTextLog(line));
const metrics = aggregateLogs(parsed);
const anomalies = detectAnomalies(parsed);
const timeline = computeMetrics(parsed);

console.log("📊 Metrics:", metrics);
console.log("⚠️ Anomalies:", anomalies);
console.log("⏱ Timeline:", timeline);

// Save report
fs.mkdirSync("reports", { recursive: true });
fs.writeFileSync(
  `reports/report-${Date.now()}.json`,
  JSON.stringify({ metrics, anomalies, timeline }, null, 2)
);
console.log("✅ Report saved to reports/");
