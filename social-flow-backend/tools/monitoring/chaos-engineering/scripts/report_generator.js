/**
 * report_generator.js
 * Aggregates chaos experiment logs into a summary.
 */

import fs from "fs";
import glob from "glob";

function summarize(logs) {
  return logs.map((file) => ({
    file,
    size: fs.statSync(file).size,
    createdAt: fs.statSync(file).ctime
  }));
}

const reports = glob.sync(process.argv[2] || "reports/*.json");
if (reports.length === 0) {
  console.error("⚠️ No reports found.");
  process.exit(0);
}

const summary = summarize(reports);
console.log("📊 Chaos Experiment Reports Summary:");
console.table(summary);

fs.writeFileSync("./reports/summary.json", JSON.stringify(summary, null, 2));
console.log("✅ Written to ./reports/summary.json");
