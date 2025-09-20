import fs from "fs";
import glob from "glob";

function summarize(files) {
  return files.map(file => {
    const data = JSON.parse(fs.readFileSync(file, "utf8"));
    return {
      file,
      total: data.metrics.total,
      errors: data.metrics.errors,
      errorRate: data.metrics.errorRate
    };
  });
}

const files = glob.sync(process.argv[2] || "reports/*.json");
if (files.length === 0) {
  console.error("⚠️ No reports found.");
  process.exit(0);
}

const summary = summarize(files);
console.table(summary);

fs.writeFileSync("reports/summary.json", JSON.stringify(summary, null, 2));
console.log("✅ Written to reports/summary.json");
