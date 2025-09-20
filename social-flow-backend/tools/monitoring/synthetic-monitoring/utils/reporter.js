import fs from "fs";

export function saveReport(result) {
  fs.mkdirSync("reports", { recursive: true });
  const file = `reports/result-${Date.now()}.json`;
  fs.writeFileSync(file, JSON.stringify(result, null, 2));
  console.log(`✅ Report saved to ${file}`);
}
