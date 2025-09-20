import path from "path";
import { saveReport } from "../utils/reporter.js";

const file = process.argv[2];
if (!file) {
  console.error("❌ Please provide scenario file");
  process.exit(1);
}

async function main() {
  const scenarioPath = path.resolve(file);
  const scenario = (await import(scenarioPath)).default;
  const result = await scenario();
  console.log("📊 Result:", result);
  saveReport(result);

  if (result.status === "FAIL") process.exit(1);
}
main();
