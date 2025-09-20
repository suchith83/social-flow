import fs from "fs";
import readline from "readline";
import { parseJSONLog } from "../parsers/jsonParser.js";
import { parseTextLog } from "../parsers/textParser.js";
import chalk from "chalk";

const file = process.argv[2];
if (!file) {
  console.error("❌ Provide log file path");
  process.exit(1);
}

const stream = fs.createReadStream(file, { encoding: "utf8", flags: "r" });
const rl = readline.createInterface({ input: stream });

rl.on("line", (line) => {
  const log = parseJSONLog(line) || parseTextLog(line);
  if (log.level === "ERROR") console.log(chalk.red(`[ERROR] ${log.message}`));
  else if (log.level === "WARN") console.log(chalk.yellow(`[WARN] ${log.message}`));
  else console.log(chalk.green(`[INFO] ${log.message}`));
});
