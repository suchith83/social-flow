/**
 * run_experiment.js
 * Dynamically executes chaos experiments (system-level or Kubernetes-based).
 */

import fs from "fs";
import yaml from "js-yaml";
import { spawn } from "child_process";
import { v4 as uuidv4 } from "uuid";

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("❌ Please provide experiment file path");
  process.exit(1);
}
const file = args[0];

function runSystemExperiment(config) {
  console.log(`🚀 Running system experiment: ${config.name}`);
  const child = spawn("bash", ["-c", config.command], { stdio: "inherit" });
  child.on("exit", (code) => {
    console.log(`✅ System experiment finished with code ${code}`);
  });
}

function runKubernetesExperiment(yamlConfig) {
  console.log(`🚀 Applying Kubernetes chaos experiment...`);
  const tmpFile = `/tmp/chaos-${uuidv4()}.yml`;
  fs.writeFileSync(tmpFile, yamlConfig);

  const child = spawn("kubectl", ["apply", "-f", tmpFile], { stdio: "inherit" });
  child.on("exit", (code) => {
    console.log(`✅ Kubernetes chaos experiment applied (exit code ${code})`);
  });
}

if (file.endsWith(".json")) {
  const config = JSON.parse(fs.readFileSync(file, "utf8"));
  runSystemExperiment(config);
} else if (file.endsWith(".yml") || file.endsWith(".yaml")) {
  const yamlConfig = fs.readFileSync(file, "utf8");
  runKubernetesExperiment(yamlConfig);
} else {
  console.error("❌ Unsupported file type (use .json or .yml)");
}
