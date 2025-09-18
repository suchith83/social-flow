#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# run series of scenarios sequentially (useful in CI)
./scripts/run-auth.sh
./scripts/run-baseline.sh
sbt "gatling:testOnly simulations.StorageSimulation"
sbt "gatling:testOnly simulations.FailoverSimulation"
