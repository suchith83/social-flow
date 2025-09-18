#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sbt "gatling:testOnly simulations.BaselineSimulation" \
  -DbaseUrl="${BASE_URL:-http://localhost:3000/api}" \
  -Dusers="${USERS:-100}" -DrampUpSeconds="${RAMP_UP_SECONDS:-30}" -DdurationSeconds="${DURATION_SECONDS:-60}"
