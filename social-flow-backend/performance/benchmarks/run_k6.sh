#!/usr/bin/env bash
# Simple runner for the bundled k6 script.
# Usage: ./run_k6.sh --vus 50 --duration 60s

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
K6_SCRIPT="${SCRIPT_DIR}/k6/social_flow_test.js"

# parse args into env vars for k6 options
while [[ $# -gt 0 ]]; do
  case $1 in
    --vus) VUS="$2"; shift 2;;
    --duration) DURATION="$2"; shift 2;;
    --base) BASE_URL="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

export K6_VUS="${VUS:-20}"
export K6_DURATION="${DURATION:-30s}"
export BASE_URL="${BASE_URL:-http://localhost:8003}"

if ! command -v k6 >/dev/null 2>&1; then
  echo "k6 not found. Install from https://k6.io/docs/getting-started/installation/"
  exit 1
fi

echo "Running k6: VUS=$K6_VUS DURATION=$K6_DURATION BASE_URL=$BASE_URL"
k6 run "$K6_SCRIPT"
