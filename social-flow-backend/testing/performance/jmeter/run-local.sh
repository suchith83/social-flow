#!/usr/bin/env bash
set -euo pipefail
# Simple runner that runs JMeter in Docker in non-GUI mode and generates an HTML report.
# Usage: ./run-local.sh test-plans/master-test-plan.jmx

PLAN="${1:-test-plans/master-test-plan.jmx}"
IMAGE="${JMETER_IMAGE:-perf-jmeter:latest}"
OUT_DIR="./reports/$(basename "${PLAN%.*}")-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT_DIR}"
echo "Running plan: ${PLAN}"
echo "Output dir: ${OUT_DIR}"

docker run --rm -v "$PWD":/work -w /work \
  -e JMETER_OPTS="-Xms1g -Xmx3g" \
  "${IMAGE}" \
  /bin/bash -c "jmeter -n -t ${PLAN} -l ${OUT_DIR}/results.jtl -j ${OUT_DIR}/jmeter.log && jmeter -g ${OUT_DIR}/results.jtl -o ${OUT_DIR}/html_report || true"

echo "Report: ${OUT_DIR}/html_report/index.html"
