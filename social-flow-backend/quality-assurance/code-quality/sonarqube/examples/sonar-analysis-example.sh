#!/usr/bin/env bash
set -euo pipefail

# Simple local sonar-scanner invocation that uses env vars and sonar-project.properties
# Usage:
#  SONAR_HOST_URL=http://localhost:9000 SONAR_TOKEN=xxx ./examples/sonar-analysis-example.sh

if [ -z "${SONAR_HOST_URL:-}" ] || [ -z "${SONAR_TOKEN:-}" ]; then
  echo "Please set SONAR_HOST_URL and SONAR_TOKEN environment variables."
  echo "e.g. SONAR_HOST_URL=http://localhost:9000 SONAR_TOKEN=xxx $0"
  exit 2
fi

# If you have sonar-scanner installed locally:
if command -v sonar-scanner >/dev/null 2>&1; then
  sonar-scanner \
    -Dsonar.host.url="${SONAR_HOST_URL}" \
    -Dsonar.login="${SONAR_TOKEN}"
  exit $?
fi

# Fallback: use official docker image
docker run --rm \
  -e SONAR_HOST_URL="${SONAR_HOST_URL}" \
  -e SONAR_TOKEN="${SONAR_TOKEN}" \
  -v "$(pwd)":/usr/src \
  -w /usr/src \
  sonarsource/sonar-scanner-cli:latest \
  -Dsonar.host.url="${SONAR_HOST_URL}" \
  -Dsonar.login="${SONAR_TOKEN}"
