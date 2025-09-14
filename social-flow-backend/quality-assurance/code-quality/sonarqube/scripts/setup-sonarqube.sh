#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   SONAR_HOST_URL=http://localhost:9000 SONAR_TOKEN=<admin token> ./scripts/setup-sonarqube.sh
#
# This script needs admin privileges on Sonar (token with 'admin' rights).
# It will:
#  - wait for Sonar to start
#  - create a token for CI (if not exists)
#  - create or update a quality gate from config/sonar-quality-gate.json
#  - optionally import quality profile via import-quality-profile.py

SONAR_HOST_URL="${SONAR_HOST_URL:-http://localhost:9000}"
SONAR_TOKEN="${SONAR_TOKEN:-}"
CI_TOKEN_NAME="${CI_TOKEN_NAME:-ci-token}"
QUAL_GATE_CONFIG="config/sonar-quality-gate.json"
PROFILE_FILE="config/sonar-quality-profile.xml"

if [ -z "$SONAR_TOKEN" ]; then
  echo "Error: SONAR_TOKEN must be set (admin token). Export SONAR_TOKEN and rerun."
  exit 2
fi

# Wait for SonarQube to be ready
echo "Waiting for SonarQube at ${SONAR_HOST_URL}..."
until curl -sS -u "${SONAR_TOKEN}:" "${SONAR_HOST_URL}/api/system/health" | grep -q '"status":"GREEN"'; do
  echo "Sonar not ready yet — sleeping 5s..."
  sleep 5
done
echo "SonarQube is ready."

# Create a CI token via API (tokens are not retrievable after creation)
echo "Creating CI token '${CI_TOKEN_NAME}'..."
create_token_response=$(curl -sS -u "${SONAR_TOKEN}:" -X POST "${SONAR_HOST_URL}/api/user_tokens/generate" --data "name=${CI_TOKEN_NAME}")
# If a token with the name already exists, Sonar returns error 409. Try to delete first as an option.
if echo "${create_token_response}" | grep -q '"error";'; then
  echo "Token creation returned error; attempting to list and delete existing token with that name..."
  # List tokens for admin user (note: listing tokens may not be supported in all Sonar versions)
fi

# Extract token if returned
TOKEN_VALUE=$(echo "${create_token_response}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || true)
if [ -n "${TOKEN_VALUE}" ]; then
  echo "Created CI token: (hidden) — you should store this securely for CI use."
  echo "Token (print once): ${TOKEN_VALUE}"
else
  echo "No token created (it may already exist). You should manually create a token in Sonar UI or reuse an existing one."
fi

# Create / update quality gate via REST API
if [ -f "$QUAL_GATE_CONFIG" ]; then
  echo "Applying quality gate from ${QUAL_GATE_CONFIG}"
  python3 scripts/create-quality-gate.py --sonar-host "${SONAR_HOST_URL}" --token "${SONAR_TOKEN}" --config "${QUAL_GATE_CONFIG}"
else
  echo "Quality gate config ${QUAL_GATE_CONFIG} not found — skipping quality gate creation."
fi

# Optionally import a quality profile (language-specific)
if [ -f "$PROFILE_FILE" ]; then
  echo "Importing quality profile from ${PROFILE_FILE}"
  python3 scripts/import-quality-profile.py --sonar-host "${SONAR_HOST_URL}" --token "${SONAR_TOKEN}" --file "${PROFILE_FILE}"
else
  echo "Quality profile ${PROFILE_FILE} not found — skipping profile import."
fi

echo "SonarQube setup complete. Remember to save tokens securely and configure CI to use them."
