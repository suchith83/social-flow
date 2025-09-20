#!/usr/bin/env bash
# Lightweight secret scanner (grep-based). Use dedicated tools (git-secrets, truffleHog) in production.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Running lightweight secret scan in ${ROOT_DIR}"

# Patterns to check (simple heuristics)
PATTERNS=(
  "AWS_SECRET_ACCESS_KEY"
  "AWS_ACCESS_KEY_ID"
  "AKIA[0-9A-Z]{16}"
  "BEGIN RSA PRIVATE KEY"
  "BEGIN PRIVATE KEY"
  "-----BEGIN OPENSSH PRIVATE KEY-----"
  "api_key"
  "secret_key"
  "stripe_key"
  "sk_live_"
  "sk_test_"
  "password\s*="
)

EXIT_CODE=0
for p in "${PATTERNS[@]}"; do
  echo "Searching for pattern: $p"
  # search excluding common safe paths
  matches=$(grep -RIn --exclude-dir={.git,node_modules,.venv,build,dist,data,docs} -E "$p" "$ROOT_DIR" || true)
  if [ -n "$matches" ]; then
    echo "Potential secrets found for pattern [$p]:"
    echo "$matches"
    EXIT_CODE=2
  fi
done

if [ $EXIT_CODE -ne 0 ]; then
  echo "Secret scan detected potential secrets. Review output and remove/rotate any secrets found."
else
  echo "No obvious secrets detected by lightweight scan."
fi

exit $EXIT_CODE
