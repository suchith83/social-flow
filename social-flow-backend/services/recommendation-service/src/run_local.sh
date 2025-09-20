#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COMPOSE_FILE="$ROOT/deployment/docker-compose.dev.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Compose file not found at $COMPOSE_FILE"
  exit 1
fi

echo "Starting local development stack using $COMPOSE_FILE"
docker compose -f "$COMPOSE_FILE" up --build
