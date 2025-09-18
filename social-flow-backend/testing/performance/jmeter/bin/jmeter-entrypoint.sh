#!/usr/bin/env bash
set -euo pipefail

# Entrypoint for Docker container. Pass-through to jmeter command by default.
# Also provides convenience environment variable handling.

# Source user.properties if present
if [ -f /work/user.properties ]; then
  echo "Loading user.properties"
  export $(sed -n 's/^\([^#].*\)=.*/\1/p' user.properties | xargs)
fi

exec "$@"
