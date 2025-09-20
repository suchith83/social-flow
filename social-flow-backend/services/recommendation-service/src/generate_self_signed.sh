#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)/.certs"
mkdir -p "$OUT_DIR"
CRT="$OUT_DIR/dev.crt"
KEY="$OUT_DIR/dev.key"

echo "Generating self-signed cert in $OUT_DIR"
if [ -f "$CRT" ] && [ -f "$KEY" ]; then
  echo "Cert/key already exist; skipping. Remove files to regenerate."
  echo "$CRT"
  echo "$KEY"
  exit 0
fi

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "$KEY" -out "$CRT" -subj "/C=US/ST=Dev/L=Dev/O=SocialFlow/OU=Dev/CN=localhost"

echo "Generated:"
ls -l "$CRT" "$KEY"
