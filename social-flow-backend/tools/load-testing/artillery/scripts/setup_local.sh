#!/bin/bash
# Setup local environment for running artillery tests

set -e

echo "Installing dependencies..."
npm install

mkdir -p reports

echo "✅ Local setup complete. Run tests with: npm run smoke"
