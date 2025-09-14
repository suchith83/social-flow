# scripts/setup/README.md

# Social Flow — Setup package

This directory contains idempotent, safe helpers to bootstrap developer machines and hosts for Social Flow.

## Purpose
- Provision OS packages & language runtimes
- Create service users
- Configure Docker & Kubernetes contexts
- Create TLS certs for development
- Bootstrap secrets placeholders (do NOT store sensitive values in repo)
- Render .env files for systemd services

## Usage
1. Create a `setup.yaml` describing the desired state (see ConfigLoader docstring in code).
2. Run:
