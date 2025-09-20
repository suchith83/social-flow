# Deployment

This folder contains examples for local development, Kubernetes manifests, a Helm chart skeleton and Terraform snippets.

Quick start (local with Docker Compose)
- Install Docker & Docker Compose.
- Copy config/.env.example to .env and update values.
- Start local infra and services:
  docker compose -f deployment/docker-compose.dev.yml up --build

Kubernetes example
- Manifests live under `deployment/k8s/`. Adjust image names, namespaces and secrets before applying:
  kubectl apply -f deployment/k8s/recommendation-deployment.yaml

Helm
- A minimal Helm chart skeleton is provided in `deployment/helm/`. Use `helm install --dry-run` to iterate.

Terraform
- `deployment/terraform/` contains provider and example resources. These are templates — inject credentials via environment or CI secrets.

Notes
- These artifacts are opinionated starting points. Replace placeholders (image names, hostnames, secrets) for your environment.
- For production, add readiness/liveness probes, resource limits, secrets management, network policies and proper CI/CD image signing.
