# Generate CI/CD pipeline (GitHub Actions) YAMLs
# ================================================================
# File: ci_cd.py
# Purpose: Generate a GitHub Actions workflow that builds docker image,
#          runs tests, pushes image, and applies k8s manifests to a cluster.
# Note: This generator writes files that require secrets (DOCKERHUB, KUBE_CONFIG).
# ================================================================

from utils import write_file, logger

GITHUB_ACTION_TPL = """
name: CI/CD - Model Deploy

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest -q

  build-and-push:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v2
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to registry
        uses: docker/login-action@v2
        with:
          registry: ${{{{ env.REGISTRY }}}}
          username: ${{{{ secrets.REGISTRY_USERNAME }}}}
          password: ${{{{ secrets.REGISTRY_PASSWORD }}}}
      - name: Build and push image
        env:
          IMAGE: ${{{{ env.REGISTRY }}}}/${{{{{ github.repository }}}}}/model:${{{{ github.sha }}}}
        run: |
          docker build -t $IMAGE .
          docker push $IMAGE

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: '1.27.0'
      - name: Configure Kubeconfig
        env:
          KUBE_CONFIG_DATA: ${{{{ secrets.KUBE_CONFIG_DATA }}}}
        run: |
          echo "$KUBE_CONFIG_DATA" | base64 --decode > kubeconfig
          export KUBECONFIG=$PWD/kubeconfig
          kubectl apply -f manifests/
"""

def generate_workflow(out_path: str = ".github/workflows/ci-cd.yaml"):
    write_file(out_path, GITHUB_ACTION_TPL)
    logger.info("Generated GitHub Actions CI/CD workflow.")
    return out_path
