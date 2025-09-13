# Create Helm chart skeleton & values
# ================================================================
# File: helm_helper.py
# Purpose: Generate a minimal Helm chart skeleton for the model service.
# Produces: Chart.yaml, values.yaml, templates/*
# ================================================================

from typing import Dict
from utils import write_file, render_template, logger

CHART_YAML_TPL = """
apiVersion: v2
name: {name}
description: A Helm chart for deploying the {name} model
type: application
version: {chart_version}
appVersion: "{app_version}"
"""

VALUES_YAML_TPL = """
image:
  repository: {image_repo}
  tag: {image_tag}
  pullPolicy: IfNotPresent

service:
  type: {service_type}
  port: {port}

resources:
  limits:
    cpu: {cpu_limit}
    memory: {mem_limit}
  requests:
    cpu: {cpu_request}
    memory: {mem_request}

replicaCount: {replica_count}
ingress:
  enabled: {ingress_enabled}
  host: {ingress_host}
"""

DEPLOYMENT_TEMPLATE = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "{name}.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ include "{name}.name" . }}
  template:
    metadata:
      labels:
        app: {{ include "{name}.name" . }}
    spec:
      containers:
        - name: {{ include "{name}.name" . }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          ports:
            - containerPort: {{ .Values.service.port }}
"""

def generate_chart(name: str, out_dir: str = "charts", values: Dict = None):
    chart_dir = f"{out_dir}/{name}"
    templates_dir = f"{chart_dir}/templates"
    # Chart.yaml
    write_file(f"{chart_dir}/Chart.yaml", CHART_YAML_TPL.format(name=name, chart_version="0.1.0", app_version="1.0.0"))
    # values.yaml
    vals = values or {
        "image_repo": "myregistry/models",
        "image_tag": "latest",
        "service_type": "ClusterIP",
        "port": 8080,
        "cpu_limit": "500m",
        "mem_limit": "512Mi",
        "cpu_request": "200m",
        "mem_request": "256Mi",
        "replica_count": 2,
        "ingress_enabled": False,
        "ingress_host": "model.example.com",
    }
    write_file(f"{chart_dir}/values.yaml", VALUES_YAML_TPL.format(**vals))
    # deployment template
    write_file(f"{templates_dir}/deployment.yaml", DEPLOYMENT_TEMPLATE.format(name=name))
    logger.info(f"Generated Helm chart skeleton at {chart_dir}")
    return chart_dir
