# Generate k8s manifests (Deployment, Service, HPA, Ingress)
# ================================================================
# File: k8s_generator.py
# Purpose: Generate Kubernetes manifests for model deployment.
# Produces: Deployment, Service, HorizontalPodAutoscaler, Ingress (optional)
# ================================================================

from typing import Dict, Any
from utils import render_template, write_file, logger
import yaml


DEPLOYMENT_TPL = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
    model: "{model_name}"
    version: "{version}"
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
        model: "{model_name}"
        version: "{version}"
    spec:
      containers:
        - name: {name}-container
          image: "{image}"
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: {port}
          readinessProbe:
            httpGet:
              path: {readiness_path}
              port: {port}
            initialDelaySeconds: 5
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: {liveness_path}
              port: {port}
            initialDelaySeconds: 15
            periodSeconds: 10
          resources:
            limits:
              cpu: "{cpu_limit}"
              memory: "{mem_limit}"
            requests:
              cpu: "{cpu_request}"
              memory: "{mem_request}"
          env:
{env_block}
"""

SERVICE_TPL = """
apiVersion: v1
kind: Service
metadata:
  name: {name}-svc
  labels:
    app: {name}
spec:
  selector:
    app: {name}
  ports:
    - protocol: TCP
      port: {service_port}
      targetPort: {port}
  type: {service_type}
"""

HPA_TPL = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {name}
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {cpu_utilization}
"""

INGRESS_TPL = """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
    - host: "{host}"
      http:
        paths:
          - path: {path}
            pathType: Prefix
            backend:
              service:
                name: {name}-svc
                port:
                  number: {service_port}
"""

def _env_block(env: Dict[str, str]) -> str:
    lines = []
    for k, v in env.items():
        lines.append("            - name: {k}\n              value: \"{v}\"".format(k=k, v=v))
    return "\n".join(lines) if lines else "            # no env vars"

class K8sGenerator:
    """
    Generate k8s YAMLs for a model service.
    """

    def __init__(self, out_dir: str = "manifests"):
        self.out_dir = out_dir

    def generate(self, spec: Dict[str, Any]) -> Dict[str, str]:
        """
        Accepts a spec dictionary and returns dict of file paths -> content.
        spec keys: name, model_name, version, image, port, service_port, replicas,
                   resources, env, hpa, ingress
        """
        context = {
            "name": spec["name"],
            "model_name": spec["model_name"],
            "version": spec["version"],
            "image": spec["image"],
            "port": spec.get("port", 8080),
            "service_port": spec.get("service_port", spec.get("port", 8080)),
            "replicas": spec.get("replicas", 2),
            "readiness_path": spec.get("readiness_path", "/health"),
            "liveness_path": spec.get("liveness_path", "/health"),
            "cpu_limit": spec.get("resources", {}).get("limits", {}).get("cpu", "500m"),
            "mem_limit": spec.get("resources", {}).get("limits", {}).get("memory", "512Mi"),
            "cpu_request": spec.get("resources", {}).get("requests", {}).get("cpu", "200m"),
            "mem_request": spec.get("resources", {}).get("requests", {}).get("memory", "256Mi"),
            "env_block": _env_block(spec.get("env", {})),
            "service_type": spec.get("service_type", "ClusterIP"),
        }

        deployment_yaml = render_template(DEPLOYMENT_TPL, context)
        service_yaml = render_template(SERVICE_TPL, context)

        files = {
            f"{self.out_dir}/{spec['name']}_deployment.yaml": deployment_yaml,
            f"{self.out_dir}/{spec['name']}_service.yaml": service_yaml,
        }

        if spec.get("hpa"):
            hpa_context = {
                "name": spec["name"],
                "min_replicas": spec["hpa"].get("min_replicas", 2),
                "max_replicas": spec["hpa"].get("max_replicas", 10),
                "cpu_utilization": spec["hpa"].get("cpu_utilization", 70),
            }
            files[f"{self.out_dir}/{spec['name']}_hpa.yaml"] = render_template(HPA_TPL, hpa_context)

        if spec.get("ingress"):
            ing = spec["ingress"]
            ingress_ctx = {
                "name": spec["name"],
                "host": ing.get("host", "model.example.com"),
                "path": ing.get("path", "/"),
                "service_port": context["service_port"],
            }
            files[f"{self.out_dir}/{spec['name']}_ingress.yaml"] = render_template(INGRESS_TPL, ingress_ctx)

        # Write files to disk
        for path, content in files.items():
            write_file(path, content)

        logger.info(f"Generated k8s manifests for {spec['name']} in {self.out_dir}")
        return files
