# Monitors and gathers Kubernetes cluster metrics
"""
Kubernetes metrics collection helpers.

If running inside Kubernetes and ENABLE_K8S_COLLECTION is True, this module uses the kubernetes client
to gather higher-level data:
 - pod counts per namespace
 - pod restart counts
 - node conditions (if accessible)
 - resource requests/limits summaries (best-effort)

This module is optional; errors are handled gracefully so it will not break non-k8s environments.
"""

import logging
from typing import Dict, Optional
from prometheus_client import Gauge
from .config import InfraMetricsConfig

logger = logging.getLogger("infra_metrics.k8s")

try:
    from kubernetes import client as k8s_client, config as k8s_config
    K8S_AVAILABLE = True
except Exception:
    K8S_AVAILABLE = False
    logger.info("kubernetes client not available; K8s metrics disabled unless the library is installed.")


class K8sMetricsCollector:
    """
    Lightweight k8s collector. Call .collect() to perform one synchronous poll.
    """

    def __init__(self):
        # Gauges with namespace label
        self.pods_count = Gauge("infra_k8s_pods_count", "Number of pods in namespace", ["namespace"])
        self.pod_restart_count = Gauge("infra_k8s_pod_restart_count", "Pod restart counts", ["namespace", "pod"])
        self.node_ready = Gauge("infra_k8s_node_ready", "Node ready condition", ["node"])
        self._api = None
        if K8S_AVAILABLE:
            try:
                # Load in-cluster config if possible, else fallback to kubeconfig
                try:
                    k8s_config.load_incluster_config()
                    logger.debug("Loaded in-cluster kube config")
                except Exception:
                    k8s_config.load_kube_config()
                    logger.debug("Loaded kubeconfig from environment")
                self._api = k8s_client.CoreV1Api()
            except Exception:
                logger.exception("Failed to initialize Kubernetes API client")
                self._api = None

    def collect(self):
        if not (InfraMetricsConfig.ENABLE_K8S_COLLECTION and K8S_AVAILABLE and self._api):
            logger.debug("K8s collection disabled or kube client not available")
            return
        # Collect pods per namespace
        try:
            pods = self._api.list_pod_for_all_namespaces(watch=False)
            namespace_counts = {}
            for pod in pods.items:
                ns = pod.metadata.namespace
                namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
                # record restarts per container
                total_restarts = sum((c.restart_count or 0) for c in pod.status.container_statuses or [])
                self.pod_restart_count.labels(namespace=ns, pod=pod.metadata.name).set(total_restarts)
            for ns, count in namespace_counts.items():
                self.pods_count.labels(namespace=ns).set(count)
        except Exception:
            logger.exception("Error collecting k8s pod information")

        # Nodes
        try:
            nodes = self._api.list_node()
            for node in nodes.items:
                ready = 0
                for cond in node.status.conditions or []:
                    if cond.type == "Ready":
                        ready = 1 if cond.status == "True" else 0
                self.node_ready.labels(node=node.metadata.name).set(ready)
        except Exception:
            logger.exception("Error collecting k8s node information")
