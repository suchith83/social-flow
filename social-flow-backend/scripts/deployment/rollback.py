# scripts/deployment/rollback.py
import logging
from .config_loader import ConfigLoader
from .k8s_deployer import K8sDeployer
from .notifier import Notifier


def rollback(deployment: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = ConfigLoader("deployment.yaml").load()

    notifier = Notifier(config)
    k8s = K8sDeployer(config)

    try:
        k8s.rollback(deployment)
        notifier.notify(f"✅ Rollback completed for {deployment}")
    except Exception as e:
        notifier.notify(f"❌ Rollback failed: {e}")
        raise


if __name__ == "__main__":
    rollback("socialflow-backend")
