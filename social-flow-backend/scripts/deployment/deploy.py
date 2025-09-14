# scripts/deployment/deploy.py
import logging
from .config_loader import ConfigLoader
from .infra_manager import InfraManager
from .docker_manager import DockerManager
from .k8s_deployer import K8sDeployer
from .notifier import Notifier
from .health_checker import HealthChecker


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = ConfigLoader("deployment.yaml").load()

    notifier = Notifier(config)
    notifier.notify("🚀 Starting deployment...")

    try:
        infra = InfraManager(config)
        docker = DockerManager(config)
        k8s = K8sDeployer(config)
        health = HealthChecker(config)

        infra.provision()
        docker.build()
        docker.push()
        k8s.deploy()
        health.check()

        notifier.notify("✅ Deployment succeeded!")

    except Exception as e:
        logging.error(f"Deployment failed: {e}")
        notifier.notify(f"❌ Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
