# scripts/setup/setup_runner.py
import logging
import argparse
import sys
from typing import Dict, Any

from .config_loader import ConfigLoader
from .dependency_installer import DependencyInstaller
from .docker_setup import DockerSetup
from .k8s_context import KubeContextSetup
from .user_setup import UserSetup
from .cert_manager import CertManager
from .secrets_bootstrap import SecretsBootstrap
from .env_manager import EnvManager
from .cleanup import SetupCleanup
from .utils import which

logger = logging.getLogger("setup.runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SetupRunner:
    """
    Orchestrates the entire setup flow. Designed to be idempotent:
     - run() will attempt safe operations and log what changed
     - uses config to control stages
    """

    def __init__(self, config_path: str = "setup.yaml"):
        loader = ConfigLoader(config_path)
        self.config = loader.load()
        self.deps = DependencyInstaller(self.config)
        self.docker = DockerSetup(self.config)
        self.kube = KubeContextSetup(self.config)
        self.users = UserSetup(self.config)
        self.certs = CertManager(self.config)
        self.secrets = SecretsBootstrap(self.config)
        self.env = EnvManager(self.config)
        self.cleanup = SetupCleanup(self.config)

    def run(self, stages: list = None):
        """
        stages: optional list to limit actions (install_deps, users, docker, kube, certs, secrets, env, cleanup)
        """
        stages = stages or ["install_deps", "users", "docker", "kube", "certs", "secrets", "env", "cleanup"]
        logger.info("Beginning setup run (stages=%s)", stages)

        if "install_deps" in stages:
            self.deps.run()

        if "users" in stages:
            self.users.run()

        if "docker" in stages:
            self.docker.run()

        if "kube" in stages:
            self.kube.ensure_tools()
            self.kube.install_kubeconfigs()
            # add symlink for each created user
            for u in self.config.get("setup", {}).get("users", []):
                self.kube.add_user_symlink(u.get("name"))

        if "certs" in stages:
            hosts = self.config.get("setup", {}).get("certificates", {}).get("hosts", [])
            if hosts:
                self.certs.ensure_certificates(hosts)

        if "secrets" in stages and self.config.get("setup", {}).get("secrets", {}).get("bootstrap", False):
            mapping = self.config.get("setup", {}).get("secrets", {}).get("mapping", {})
            if mapping:
                self.secrets.bootstrap_from_env(mapping)

        if "env" in stages and self.config.get("setup", {}).get("env"):
            # write environment files per service config
            env_cfg = self.config.get("setup", {}).get("env", {})
            for sname, mapping in env_cfg.get("services", {}).items():
                path = mapping.get("path") or f"/etc/socialflow/env/{sname}.env"
                self.env.render_dotenv(mapping.get("vars", {}), path, override=mapping.get("override", False))

        if "cleanup" in stages:
            self.cleanup.run()

        logger.info("Setup run complete")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run system/setup provisioning for Social Flow")
    parser.add_argument("--config", default="setup.yaml", help="Path to setup config file")
    parser.add_argument("--stages", nargs="*", help="Optional stages to run (install_deps users docker kube certs secrets env cleanup)")
    args = parser.parse_args(argv)
    runner = SetupRunner(args.config)
    try:
        runner.run(stages=args.stages)
    except Exception as e:
        logging.exception("Setup run failed: %s", e)
        sys.exit(2)

if __name__ == "__main__":
    main()
