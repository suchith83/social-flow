# scripts/setup/docker_setup.py
import logging
from typing import Dict, Any
from .utils import which, run, ensure_dir

logger = logging.getLogger("setup.docker")

class DockerSetup:
    """
    Configure Docker runtime on host and provide registry login helper.

    Actions:
      - Install docker engine (if distro supports)
      - Create /etc/docker/daemon.json tweaks (insecure registries, log-driver)
      - Ensure user added to docker group
      - Login to registry using provided credentials (registry + username + password from env or config)
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("setup", {}).get("docker", {})
        self.registry = self.cfg.get("registry")
        self.docker_group = "docker"
        self.daemon_conf_path = "/etc/docker/daemon.json"

    def ensure_docker_engine(self):
        if which("docker"):
            logger.info("Docker already installed")
            return
        # Try simple install instructions for apt-based distros (best-effort)
        if which("apt-get"):
            logger.info("Installing docker via apt-get (best-effort)")
            run(["sudo", "apt-get", "update"])
            run(["sudo", "apt-get", "install", "-y", "docker.io"])
            run(["sudo", "systemctl", "enable", "--now", "docker"])
        else:
            logger.warning("Automatic Docker install not available for this distro. Please install manually.")

    def configure_daemon(self):
        # Example: set log-driver and optional insecure registry for local dev
        import json
        conf = {"log-driver": "json-file", "log-opts": {"max-size": "50m", "max-file": "5"}}
        insecure = self.cfg.get("insecure_registries")
        if insecure:
            conf["insecure-registries"] = insecure
        ensure_dir("/etc/docker")
        try:
            with open(self.daemon_conf_path, "w") as fh:
                json.dump(conf, fh, indent=2)
            logger.info("Wrote Docker daemon config to %s", self.daemon_conf_path)
            run(["sudo", "systemctl", "restart", "docker"], check=False)
        except Exception:
            logger.exception("Failed writing docker daemon config")

    def add_user_to_group(self, user: str):
        try:
            run(["sudo", "usermod", "-aG", self.docker_group, user], check=False)
            logger.info("Added %s to docker group", user)
        except Exception:
            logger.exception("Failed to add user to docker group")

    def login_registry(self, username: str = None, password: str = None):
        if not self.registry:
            logger.info("No registry configured; skipping docker login")
            return
        if not which("docker"):
            logger.warning("Docker not available to perform login")
            return
        # We don't log credentials here; prefer env-driven login
        if not username:
            username = ""
        if password is None:
            # Attempt to use docker credential helpers or skip if no password provided
            logger.info("No password provided for registry login; try 'docker login' manually or configure credentials helper")
            return
        try:
            run(["docker", "login", self.registry, "-u", username, "--password-stdin"], capture_output=False, check=True, env=None, cwd=None)
        except Exception:
            # For security we avoid printing creds — instruct manual login
            logger.warning("docker login failed; run `docker login %s` manually", self.registry)

    def run(self):
        if not self.cfg.get("install", True):
            logger.info("Docker install/config disabled")
            return
        self.ensure_docker_engine()
        self.configure_daemon()
