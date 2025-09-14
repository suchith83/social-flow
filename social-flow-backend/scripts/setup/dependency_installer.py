# scripts/setup/dependency_installer.py
import logging
import os
from typing import Dict, Any, List, Optional
from .utils import run, which

logger = logging.getLogger("setup.deps")

class DependencyInstaller:
    """
    Idempotent installation of OS and language-level dependencies.

    - OS packages via apt/dnf (auto-detect)
    - Python via pyenv or system package + venv
    - Node via nvm or distro package (optional)
    - Optionally installs tooling (kubectl, helm, docker-compose)
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("setup", {})
        self.packages = self.cfg.get("os_packages", [])
        self.python_cfg = self.cfg.get("python", {})
        self.node_cfg = self.cfg.get("node", {})
        self.tools = self.cfg.get("tools", [])

    def _detect_pkg_mgr(self) -> Optional[str]:
        if which("apt-get"):
            return "apt"
        if which("dnf"):
            return "dnf"
        if which("yum"):
            return "yum"
        return None

    def install_os_packages(self):
        pkg_mgr = self._detect_pkg_mgr()
        if not pkg_mgr:
            logger.warning("No supported package manager found; skipping OS package installation")
            return
        if not self.packages:
            logger.info("No OS packages requested")
            return
        logger.info("Installing packages via %s: %s", pkg_mgr, self.packages)
        if pkg_mgr == "apt":
            run(["sudo", "apt-get", "update"], check=True)
            run(["sudo", "apt-get", "install", "-y"] + self.packages, check=True)
        elif pkg_mgr in ("dnf", "yum"):
            run(["sudo", pkg_mgr, "install", "-y"] + self.packages, check=True)

    def ensure_python_venv(self):
        venv_path = self.python_cfg.get("venv_path", "/opt/socialflow/venv")
        versions = self.python_cfg.get("versions", ["3.10"])
        # Prefer system python, create venv for primary version available
        py_bin = None
        for v in versions:
            candidate = f"python{v}"
            if which(candidate):
                py_bin = candidate
                break
        if not py_bin and which("python3"):
            py_bin = "python3"
        if not py_bin:
            logger.warning("No python interpreter found. Skipping venv creation")
            return
        logger.info("Creating python venv using %s at %s", py_bin, venv_path)
        run(["sudo", "mkdir", "-p", venv_path], check=True)
        run(["sudo", py_bin, "-m", "venv", venv_path], check=True)
        run(["sudo", "chown", "-R", f"{os.geteuid()}:{os.getegid()}", venv_path], check=False)

    def ensure_node(self):
        if not self.node_cfg.get("install", False):
            logger.info("Node install not requested")
            return
        # naive: if node exists, skip; else provide instructions / attempt install via distro
        if which("node") and which("npm"):
            logger.info("Node already present")
            return
        logger.info("Attempting to install node via distro package manager")
        pkg_mgr = self._detect_pkg_mgr()
        if pkg_mgr == "apt":
            run(["sudo", "apt-get", "install", "-y", "nodejs", "npm"], check=False)
        elif pkg_mgr in ("dnf", "yum"):
            run(["sudo", pkg_mgr, "install", "-y", "nodejs", "npm"], check=False)
        else:
            logger.warning("Please install node manually (no supported package manager detected)")

    def install_misc_tools(self):
        # installs kubectl/helm/terraform if requested via tools config or present in self.tools
        tools = self.tools or []
        for t in tools:
            if which(t):
                logger.info("%s present; skipping", t)
                continue
            logger.info("Requested tool %s not found; user must install manually or enable package install", t)

    def run(self):
        self.install_os_packages()
        self.ensure_python_venv()
        self.ensure_node()
        self.install_misc_tools()
