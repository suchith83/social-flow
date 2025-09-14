# scripts/setup/k8s_context.py
import logging
import os
from typing import Dict, Any
from .utils import run, which

logger = logging.getLogger("setup.kube")

class KubeContextSetup:
    """
    Manage kubeconfig contexts and helpers:

      - Copy supplied kubeconfig files to /etc/kubernetes/<name>.kubeconfig
      - Create symlinks per-user in ~/.kube/config or append contexts
      - Optionally install kubectl/helm if missing
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("setup", {}).get("kube", {})
        self.contexts = self.cfg.get("contexts", [])

    def ensure_tools(self):
        if not which("kubectl"):
            logger.warning("kubectl not found; please install for full kube functionality")
        if not which("helm"):
            logger.info("helm not found; install if you plan to use helm charts")

    def install_kubeconfigs(self):
        for ctx in self.contexts:
            name = ctx.get("name")
            kube_src = ctx.get("kubeconfig")
            if not name or not kube_src:
                logger.warning("Invalid kube context entry (skipping): %s", ctx)
                continue
            dest = f"/etc/kubernetes/{name}.kubeconfig"
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.exists(kube_src):
                    # copy only if different
                    import filecmp
                    if not os.path.exists(dest) or not filecmp.cmp(kube_src, dest):
                        run(["sudo", "cp", kube_src, dest], check=True)
                        run(["sudo", "chmod", "0640", dest], check=False)
                        logger.info("Installed kubeconfig %s -> %s", kube_src, dest)
                else:
                    logger.warning("kubeconfig source does not exist: %s", kube_src)
            except Exception:
                logger.exception("Failed installing kubeconfig for %s", name)

    def add_user_symlink(self, username: str):
        # create ~/.kube dir and symlink admin context if available for user convenience
        try:
            import pwd
            pw = pwd.getpwnam(username)
            home = pw.pw_dir
            kube_dir = os.path.join(home, ".kube")
            os.makedirs(kube_dir, exist_ok=True)
            # if there's a default cluster installed at /etc/kubernetes/dev.kubeconfig, symlink
            for dest_candidate in os.listdir("/etc/kubernetes") if os.path.exists("/etc/kubernetes") else []:
                if dest_candidate.endswith(".kubeconfig"):
                    src = os.path.join("/etc/kubernetes", dest_candidate)
                    link = os.path.join(kube_dir, "config")
                    if not os.path.exists(link):
                        run(["ln", "-s", src, link], check=False)
                        run(["chown", f"{username}:{username}", link], check=False)
                        logger.info("Symlinked kubeconfig %s to %s for %s", src, link, username)
                        break
        except KeyError:
            logger.warning("User %s not found for kube symlink creation", username)
        except Exception:
            logger.exception("Failed adding user kube symlink")
