# scripts/setup/cert_manager.py
import ssl
import os
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
from .utils import write_file, ensure_dir

logger = logging.getLogger("setup.certs")

class CertManager:
    """
    Manage TLS certificates for dev/staging/production.

    Capabilities:
      - Create self-signed certs for dev hosts (idempotent)
      - Store certs under configured cert_dir
      - (Optional) Integrate with ACME (Let's Encrypt) using certbot if auto=True
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("setup", {}).get("certificates", {})
        self.cert_dir = self.cfg.get("cert_dir", "/etc/ssl/socialflow")
        ensure_dir(self.cert_dir)

    def _self_signed(self, common_name: str, days: int = 3650):
        """
        Create a self-signed certificate (PEM) using Python ssl APIs (RFC-compliant).
        Idempotent: will not overwrite if files exist (unless override True).
        """
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import rsa

        key_path = os.path.join(self.cert_dir, f"{common_name}.key.pem")
        cert_path = os.path.join(self.cert_dir, f"{common_name}.cert.pem")
        if os.path.exists(key_path) and os.path.exists(cert_path):
            logger.info("Cert and key for %s already exist at %s", common_name, self.cert_dir)
            return cert_path, key_path

        # generate RSA key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow() - timedelta(minutes=5))
            .not_valid_after(datetime.utcnow() + timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        # write key
        with open(key_path, "wb") as fh:
            fh.write(key.private_bytes(encoding=serialization.Encoding.PEM,
                                      format=serialization.PrivateFormat.TraditionalOpenSSL,
                                      encryption_algorithm=serialization.NoEncryption()))
        # write cert
        with open(cert_path, "wb") as fh:
            fh.write(cert.public_bytes(serialization.Encoding.PEM))
        logger.info("Generated self-signed cert for %s", common_name)
        return cert_path, key_path

    def ensure_certificates(self, hosts: list):
        if not hosts:
            logger.info("No hosts provided for certificates")
            return []
        created = []
        for h in hosts:
            try:
                cert, key = self._self_signed(h)
                created.append({"host": h, "cert": cert, "key": key})
            except Exception:
                logger.exception("Failed generating cert for %s", h)
        return created

    def run_acme(self):
        # highly environment specific; only provide guidance and not automatic ACME request here
        if not self.cfg.get("auto", False):
            logger.info("ACME auto mode disabled")
            return
        logger.warning("ACME flow not implemented — use certbot or your ACME client in production")
