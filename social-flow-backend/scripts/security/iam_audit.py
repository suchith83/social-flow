# scripts/security/iam_audit.py
import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger("security.iam")


class IAMAudit:
    """
    Lightweight IAM audit helpers. Meant to run with AWS SDK (boto3) if credentials exist.
    - Enumerate IAM users / roles
    - Flag overly-permissive policies (wildcards in Action/Resource)
    - Suggest least-privilege notes
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("security", {}).get("iam", {})
        self.enabled = bool(self.cfg.get("enabled", False))
        # boto3 import on demand to avoid mandatory dependency
        self._client = None

    def _init_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("iam")
            except Exception:
                logger.exception("boto3 not available or cannot create IAM client")
                self._client = None

    def list_roles(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            logger.info("IAM audit disabled")
            return []
        self._init_client()
        if not self._client:
            return []
        results = []
        paginator = self._client.get_paginator("list_roles")
        for page in paginator.paginate():
            for r in page.get("Roles", []):
                results.append({"RoleName": r.get("RoleName"), "Arn": r.get("Arn")})
        return results

    def analyze_policy_document(self, doc: Dict[str, Any]) -> List[str]:
        """
        Basic check for wildcard actions/resources.
        Returns list of warnings.
        """
        warnings = []
        statements = doc.get("Statement", [])
        if not isinstance(statements, list):
            statements = [statements]
        for s in statements:
            actions = s.get("Action", [])
            resources = s.get("Resource", [])
            if isinstance(actions, str):
                actions = [actions]
            if isinstance(resources, str):
                resources = [resources]
            for a in actions:
                if a == "*" or a.endswith(":*"):
                    warnings.append("Wildcard action detected: %s" % a)
            for r in resources:
                if r == "*" or r.endswith(":*"):
                    warnings.append("Wildcard resource detected: %s" % r)
        return warnings

    def run(self) -> Dict[str, Any]:
        """
        Example output:
          {"roles": [...], "policy_warnings": {...}}
        """
        if not self.enabled:
            return {}
        self._init_client()
        if not self._client:
            return {}
        roles = self.list_roles()
        policy_issues = {}
        for r in roles:
            try:
                arn = r.get("Arn")
                name = r.get("RoleName")
                policy_issues[name] = []
                attached = self._client.list_attached_role_policies(RoleName=name)
                for p in attached.get("AttachedPolicies", []):
                    doc = self._client.get_policy_version(PolicyArn=p["PolicyArn"], VersionId=self._client.get_policy(PolicyArn=p["PolicyArn"])["Policy"]["DefaultVersionId"])  # simplified; may need pagination handling
                    # note: above is heavy; this is illustrative
                    # real implementation should call get_policy -> list_policy_versions -> get_policy_version
                    policy_doc = {}  # placeholder
                    # analyze if policy_doc present
                    if policy_doc:
                        warnings = self.analyze_policy_document(policy_doc)
                        if warnings:
                            policy_issues[name].extend(warnings)
            except Exception:
                logger.exception("Failed analyzing role %s", r.get("RoleName"))
        return {"roles": roles, "policy_warnings": policy_issues}
