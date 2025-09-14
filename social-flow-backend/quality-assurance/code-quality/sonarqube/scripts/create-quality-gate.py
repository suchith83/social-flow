#!/usr/bin/env python3
"""
Create or update a SonarQube Quality Gate based on JSON config.

Usage:
  python3 scripts/create-quality-gate.py --sonar-host http://localhost:9000 --token <admin-token> --config config/sonar-quality-gate.json

The JSON structure is simple:
{
  "name": "MyQualityGate",
  "conditions": [
    {"metric": "coverage", "operator": "LT", "errorThreshold": "80"},
    {"metric": "new_coverage", "operator": "LT", "errorThreshold": "80"},
    {"metric": "bugs", "operator": "GT", "errorThreshold": "0", "onNewCode": true}
  ]
}
"""
import argparse
import json
import requests
from urllib.parse import urljoin

def get_quality_gate_id(session, base_url, name):
    r = session.get(urljoin(base_url, "/api/qualitygates/search"))
    r.raise_for_status()
    data = r.json()
    for gate in data.get("qualitygates", []):
        if gate.get("name") == name:
            return gate.get("id")
    return None

def delete_quality_gate(session, base_url, gate_id):
    r = session.post(urljoin(base_url, "/api/qualitygates/delete"), data={"id": gate_id})
    r.raise_for_status()
    return r.json()

def create_quality_gate(session, base_url, name):
    r = session.post(urljoin(base_url, "/api/qualitygates/create"), data={"name": name})
    r.raise_for_status()
    return r.json()

def create_condition(session, base_url, gate_id, condition):
    payload = {
        "gateId": gate_id,
        "metric": condition["metric"],
        "op": condition["operator"],
        "error": condition["errorThreshold"]
    }
    # SonarQube older/newer APIs use slightly different field names; this is a solid base.
    r = session.post(urljoin(base_url, "/api/qualitygates/create_condition"), data=payload)
    r.raise_for_status()
    return r.json()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sonar-host", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    base = args.sonar_host.rstrip("/")
    session = requests.Session()
    session.auth = (args.token, "")

    cfg = json.load(open(args.config, "r"))
    name = cfg["name"]
    print(f"Ensuring quality gate '{name}'")

    existing_id = get_quality_gate_id(session, base, name)
    if existing_id:
        print(f"Quality gate '{name}' exists (id={existing_id}), deleting and recreating to replace conditions.")
        delete_quality_gate(session, base, existing_id)

    resp = create_quality_gate(session, base, name)
    gate_id = resp.get("id")
    print(f"Created quality gate id={gate_id}")

    for cond in cfg.get("conditions", []):
        print(f"Adding condition: {cond}")
        create_condition(session, base, gate_id, cond)

    print("Quality gate setup completed.")

if __name__ == "__main__":
    main()
