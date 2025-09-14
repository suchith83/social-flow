#!/usr/bin/env python3
"""
Import a SonarQube Quality Profile XML.

Usage:
  python3 scripts/import-quality-profile.py --sonar-host http://localhost:9000 --token <admin-token> --file config/sonar-quality-profile.xml

This uses the /api/qualityprofiles/restore endpoint (which might require admin privileges).
"""
import argparse
import requests
from urllib.parse import urljoin
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sonar-host", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--file", required=True)
    args = p.parse_args()

    base = args.sonar_host.rstrip("/")
    session = requests.Session()
    session.auth = (args.token, "")

    url = urljoin(base, "/api/qualityprofiles/restore")
    with open(args.file, "rb") as fh:
        files = {'backup': fh}
        r = session.post(url, files=files)
        if r.status_code == 200:
            print("Quality profile imported successfully.")
            sys.exit(0)
        else:
            print("Failed to import quality profile. Status:", r.status_code)
            print("Response:", r.text)
            sys.exit(2)

if __name__ == "__main__":
    main()
