#!/usr/bin/env python3
"""
Minimal prediction server (stdlib) to serve JSON model artifacts for local testing.

Endpoints:
- GET /predict?user_id=...&limit=...
- GET /health
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List


class SimpleModelServer(BaseHTTPRequestHandler):
    model: Dict[str, Any] = {}

    def _write(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write(200, {"status": "ok"})
            return

        if parsed.path == "/predict":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["anonymous"])[0]
            lim = int(qs.get("limit", [10])[0])
            candidates = SimpleModelServer.model.get("candidates", [])
            # simple personalization: rotate based on user hash
            offset = sum(ord(c) for c in user_id) % max(1, len(candidates))
            selected = [candidates[(offset + i) % len(candidates)] for i in range(min(lim, len(candidates)))]
            return self._write(200, {"user_id": user_id, "recommendations": selected})
        return self._write(404, {"error": "not found"})


def load_model(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()

    model = load_model(args.model_path)
    SimpleModelServer.model = model

    server = HTTPServer((args.host, args.port), SimpleModelServer)
    print(f"Serving model {args.model_path} on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
