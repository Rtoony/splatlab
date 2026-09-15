#!/usr/bin/env python3
"""Serve one sealed private atlas on loopback, with a bounded lifetime."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import capture_atlas


def make_server(atlas_id: str, port: int = 0) -> HTTPServer:
    capture_atlas.load_atlas(atlas_id)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def do_GET(self):
            self.send_artifact()

        def do_HEAD(self):
            self.send_artifact(head_only=True)

        def send_artifact(self, head_only=False):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in expected:
                self.send_error(403, "Loopback host required")
                return
            filename = urlsplit(self.path).path.removeprefix("/") or "index.html"
            try:
                path = capture_atlas.atlas_file(atlas_id, filename)
                data = path.read_bytes()
            except ValueError:
                self.send_error(404)
                return
            self.send_response(200)
            content_type = "application/geo+json" if path.suffix == ".geojson" else mimetypes.guess_type(path.name)[0]
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            if not head_only:
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 1
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas_id")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=28800)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 43200:
        parser.error("Preview lifetime must be 1–43200 seconds")
    with make_server(args.atlas_id, args.port) as server:
        print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/", "seconds": args.seconds}), flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            server.handle_request()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
