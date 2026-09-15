#!/usr/bin/env python3
"""Serve one sealed reconstruction showcase privately on loopback for a bounded lifetime."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import artifact_manifest as manifests
from reference_delivery import artifact_path


def make_server(directory, port=0):
    directory = directory.resolve()
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt.get("schema") != "dev.splatlab.real-to-model-showcase/v1":
        raise ValueError("Expected a sealed reconstruction showcase")
    for name, digest in receipt["files"].items():
        if manifests.sha256_file(artifact_path(directory, name)) != digest:
            raise ValueError("Showcase artifact changed")
    allowed = {*receipt["files"], "receipt.json"}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def do_GET(self):
            self.send_artifact()

        def do_HEAD(self):
            self.send_artifact(True)

        def send_artifact(self, head_only=False):
            if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                self.send_error(403)
                return
            name = unquote(urlsplit(self.path).path).removeprefix("/") or "index.html"
            if name == "favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if name not in allowed:
                self.send_error(404)
                return
            try:
                path = artifact_path(directory, name)
            except ValueError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            if not head_only:
                try:
                    with path.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile, 65536)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 1
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=32852)
    parser.add_argument("--seconds", type=int, default=21600)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 43200:
        parser.error("Preview lifetime must be 1–43200 seconds")
    with make_server(args.directory, args.port) as server:
        print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/", "seconds": args.seconds}), flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            server.handle_request()


if __name__ == "__main__":
    main()
