#!/usr/bin/env python3
"""Loopback static server for the Spark agreement page: /  -> index.html,
/node_modules/* -> frontend/node_modules (three + spark), /ply/<name>.ply -> registered files."""
from __future__ import annotations
import http.server, json, os, socketserver, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NODE_MODULES = REPO / "frontend" / "node_modules"


def make_handler(ply_map: dict[str, Path]):
    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(HERE / "index.html", "text/html")
            if path.startswith("/node_modules/"):
                target = (NODE_MODULES / path[len("/node_modules/"):]).resolve()
                if NODE_MODULES.resolve() in target.parents and target.is_file():
                    ctype = "text/javascript" if target.suffix in (".js", ".mjs") else "application/wasm" if target.suffix == ".wasm" else "application/octet-stream"
                    return self._send(target, ctype)
            if path.startswith("/ply/"):
                target = ply_map.get(path[len("/ply/"):])
                if target and target.is_file():
                    return self._send(target, "application/octet-stream")
            self.send_error(404)
        def _send(self, p: Path, ctype: str):
            data = p.read_bytes()
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)
    return H


def start(ply_map: dict[str, Path]) -> tuple[socketserver.TCPServer, int]:
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), make_handler(ply_map))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


if __name__ == "__main__":
    ply_map = {k: Path(v) for k, v in json.loads(sys.argv[1]).items()} if len(sys.argv) > 1 else {}
    srv, port = start(ply_map); print(f"http://127.0.0.1:{port}/", flush=True)
    try: threading.Event().wait()
    except KeyboardInterrupt: srv.shutdown()
