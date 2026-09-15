from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from test_capture_atlas import packet
import capture_atlas as atlas


def test_private_server_rejects_foreign_hosts_and_arbitrary_files(packet):
    handoff, capture = packet
    document = atlas.write_atlas(handoff, [capture], "Private test")
    path = Path(__file__).resolve().parents[2] / "tools/serve-capture-atlas.py"
    spec = importlib.util.spec_from_file_location("atlas_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.make_server(document["atlas_id"]) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            origin = f"http://127.0.0.1:{server.server_port}"
            with urlopen(origin + "/atlas.json", timeout=3) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "private, no-store"
                assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
            with urlopen(Request(origin + "/", method="HEAD"), timeout=3) as response:
                assert response.status == 200
                assert response.read() == b""
            for route in ["/receipt.json", "/../outside", "/file%2fprivate"]:
                with pytest.raises(HTTPError) as error:
                    urlopen(origin + route, timeout=3)
                assert error.value.code == 404
            with pytest.raises(HTTPError) as error:
                urlopen(Request(origin + "/", headers={"Host": "outside.example"}), timeout=3)
            assert error.value.code == 403
        finally:
            server.shutdown()
            thread.join(timeout=3)
