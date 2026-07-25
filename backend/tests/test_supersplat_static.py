"""Self-hosted SuperSplat editor: auth gate + path-traversal guard + root-only
index fallback (main.supersplat_static replaces the old portal reverse-proxy).

NOTE: TestClient(main.app) is used WITHOUT a `with` block on purpose — entering
the client would run main's lifespan (legacy-meta migration + orphan-job
recovery) against the live splatcli data tree, which pytest must never touch.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as splat_main  # noqa: E402

TOKEN = "test-supersplat-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
INDEX_MARK = "SUPERSPLAT-EDITOR-INDEX"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    dist = tmp_path / "dist"
    (dist / "static").mkdir(parents=True)
    (dist / "index.html").write_text(f"<html>{INDEX_MARK}</html>")
    (dist / "index.js").write_text("console.log('supersplat');")
    (dist / "static" / "app.css").write_text("body{color:#fff}")
    # a file OUTSIDE the dist root that a traversal would try to reach
    (tmp_path / "secret.txt").write_text("OUTSIDE-DIST-SECRET")
    monkeypatch.setattr(splat_main, "SUPERSPLAT_DIST", dist)
    monkeypatch.setattr(splat_main, "PORTAL_TOKEN", TOKEN)
    return TestClient(splat_main.app)


def test_unauthed_root_redirects_to_login(client: TestClient) -> None:
    """Mirrors the SPA handler: a human with an expired cookie gets the login
    page (303), not a bare 401 error body."""
    resp = client.get("/supersplat/", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"
    assert INDEX_MARK not in resp.text


def test_unauthed_asset_redirects_to_login(client: TestClient) -> None:
    resp = client.get("/supersplat/index.js", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_authed_root_serves_editor_index(client: TestClient) -> None:
    resp = client.get("/supersplat/", headers=AUTH)
    assert resp.status_code == 200, resp.text
    assert INDEX_MARK in resp.text
    assert resp.headers["content-type"].startswith("text/html")


def test_load_query_contract_reaches_index(client: TestClient) -> None:
    """The frontend contract is /supersplat/?load=<url>&filename=<name> — static
    serving ignores the query string; the request must still land on index.html."""
    resp = client.get(
        "/supersplat/?load=%2Fapi%2Fsplat%2Fjobs%2Fsplat_abc%2Fpreview%2Ffile&filename=splat_abc.ply",
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    assert INDEX_MARK in resp.text


def test_authed_asset_served_with_media_type(client: TestClient) -> None:
    resp = client.get("/supersplat/index.js", headers=AUTH)
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    nested = client.get("/supersplat/static/app.css", headers=AUTH)
    assert nested.status_code == 200
    assert "css" in nested.headers["content-type"]


def test_deep_miss_is_404_not_index_fallback(client: TestClient) -> None:
    """SuperSplat is an editor SPA, not a router app: only the bare root falls
    back to index.html; a deep path that misses a real file is an honest 404."""
    resp = client.get("/supersplat/no-such-asset.js", headers=AUTH)
    assert resp.status_code == 404
    assert INDEX_MARK not in resp.text
    deep = client.get("/supersplat/editor/scene/42", headers=AUTH)
    assert deep.status_code == 404
    assert INDEX_MARK not in deep.text


def test_directory_path_is_404(client: TestClient) -> None:
    assert client.get("/supersplat/static", headers=AUTH).status_code == 404


def test_traversal_is_guarded_at_the_handler(client: TestClient) -> None:
    """The resolve()+is_relative_to guard must 404 dot-segment escapes. Exercised
    at the handler directly because HTTP clients normalize ../ away before the
    route ever sees it (which would silently test nothing)."""
    # "/etc/hostname" covers the pathlib absolute-join replacement (base / "/x" == "/x").
    # The handler now takes the request for its 303-to-login auth check — a
    # minimal authed stub keeps this a direct handler-level exercise.
    from types import SimpleNamespace

    authed_request = SimpleNamespace(cookies={}, headers={"authorization": f"Bearer {TOKEN}"})
    for attempt in ("../secret.txt", "static/../../secret.txt", "..%2Fsecret.txt/..", "/etc/hostname"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(splat_main.supersplat_static(attempt, authed_request))
        assert exc.value.status_code == 404


def test_traversal_over_http_never_leaks(client: TestClient) -> None:
    """Belt and braces at the HTTP layer: whatever the client/router does with
    encoded dot segments, the out-of-root file content must never come back."""
    resp = client.get("/supersplat/%2e%2e/secret.txt", headers=AUTH)
    assert "OUTSIDE-DIST-SECRET" not in resp.text
