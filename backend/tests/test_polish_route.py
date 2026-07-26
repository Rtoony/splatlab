"""Polish round-trip routes: GLB validation, atomic landing, versioning, and
the untouched-tree guarantee on every rejection path."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glb_check  # noqa: E402
import polish_route  # noqa: E402
import splat_route  # noqa: E402


def _minimal_glb(*, meshes: int = 1, bin_bytes: int = 4, version: str = "2.0") -> bytes:
    doc = {
        "asset": {"version": version, "generator": "splatlab-test"},
        "meshes": [{"primitives": []} for _ in range(meshes)],
        "nodes": [{} for _ in range(max(meshes, 1))],
        "buffers": [{"byteLength": bin_bytes}],
    }
    payload = json.dumps(doc).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    bin_chunk = b"\x00" * bin_bytes
    body = (
        struct.pack("<I4s", len(payload), b"JSON") + payload
        + struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
    )
    total = 12 + len(body)
    return struct.pack("<4sII", b"glTF", 2, total) + body


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    app.include_router(polish_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_0c0001") -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "output_dir": str(job_dir),
                "status": "completed",
                "mode": "3d",
            }
        )
    )
    return job_dir


def _mk_object(job_dir: Path, slug: str = "crate") -> Path:
    obj_dir = job_dir / "_objects" / slug
    (obj_dir / "mesh").mkdir(parents=True)
    (obj_dir / "object.json").write_text(json.dumps({"slug": slug}))
    return obj_dir


def _mk_world(job_dir: Path) -> Path:
    world = job_dir / "_world"
    (world / "elements").mkdir(parents=True)
    (world / "world.json").write_text(
        json.dumps(
            {
                "v": 1,
                "job_id": job_dir.name,
                "shell": {"built": True, "glb": "shell.glb"},
                "elements": [
                    {"slug": "crate", "built": True, "glb": "crate.glb"}
                ],
            }
        )
    )
    (world / "shell.glb").write_bytes(b"old-shell")
    (world / "elements" / "crate.glb").write_bytes(b"old-crate")
    return world


def _post(http, url: str, payload: bytes, filename: str = "polished.glb"):
    return http.post(url, files={"file": (filename, payload, "model/gltf-binary")})


# ── glb_check unit surface ──────────────────────────────────────────────────────


def test_validate_glb_accepts_minimal_file(tmp_path: Path) -> None:
    path = tmp_path / "ok.glb"
    path.write_bytes(_minimal_glb(meshes=2))
    summary = glb_check.validate_glb(path)
    assert summary["meshes"] == 2
    assert summary["gltf_version"] == "2.0"
    assert summary["bin_bytes"] == 4


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: b"NOPE" + raw[4:], "magic"),
        (lambda raw: raw[:-3], "declared length"),
        (lambda raw: raw[:12] + b"\xff" + raw[13:], "chunk"),
    ],
)
def test_validate_glb_rejects_structural_damage(
    tmp_path: Path, mutate, message
) -> None:
    path = tmp_path / "bad.glb"
    path.write_bytes(mutate(_minimal_glb()))
    with pytest.raises(ValueError):
        glb_check.validate_glb(path)


def test_validate_glb_rejects_zero_meshes(tmp_path: Path) -> None:
    path = tmp_path / "empty.glb"
    path.write_bytes(_minimal_glb(meshes=0))
    with pytest.raises(ValueError, match="no meshes"):
        glb_check.validate_glb(path)


def test_validate_glb_rejects_missing_bin_for_urilless_buffer(tmp_path: Path) -> None:
    doc = json.dumps(
        {"asset": {"version": "2.0"}, "meshes": [{}], "buffers": [{"byteLength": 8}]}
    ).encode()
    doc += b" " * ((4 - len(doc) % 4) % 4)
    body = struct.pack("<I4s", len(doc), b"JSON") + doc
    raw = struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body
    path = tmp_path / "nobin.glb"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="binary chunk"):
        glb_check.validate_glb(path)


# ── object polish ───────────────────────────────────────────────────────────────


def test_object_polish_lands_and_lists(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    obj_dir = _mk_object(job_dir)
    response = _post(
        http, "/api/splat/jobs/splat_0c0001/objects/crate/polish", _minimal_glb()
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["url"].endswith("fmt=polished")
    assert body["receipt"]["provenance"] == "polished"
    assert body["receipt"]["supersedes"] is None
    assert (obj_dir / "mesh" / "polished.glb").is_file()
    receipt = json.loads((obj_dir / "mesh" / "polished.json").read_text())
    assert receipt["schema"] == polish_route.POLISH_SCHEMA

    listing = http.get("/api/splat/jobs/splat_0c0001/objects").json()
    files = listing["objects"][0]["files"]
    assert "polished" in files and "polish-receipt" in files

    served = http.get(
        "/api/splat/jobs/splat_0c0001/objects/crate/file", params={"fmt": "polished"}
    )
    assert served.status_code == 200
    assert served.content == _minimal_glb()


def test_object_polish_versions_prior_upload(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    obj_dir = _mk_object(job_dir)
    first = _minimal_glb(meshes=1)
    second = _minimal_glb(meshes=2)
    assert _post(
        http, "/api/splat/jobs/splat_0c0001/objects/crate/polish", first
    ).status_code == 200
    response = _post(
        http, "/api/splat/jobs/splat_0c0001/objects/crate/polish", second
    )
    assert response.status_code == 200
    assert response.json()["receipt"]["supersedes"] is not None
    versions = sorted((obj_dir / "mesh" / "versions").glob("polished-v*.glb"))
    assert [v.name for v in versions] == ["polished-v0001.glb"]
    assert versions[0].read_bytes() == first
    assert (obj_dir / "mesh" / "polished.glb").read_bytes() == second


@pytest.mark.parametrize(
    ("payload", "filename", "status"),
    [
        (b"not a glb at all", "polished.glb", 400),
        (_minimal_glb()[:-5], "polished.glb", 400),
        (_minimal_glb(meshes=0), "polished.glb", 400),
        (b"", "polished.glb", 400),
        (_minimal_glb(), "polished.ply", 400),
    ],
)
def test_object_polish_rejections_leave_tree_untouched(
    client, payload, filename, status
) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    obj_dir = _mk_object(job_dir)
    response = _post(
        http,
        "/api/splat/jobs/splat_0c0001/objects/crate/polish",
        payload,
        filename=filename,
    )
    assert response.status_code == status, response.text
    mesh_dir = obj_dir / "mesh"
    assert not (mesh_dir / "polished.glb").exists()
    assert not (mesh_dir / "polished.json").exists()
    assert not list(mesh_dir.glob(".building-*"))


def test_object_polish_413_over_cap(client, monkeypatch) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    obj_dir = _mk_object(job_dir)
    monkeypatch.setattr(splat_route, "MAX_UPLOAD_BYTES", 16)
    response = _post(
        http, "/api/splat/jobs/splat_0c0001/objects/crate/polish", _minimal_glb()
    )
    assert response.status_code == 413
    assert not (obj_dir / "mesh" / "polished.glb").exists()
    assert not list((obj_dir / "mesh").glob(".building-*"))


def test_object_polish_409_while_mesh_lock_held(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    _mk_object(job_dir)
    # TestClient owns the event loop, so the lock cannot be awaited from here;
    # flipping the asyncio.Lock's internal flag simulates a running build.
    lock = splat_route._mesh_export_lock("splat_0c0001")
    lock._locked = True  # noqa: SLF001 - simulate a running build
    try:
        response = _post(
            http, "/api/splat/jobs/splat_0c0001/objects/crate/polish", _minimal_glb()
        )
        assert response.status_code == 409
    finally:
        lock._locked = False  # noqa: SLF001


def test_object_polish_404s(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    _mk_object(job_dir)
    cases = [
        "/api/splat/jobs/splat_0c9999/objects/crate/polish",
        "/api/splat/jobs/splat_0c0001/objects/unknown/polish",
        "/api/splat/jobs/splat_0c0001/objects/UPPER..slug/polish",
    ]
    for url in cases:
        assert _post(http, url, _minimal_glb()).status_code == 404, url


# ── world element polish ────────────────────────────────────────────────────────


def test_world_element_polish_replaces_and_marks(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    world = _mk_world(job_dir)
    payload = _minimal_glb(meshes=3)
    response = _post(
        http, "/api/splat/jobs/splat_0c0001/world/elements/crate/polish", payload
    )
    assert response.status_code == 200, response.text
    assert (world / "elements" / "crate.glb").read_bytes() == payload
    versions = list((world / "versions").glob("crate-v*.glb"))
    assert len(versions) == 1
    assert versions[0].read_bytes() == b"old-crate"
    marker = json.loads((world / "elements" / "crate.polish.json").read_text())
    assert marker["atlas_superseded"] is True

    manifest = http.get("/api/splat/jobs/splat_0c0001/world/manifest").json()
    element = next(e for e in manifest["elements"] if e["slug"] == "crate")
    assert element["polished"]["atlas_superseded"] is True
    assert element["polished"]["source_filename"] == "polished.glb"


def test_world_shell_polish_replaces_and_marks(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    world = _mk_world(job_dir)
    payload = _minimal_glb(meshes=2)
    response = _post(
        http, "/api/splat/jobs/splat_0c0001/world/elements/shell/polish", payload
    )
    assert response.status_code == 200, response.text
    assert (world / "shell.glb").read_bytes() == payload
    assert (world / "versions" / "shell-v0001.glb").read_bytes() == b"old-shell"
    manifest = http.get("/api/splat/jobs/splat_0c0001/world/manifest").json()
    assert manifest["shell"]["polished"]["atlas_superseded"] is True


def test_world_polish_404_when_element_not_built(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    _mk_world(job_dir)
    response = _post(
        http, "/api/splat/jobs/splat_0c0001/world/elements/ghost/polish", _minimal_glb()
    )
    assert response.status_code == 404
    assert "polish replaces" in response.json()["detail"]


def test_world_polish_rejection_leaves_original(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    world = _mk_world(job_dir)
    response = _post(
        http,
        "/api/splat/jobs/splat_0c0001/world/elements/crate/polish",
        b"garbage-bytes",
    )
    assert response.status_code == 400
    assert (world / "elements" / "crate.glb").read_bytes() == b"old-crate"
    assert not (world / "versions").exists()
    assert not list(world.glob(".building-*"))
