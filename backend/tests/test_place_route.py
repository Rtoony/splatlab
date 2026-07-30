"""The set-dressing door: create refuses to replace, delete refuses captured,
every rejection leaves the tree untouched, and a landed asset is picked up
by world_slugs / affordances / restyle for free."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import place_route  # noqa: E402
import splat_route  # noqa: E402
import world_affordances  # noqa: E402
import world_interactions as wi  # noqa: E402
import world_placed as wp  # noqa: E402
import world_restyle as wr  # noqa: E402

JOB = "splat_dec0de"  # hex-only — a readable slug silently 404s every route


def _glb_from_doc(doc: dict) -> bytes:
    payload = json.dumps(doc).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    body = struct.pack("<I4s", len(payload), b"JSON") + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _authored_glb(*, aabb_min=(1.0, 0.0, -2.0), aabb_max=(1.4, 0.45, -1.6),
                  nodes=None) -> bytes:
    return _glb_from_doc({
        "asset": {"version": "2.0", "generator": "splatlab-test"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"type": "VEC3", "componentType": 5126, "count": 3,
                       "min": list(aabb_min), "max": list(aabb_max)}],
        "nodes": nodes if nodes is not None else [{"mesh": 0}],
    })


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    app.include_router(place_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = JOB) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir),
        "status": "completed", "mode": "3d",
    }))
    return job_dir


def _mk_world(job_dir: Path, *, units: str = "meters", graded: bool = True) -> Path:
    world = job_dir / "_world"
    (world / "elements").mkdir(parents=True)
    (world / "world.json").write_text(json.dumps({
        "v": 1, "job_id": job_dir.name, "units": units,
        "meters_per_unit": 1.0 if units == "meters" else None,
        "shell": {"built": True, "glb": "shell.glb", "extent": [8, 3, 8]},
        "elements": [
            {"slug": "crate", "built": True, "glb": "crate.glb",
             "extent": [1, 1, 1]},
            {"slug": "bench", "built": False, "reason": "not isolated"},
        ],
    }))
    if graded:
        (world / "world_manifest.json").write_text(json.dumps({
            "v": 1, "job_id": job_dir.name,
            "shell": {"slug": "shell", "role": "static", "extent": [8, 3, 8]},
            "elements": [
                {"slug": "crate", "label": "crate", "role": "prop",
                 "glb": "crate.glb"},
                {"slug": "bench", "role": "unbuilt", "reason": "not isolated"},
            ],
            "counts": {"props": 1, "static": 0, "unbuilt": 1},
        }))
    (world / "elements" / "crate.glb").write_bytes(b"captured-crate")
    return world


def _place(http, slug: str, payload: bytes, *, role: str = "static",
           label: str = "", job: str = JOB):
    return http.post(
        f"/api/splat/jobs/{job}/world/elements",
        params={"slug": slug, "role": role, "label": label},
        files={"file": ("torch.glb", payload, "model/gltf-binary")})


# ── create ──────────────────────────────────────────────────────────────────────


def test_place_lands_glb_registry_manifest_and_receipt(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    world = _mk_world(job_dir)
    payload = _authored_glb()

    response = _place(http, "torch-sconce", payload, label="torch sconce")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["element"]["provenance"] == "authored"
    assert body["receipt"]["supersedes"] is None
    assert body["receipt"]["sha256"]
    assert body["warnings"] == []

    assert (world / "elements" / "torch-sconce.glb").read_bytes() == payload
    registry = json.loads((world / wp.PLACED_FILENAME).read_text())
    assert registry["elements"][0]["glb"] == "torch-sconce.glb"

    manifest = json.loads((world / "world_manifest.json").read_text())
    torch = next(e for e in manifest["elements"] if e["slug"] == "torch-sconce")
    assert torch["provenance"] == "authored"
    assert torch["collision"]["strategy"] == "complex_as_simple"
    assert manifest["counts"]["authored"] == 1
    assert not list(world.glob(".building-*"))

    # One append point, six consumers: slugs / affordances / restyle all see
    # the placed asset with zero extra wiring.
    assert "torch-sconce" in wi.world_slugs(manifest)
    proposals = world_affordances.propose_affordances(manifest)
    torch_prop = next(p for p in proposals["proposals"]
                      if p["record"]["slug"] == "torch-sconce")
    assert torch_prop["record"]["verb"] == "inspect"  # static -> inspect
    wr.validate_restyle({"elements": {"torch-sconce": {"tint": "#112233"}}},
                        known_slugs=wi.world_slugs(manifest))


def test_place_refuses_every_existing_slug_shape(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    world = _mk_world(job_dir)
    for slug in ("crate", "bench"):  # captured built AND unbuilt instance
        response = _place(http, slug, _authored_glb())
        assert response.status_code == 409, slug
        assert "already exists" in response.json()["detail"]
    assert _place(http, "torch-sconce", _authored_glb()).status_code == 200
    assert _place(http, "torch-sconce", _authored_glb()).status_code == 409
    # A stray file with no record anywhere still blocks — never overwrite.
    (world / "elements" / "stray.glb").write_bytes(b"stray")
    assert _place(http, "stray", _authored_glb()).status_code == 409


def test_place_404_bad_slug_and_400_bad_role(client) -> None:
    http, outputs = client
    _mk_world(_mk_job(outputs))
    assert _place(http, "shell", _authored_glb()).status_code == 404
    assert _place(http, "UPPER", _authored_glb()).status_code == 404
    assert _place(http, "torch", _authored_glb(), role="vehicle").status_code == 400


def test_place_409_without_world_or_grading(client) -> None:
    http, outputs = client
    _mk_job(outputs)
    response = _place(http, "torch", _authored_glb())
    assert response.status_code == 409
    assert "solidify" in response.json()["detail"]

    _mk_world(outputs / JOB, graded=False)
    response = _place(http, "torch", _authored_glb())
    assert response.status_code == 409
    assert "ungraded" in response.json()["detail"]


def test_place_bad_glb_400_leaves_tree_untouched(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    world = _mk_world(job_dir)
    before = (world / "world_manifest.json").read_bytes()
    response = _place(http, "torch", b"garbage-bytes")
    assert response.status_code == 400
    assert not (world / "elements" / "torch.glb").exists()
    assert not (world / wp.PLACED_FILENAME).exists()
    assert (world / "world_manifest.json").read_bytes() == before
    assert not list(world.glob(".building-*"))


def test_place_prop_with_node_transforms_422_static_warns(client) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    moved = _authored_glb(nodes=[{"mesh": 0, "translation": [1.0, 0.0, 0.0]}])

    response = _place(http, "torch-prop", moved, role="prop")
    assert response.status_code == 422
    assert "bake_world_transform" in response.json()["detail"]
    assert not (world / "elements" / "torch-prop.glb").exists()

    response = _place(http, "torch-static", moved, role="static")
    assert response.status_code == 200
    assert any("node transforms" in w for w in response.json()["warnings"])


def test_place_413_and_busy_lock_409(client, monkeypatch) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    monkeypatch.setattr(splat_route, "MAX_UPLOAD_BYTES", 16)
    assert _place(http, "torch", _authored_glb()).status_code == 413
    monkeypatch.setattr(splat_route, "MAX_UPLOAD_BYTES", 2 * 1024 ** 3)

    lock = splat_route._mesh_export_lock(JOB)
    lock._locked = True  # noqa: SLF001 - simulate a running build
    try:
        assert _place(http, "torch", _authored_glb()).status_code == 409
    finally:
        lock._locked = False  # noqa: SLF001
    assert not (world / "elements" / "torch.glb").exists()


def test_place_warns_outside_shell_and_uncalibrated(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    _mk_world(job_dir, units="scene-units (uncalibrated)")
    far = _authored_glb(aabb_min=(99.0, 0.0, 99.0), aabb_max=(99.5, 0.5, 99.5))
    response = _place(http, "far-obelisk", far)
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert any("half-diagonal" in w for w in warnings)
    assert any("uncalibrated" in w for w in warnings)


def test_place_409_when_registry_full(client, monkeypatch) -> None:
    http, outputs = client
    _mk_world(_mk_job(outputs))
    monkeypatch.setattr(wp, "MAX_PLACED", 0)
    response = _place(http, "torch", _authored_glb())
    assert response.status_code == 409
    assert "full" in response.json()["detail"]


# ── delete ──────────────────────────────────────────────────────────────────────


def test_delete_authored_tombstones_and_cleans_everything(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    world = _mk_world(job_dir)
    payload = _authored_glb()
    assert _place(http, "torch-sconce", payload).status_code == 200

    response = http.delete(f"/api/splat/jobs/{JOB}/world/elements/torch-sconce")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tombstone"] == "versions/torch-sconce-v0001.glb"
    assert (world / "versions" / "torch-sconce-v0001.glb").read_bytes() == payload
    assert not (world / "elements" / "torch-sconce.glb").exists()
    assert not (world / "elements" / "torch-sconce.placed.json").exists()
    removal = json.loads(
        (world / "elements" / "torch-sconce.removed.json").read_text())
    assert removal["supersedes"]["sha256"]

    registry = json.loads((world / wp.PLACED_FILENAME).read_text())
    assert registry["elements"] == []
    manifest = json.loads((world / "world_manifest.json").read_text())
    assert all(e["slug"] != "torch-sconce" for e in manifest["elements"])
    assert manifest["counts"]["authored"] == 0

    # The slug is free again — placement after removal is legal.
    assert _place(http, "torch-sconce", payload).status_code == 200


def test_delete_captured_409_and_unknown_404(client) -> None:
    http, outputs = client
    _mk_world(_mk_job(outputs))
    response = http.delete(f"/api/splat/jobs/{JOB}/world/elements/crate")
    assert response.status_code == 409
    assert "only placed assets" in response.json()["detail"]
    assert http.delete(
        f"/api/splat/jobs/{JOB}/world/elements/ghost").status_code == 404
    assert http.delete(
        f"/api/splat/jobs/{JOB}/world/elements/shell").status_code == 404


# ── environment role (Lane 6, layer 1) ──────────────────────────────────────────


def test_place_environment_lands_and_422s_on_node_transforms(client) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))

    response = _place(http, "terrain-skirt", _authored_glb(), role="environment")
    assert response.status_code == 200
    manifest = json.loads((world / "world_manifest.json").read_text())
    rec = next(e for e in manifest["elements"] if e["slug"] == "terrain-skirt")
    assert rec["role"] == "environment"
    assert rec["provenance"] == "authored"
    assert rec["collision"] == {"ok": True, "strategy": "complex_as_simple",
                                "hulls": 0}
    assert rec["classification"] == [
        "authored environment geometry (placed.json)"]

    # Environment is held to the prop standard: the recorded AABB must match
    # the walked-on collider, so node transforms are a 422, not a warning.
    moved = _authored_glb(nodes=[{"mesh": 0, "translation": [1.0, 0.0, 0.0]}])
    response = _place(http, "moved-skirt", moved, role="environment")
    assert response.status_code == 422
    assert "bake_world_transform" in response.json()["detail"]
    assert not (world / "elements" / "moved-skirt.glb").exists()


def test_environment_origin_warning_relaxed(client) -> None:
    http, outputs = client
    _mk_world(_mk_job(outputs))
    # Shell half-diagonal ~5.85: centre (10, 0.25, 10) is ~14.1 away — beyond
    # the 1.5x static band, inside the 4x environment band.
    mid = _authored_glb(aabb_min=(9.75, 0.0, 9.75), aabb_max=(10.25, 0.5, 10.25))
    warns = _place(http, "mid-static", mid, role="static").json()["warnings"]
    assert any("half-diagonal" in w for w in warns)
    warns = _place(http, "mid-skirt", mid, role="environment").json()["warnings"]
    assert not any("half-diagonal" in w for w in warns)
    # Past 4x it warns for environment too.
    far = _authored_glb(aabb_min=(99.0, 0.0, 99.0), aabb_max=(99.5, 0.5, 99.5))
    warns = _place(http, "far-skirt", far, role="environment").json()["warnings"]
    assert any("half-diagonal" in w for w in warns)


def test_affordances_skip_environment_with_honest_reason(client) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    assert _place(http, "terrace", _authored_glb(),
                  role="environment").status_code == 200
    manifest = json.loads((world / "world_manifest.json").read_text())
    result = world_affordances.propose_affordances(manifest, set())
    skip = next(s for s in result["skipped"] if s["slug"] == "terrace")
    assert "walked on" in skip["reason"]
    assert all(p["record"]["slug"] != "terrace" for p in result["proposals"])


# ── replace-with-asset (triage lane, slice 2) ───────────────────────────────────


LIBRARY = Path(__file__).resolve().parents[2] / "assets" / "library"


def _mk_real_captured(world: Path, slug: str = "crate") -> None:
    """The pose source must be a real GLB — reuse a committed library asset."""
    (world / "elements" / f"{slug}.glb").write_bytes(
        (LIBRARY / "dungeon-torch.glb").read_bytes())


def test_replace_lands_scaled_at_the_captured_pose(client) -> None:
    import glb_check
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    _mk_real_captured(world)  # crate is a prop in the fixture manifest

    r = http.post(f"/api/splat/jobs/{JOB}/world/elements/crate/replace",
                  params={"asset": "graveyard-lantern-standing"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == "re-crate" and body["replaces"] == "crate"
    assert body["element"]["replaces"] == "crate"
    assert body["element"]["role"] == "prop"  # inherited from the captured

    # The replacement stands at the captured floor point at captured height.
    cap = glb_check.position_bounds(world / "elements" / "crate.glb")
    new = glb_check.position_bounds(world / "elements" / "re-crate.glb")
    cap_c = [(a + b) / 2 for a, b in zip(cap["aabb"]["min"], cap["aabb"]["max"])]
    new_c = [(a + b) / 2 for a, b in zip(new["aabb"]["min"], new["aabb"]["max"])]
    assert abs(new["aabb"]["min"][1] - cap["aabb"]["min"][1]) < 0.01  # floor
    assert abs(new_c[0] - cap_c[0]) < 0.01 and abs(new_c[2] - cap_c[2]) < 0.01
    assert abs(new["extent"][1] - cap["extent"][1]) < 0.02  # height-matched

    # One replacement per captured element.
    dup = http.post(f"/api/splat/jobs/{JOB}/world/elements/crate/replace",
                    params={"asset": "dungeon-torch"})
    assert dup.status_code == 409

    # Deleting the replacement frees the captured element again.
    assert http.delete(
        f"/api/splat/jobs/{JOB}/world/elements/re-crate").status_code == 200
    again = http.post(f"/api/splat/jobs/{JOB}/world/elements/crate/replace",
                      params={"asset": "dungeon-torch"})
    assert again.status_code == 200


def test_replace_refusals(client) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    _mk_real_captured(world)

    # Unknown asset names the library's actual contents.
    r = http.post(f"/api/splat/jobs/{JOB}/world/elements/crate/replace",
                  params={"asset": "excalibur"})
    assert r.status_code == 404

    # Unbuilt captured element (bench has no GLB) -> 409.
    r = http.post(f"/api/splat/jobs/{JOB}/world/elements/bench/replace",
                  params={"asset": "dungeon-torch"})
    assert r.status_code == 409

    # Authored elements are not replaceable.
    assert _place(http, "torch-a", _authored_glb()).status_code == 200
    r = http.post(f"/api/splat/jobs/{JOB}/world/elements/torch-a/replace",
                  params={"asset": "dungeon-torch"})
    assert r.status_code == 409


def test_asset_library_route_lists_catalogued_assets(client) -> None:
    http, _outputs = client
    r = http.get("/api/splat/assets/library")
    assert r.status_code == 200
    assets = {a["name"]: a for a in r.json()["assets"]}
    assert "dungeon-torch" in assets
    assert assets["dungeon-torch"]["catalog"]["license"] == "CC0-1.0"


def test_glb_transform_translate_scale_roundtrip() -> None:
    import glb_check
    import glb_transform
    src = (LIBRARY / "dungeon-torch.glb").read_bytes()
    before = glb_check.position_bounds_bytes if False else None  # noqa: F841
    out = glb_transform.translate_scale_glb(src, (2.0, 0.5, -1.0), 2.0)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".glb") as fh:
        fh.write(out)
        fh.flush()
        glb_check.validate_glb(Path(fh.name))
        b = glb_check.position_bounds(Path(fh.name))
    assert abs(b["aabb"]["min"][1] - 0.5) < 1e-3          # floor translated
    assert abs(b["extent"][1] - 2 * 1.04) < 0.03          # torch 1.04m doubled
    centre_x = (b["aabb"]["min"][0] + b["aabb"]["max"][0]) / 2
    assert abs(centre_x - 2.0) < 1e-3
