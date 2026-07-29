"""Restyle-bake routes: versioned landing, receipts, overlay clearing, the
all-or-nothing guarantee, revert, and provenance surfacing.

The mesh-env baker is faked at the subprocess seam (the same seam the route
uses), writing staged files exactly where the real CLI would — the real
baker's math is covered by test_restyle_bake_math.py and its mesh-env e2e.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_route  # noqa: E402
import opregistry  # noqa: E402
import restyle_bake_route  # noqa: E402
import splat_route  # noqa: E402

JOB = "splat_ba3e01"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(splat_route, "MESH_ENV_PYTHON", Path(sys.executable))
    opregistry.configure_storage(tmp_path / "ops")
    opregistry.init_db()
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    app.include_router(restyle_bake_route.router, prefix="/api/splat")
    yield TestClient(app), outputs
    opregistry.configure_storage(Path(opregistry.PROJECT_ROOT) / "data" / "ops")


def _mk_job(outputs: Path, job_id: str = JOB) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir),
        "status": "completed", "mode": "3d",
    }))
    return job_dir


def _mk_world(job_dir: Path, *, units: str = "meters") -> Path:
    world = job_dir / "_world"
    (world / "elements").mkdir(parents=True)
    (world / "world.json").write_text(json.dumps({
        "v": 1, "job_id": job_dir.name, "units": units,
        "shell": {"built": True, "glb": "shell.glb"},
        "elements": [{"slug": "crate", "built": True, "glb": "crate.glb"}],
    }))
    (world / "world_manifest.json").write_text(json.dumps({
        "v": 1, "job_id": job_dir.name,
        "elements": [{"slug": "crate", "label": "crate", "role": "prop"}],
        "shell": {"slug": "shell", "role": "static"},
    }))
    (world / "shell.glb").write_bytes(b"old-shell")
    (world / "shell_atlas.png").write_bytes(b"old-shell-atlas")
    (world / "elements" / "crate.glb").write_bytes(b"old-crate")
    (world / "elements" / "crate_atlas.png").write_bytes(b"old-crate-atlas")
    return world


RESTYLE = {
    "v": 1, "job_id": JOB,
    "elements": {"crate": {"material": "cobblestone", "material_scale": 1.5},
                 "shell": {"tint": "#ff3355"}},
    "lighting": {"preset": "dungeon", "intensity": 1.2},
}


def _write_restyle(world: Path, doc: dict | None = None) -> dict:
    doc = doc or json.loads(json.dumps(RESTYLE))
    (world / "restyle.json").write_text(json.dumps(doc))
    return doc


def _fake_bake(monkeypatch: pytest.MonkeyPatch, *, fail_slug: str | None = None):
    """Stand in for the mesh-env baker at the subprocess seam: stage exactly
    the files the real CLI stages, emit the same report shape."""
    async def fake(cmd: list[str]):
        staging = Path(cmd[cmd.index("--staging-dir") + 1])
        job_dir = Path(cmd[2])
        doc = json.loads((job_dir / "_world" / "restyle.json").read_text())
        elements, all_ok = [], True
        for slug in sorted(doc.get("elements") or {}):
            entry = doc["elements"][slug] or {}
            if not entry.get("tint") and not entry.get("material"):
                continue
            if slug == fail_slug:
                all_ok = False
                elements.append({"slug": slug, "ok": False, "error": "boom"})
                continue
            staging.mkdir(parents=True, exist_ok=True)
            (staging / f"{slug}.glb").write_bytes(f"baked-{slug}".encode())
            (staging / f"{slug}_atlas.png").write_bytes(
                f"baked-{slug}-atlas".encode())
            elements.append({
                "slug": slug, "ok": True,
                "verbs": (["material"] if entry.get("material") else [])
                         + (["tint"] if entry.get("tint") else []),
                "material": entry.get("material"),
                "material_scale": (entry.get("material_scale", 1.0)
                                   if entry.get("material") else None),
                "tint": entry.get("tint"),
                "texels_per_unit": 341.3, "atlas_size": 4,
                "covered_frac": 0.9, "faces": 2,
                "roughness": 0.85, "metallic": 0.0,
                "staged_glb": f"{slug}.glb",
                "staged_atlas": f"{slug}_atlas.png",
                "seconds": 0.1,
            })
        report = json.dumps({"ok": all_ok, "units_per_metre": 1.0,
                             "tile_size": 512, "elements": elements})
        return (0 if all_ok else 1), report.encode(), b""
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake)


def _bake(http, job_id: str = JOB, upm: float = 1.0):
    return http.post(f"/api/splat/jobs/{job_id}/world/restyle/bake",
                     json={"units_per_metre": upm})


# ── refusals ────────────────────────────────────────────────────────────────────


def test_bake_409_without_world(client) -> None:
    http, outputs = client
    _mk_job(outputs)
    response = _bake(http)
    assert response.status_code == 409
    assert "solidify" in response.json()["detail"]


def test_bake_409_when_only_lighting_is_restyled(client) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    _write_restyle(world, {"v": 1, "job_id": JOB, "elements": {},
                           "lighting": {"preset": "dungeon", "intensity": 1}})
    response = _bake(http)
    assert response.status_code == 409
    assert "Nothing to bake" in response.json()["detail"]
    assert "never baked" in response.json()["detail"]


def test_bake_400_names_missing_element_glbs(client) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    doc = json.loads(json.dumps(RESTYLE))
    doc["elements"]["ghost"] = {"tint": "#112233"}
    _write_restyle(world, doc)
    response = _bake(http)
    assert response.status_code == 400
    assert "ghost" in response.json()["detail"]


def test_bake_409_while_mesh_lock_held(client) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    _write_restyle(world)
    lock = splat_route._mesh_export_lock(JOB)
    lock._locked = True  # noqa: SLF001 - simulate a running build
    try:
        assert _bake(http).status_code == 409
    finally:
        lock._locked = False  # noqa: SLF001


# ── the landing ─────────────────────────────────────────────────────────────────


def test_bake_lands_versions_markers_receipt_and_clears_overlay(
    client, monkeypatch
) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    _write_restyle(world)
    _fake_bake(monkeypatch)

    response = _bake(http)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True and body["lighting_baked"] is False
    assert body["uncalibrated"] is False
    assert {e["slug"] for e in body["elements"]} == {"crate", "shell"}

    # Live files replaced; priors versioned as GLB+atlas PAIRS at one seq.
    assert (world / "elements" / "crate.glb").read_bytes() == b"baked-crate"
    assert (world / "elements" / "crate_atlas.png").read_bytes() == b"baked-crate-atlas"
    assert (world / "versions" / "crate-v0001.glb").read_bytes() == b"old-crate"
    assert (world / "versions" / "crate-v0001_atlas.png").read_bytes() == b"old-crate-atlas"
    assert (world / "shell.glb").read_bytes() == b"baked-shell"
    assert (world / "versions" / "shell-v0001.glb").read_bytes() == b"old-shell"

    # Markers: provenance restyled, atlas refreshed (opposite of polish).
    marker = json.loads((world / "elements" / "crate.restyle.json").read_text())
    assert marker["provenance"] == "restyled"
    assert marker["atlas_updated"] is True
    assert marker["applied"] == {"material": "cobblestone", "material_scale": 1.5}
    assert marker["supersedes"]["sha256"]
    shell_marker = json.loads((world / "shell.restyle.json").read_text())
    assert shell_marker["applied"] == {"tint": "#ff3355"}

    # Bake receipt archives the FULL pre-bake overlay.
    receipt = json.loads((world / "restyle_bake.json").read_text())
    assert receipt["archived_restyle"]["elements"] == RESTYLE["elements"]
    assert receipt["lighting_baked"] is False

    # Overlay cleared, lighting mood preserved.
    stored = json.loads((world / "restyle.json").read_text())
    assert stored["elements"] == {}
    assert stored["lighting"] == {"preset": "dungeon", "intensity": 1.2}
    assert body["restyle"]["restyle"]["elements"] == {}

    # No staging debris.
    assert not list(world.glob(".building-*"))

    # The manifest surfaces the marker (the walker's bakedLook signal).
    manifest = http.get(f"/api/splat/jobs/{JOB}/world/manifest").json()
    element = next(e for e in manifest["elements"] if e["slug"] == "crate")
    assert element["restyled"]["atlas_updated"] is True
    assert manifest["shell"]["restyled"]["applied"] == {"tint": "#ff3355"}

    # A durable operation record.
    ops = opregistry.list_ops(job_id=JOB, kind="restyle_bake")
    assert ops and ops[0]["status"] == opregistry.SUCCEEDED


def test_bake_all_or_nothing_on_element_failure(client, monkeypatch) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    original = _write_restyle(world)
    _fake_bake(monkeypatch, fail_slug="crate")

    response = _bake(http)
    assert response.status_code == 500
    assert "crate: boom" in response.json()["detail"]

    # The tree is byte-identical: nothing landed, nothing versioned.
    assert (world / "elements" / "crate.glb").read_bytes() == b"old-crate"
    assert (world / "shell.glb").read_bytes() == b"old-shell"
    assert not (world / "versions").exists()
    assert not (world / "elements" / "crate.restyle.json").exists()
    assert not (world / "restyle_bake.json").exists()
    assert json.loads((world / "restyle.json").read_text()) == original
    assert not list(world.glob(".building-*"))

    ops = opregistry.list_ops(job_id=JOB, kind="restyle_bake")
    assert ops and ops[0]["status"] == opregistry.FAILED


# ── revert ──────────────────────────────────────────────────────────────────────


def test_revert_restores_bytes_markers_and_doc(client, monkeypatch) -> None:
    http, outputs = client
    world = _mk_world(_mk_job(outputs))
    original = _write_restyle(world)
    _fake_bake(monkeypatch)
    assert _bake(http).status_code == 200

    response = http.post(f"/api/splat/jobs/{JOB}/world/restyle/bake/revert")
    assert response.status_code == 200, response.text
    assert set(response.json()["elements"]) == {"crate", "shell"}

    # Exact pre-bake bytes back, for GLB and sidecar alike.
    assert (world / "elements" / "crate.glb").read_bytes() == b"old-crate"
    assert (world / "elements" / "crate_atlas.png").read_bytes() == b"old-crate-atlas"
    assert (world / "shell.glb").read_bytes() == b"old-shell"
    assert (world / "shell_atlas.png").read_bytes() == b"old-shell-atlas"

    # Markers gone; overlay document restored; receipt renamed, not deleted.
    assert not (world / "elements" / "crate.restyle.json").exists()
    assert not (world / "shell.restyle.json").exists()
    assert json.loads((world / "restyle.json").read_text()) == original
    assert not (world / "restyle_bake.json").exists()
    assert list(world.glob("restyle_bake.reverted-*.json"))

    # The baked state was itself versioned first — revert is un-revertable
    # only by re-baking, never by data loss.
    assert (world / "versions" / "crate-v0002.glb").read_bytes() == b"baked-crate"

    ops = opregistry.list_ops(job_id=JOB, kind="restyle_bake_revert")
    assert ops and ops[0]["status"] == opregistry.SUCCEEDED


def test_revert_404_without_a_bake(client) -> None:
    http, outputs = client
    _mk_world(_mk_job(outputs))
    response = http.post(f"/api/splat/jobs/{JOB}/world/restyle/bake/revert")
    assert response.status_code == 404


# ── export provenance ───────────────────────────────────────────────────────────


def test_world_bundle_provenance_restyled_and_newer_marker_wins(
    client, monkeypatch
) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    world = _mk_world(job_dir)
    _write_restyle(world)
    _fake_bake(monkeypatch)
    assert _bake(http).status_code == 200

    by_rel = {rel: (prov, role) for _src, rel, prov, role, _extra
              in export_route._world_bundle_candidates(job_dir)}
    assert by_rel["World/Elements/crate.glb"][0] == "restyled"
    # The bake refreshed the sidecar, so the atlas is honestly restyled too.
    assert by_rel["World/Elements/crate_atlas.png"][0] == "restyled"
    assert by_rel["World/shell.glb"][0] == "restyled"

    # A LATER polish supersedes the bake — and its atlas goes stale again.
    (world / "elements" / "crate.polish.json").write_text(json.dumps({
        "uploaded_at": "9999-01-01T00:00:00Z", "atlas_superseded": True}))
    by_rel = {rel: prov for _src, rel, prov, _role, _extra
              in export_route._world_bundle_candidates(job_dir)}
    assert by_rel["World/Elements/crate.glb"] == "polished"
    assert by_rel["World/Elements/crate_atlas.png"] == "captured-derived"
