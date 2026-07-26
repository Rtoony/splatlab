"""World-lane routes (_world/): the merged viewer manifest and by-name artifact
serving.

Fixtures mirror the real tree produced by scene_solidify.py + world_collision.py
on splat_aea04ab3 (shell + 5 elements, 8 hulls per prop, uncalibrated) — reduced
in size but not in shape, so the merge, the URL resolution and the calibration
statement are tested against the layout that actually lands on disk.

The security case is the point of the second route: `name` is free-form, so
traversal, absolute paths and symlink escapes must all be indistinguishable
404s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402

# world.json — scene_solidify.py's receipt: geometry facts, no roles.
WORLD_REPORT = {
    "v": 1,
    "job_id": "splat_0b0001",
    "units": "scene-units (uncalibrated)",
    "meters_per_unit": None,
    "up_axis": "Y",
    "counts": {"instances": 2, "props_built": 2, "props_failed": 0, "shell_built": True},
    "shell": {"role": "shell", "built": True, "glb": "shell.glb", "faces": 59999,
              "extent": [7.548, 2.451, 8.825], "texture": True, "seconds": 25.9},
    "elements": [
        {"slug": "red-bicycle", "label": "red bicycle", "role": "prop", "n_members": 6715,
         "geometry_source": "gaussians", "built": True, "glb": "red-bicycle.glb",
         "faces": 8000, "extent": [1.091, 1.043, 1.75], "texture": True},
        {"slug": "cardboard-box", "label": "cardboard box", "role": "prop", "n_members": 1399,
         "geometry_source": "gaussians", "built": True, "glb": "cardboard-box.glb",
         "faces": 7278, "extent": [0.568, 0.585, 0.583], "texture": True},
    ],
    "seconds": 48.3,
}

# world_manifest.json — world_collision.py's verdict: roles + hulls.
WORLD_MANIFEST = {
    "v": 1,
    "job_id": "splat_0b0001",
    "units": "scene-units (uncalibrated)",
    "meters_per_unit": None,
    "scene_extent": [7.548, 2.451, 8.825],
    "thresholds": {"prop_max_frac": 0.16, "prop_max_metres": None, "max_hulls": 8},
    "coacd_available": True,
    "shell": {"slug": "shell", "role": "static", "glb": "shell.glb", "faces": 59999,
              "extent": [7.548, 2.451, 8.825],
              "collision": {"ok": True, "strategy": "complex_as_simple", "hulls": 0},
              "classification": ["the shell is the environment; always static"]},
    "elements": [
        # Too big for a prop -> static, complex-as-simple, no hull files.
        {"slug": "red-bicycle", "label": "red bicycle", "role": "static",
         "extent": [1.091, 1.043, 1.75], "faces": 8000,
         "classification": ["longest 1.75 = 19.8% of scene (8.82); prop threshold 16%"],
         "glb": "red-bicycle.glb",
         "collision": {"ok": True, "strategy": "complex_as_simple", "hulls": 0}},
        {"slug": "cardboard-box", "label": "cardboard box", "role": "prop",
         "extent": [0.568, 0.585, 0.583], "faces": 7278,
         "classification": ["longest 0.58 = 6.6% of scene (8.82); prop threshold 16%"],
         "glb": "cardboard-box.glb",
         "collision": {"ok": True, "hulls": 2,
                       "files": ["UCX_cardboard-box_00.glb", "UCX_cardboard-box_01.glb"],
                       "hull_faces_total": 3090, "capped": True,
                       "surface_coverage": 1.0, "coverage_ok": True, "seconds": 21.8}},
    ],
    "counts": {"props": 1, "static": 1, "unbuilt": 0, "hulls_total": 2},
    "seconds": 87.7,
}

ELEMENT_REPORT = {"verts": 6179, "faces": 8000, "units": "scene-units (uncalibrated)",
                  "meters_per_unit": None, "up_axis": "Y", "extent": [1.091, 1.043, 1.75],
                  "texture": {"baked": True, "size": 1024, "atlas_png": "red-bicycle_atlas.png"},
                  "glb_bytes": 989276}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_0b0001", **meta) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d", **meta,
    }))
    return job_dir


def _mk_world(job_dir: Path, *, manifest: bool = True, world: dict | None = None) -> Path:
    """Build the _world/ tree scene_solidify.py writes (+ collision, optionally)."""
    world_dir = job_dir / splat_route.WORLD_DIRNAME
    (world_dir / "elements").mkdir(parents=True)
    (world_dir / "collision").mkdir()
    (world_dir / "world.json").write_text(json.dumps(world or WORLD_REPORT))
    (world_dir / "shell.glb").write_bytes(b"glTF-shell")
    (world_dir / "shell_atlas.png").write_bytes(b"\x89PNG-shell")
    (world_dir / "shell.json").write_text(json.dumps({"faces": 59999}))
    for slug in ("red-bicycle", "cardboard-box"):
        (world_dir / "elements" / f"{slug}.glb").write_bytes(b"glTF-" + slug.encode())
        (world_dir / "elements" / f"{slug}_atlas.png").write_bytes(b"\x89PNG-" + slug.encode())
        (world_dir / "elements" / f"{slug}.json").write_text(json.dumps(ELEMENT_REPORT))
    for i in range(2):
        (world_dir / "collision" / f"UCX_cardboard-box_{i:02d}.glb").write_bytes(b"glTF-hull")
    if manifest:
        (world_dir / "world_manifest.json").write_text(json.dumps(WORLD_MANIFEST))
    return world_dir


# ── manifest: absence ───────────────────────────────────────────────────────────


def test_world_manifest_unknown_job_404(client):
    http, _outputs = client
    assert http.get("/api/splat/jobs/splat_0b0001/world/manifest").status_code == 404


def test_world_manifest_404_before_solidify(client):
    """A completed scene with no _world/ is the common case — the 404 has to say
    which stage to run, not just 'not found'."""
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0b0001/world/manifest")
    assert r.status_code == 404
    assert "solidify" in r.json()["detail"]


def test_world_manifest_404_when_dir_exists_but_report_does_not(client):
    """An aborted solidify can leave the dir behind. No world.json = no world."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    (job_dir / splat_route.WORLD_DIRNAME / "elements").mkdir(parents=True)
    r = http.get("/api/splat/jobs/splat_0b0001/world/manifest")
    assert r.status_code == 404
    assert "solidify" in r.json()["detail"]


# ── manifest: shape ─────────────────────────────────────────────────────────────


def test_world_manifest_success_shape(client):
    http, outputs = client
    _mk_world(_mk_job(outputs))
    r = http.get("/api/splat/jobs/splat_0b0001/world/manifest")
    assert r.status_code == 200
    body = r.json()

    assert body["job_id"] == "splat_0b0001"
    assert body["up_axis"] == "Y"
    assert body["collision_built"] is True
    assert body["coacd_available"] is True
    assert body["scene_extent"] == [7.548, 2.451, 8.825]
    assert body["counts"]["props"] == 1 and body["counts"]["hulls_total"] == 2
    assert body["reports"]["world"].endswith("world/file?name=world.json")
    assert body["reports"]["manifest"].endswith("world/file?name=world_manifest.json")

    shell = body["shell"]
    assert shell["role"] == "static" and shell["faces"] == 59999
    assert set(shell["files"]) == {"glb", "atlas", "report"}
    assert shell["files"]["glb"].endswith("world/file?name=shell.glb")
    assert shell["collision"]["strategy"] == "complex_as_simple"

    elements = {e["slug"]: e for e in body["elements"]}
    assert set(elements) == {"red-bicycle", "cardboard-box"}
    bike = elements["red-bicycle"]
    assert bike["label"] == "red bicycle"
    assert bike["faces"] == 8000 and bike["extent"] == [1.091, 1.043, 1.75]
    assert bike["built"] is True and bike["textured"] is True
    assert bike["n_members"] == 6715 and bike["geometry_source"] == "gaussians"
    # role comes from world_manifest.json, NOT world.json (which still says "prop")
    assert bike["role"] == "static"
    assert "19.8% of scene" in bike["classification"][0]
    assert bike["files"]["glb"].endswith("world/file?name=elements/red-bicycle.glb")
    assert bike["files"]["atlas"].endswith("world/file?name=elements/red-bicycle_atlas.png")
    assert bike["collision"]["hull_urls"] == []

    box = elements["cardboard-box"]
    assert box["role"] == "prop" and box["collision"]["hulls"] == 2
    assert [u.split("name=")[1] for u in box["collision"]["hull_urls"]] == [
        "collision/UCX_cardboard-box_00.glb", "collision/UCX_cardboard-box_01.glb"]
    assert "missing_hulls" not in box["collision"]
    # every advertised URL actually serves
    for element in body["elements"]:
        for url in list(element["files"].values()) + element["collision"]["hull_urls"]:
            assert http.get(url).status_code == 200, url


def test_world_manifest_before_collision_stage(client):
    """Solidify ran, world_collision.py did not: elements are ungraded rather
    than wrongly labelled, and the flag says so."""
    http, outputs = client
    _mk_world(_mk_job(outputs), manifest=False)
    body = http.get("/api/splat/jobs/splat_0b0001/world/manifest").json()
    assert body["collision_built"] is False
    assert body["reports"] == {"world": body["reports"]["world"]}
    assert all(e["role"] is None and e["collision"] is None for e in body["elements"])
    assert all(e["classification"] == [] for e in body["elements"])
    # geometry is still fully usable without grading
    assert body["elements"][0]["files"]["glb"].endswith("elements/red-bicycle.glb")
    # the shell is the environment; static is the only sane pre-grade default
    assert body["shell"]["role"] == "static"
    assert body["scene_extent"] == [7.548, 2.451, 8.825]


def test_world_manifest_omits_artifacts_that_were_not_written(client):
    """A failed texture bake leaves the .glb but no atlas — files{} must not
    advertise a URL that 404s."""
    http, outputs = client
    world_dir = _mk_world(_mk_job(outputs))
    (world_dir / "elements" / "red-bicycle_atlas.png").unlink()
    elements = {e["slug"]: e for e in
                http.get("/api/splat/jobs/splat_0b0001/world/manifest").json()["elements"]}
    assert set(elements["red-bicycle"]["files"]) == {"glb", "report"}
    assert set(elements["cardboard-box"]["files"]) == {"glb", "atlas", "report"}


def test_world_manifest_flags_hulls_the_manifest_names_but_disk_lost(client):
    """world_collision.py rewrites collision/ per run, so a stale manifest can
    name hulls that no longer exist. Report them, never serve dead URLs."""
    http, outputs = client
    world_dir = _mk_world(_mk_job(outputs))
    (world_dir / "collision" / "UCX_cardboard-box_01.glb").unlink()
    elements = {e["slug"]: e for e in
                http.get("/api/splat/jobs/splat_0b0001/world/manifest").json()["elements"]}
    collision = elements["cardboard-box"]["collision"]
    assert len(collision["hull_urls"]) == 1
    assert collision["missing_hulls"] == ["UCX_cardboard-box_01.glb"]


def test_world_manifest_skips_unbuilt_element_without_geometry(client):
    """An element that never isolated is listed honestly (built False, reason,
    empty files) rather than dropped or faked."""
    http, outputs = client
    world = json.loads(json.dumps(WORLD_REPORT))
    world["elements"].append({"slug": "ghost-chair", "label": "ghost chair", "role": "prop",
                              "built": False, "reason": "not isolated yet (no object.ply)"})
    _mk_world(_mk_job(outputs), world=world)
    elements = {e["slug"]: e for e in
                http.get("/api/splat/jobs/splat_0b0001/world/manifest").json()["elements"]}
    ghost = elements["ghost-chair"]
    assert ghost["built"] is False
    assert ghost["files"] == {}
    assert ghost["reason"] == "not isolated yet (no object.ply)"


def test_world_manifest_drops_path_unsafe_slug(client):
    """A slug becomes a path segment in every URL below it — one that could climb
    out is not listable."""
    http, outputs = client
    world = json.loads(json.dumps(WORLD_REPORT))
    world["elements"].append({"slug": "../../../etc/passwd", "built": True, "faces": 1})
    _mk_world(_mk_job(outputs), world=world)
    slugs = [e["slug"] for e in
             http.get("/api/splat/jobs/splat_0b0001/world/manifest").json()["elements"]]
    assert slugs == ["red-bicycle", "cardboard-box"]


# ── manifest: calibration ───────────────────────────────────────────────────────


def test_world_manifest_says_scale_is_unknown(client):
    """The default: nothing in the pipeline knows metres. The viewer must be told
    outright, not left to infer it from a units string."""
    http, outputs = client
    _mk_world(_mk_job(outputs))
    body = http.get("/api/splat/jobs/splat_0b0001/world/manifest").json()
    assert body["uncalibrated"] is True
    assert body["meters_per_unit"] is None
    assert body["units"] == "scene-units (uncalibrated)"
    calibration = body["calibration"]
    assert calibration["job_meters_per_unit"] is None
    assert calibration["stale"] is False
    assert "scale is unknown" in calibration["detail"]


def test_world_manifest_flags_calibration_added_after_solidify(client):
    """POST /scale updates the job; nothing re-stamps GLBs already on disk. The
    world is still uncalibrated AND now disagrees with the job."""
    http, outputs = client
    _mk_world(_mk_job(outputs, meters_per_unit=2.5))
    body = http.get("/api/splat/jobs/splat_0b0001/world/manifest").json()
    calibration = body["calibration"]
    assert body["uncalibrated"] is True
    assert calibration["job_meters_per_unit"] == 2.5
    assert calibration["stale"] is True
    assert "Re-run solidify" in calibration["detail"]


def test_world_manifest_calibrated_world_is_not_stale(client):
    http, outputs = client
    world = json.loads(json.dumps(WORLD_REPORT))
    world["meters_per_unit"] = 2.5
    world["units"] = "meters"
    _mk_world(_mk_job(outputs, meters_per_unit=2.5), world=world)
    body = http.get("/api/splat/jobs/splat_0b0001/world/manifest").json()
    assert body["uncalibrated"] is False
    assert body["meters_per_unit"] == 2.5
    assert body["calibration"]["stale"] is False
    assert body["calibration"]["detail"] == "World carries the job's current calibration."


def test_world_manifest_recalibration_marks_the_world_stale(client):
    http, outputs = client
    world = json.loads(json.dumps(WORLD_REPORT))
    world["meters_per_unit"] = 2.5
    _mk_world(_mk_job(outputs, meters_per_unit=3.0), world=world)
    calibration = http.get(
        "/api/splat/jobs/splat_0b0001/world/manifest").json()["calibration"]
    assert calibration["meters_per_unit"] == 2.5
    assert calibration["job_meters_per_unit"] == 3.0
    assert calibration["stale"] is True


# ── file serving ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name, media, body", [
    ("shell.glb", "model/gltf-binary", b"glTF-shell"),
    ("shell_atlas.png", "image/png", b"\x89PNG-shell"),
    ("elements/red-bicycle.glb", "model/gltf-binary", b"glTF-red-bicycle"),
    ("elements/red-bicycle_atlas.png", "image/png", b"\x89PNG-red-bicycle"),
    ("collision/UCX_cardboard-box_00.glb", "model/gltf-binary", b"glTF-hull"),
])
def test_world_file_serves_binary_artifacts(client, name, media, body):
    http, outputs = client
    _mk_world(_mk_job(outputs))
    r = http.get(f"/api/splat/jobs/splat_0b0001/world/file?name={name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == media
    assert r.content == body


def test_world_file_serves_json_reports(client):
    http, outputs = client
    _mk_world(_mk_job(outputs))
    for name in ("world.json", "world_manifest.json", "elements/red-bicycle.json"):
        r = http.get(f"/api/splat/jobs/splat_0b0001/world/file?name={name}")
        assert r.status_code == 200, name
        assert r.headers["content-type"] == "application/json"
        assert isinstance(r.json(), dict)
    assert http.get(
        "/api/splat/jobs/splat_0b0001/world/file?name=world.json").json()["v"] == 1


def test_world_file_unknown_name_404(client):
    http, outputs = client
    _mk_world(_mk_job(outputs))
    assert http.get(
        "/api/splat/jobs/splat_0b0001/world/file?name=elements/nope.glb").status_code == 404


def test_world_file_requires_name(client):
    http, outputs = client
    _mk_world(_mk_job(outputs))
    assert http.get("/api/splat/jobs/splat_0b0001/world/file").status_code == 422


def test_world_file_before_solidify_404(client):
    http, outputs = client
    _mk_job(outputs)
    assert http.get(
        "/api/splat/jobs/splat_0b0001/world/file?name=shell.glb").status_code == 404


def test_world_file_unknown_job_404(client):
    http, outputs = client
    _mk_world(_mk_job(outputs))
    assert http.get(
        "/api/splat/jobs/splat_ffffff/world/file?name=shell.glb").status_code == 404
    assert http.get(
        "/api/splat/jobs/..%2F..%2Fetc/world/file?name=shell.glb").status_code == 404


# ── file serving: containment ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "../meta.json",                       # one level out of _world/
    "../../splat_other/meta.json",        # into a sibling job
    "elements/../../meta.json",           # climbs after a legitimate segment
    "/etc/passwd",                        # absolute
    "//etc/passwd",
    "elements/../../../../../../etc/passwd",
    ".",
    "..",
])
def test_world_file_path_traversal_404(client, name):
    """`name` is free-form, so this is the route's whole security surface: every
    escape must 404 exactly like a missing file, leaking nothing."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    _mk_world(job_dir)
    _mk_world(_mk_job(outputs, job_id="splat_0b0002"))
    assert (job_dir / "meta.json").is_file()   # the file traversal is aiming at
    r = http.get("/api/splat/jobs/splat_0b0001/world/file", params={"name": name})
    assert r.status_code == 404, f"{name!r} was served"
    assert r.json()["detail"] == "World artifact not found"


def test_world_file_url_encoded_traversal_404(client):
    """Percent-encoded separators decode before the route sees them, so the guard
    must reject the decoded form, not the raw query string."""
    http, outputs = client
    _mk_world(_mk_job(outputs))
    for encoded in ("..%2Fmeta.json", "%2Fetc%2Fpasswd", "elements%2F..%2F..%2Fmeta.json"):
        r = http.get(f"/api/splat/jobs/splat_0b0001/world/file?name={encoded}")
        assert r.status_code == 404, encoded


def test_world_file_symlink_escape_404(client, tmp_path):
    """A symlink planted inside _world/ resolves to its real target, which then
    fails the containment check — the reason resolve() runs on both sides."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    world_dir = _mk_world(job_dir)
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")

    (world_dir / "leak.json").symlink_to(secret)                 # file link
    (world_dir / "elements" / "out.glb").symlink_to(job_dir / "meta.json")
    (world_dir / "escape").symlink_to(tmp_path)                  # dir link

    for name in ("leak.json", "elements/out.glb", "escape/id_rsa"):
        r = http.get("/api/splat/jobs/splat_0b0001/world/file", params={"name": name})
        assert r.status_code == 404, f"{name!r} was served"
    assert secret.read_text() == "PRIVATE KEY"


def test_world_file_symlink_inside_world_still_serves(client):
    """Containment, not link-phobia: a link that stays inside _world/ is fine."""
    http, outputs = client
    world_dir = _mk_world(_mk_job(outputs))
    (world_dir / "alias.glb").symlink_to(world_dir / "shell.glb")
    r = http.get("/api/splat/jobs/splat_0b0001/world/file?name=alias.glb")
    assert r.status_code == 200
    assert r.content == b"glTF-shell"


def test_world_file_scopes_to_its_own_job(client):
    """Two solidified jobs: neither can be reached through the other's route."""
    http, outputs = client
    _mk_world(_mk_job(outputs))
    _mk_world(_mk_job(outputs, job_id="splat_0b0002"))
    assert http.get(
        "/api/splat/jobs/splat_0b0002/world/file?name=shell.glb").status_code == 200
    assert http.get("/api/splat/jobs/splat_0b0001/world/file",
                    params={"name": "../../splat_0b0002/_world/shell.glb"}).status_code == 404


# ── presence flag on the existing job surface ───────────────────────────────────


def test_status_reports_world_presence(client, monkeypatch):
    """/status is where the frontend already learns what a job has; the world
    flag rides the same payload rather than needing its own poll."""
    http, outputs = client
    monkeypatch.setattr(splat_route, "_engine_availability", lambda: {})
    job_dir = _mk_job(outputs)

    job = http.get("/api/splat/status").json()["jobs"][0]
    assert job["world_available"] is False
    assert job["world_manifest_url"] is None

    _mk_world(job_dir)
    job = http.get("/api/splat/status").json()["jobs"][0]
    assert job["world_available"] is True
    assert job["world_manifest_url"] == "/api/splat/jobs/splat_0b0001/world/manifest"
    assert http.get(job["world_manifest_url"]).status_code == 200
