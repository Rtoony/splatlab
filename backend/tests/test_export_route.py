"""Portable export, collision, and Unreal-bundle API tests."""

from __future__ import annotations

import json
import os
import subprocess
import struct
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_route  # noqa: E402
import splat_route  # noqa: E402


def _write_ply(path: Path, count: int = 2) -> None:
    properties = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {count}",
            *(f"property float {name}" for name in properties),
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for index in range(count):
            row = [
                float(index),
                float(-index),
                1.0,
                0.0,
                0.0,
                0.0,
                4.0,
                -3.0,
                -3.0,
                -3.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            handle.write(struct.pack("<14f", *row))


def _make_job(outputs: Path, job_id: str = "splat_e70001", count: int = 2) -> Path:
    job_dir = outputs / job_id
    preview = job_dir / "_preview"
    preview.mkdir(parents=True)
    _write_ply(preview / "splat.ply", count=count)
    (preview / "thumb.webp").write_bytes(b"thumb")
    (job_dir / "meta.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "output_dir": str(job_dir),
                "status": "completed",
                "mode": "3d",
                "meters_per_unit": 2.0,
                "geo": {"lat": 38.1, "lon": -122.2, "heading_deg": 5.0},
            }
        )
    )
    return job_dir


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(
        splat_route, "_splat_transform_path", lambda: "/fake/splat-transform"
    )
    state = {"calls": [], "fail": False}

    async def fake_host(command: list[str]) -> tuple[int, str]:
        state["calls"].append(command)
        if state["fail"]:
            return 17, "deliberate conversion failure"
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.name == "lod-meta.json":
            output.write_text('{"chunks": ["chunk-0.sog"]}')
            (output.parent / "chunk-0.sog").write_bytes(b"chunk")
        else:
            output.write_bytes(f"artifact:{output.suffix}".encode())
        return 0, "ok"

    monkeypatch.setattr(export_route, "_host_conversion", fake_host)

    async def fake_host_admission(*, operation):
        return await operation()

    monkeypatch.setattr(
        export_route.gpu_arbiter,
        "run_host_operation",
        fake_host_admission,
    )
    app = FastAPI()
    app.include_router(export_route.router, prefix="/api/splat")
    return TestClient(app), outputs, state


def test_build_formats_manifest_and_cache(client) -> None:
    http, outputs, state = client
    job_dir = _make_job(outputs)
    body = {"formats": ["spz", "sog", "gltf"]}

    response = http.post("/api/splat/jobs/splat_e70001/exports", json=body)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["result"]["built"] == ["spz", "sog", "gltf"]
    assert payload["source"]["gaussian_count"] == 2
    assert payload["source"]["bounds_scene"]["extent"] == [1.0, 1.0, 0.0]
    assert payload["coordinate_system"]["source_frame"]["meters_per_unit"] == 2.0
    assert (job_dir / "_exports" / "scene.spz").is_file()
    assert (job_dir / "_exports" / "scene.sog").is_file()
    assert (job_dir / "_exports" / "scene.glb").is_file()

    manifest = json.loads((job_dir / "_exports" / "manifest.json").read_text())
    assert manifest["schema"] == "dev.splatlab.exports/v1"
    assert len(manifest["artifacts"]["spz"]["files"][0]["sha256"]) == 64
    assert http.get("/api/splat/jobs/splat_e70001/exports/file/spz").status_code == 200

    cached = http.post("/api/splat/jobs/splat_e70001/exports", json=body)
    assert cached.status_code == 200
    assert cached.json()["result"]["cached"] == ["spz", "sog", "gltf"]
    assert len(state["calls"]) == 3


def test_streamed_sog_command_tags_lod_level() -> None:
    """Untagged input dies in splat-transform's writeLod ("Missing lod
    assignment") — the streamed-sog command must inject `-l 0` after the
    input so every gaussian carries an LOD tag. Proven live 2026-07-25."""
    import export_route

    cmd = export_route._conversion_command(
        "splat-transform",
        Path("/in/splat.ply"),
        Path("/out/lod-meta.json"),
        "streamed-sog",
        export_route.ExportRequest(),
    )
    src = cmd.index("/in/splat.ply")
    assert cmd[src + 1 : src + 3] == ["-l", "0"], cmd
    assert cmd[-1] == "/out/lod-meta.json"


def test_streamed_sog_skips_small_scene_unless_forced(client) -> None:
    http, outputs, state = client
    job_dir = _make_job(outputs)

    skipped = http.post(
        "/api/splat/jobs/splat_e70001/exports",
        json={"formats": ["streamed-sog"]},
    )
    assert skipped.status_code == 200
    assert skipped.json()["result"]["skipped"] == ["streamed-sog"]
    assert state["calls"] == []

    forced = http.post(
        "/api/splat/jobs/splat_e70001/exports",
        json={"formats": ["streamed-sog"], "force_streamed_sog": True},
    )
    assert forced.status_code == 200, forced.text
    assert (job_dir / "_exports" / "streamed-sog" / "lod-meta.json").is_file()
    assert (job_dir / "_exports" / "streamed-sog" / "chunk-0.sog").is_file()
    assert forced.json()["artifacts"]["streamed-sog"]["url"].endswith(
        "/exports/streamed-sog/lod-meta.json"
    )
    assert (
        http.get(
            "/api/splat/jobs/splat_e70001/exports/streamed-sog/chunk-0.sog"
        ).content
        == b"chunk"
    )
    assert (
        http.get(
            "/api/splat/jobs/splat_e70001/exports/streamed-sog/../manifest.json"
        ).status_code
        == 404
    )


def test_source_change_makes_exports_unservable(client) -> None:
    http, outputs, _state = client
    job_dir = _make_job(outputs)
    assert (
        http.post(
            "/api/splat/jobs/splat_e70001/exports", json={"formats": ["spz"]}
        ).status_code
        == 200
    )

    source = job_dir / "_preview" / "splat.ply"
    before = source.stat()
    _write_ply(source, count=3)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))

    status = http.get("/api/splat/jobs/splat_e70001/exports")
    assert status.status_code == 200
    assert status.json()["status"] == "stale"
    served = http.get("/api/splat/jobs/splat_e70001/exports/file/spz")
    assert served.status_code == 409


def test_failed_overwrite_preserves_previous_artifact(client) -> None:
    http, outputs, state = client
    job_dir = _make_job(outputs)
    url = "/api/splat/jobs/splat_e70001/exports"
    assert http.post(url, json={"formats": ["spz"]}).status_code == 200
    artifact = job_dir / "_exports" / "scene.spz"
    original = artifact.read_bytes()

    state["fail"] = True
    failed = http.post(url, json={"formats": ["spz"], "overwrite": True})
    assert failed.status_code == 500
    assert artifact.read_bytes() == original
    manifest = json.loads((job_dir / "_exports" / "manifest.json").read_text())
    assert manifest["artifacts"]["spz"]["status"] == "ready"
    assert "deliberate conversion failure" in manifest["artifacts"]["spz"]["last_error"]


def test_collision_command_has_one_typed_mode() -> None:
    request = export_route.CollisionRequest(
        mode="interior", seed_position=(1, 2, 3), carve=True
    )
    command = export_route._collision_command(
        "/tool", Path("source.ply"), Path("scene.voxel.json"), request
    )
    assert "--voxel-external-fill" in command
    assert "--voxel-floor-fill" not in command
    assert "--voxel-carve" in command
    # splat-transform's contract is `input [ACTIONS] ... output`: the source
    # must LEAD so ordered actions apply to it, and the output trails.
    assert command[1] == "source.ply"
    assert command[-1] == "scene.voxel.json"


def test_collision_command_rotates_capture_frame_into_engine_frame() -> None:
    """SplatLab captures are Z-up; splat-transform voxelizes Y-up.

    Without the rotation, --voxel-floor-fill ("fill columns upward from
    bottom") fills along the wrong axis and the resulting floor is unwalkable.
    The rotation is an ACTION, so it must come after the input file.
    """
    command = export_route._collision_command(
        "/tool", Path("source.ply"), Path("scene.voxel.json"),
        export_route.CollisionRequest(),
    )
    assert command[1] == "source.ply"
    assert command[2:4] == ["-r", "-90,0,0"]
    assert command.index("-r") < command.index("--voxel-floor-fill")


def test_collision_command_rotates_the_seed_with_the_cloud() -> None:
    """Everything after -r is evaluated in the rotated frame, so a seed given
    in capture coords must be rotated too or it silently points elsewhere."""
    command = export_route._collision_command(
        "/tool", Path("source.ply"), Path("scene.voxel.json"),
        export_route.CollisionRequest(seed_position=(1, 2, 3)),
    )
    assert command[command.index("--seed-pos") + 1] == "1,3,-2"


def _write_binary_ply(path: Path, points) -> None:
    """Minimal binary_little_endian PLY with a couple of extra properties, so
    the reader is exercised on a strided layout rather than a bare xyz block."""
    import struct
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty float opacity\n"
        "end_header\n"
    ).encode("ascii")
    body = b"".join(struct.pack("<fffBf", x, y, z, 7, 0.5) for x, y, z in points)
    path.write_bytes(header + body)


def test_ply_xyz_reads_strided_binary_ply(tmp_path: Path) -> None:
    """The backend venv has numpy but NOT trimesh/plyfile — the reader has to
    stand alone, or the crop silently never happens in production."""
    pts = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)]
    ply = tmp_path / "m.ply"
    _write_binary_ply(ply, pts)
    got = export_route._ply_xyz(ply)
    assert got is not None and got.shape == (3, 3)
    assert got[1].tolist() == [1.0, 2.0, 3.0]


def test_ply_xyz_rejects_non_ply_without_raising(tmp_path: Path) -> None:
    bad = tmp_path / "nope.ply"
    bad.write_bytes(b"this is not a ply")
    assert export_route._ply_xyz(bad) is None


def test_derive_filter_box_prefers_the_mesh_over_the_splat(tmp_path: Path) -> None:
    """The TSDF mesh already dropped most far-field, so it gives the tighter
    (correct) crop; the splat is only a fallback."""
    job = tmp_path / "job"
    (job / "_mesh").mkdir(parents=True)
    _write_binary_ply(job / "_mesh" / "mesh.ply",
                      [(float(i % 3), float(i % 5), float(i % 2)) for i in range(64)])
    splat = job / "splat.ply"
    _write_binary_ply(splat, [(100.0 * i, 100.0 * i, 100.0 * i) for i in range(64)])
    box = export_route._derive_filter_box(job, splat, 2.0)
    assert box is not None
    lo_x = float(box.split(",")[0])
    assert lo_x < 10, f"box came from the splat, not the mesh: {box}"


def test_derive_filter_box_falls_back_to_the_splat(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    splat = job / "splat.ply"
    _write_binary_ply(splat, [(float(i), float(i), float(i)) for i in range(64)])
    assert export_route._derive_filter_box(job, splat, 2.0) is not None


def test_derive_filter_box_returns_none_when_nothing_readable(tmp_path: Path) -> None:
    assert export_route._derive_filter_box(tmp_path, tmp_path / "missing.ply", 2.0) is None


def test_collision_command_crops_before_rotating() -> None:
    """--filter-box is in the CAPTURE frame, so it must precede -r; applied
    after the rotation it would crop the wrong region entirely."""
    command = export_route._collision_command(
        "/tool", Path("source.ply"), Path("scene.voxel.json"),
        export_route.CollisionRequest(), "-1,-2,-3,4,5,6",
    )
    assert command.index("--filter-box") < command.index("-r")
    assert command[command.index("--filter-box") + 1] == "-1,-2,-3,4,5,6"


def test_collision_command_explicit_box_beats_derived() -> None:
    command = export_route._collision_command(
        "/tool", Path("source.ply"), Path("scene.voxel.json"),
        export_route.CollisionRequest(filter_box="9,9,9,10,10,10"), None,
    )
    assert command[command.index("--filter-box") + 1] == "9,9,9,10,10,10"


def test_collision_command_box_can_be_disabled() -> None:
    command = export_route._collision_command(
        "/tool", Path("source.ply"), Path("scene.voxel.json"),
        export_route.CollisionRequest(auto_filter_box=False), None,
    )
    assert "--filter-box" not in command


def test_collision_command_rotation_can_be_disabled() -> None:
    """An already-Y-up source must be able to opt out."""
    command = export_route._collision_command(
        "/tool", Path("source.ply"), Path("scene.voxel.json"),
        export_route.CollisionRequest(rotate="", seed_position=(1, 2, 3)),
    )
    assert "-r" not in command
    assert command[command.index("--seed-pos") + 1] == "1,2,3"


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_TRANSFORM_TESTS") != "1",
    reason="set SPLATLAB_RUN_TRANSFORM_TESTS=1 for the real converter smoke test",
)
def test_real_transform_round_trips_cpu_formats(tmp_path: Path) -> None:
    transform = splat_route._splat_transform_path()
    if not transform:
        pytest.skip("splat-transform is unavailable")

    source = tmp_path / "source.ply"
    _write_ply(source)
    request = export_route.ExportRequest(sog_iterations=1)
    outputs = {
        "spz": tmp_path / "scene.spz",
        "sog": tmp_path / "scene.sog",
        "gltf": tmp_path / "scene.glb",
    }
    for fmt, output in outputs.items():
        result = subprocess.run(
            export_route._conversion_command(
                transform,
                source,
                output,
                fmt,
                request,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.stat().st_size > 0

    with outputs["gltf"].open("rb") as handle:
        magic, version, _length = struct.unpack("<4sII", handle.read(12))
        json_length, json_type = struct.unpack("<II", handle.read(8))
        gltf = json.loads(handle.read(json_length).decode().rstrip("\x00 \t\r\n"))
    assert magic == b"glTF"
    assert version == 2
    assert json_type == 0x4E4F534A
    assert "KHR_gaussian_splatting" in gltf["extensionsUsed"]

    for compressed in (outputs["spz"], outputs["sog"]):
        round_trip = tmp_path / f"{compressed.suffix[1:]}-roundtrip.ply"
        result = subprocess.run(
            [transform, str(compressed), str(round_trip)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert export_route.manifests.inspect_ply(round_trip)["gaussian_count"] == 2


def test_collision_route_publishes_voxel_and_mesh(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    http, outputs, _state = client
    job_dir = _make_job(outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)

    async def fake_execute(command: list[str]) -> tuple[int, str]:
        output = Path(command[-1])
        output.write_text("{}")
        output.with_name("scene.voxel.bin").write_bytes(b"voxel")
        output.with_name("scene.collision.glb").write_bytes(b"glb")
        return 0, "ok"

    async def fake_gpu_operation(**kwargs):
        return await kwargs["operation"]()

    monkeypatch.setattr(export_route, "_execute_command", fake_execute)
    monkeypatch.setattr(
        export_route.gpu_arbiter, "run_gpu_operation", fake_gpu_operation
    )

    response = http.post(
        "/api/splat/jobs/splat_e70001/collision",
        json={"mode": "exterior", "seed_position": [0, 0, 0]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ready"
    assert (job_dir / "_exports" / "collision" / "scene.voxel.json").is_file()
    assert (job_dir / "_exports" / "collision" / "scene.collision.glb").is_file()
    assert (
        http.get("/api/splat/jobs/splat_e70001/exports/file/collision-mesh").status_code
        == 200
    )


def test_unreal_bundle_is_portable_and_manifested(client) -> None:
    http, outputs, _state = client
    job_dir = _make_job(outputs)
    assert (
        http.post(
            "/api/splat/jobs/splat_e70001/exports",
            json={"formats": ["spz", "gltf"]},
        ).status_code
        == 200
    )

    response = http.post(
        "/api/splat/jobs/splat_e70001/unreal-bundle",
        json={"include_zip": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ready"
    bundle_dir = job_dir / "_exports" / "unreal"
    bundle = json.loads((bundle_dir / "manifest.json").read_text())
    assert bundle["target"] == {
        "engine": "Unreal Engine",
        "version": "5.6",
        "platform": "Windows",
        "renderer_probe_order": [
            "NanoGS",
            "MLSLabsRenderer",
            "UnrealSplat",
        ],
    }
    assert bundle["import_contract"]["centimeters_per_source_unit"] == 200.0
    assert (bundle_dir / "Gaussian" / "scene.ply").is_file()
    assert (job_dir / "_exports" / "splat_e70001-unreal-5.6.zip").is_file()
    assert (
        http.get("/api/splat/jobs/splat_e70001/unreal-bundle/manifest").status_code
        == 200
    )
    assert (
        http.get("/api/splat/jobs/splat_e70001/unreal-bundle/file").status_code == 200
    )


def test_unreal_bundle_preserves_nested_streamed_sog_paths(client) -> None:
    http, outputs, _state = client
    job_dir = _make_job(outputs)
    response = http.post(
        "/api/splat/jobs/splat_e70001/exports",
        json={"formats": ["streamed-sog"], "force_streamed_sog": True},
    )
    assert response.status_code == 200, response.text

    stream_root = job_dir / "_exports" / "streamed-sog"
    manifest_path = job_dir / "_exports" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest["artifacts"]["streamed-sog"]["files"]
    for level, payload in (("lod-0", b"near"), ("lod-1", b"far")):
        chunk = stream_root / level / "chunk-0.sog"
        chunk.parent.mkdir()
        chunk.write_bytes(payload)
        files.append(
            export_route.manifests.artifact_file_record(
                chunk,
                job_dir / "_exports",
                media_type="application/octet-stream",
            )
        )
    manifest_path.write_text(json.dumps(manifest))

    response = http.post(
        "/api/splat/jobs/splat_e70001/unreal-bundle",
        json={"include_zip": False},
    )
    assert response.status_code == 200, response.text
    gaussian = job_dir / "_exports" / "unreal" / "Gaussian" / "streamed-sog"
    assert (gaussian / "lod-0" / "chunk-0.sog").read_bytes() == b"near"
    assert (gaussian / "lod-1" / "chunk-0.sog").read_bytes() == b"far"


def test_unreal_bundle_fails_closed_when_host_admission_is_unavailable(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    http, outputs, _state = client
    _make_job(outputs)
    response = http.post(
        "/api/splat/jobs/splat_e70001/exports",
        json={"formats": ["spz"]},
    )
    assert response.status_code == 200, response.text

    async def reject_host_admission(*, operation):
        del operation
        raise export_route.gpu_arbiter.GPUArbiterUnavailable("backup active")

    monkeypatch.setattr(
        export_route.gpu_arbiter,
        "run_host_operation",
        reject_host_admission,
    )
    response = http.post(
        "/api/splat/jobs/splat_e70001/unreal-bundle",
        json={"include_zip": True},
    )

    assert response.status_code == 503
    assert response.json()["detail"].endswith("backup active")
