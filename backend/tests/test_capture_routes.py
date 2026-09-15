from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import capture_records as captures
import capture_route
import route_builder as routes


def test_gps_keeps_document_grouping_and_rejects_invalid_fixes():
    metadata = {"Doc4:GPSLatitude": 38.0, "Doc7:GPSLatitude": 39.0,
                "Doc7:GPSLongitude": -122.0, "Doc4:GPSLongitude": -123.0,
                "Doc4:GPSDateTime": "2026:08:28 15:16:56.9Z",
                "Doc7:GPSDateTime": "2026:08:28 15:16:59.9Z",
                "Doc8:GPSLatitude": float("nan"), "Doc8:GPSLongitude": 0,
                "Doc8:GPSDateTime": "2026:08:28 15:17:00Z"}
    result = captures.gps_records(metadata)
    assert result["count"] == 2
    assert result["samples"][0]["longitude"] == -123
    assert result["rejected_records"] == 1
    assert result["gaps"][0]["duration_s"] == 3
    assert result["accuracy_known"] is False


def test_gps_alignment_uses_video_clock_and_never_extrapolates():
    document = {"clock": {"video_start_utc_s": 100, "offset_s": 0, "verified": False},
                "gps": {"samples": [
                    {"utc_s": 105, "latitude": 38, "longitude": 179, "altitude_m": 10},
                    {"utc_s": 107, "latitude": 40, "longitude": -179, "altitude_m": 12},
                    {"utc_s": 120, "latitude": 41, "longitude": -178, "altitude_m": 13}]}}
    assert captures.location_at(document, 0) is None
    assert captures.location_at(document, 15) is None
    assert captures.location_at(document, 21) is None
    middle = captures.location_at(document, 6)
    assert middle["latitude"] == 39
    assert abs(middle["longitude"]) == 180
    assert middle["altitude_m"] == 11
    assert middle["alignment_verified"] is False


@pytest.mark.parametrize("value", ["bad", "2026:99:99 00:00:00", None])
def test_invalid_utc_has_no_silent_clock(value):
    assert captures.utc_seconds(value) is None


@pytest.fixture()
def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(captures, "DATA_ROOT", tmp_path / "spatial")
    source = tmp_path / "input.insv"
    source.write_bytes(b"source")
    document = {"schema": captures.CAPTURE_SCHEMA, "capture_id": "capture_" + "a" * 24,
                "duration_s": 31, "source": {"path": str(source), "name": source.name,
                                             **manifests.file_identity(source)},
                "streams": [{"index": 0, "width": 3840, "height": 3840},
                            {"index": 1, "width": 3840, "height": 3840}],
                "clock": {"video_start_utc_s": None, "verified": False},
                "gps": {"samples": [], "count": 0}}
    manifests.atomic_write_json(captures.capture_path(document["capture_id"]), document)
    return document


def test_route_segment_bounds_and_persisted_points(capture):
    spec = routes.RouteSpec(capture_id=capture["capture_id"], start_s=10, end_s=100, interval_s=10)
    document = routes.create_route(spec)
    assert [point["time_s"] for point in document["points"]] == [10, 20, 30]
    assert routes.load_route(document["route_id"])["source_sha256"] == capture["source"]["sha256"]
    with pytest.raises(ValueError, match="outside"):
        routes.create_route(routes.RouteSpec(capture_id=capture["capture_id"], start_s=100))


def test_source_change_refuses_resume(capture):
    document = routes.create_route(routes.RouteSpec(capture_id=capture["capture_id"]))
    Path(capture["source"]["path"]).write_bytes(b"different")
    with pytest.raises(ValueError, match="source changed"):
        routes.build_route(document["route_id"])


def test_clock_change_refuses_resume(capture):
    document = routes.create_route(routes.RouteSpec(capture_id=capture["capture_id"]))
    capture["clock"]["video_start_utc_s"] = 100
    manifests.atomic_write_json(captures.capture_path(capture["capture_id"]), capture)
    with pytest.raises(ValueError, match="clock changed"):
        routes.build_route(document["route_id"])


def test_panorama_command_maps_both_lenses_without_shell(capture, tmp_path, monkeypatch):
    monkeypatch.setattr(routes.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    source = Path("/tmp/my capture; $(touch injected).insv")
    command = routes.panorama_command(source, capture["streams"], routes.RouteSpec(capture_id=capture["capture_id"]), 12.5, tmp_path / "out.jpg")
    assert str(source) in command
    assert "[0:0][0:1]hstack=inputs=2" in command[command.index("-filter_complex") + 1]
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-hwaccel") + 1] == "none"


def test_completed_frames_are_checksum_verified_before_skip(capture, monkeypatch):
    document = routes.create_route(routes.RouteSpec(capture_id=capture["capture_id"], end_s=1))
    point = document["points"][0]
    destination = routes.route_dir(document["route_id"]) / "pano-00000.jpg"
    destination.write_bytes(b"retained")
    point.update(status="ready", sha256=manifests.sha256_file(destination))
    routes.save_route(document)
    monkeypatch.setattr(routes, "panorama_command", lambda *args: pytest.fail("Should reuse verified frame"))
    assert routes.build_route(document["route_id"])["status"] == "completed"


def test_route_api_rejects_unregistered_sources_and_invalid_panorama(capture, monkeypatch):
    monkeypatch.setattr(capture_route.splat_route, "_transfers_entries", lambda: [])
    monkeypatch.setattr(capture_route.splat_route, "_all_metas", lambda: [])
    app = FastAPI()
    app.include_router(capture_route.router, prefix="/api/splat")
    with TestClient(app) as client:
        assert client.post("/api/splat/captures/inspect", json={"input_path": "/etc/passwd"}).status_code == 400
        assert client.get("/api/splat/routes/not-valid").status_code == 404
        document = routes.create_route(routes.RouteSpec(capture_id=capture["capture_id"]))
        assert client.get(f"/api/splat/routes/{document['route_id']}/panoramas/-1").status_code == 404
        assert client.get(f"/api/splat/routes/{document['route_id']}/panoramas/0").status_code == 404


def test_worker_environment_does_not_inherit_vault_secrets(monkeypatch):
    monkeypatch.setenv("PORTAL_TOKEN", "test-not-a-real-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-secret")
    assert "PORTAL_TOKEN" not in captures.worker_env()
    assert "OPENAI_API_KEY" not in captures.worker_env()


def test_restart_marks_running_routes_paused(capture):
    document = routes.create_route(routes.RouteSpec(capture_id=capture["capture_id"]))
    document["status"] = "running"
    routes.save_route(document)
    capture_route.reconcile_routes()
    assert routes.load_route(document["route_id"])["status"] == "paused"


def test_stop_before_worker_starts_is_not_lost(capture, monkeypatch):
    document = routes.create_route(routes.RouteSpec(capture_id=capture["capture_id"]))
    (routes.route_dir(document["route_id"]) / "stop-requested").touch()
    monkeypatch.setattr(routes, "extract_point", lambda *args: pytest.fail("Stopped worker must not decode"))
    assert routes.build_route(document["route_id"])["status"] == "paused"


def test_bad_frame_retains_other_viewpoints_and_resumes(capture, monkeypatch):
    document = routes.create_route(routes.RouteSpec(capture_id=capture["capture_id"], end_s=12))
    attempts = []

    def extract(source, streams, spec, point, directory, deadline):
        attempts.append(point["index"])
        if len(attempts) == 1:
            raise routes.FrameError("Decode failed")
        path = directory / f"pano-{point['index']:05d}.jpg"
        path.write_bytes(b"verified-output")
        return {"status": "ready", "sha256": manifests.sha256_file(path)}

    monkeypatch.setattr(routes, "extract_point", extract)
    result = routes.build_route(document["route_id"])
    assert result["status"] == "partial"
    assert [point["status"] for point in result["points"]] == ["failed", "ready", "ready"]
    assert routes.build_route(document["route_id"])["status"] == "completed"
    assert attempts == [0, 1, 2, 0]
