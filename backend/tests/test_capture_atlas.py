from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import artifact_manifest as manifests
import capture_atlas as atlas
import capture_records as captures
import capture_route


def sample(timestamp=100, longitude=-122, latitude=38):
    return {"utc_s": timestamp, "longitude": longitude, "latitude": latitude,
            "altitude_m": 20, "horizontal_accuracy_m": None}


@pytest.fixture()
def packet(tmp_path, monkeypatch):
    monkeypatch.setattr(captures, "DATA_ROOT", tmp_path / "spatial")
    handoff = tmp_path / "handoff"
    (handoff / "metadata").mkdir(parents=True)
    (handoff / "review").mkdir()
    image = handoff / "review/frame.jpg"
    Image.new("RGB", (64, 32), "orange").save(image)
    source = tmp_path / "example.insv"
    source.write_bytes(b"synthetic source fixture")
    source_identity = manifests.file_identity(source)
    capture = {"schema": captures.CAPTURE_SCHEMA, "capture_id": "capture_" + source_identity["sha256"][:24],
               "source": {"name": source.name, "path": str(source), **source_identity},
               "duration_s": 10, "clock": {"video_start_utc_s": 100, "offset_s": 0, "verified": False},
               "streams": [{"index": 0, "width": 3840, "height": 3840}],
               "gps": {"count": 2, "samples": [sample(), sample(101, -121.99999)], "accuracy_known": False}}
    manifest = {"schema": "dev.roonytoony.condo-360-handoff/v1",
                "raw_files": [{"filename": source.name, **source_identity}],
                "derivatives": [{"source": source.name, "path": "review/frame.jpg", "kind": "review_frame",
                                 "approximate_source_time_s": 0.5, "quality_flags": ["not_vendor_calibrated"]},
                                {"source": source.name, "path": "reference/missing.jpg", "kind": "reference_panorama"}]}
    manifests.atomic_write_json(handoff / "metadata/manifest.json", manifest)
    manifests.atomic_write_json(captures.capture_path(capture["capture_id"]), capture)
    return handoff, capture


def test_geographic_frame_has_metre_units_without_using_altitude():
    origin = {"longitude": -122, "latitude": 38}
    assert atlas.local_position(-122, 38, origin) == {"east_m": 0, "north_m": 0}
    assert 87 < atlas.local_position(-121.999, 38, origin)["east_m"] < 89
    assert 110 < atlas.local_position(-122, 38.001, origin)["north_m"] < 112


def test_dateline_does_not_wrap_display_around_earth():
    point = atlas.local_position(-179.999, 0, {"longitude": 179.999, "latitude": 0})
    assert 220 < point["east_m"] < 224


def test_tracks_split_invalid_fixes_gaps_jumps_and_reversals():
    samples = [sample(), sample(101), sample(104), sample(105, -120), sample(106, float("nan")), sample(107), sample(107)]
    segments, breaks = atlas.track_segments(samples, {"longitude": -122, "latitude": 38})
    assert [len(segment) for segment in segments] == [2, 1, 1, 1, 1]
    assert [item["reason"] for item in breaks] == ["telemetry_gap", "position_jump", "invalid_fix", "non_increasing_timestamp"]


def test_location_is_tentative_and_never_crosses_rejected_jump(packet):
    _, capture = packet
    origin = {"longitude": -122, "latitude": 38}
    segments, _ = atlas.track_segments(capture["gps"]["samples"], origin)
    location = atlas.viewpoint_location(capture, .5, segments, origin)
    assert location["status"] == "tentative_clock_unverified"
    assert location["alignment_verified"] is False
    assert atlas.viewpoint_location(capture, 5, segments, origin) is None
    capture["gps"]["samples"][1]["longitude"] = -120
    segments, _ = atlas.track_segments(capture["gps"]["samples"], origin)
    assert atlas.viewpoint_location(capture, .5, segments, origin) is None


def test_unknown_video_clock_keeps_tracks_but_not_image_locations(packet):
    handoff, capture = packet
    capture["clock"]["video_start_utc_s"] = None
    document, _ = atlas.build_document(handoff, [capture], "Test")
    assert document["clips"][0]["segments"]
    assert document["views"][0]["location"] is None


def test_clock_mismatch_is_not_silently_corrected(packet):
    handoff, capture = packet
    capture["clock"]["video_start_utc_s"] = 1
    document, _ = atlas.build_document(handoff, [capture], "Test")
    diagnostic = document["clips"][0]["clock_diagnostic"]
    assert diagnostic["status"] == "no_time_overlap"
    assert diagnostic["first_fix_minus_container_start_s"] == 99
    assert diagnostic["automatic_clock_correction"] is False
    assert document["summary"]["located_views"] == 0
    capture["clock"]["offset_s"] = 99
    segments, _ = atlas.track_segments(capture["gps"]["samples"], {"longitude": -122, "latitude": 38})
    assert atlas.clock_diagnostic(capture, segments)["overlap_s"] == 1


def test_packet_preserves_missing_files_clock_and_source_identity(packet):
    handoff, capture = packet
    before = (handoff / "metadata/manifest.json").read_bytes()
    document, copies = atlas.build_document(handoff, [capture], "Test")
    assert document["summary"]["missing_derivatives"] == 1
    assert document["summary"]["review_views"] == 1
    assert len(copies) == 1
    assert document["clips"][0]["source_sha256"] == capture["source"]["sha256"]
    assert document["registration"] == {"status": "unregistered", "model_transform": None}
    assert document["coordinate_frame"]["altitude_used"] is False
    assert (handoff / "metadata/manifest.json").read_bytes() == before
    assert "path" not in document["clips"][0]


@pytest.mark.parametrize("relative", ["../outside.jpg", "/etc/passwd"])
def test_escaping_derivative_paths_refused(packet, relative):
    handoff, capture = packet
    manifest = manifests.read_json(handoff / "metadata/manifest.json")
    manifest["derivatives"][0]["path"] = relative
    manifests.atomic_write_json(handoff / "metadata/manifest.json", manifest)
    with pytest.raises(ValueError, match="root|relative"):
        atlas.build_document(handoff, [capture], "Test")


def test_symlink_escape_refused(packet, tmp_path):
    handoff, _ = packet
    (handoff / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        atlas.contained_file(handoff, "escape/private.jpg")


def test_missing_capture_and_hash_mismatch_refused(packet):
    handoff, capture = packet
    with pytest.raises(ValueError, match="every original"):
        atlas.build_document(handoff, [], "Test")
    capture["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        atlas.build_document(handoff, [capture], "Test")


def test_out_of_range_viewpoint_time_refused(packet):
    handoff, capture = packet
    manifest = manifests.read_json(handoff / "metadata/manifest.json")
    manifest["derivatives"][0]["approximate_source_time_s"] = 999
    manifests.atomic_write_json(handoff / "metadata/manifest.json", manifest)
    with pytest.raises(ValueError, match="outside"):
        atlas.build_document(handoff, [capture], "Test")


def test_inspection_reuses_unchanged_hashed_capture_not_removable_media(packet, monkeypatch):
    handoff, capture = packet
    monkeypatch.setattr(captures, "inspect_capture", lambda *args: pytest.fail("Do not rehash unchanged inspected source"))
    assert atlas.inspect_sources(handoff, handoff.parent)[0]["capture_id"] == capture["capture_id"]


def test_portable_snapshot_safe_embedding_and_tamper_refusal(packet):
    handoff, capture = packet
    hostile_title = "Condo </script><script>alert(1)</script>"
    document = atlas.write_atlas(handoff, [capture], hostile_title)
    page = atlas.atlas_file(document["atlas_id"], "index.html").read_text()
    assert "<script>alert(1)</script>" not in page
    assert "data:image/jpeg;base64," in page
    assert "__ATLAS_" not in page
    assert "__MATH__" not in page
    assert "__VIEWER__" not in page
    assert "fetch(" not in page
    assert atlas.list_atlases()[0]["atlas_id"] == document["atlas_id"]
    directory = atlas.atlas_dir(document["atlas_id"])
    (directory / "view-0000.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        atlas.atlas_file(document["atlas_id"], "view-0000.jpg")
    with pytest.raises(ValueError, match="not found"):
        atlas.atlas_file(document["atlas_id"], "receipt.json")


def test_geojson_retains_times_and_does_not_promote_altitude_to_elevation(packet):
    handoff, capture = packet
    document, _ = atlas.build_document(handoff, [capture], "Test")
    feature = atlas.geojson(document)["features"][0]
    assert feature["properties"]["utc_seconds"] == [100, 101]
    assert all(len(coordinate) == 2 for coordinate in feature["geometry"]["coordinates"])


def test_api_exposes_only_sealed_files(packet):
    handoff, capture = packet
    document = atlas.write_atlas(handoff, [capture], "Test")
    app = FastAPI()
    app.include_router(capture_route.router, prefix="/api/splat")
    with TestClient(app) as client:
        assert len(client.get("/api/splat/capture-atlases").json()["atlases"]) == 1
        prefix = f"/api/splat/capture-atlases/{document['atlas_id']}"
        response = client.get(prefix + "/index.html")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "private, no-store"
        assert client.get(prefix + "/receipt.json").status_code == 404
        assert client.get(prefix + "/other.jpg").status_code == 404
        assert client.get("/api/splat/capture-atlases/not-valid/index.html").status_code == 404
