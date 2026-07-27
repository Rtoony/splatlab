"""Scale as the spine: evidence-bound calibration, averaging, uncertainty, and
the staleness generation counter.

`meters_per_unit` is the one scalar every metric claim in SplatLab depends on —
dimensions, world_collision's 1.5 m prop rule, the scale_sanity world gate, the
DXF and LandXML survey exports. It used to be a bare float with no record of
where it came from; STATUS.md:294 has a calibration derived from a made-up 5 ft
reference that nothing in the system could have caught.

These tests pin the arithmetic (in scale_calibration, which is pure) and the
route contract: derivation from stored measurements, multi-reference averaging,
honest uncertainty, refusal to silently downgrade evidence, and a generation
counter downstream artifacts can compare against.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dimensions_route  # noqa: E402
import scale_calibration as sc  # noqa: E402
import splat_route  # noqa: E402


JOB = "splat_5ca1e0"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    app.include_router(dimensions_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = JOB) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps(
        {"job_id": job_id, "output_dir": str(job_dir), "status": "completed"}))
    return job_dir


def _dimension(job_dir: Path, dim_id: str, length_units: float, label: str = ""):
    """A measurement lying along +X, so its scene length is exactly `length_units`."""
    dimensions_route.upsert_dimension(
        job_dir, dim_id=dim_id, a=[0.0, 0.0, 0.0], b=[length_units, 0.0, 0.0], label=label)


def _meta(job_dir: Path) -> dict:
    return json.loads((job_dir / "meta.json").read_text())


# ---------------------------------------------------------------------------
# Pure arithmetic
# ---------------------------------------------------------------------------

def test_scene_length_is_euclidean():
    assert sc.scene_length([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)


def test_scene_length_rejects_non_finite_endpoints():
    with pytest.raises(sc.CalibrationError, match="finite"):
        sc.scene_length([0, 0, 0], [float("nan"), 0, 0])


@pytest.mark.parametrize("bad", ["banana", None, 0, -1.5, float("nan"), float("inf"), 1e9])
def test_validate_factor_rejects_garbage(bad):
    with pytest.raises(sc.CalibrationError):
        sc.validate_factor(bad)


def test_single_reference_derives_the_factor():
    dimensions = [{"id": "d1", "a": [0, 0, 0], "b": [2, 0, 0], "label": "door"}]
    record = sc.calibrate_from_dimensions(
        dimensions, [{"dimension_id": "d1", "real_length_m": 0.9}])

    assert record["meters_per_unit"] == pytest.approx(0.45)
    assert record["method"] == sc.METHOD_DIMENSION
    assert record["references"][0]["dimension_id"] == "d1"
    assert record["references"][0]["label"] == "door"
    assert record["references"][0]["scene_length"] == pytest.approx(2.0)


def test_multiple_references_average():
    dimensions = [
        {"id": "d1", "a": [0, 0, 0], "b": [2, 0, 0]},
        {"id": "d2", "a": [0, 0, 0], "b": [4, 0, 0]},
    ]
    record = sc.calibrate_from_dimensions(dimensions, [
        {"dimension_id": "d1", "real_length_m": 1.0},   # 0.50
        {"dimension_id": "d2", "real_length_m": 2.4},   # 0.60
    ])

    assert record["meters_per_unit"] == pytest.approx(0.55)
    assert record["uncertainty"]["n"] == 2


def test_agreeing_references_report_low_uncertainty():
    dimensions = [{"id": f"d{i}", "a": [0, 0, 0], "b": [2, 0, 0]} for i in range(3)]
    record = sc.calibrate_from_dimensions(dimensions, [
        {"dimension_id": "d0", "real_length_m": 1.00},
        {"dimension_id": "d1", "real_length_m": 1.01},
        {"dimension_id": "d2", "real_length_m": 0.99},
    ])

    unc = record["uncertainty"]
    assert unc["relative"] < 0.02
    assert unc["disagreement"] is False
    assert unc["spread_ratio"] == pytest.approx(1.01 / 0.99, rel=1e-3)


def test_a_made_up_reference_shows_up_as_disagreement():
    """The STATUS.md:294 failure mode: one invented length among real ones.
    It cannot be auto-detected, but it must not hide."""
    dimensions = [{"id": f"d{i}", "a": [0, 0, 0], "b": [2, 0, 0]} for i in range(3)]
    record = sc.calibrate_from_dimensions(dimensions, [
        {"dimension_id": "d0", "real_length_m": 1.00},
        {"dimension_id": "d1", "real_length_m": 1.02},
        {"dimension_id": "d2", "real_length_m": 1.52},   # the guess
    ])

    unc = record["uncertainty"]
    assert unc["disagreement"] is True
    assert "disagree" in sc.describe(record)
    worst = max(record["references"], key=lambda r: abs(r["deviation_from_mean"]))
    assert worst["dimension_id"] == "d2", "the outlier must be identifiable"


def test_one_reference_reports_unknown_uncertainty_not_zero():
    """A comfortable 0.0 would read as certainty. It is not."""
    dimensions = [{"id": "d1", "a": [0, 0, 0], "b": [2, 0, 0]}]
    record = sc.calibrate_from_dimensions(
        dimensions, [{"dimension_id": "d1", "real_length_m": 1.0}])

    assert record["uncertainty"]["n"] == 1
    assert record["uncertainty"]["stddev"] is None
    assert record["uncertainty"]["relative"] is None
    assert "no cross-check" in sc.describe(record)


def test_unknown_dimension_id_is_refused():
    with pytest.raises(sc.CalibrationError, match="no stored dimension"):
        sc.calibrate_from_dimensions([], [{"dimension_id": "ghost", "real_length_m": 1.0}])


def test_zero_length_measurement_cannot_calibrate():
    dimensions = [{"id": "d1", "a": [1, 1, 1], "b": [1, 1, 1]}]
    with pytest.raises(sc.CalibrationError, match="zero length"):
        sc.calibrate_from_dimensions(
            dimensions, [{"dimension_id": "d1", "real_length_m": 1.0}])


@pytest.mark.parametrize("bad", [0, -1, float("nan"), "banana", None])
def test_bad_real_length_is_refused(bad):
    dimensions = [{"id": "d1", "a": [0, 0, 0], "b": [2, 0, 0]}]
    with pytest.raises(sc.CalibrationError):
        sc.calibrate_from_dimensions(
            dimensions, [{"dimension_id": "d1", "real_length_m": bad}])


def test_no_references_is_refused():
    with pytest.raises(sc.CalibrationError, match="at least one"):
        sc.calibrate_from_dimensions([], [])


# ---------------------------------------------------------------------------
# Evidence ranking
# ---------------------------------------------------------------------------

def test_map_eyeballing_is_weaker_than_a_measurement():
    measured = sc.calibrate_manual(0.5, method=sc.METHOD_DIMENSION)
    eyeballed = sc.calibrate_manual(0.7, method=sc.METHOD_MAP)

    assert sc.downgrades_evidence(measured, eyeballed) is True
    assert sc.downgrades_evidence(eyeballed, measured) is False


def test_same_method_is_not_a_downgrade():
    first = sc.calibrate_manual(0.5, method=sc.METHOD_DIMENSION)
    second = sc.calibrate_manual(0.6, method=sc.METHOD_DIMENSION)
    assert sc.downgrades_evidence(first, second) is False


def test_no_prior_calibration_is_never_a_downgrade():
    assert sc.downgrades_evidence(None, sc.calibrate_manual(0.5, method=sc.METHOD_MAP)) is False


def test_describe_handles_the_uncalibrated_case():
    assert sc.describe(None) == "not calibrated"
    assert sc.describe({}) == "not calibrated"


# ---------------------------------------------------------------------------
# Route: derivation from stored measurements
# ---------------------------------------------------------------------------

def test_scale_derived_from_a_stored_dimension(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _dimension(job_dir, "d1", 2.0, label="garage door")

    r = http.post(f"/api/splat/jobs/{JOB}/scale",
                  json={"references": [{"dimension_id": "d1", "real_length_m": 0.9}]})

    assert r.status_code == 200, r.text
    assert r.json()["meters_per_unit"] == pytest.approx(0.45)

    stored = _meta(job_dir)
    assert stored["meters_per_unit"] == pytest.approx(0.45)
    calibration = stored["scale_calibration"]
    assert calibration["method"] == "dimension"
    assert calibration["references"][0]["dimension_id"] == "d1"
    assert calibration["references"][0]["label"] == "garage door"
    assert calibration["set_at"]


def test_scale_averages_several_stored_dimensions(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _dimension(job_dir, "d1", 2.0)
    _dimension(job_dir, "d2", 4.0)

    r = http.post(f"/api/splat/jobs/{JOB}/scale", json={"references": [
        {"dimension_id": "d1", "real_length_m": 1.0},
        {"dimension_id": "d2", "real_length_m": 2.4},
    ]})

    assert r.json()["meters_per_unit"] == pytest.approx(0.55)
    assert _meta(job_dir)["scale_calibration"]["uncertainty"]["n"] == 2


def test_route_refuses_an_unknown_dimension(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.post(f"/api/splat/jobs/{JOB}/scale",
                  json={"references": [{"dimension_id": "ghost", "real_length_m": 1.0}]})

    assert r.status_code == 400
    assert "no stored dimension" in r.json()["detail"]


def test_references_must_be_a_list(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.post(f"/api/splat/jobs/{JOB}/scale", json={"references": "d1"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Route: provenance on the direct path, and the downgrade gate
# ---------------------------------------------------------------------------

def test_direct_set_still_works_and_is_recorded_as_manual(client):
    http, outputs = client
    job_dir = _mk_job(outputs)

    r = http.post(f"/api/splat/jobs/{JOB}/scale", json={"meters_per_unit": 0.3125})

    assert r.status_code == 200
    assert _meta(job_dir)["scale_calibration"]["method"] == "manual"


def test_map_source_is_recorded_as_map(client):
    http, outputs = client
    job_dir = _mk_job(outputs)

    http.post(f"/api/splat/jobs/{JOB}/scale",
              json={"meters_per_unit": 0.5, "source": "map"})

    assert _meta(job_dir)["scale_calibration"]["method"] == "map"


def test_map_scale_cannot_silently_overwrite_a_measured_one(client):
    """The Locate modal's footprint drag is an accuracy downgrade behind a
    checkbox. It must not land by accident."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    _dimension(job_dir, "d1", 2.0)
    http.post(f"/api/splat/jobs/{JOB}/scale",
              json={"references": [{"dimension_id": "d1", "real_length_m": 0.9}]})

    r = http.post(f"/api/splat/jobs/{JOB}/scale",
                  json={"meters_per_unit": 0.7, "source": "map"})

    assert r.status_code == 409
    assert "force" in r.json()["detail"]
    assert _meta(job_dir)["meters_per_unit"] == pytest.approx(0.45), "unchanged"


def test_the_downgrade_can_be_forced_explicitly(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _dimension(job_dir, "d1", 2.0)
    http.post(f"/api/splat/jobs/{JOB}/scale",
              json={"references": [{"dimension_id": "d1", "real_length_m": 0.9}]})

    r = http.post(f"/api/splat/jobs/{JOB}/scale",
                  json={"meters_per_unit": 0.7, "source": "map", "force": True})

    assert r.status_code == 200
    assert _meta(job_dir)["meters_per_unit"] == pytest.approx(0.7)
    assert _meta(job_dir)["scale_calibration"]["method"] == "map"


def test_a_better_measurement_replaces_a_map_scale_without_force(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _dimension(job_dir, "d1", 2.0)
    http.post(f"/api/splat/jobs/{JOB}/scale",
              json={"meters_per_unit": 0.7, "source": "map"})

    r = http.post(f"/api/splat/jobs/{JOB}/scale",
                  json={"references": [{"dimension_id": "d1", "real_length_m": 0.9}]})

    assert r.status_code == 200
    assert _meta(job_dir)["meters_per_unit"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Route: the staleness generation counter
# ---------------------------------------------------------------------------

def test_generation_increments_on_every_change(client):
    http, outputs = client
    job_dir = _mk_job(outputs)

    assert _meta(job_dir).get("scale_generation") is None
    http.post(f"/api/splat/jobs/{JOB}/scale", json={"meters_per_unit": 0.5})
    assert _meta(job_dir)["scale_generation"] == 1
    http.post(f"/api/splat/jobs/{JOB}/scale", json={"meters_per_unit": 0.6})
    assert _meta(job_dir)["scale_generation"] == 2


def test_clearing_the_scale_also_bumps_the_generation(client):
    """An artifact built at a known scale is just as stale when the scale is
    removed as when it changes."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    http.post(f"/api/splat/jobs/{JOB}/scale", json={"meters_per_unit": 0.5})

    r = http.post(f"/api/splat/jobs/{JOB}/scale", json={"meters_per_unit": None})

    assert r.status_code == 200
    stored = _meta(job_dir)
    assert stored["meters_per_unit"] is None
    assert stored["scale_calibration"] is None
    assert stored["scale_generation"] == 2


def test_a_rejected_change_does_not_bump_the_generation(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    http.post(f"/api/splat/jobs/{JOB}/scale", json={"meters_per_unit": 0.5})
    before = _meta(job_dir)["scale_generation"]

    http.post(f"/api/splat/jobs/{JOB}/scale", json={"meters_per_unit": -1})

    assert _meta(job_dir)["scale_generation"] == before


# ---------------------------------------------------------------------------
# Dimensions: NaN guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_dimension_endpoints_must_be_finite(client, bad):
    """A NaN endpoint silently poisons every length derived from it, including
    a scale calibration that cites the measurement as its evidence."""
    http, outputs = client
    _mk_job(outputs)
    body = json.dumps({"id": "d1", "a": [0.0, 0.0, 0.0], "b": [bad, 0.0, 0.0],
                       "label": ""}, allow_nan=True).encode()

    r = http.post(f"/api/splat/jobs/{JOB}/dimensions", content=body,
                  headers={"Content-Type": "application/json"})

    assert r.status_code == 400


def test_finite_dimensions_are_still_accepted(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.post(f"/api/splat/jobs/{JOB}/dimensions",
                  json={"id": "d1", "a": [0, 0, 0], "b": [2, 0, 0], "label": "door"})
    assert r.status_code == 200
    assert math.isfinite(r.json()["dimension"]["b"][0])
