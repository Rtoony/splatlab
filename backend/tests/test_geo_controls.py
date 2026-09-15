import math

import pytest

from geo_controls import fit_controls


def test_known_scale_and_heading_with_independent_control():
    degrees_per_meter = 180 / (math.pi * 6378137)
    controls = [{"scene": [0, 0], "lat": 0, "lon": 0},
                {"scene": [10, 0], "lat": 0, "lon": 20 * degrees_per_meter},
                {"scene": [20, 0], "lat": 0, "lon": 40 * degrees_per_meter}]
    result = fit_controls(controls)
    assert result["meters_per_unit"] == pytest.approx(2, abs=1e-8)
    assert result["geo"]["heading_deg"] == pytest.approx(0)
    assert result["max_residual_m"] < 1e-6
    assert result["redundant_control_check"]
    assert result["independent_check_points"] == 0
    assert result["dimensions"] == "horizontal-only"


def test_two_controls_do_not_claim_independent_residual_check():
    result = fit_controls([{"scene": [0, 0], "lat": 38, "lon": -122},
                           {"scene": [10, 0], "lat": 38.001, "lon": -122}])
    assert not result["redundant_control_check"]
    assert result["geo"]["heading_deg"] == pytest.approx(270, abs=1e-5)


def test_duplicate_controls_refused():
    with pytest.raises(ValueError, match="constrain"):
        fit_controls([{"scene": [0, 0], "lat": 38, "lon": -122}] * 3)


def test_inconsistent_third_control_reports_residual():
    result = fit_controls([{"scene": [0, 0], "lat": 38, "lon": -122},
                           {"scene": [10, 0], "lat": 38, "lon": -121.9999},
                           {"scene": [20, 0], "lat": 38.0001, "lon": -121.9998}])
    assert result["rms_m"] > 1
