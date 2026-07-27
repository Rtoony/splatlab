"""Metric scale calibration: evidence, averaging, and uncertainty.

Why this module exists
----------------------
`meters_per_unit` is the single scalar bridging a non-metric nerfstudio scene to
the real world. Every metric claim the tool makes hangs off it — dimensions,
world_collision's 1.5 m prop rule, the world gates' scale_sanity check, DXF and
LandXML survey exports. It was also the least evidenced artifact in the system:
a bare float, posted by the browser after dividing two numbers, with no record
of which measurement produced it, how, or how well the measurements agreed.
STATUS.md:294 records a calibration derived from a made-up 5 ft reference, and
nothing in the system could have told you.

This module makes scale the spine:

  * **Bound to evidence.** The server computes the factor from stored
    `dimensions.json` records, so the reference is a real measurement with an
    id, not a number typed into a form.
  * **Multi-measurement.** Several references average, which is both more
    accurate and the only way to see disagreement at all.
  * **Uncertainty is reported, not implied.** Spread between references is the
    honest signal that one of them is wrong — a 5 ft guess next to a measured
    door shows up as a spread ratio, loudly.
  * **Provenance travels with the value**, so a later reader can ask where a
    calibration came from and when.

Pure and stdlib-only: no FastAPI, no filesystem, no I/O. All arithmetic here is
directly unit-tested.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

# How a calibration was arrived at, strongest evidence first. The ordering is
# meaningful: `dimension` is derived by the server from stored measurements,
# `manual` is a number a human supplied directly, `map` is eyeballed against
# satellite imagery in the Locate modal.
METHOD_DIMENSION = "dimension"
METHOD_MANUAL = "manual"
METHOD_MAP = "map"

_METHOD_RANK = {METHOD_MAP: 0, METHOD_MANUAL: 1, METHOD_DIMENSION: 2}

# Above this relative standard deviation the references disagree enough that the
# mean is not trustworthy on its own; surfaced as a warning, never silently.
DISAGREEMENT_WARN_RELATIVE = 0.05


class CalibrationError(ValueError):
    """Bad calibration input. Always actionable and always surfaced."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def scene_length(a: Iterable[float], b: Iterable[float]) -> float:
    """Euclidean distance between two scene-space points."""
    pa, pb = [float(v) for v in a], [float(v) for v in b]
    if len(pa) != 3 or len(pb) != 3:
        raise CalibrationError("measurement endpoints must be 3D points")
    if not all(math.isfinite(v) for v in (*pa, *pb)):
        raise CalibrationError("measurement endpoints must be finite")
    return math.dist(pa, pb)


def validate_factor(value: Any) -> float:
    """The single validity rule for a meters-per-unit factor."""
    try:
        factor = float(value)
    except (TypeError, ValueError):
        raise CalibrationError("meters_per_unit must be a number") from None
    if not (math.isfinite(factor) and 0.0 < factor < 1e6):
        raise CalibrationError("meters_per_unit must be finite and > 0")
    return factor


def calibrate_from_dimensions(
    dimensions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    source: str = "",
) -> dict[str, Any]:
    """Derive meters_per_unit from stored measurements.

    `references` is `[{"dimension_id": str, "real_length_m": float}, ...]`.
    Each contributes one estimate; the result is their mean plus the spread
    between them. Returns the full provenance record — the caller stores it
    next to the factor rather than storing the factor alone.
    """
    if not references:
        raise CalibrationError("at least one reference measurement is required")

    by_id = {str(record.get("id")): record for record in dimensions}
    estimates: list[dict[str, Any]] = []

    for reference in references:
        dim_id = str(reference.get("dimension_id") or "")
        if not dim_id:
            raise CalibrationError("each reference needs a dimension_id")
        record = by_id.get(dim_id)
        if record is None:
            raise CalibrationError(f"no stored dimension with id {dim_id!r}")

        try:
            real_m = float(reference.get("real_length_m"))
        except (TypeError, ValueError):
            raise CalibrationError(
                f"real_length_m for {dim_id!r} must be a number") from None
        if not (math.isfinite(real_m) and real_m > 0):
            raise CalibrationError(
                f"real_length_m for {dim_id!r} must be finite and > 0")

        units = scene_length(record.get("a", []), record.get("b", []))
        if units <= 1e-9:
            raise CalibrationError(
                f"dimension {dim_id!r} has zero length in scene units; it cannot "
                f"calibrate anything")

        estimates.append({
            "dimension_id": dim_id,
            "label": str(record.get("label", "")),
            "scene_length": round(units, 8),
            "real_length_m": real_m,
            "meters_per_unit": real_m / units,
        })

    factors = [item["meters_per_unit"] for item in estimates]
    mean = sum(factors) / len(factors)
    validate_factor(mean)

    for item in estimates:
        # How far this reference sits from the consensus. The outlier in a
        # disagreeing set is the one to re-measure.
        item["deviation_from_mean"] = round(item["meters_per_unit"] / mean - 1.0, 6)
        item["meters_per_unit"] = round(item["meters_per_unit"], 8)

    return {
        "meters_per_unit": mean,
        "method": METHOD_DIMENSION,
        "source": str(source or ""),
        "references": estimates,
        "uncertainty": uncertainty(factors),
        "set_at": _now(),
    }


def calibrate_manual(value: Any, *, method: str = METHOD_MANUAL,
                     source: str = "") -> dict[str, Any]:
    """Provenance record for a factor supplied directly rather than derived."""
    if method not in _METHOD_RANK:
        raise CalibrationError(f"unknown calibration method {method!r}")
    return {
        "meters_per_unit": validate_factor(value),
        "method": method,
        "source": str(source or ""),
        "references": [],
        "uncertainty": uncertainty([]),
        "set_at": _now(),
    }


def uncertainty(factors: list[float]) -> dict[str, Any]:
    """Agreement between independent estimates.

    One reference yields no uncertainty information, and says so, rather than
    reporting a comfortable zero that would read as certainty.
    """
    count = len(factors)
    if count == 0:
        return {"n": 0, "stddev": None, "relative": None, "spread_ratio": None,
                "min": None, "max": None, "disagreement": False}
    lo, hi = min(factors), max(factors)
    if count == 1:
        return {"n": 1, "stddev": None, "relative": None, "spread_ratio": 1.0,
                "min": round(lo, 8), "max": round(hi, 8), "disagreement": False}

    mean = sum(factors) / count
    variance = sum((f - mean) ** 2 for f in factors) / (count - 1)
    stddev = math.sqrt(variance)
    relative = stddev / mean if mean else None
    return {
        "n": count,
        "stddev": round(stddev, 8),
        "relative": round(relative, 6) if relative is not None else None,
        "spread_ratio": round(hi / lo, 6) if lo > 0 else None,
        "min": round(lo, 8),
        "max": round(hi, 8),
        "disagreement": bool(relative is not None and relative > DISAGREEMENT_WARN_RELATIVE),
    }


def downgrades_evidence(existing: dict[str, Any] | None, proposed: dict[str, Any]) -> bool:
    """True when `proposed` would replace a better-evidenced calibration.

    The case this exists for: the Locate modal can overwrite a measured
    calibration with a footprint scale eyeballed against satellite tiles. That
    is an accuracy downgrade and must be an explicit choice, not a side effect
    of dragging a map.
    """
    if not isinstance(existing, dict):
        return False
    have = _METHOD_RANK.get(str(existing.get("method", "")), -1)
    want = _METHOD_RANK.get(str(proposed.get("method", "")), -1)
    return want < have


def describe(record: dict[str, Any] | None) -> str:
    """One human line for a calibration, for receipts and UI."""
    if not isinstance(record, dict) or record.get("meters_per_unit") is None:
        return "not calibrated"
    factor = record["meters_per_unit"]
    method = record.get("method", "unknown")
    unc = record.get("uncertainty") or {}
    count = unc.get("n") or 0
    if count >= 2:
        relative = unc.get("relative")
        spread = f" +/- {relative:.1%}" if isinstance(relative, (int, float)) else ""
        detail = f"{count} references{spread}"
        if unc.get("disagreement"):
            detail += " -- references disagree, re-measure the outlier"
    elif count == 1:
        detail = "1 reference (no cross-check)"
    else:
        detail = "no stored reference"
    return f"{factor:.6g} m/unit via {method} ({detail})"
