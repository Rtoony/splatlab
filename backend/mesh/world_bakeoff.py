#!/usr/bin/env python3
"""Candidate bake-off: score every competing reconstruction of ONE object
against the real photographs, in ONE coordinate frame.

`mesh_gate.py` renders through the capture's real TRAIN cameras, so it only
ever produces an honest number for geometry that is expressed in the CAPTURE
frame (Z-up, scene units). Our finished artifacts are not: `twin_finish.py`
and `object_texture.py` both export Y-up glTF, optionally rescaled to metres.
Handing those straight to the gate draws the object sideways, at the wrong
size, somewhere else in the scene, and scores it against whatever background
it happens to cover. The resulting PSNR is not a weak score — it is not a
score at all. Registering first is the whole point of this module.

Two registration modes, and the difference is reported, never blurred:

  exact_recorded  -- the artifact was written by one of OUR exporters, whose
                     convention is `v_export = R_YUP @ (meters_per_unit *
                     v_scene)`. The build's own receipt (twin_finish.json /
                     object_texture.json) records the factor it actually used,
                     so the inverse is a known transform, not a guess.

  approximate_bbox -- the artifact is of unknown provenance. Fall back to a
                     similarity fit (uniform scale + translation, plus a yaw
                     sweep) derived from the reference object's bounding box.
                     An axis-aligned bbox cost is 180-degenerate in yaw, so
                     yaw is NOT uniquely determined; such a candidate is
                     reported with its numbers but is NOT ranked against the
                     exactly-registered ones. Silently presenting a fitted
                     alignment as an exact one is the failure mode this whole
                     file exists to prevent.

Every registration is then VERIFIED against the bbox the renderer actually
drew (`bbox_render_frame`, reported back by mesh_gate.py) — not against the
bbox this module hoped it had produced — and a candidate whose registered
bbox does not land on the reference object is marked `untrusted` instead of
being ranked.

Ranking uses median PSNR over COVERED pixels, which is meaningful per-object.
SSIM is recorded but never ranked on: full-frame SSIM against a black-backed
render of a single small object is dominated by the background.

Usage: world_bakeoff.py <job_dir> --object <slug> [--cams 8]
         [--candidates NAME=PATH ...] [--skip NAME ...] [--out-dir DIR]
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

MESH_GATE = Path(__file__).resolve().parent / "mesh_gate.py"

# The convention BOTH of our exporters hard-code (twin_finish.py:131,
# object_texture.py:721):  yup = (x, z, -y) applied after scaling by
# meters_per_unit. As a matrix, v_export = R_YUP @ (s * v_scene).
R_YUP = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]])
# Inverse rotation: R_YUP.T @ (X, Y, Z) = (X, -Z, Y). Independently re-derived
# by the mesh-quality lab (twin_gate_prep.py) after a first smoke test rendered
# the hydrant lying on its side — the cheapest possible confirmation that this
# is the right inverse and not its mirror.
R_YUP_INV = R_YUP.T

# Registration is trusted only if the registered bbox actually lands on the
# reference object. Generous, because a legitimately CROPPED candidate (the
# textured build crops to the object's tight bbox) shifts its centre a little
# and shrinks a lot; the thing being caught here is a whole wrong frame, which
# misses by a large fraction of the object or blows the scale up.
MAX_CENTRE_OFFSET_FRAC = 0.25   # of the reference bbox diagonal
MAX_EXTENT_RATIO = 1.5          # registered candidate may not dwarf the reference
# A similarity fit is only worth reporting if one uniform scale explains all
# three axes. Beyond this the shape disagreement means the fit is meaningless.
MAX_FIT_ANISOTROPY = 1.35
# Predicted-vs-rendered bbox agreement, as a fraction of the reference
# diagonal. Anything above this means this module and the renderer disagree
# about what geometry was drawn, which invalidates the number outright.
BBOX_VERIFY_TOL_FRAC = 0.02
# A bbox can agree while the silhouettes land on different pixels, so the
# registration is ALSO checked in image space against the reference's own
# render. Legitimately cropped candidates score well under 1.0 here, so this
# floor only catches "not scoring the same object at all".
MIN_SILHOUETTE_IOU = 0.10
# A camera whose pose is inconsistent with the reconstruction penalises EVERY
# candidate at once. That shared-mode dip is detectable and is a property of
# the capture, not of any candidate — flagged, and excluded from the paired
# comparison, rather than silently averaged in.
SUSPECT_CAM_DB = -2.0
# Below this the gate itself refuses to score a frame; used to recognise the
# same "camera saw almost nothing" condition when pairing cameras.
COVERAGE_FLOOR = 0.05


def _affine(linear: np.ndarray, translation=None) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    if translation is not None:
        matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _vertices(path: Path) -> np.ndarray:
    """Vertices in the file's OWN frame, scene-graph transforms applied.

    trimesh, not Open3D, deliberately: Open3D materialises an empty texture
    placeholder for every image-less material and touching one aborts the
    process (uncatchable C++ terminate). The gate handles that; this planner
    has no reason to go near it. The two agree on the bbox of every artifact
    in this pipeline, and where they might not, the rendered-bbox verification
    below is what settles it.
    """
    mesh = trimesh.load(str(path), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        raise ValueError(f"{path} has no vertices")
    return verts


def _bbox(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return verts.min(axis=0), verts.max(axis=0)


def _apply(matrix: np.ndarray, verts: np.ndarray) -> np.ndarray:
    return verts @ matrix[:3, :3].T + matrix[:3, 3]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _our_convention(receipt: dict | None, job_mpu: float | None) -> dict | None:
    """The scale + up-axis OUR exporter actually used for this artifact.

    Read from the build's own receipt rather than inferred from the file, and
    the provenance of each field is recorded, because the two are not equally
    strong: `meters_per_unit: null` in a receipt is a positive statement that
    the build ran uncalibrated (scale 1.0), whereas an ABSENT field only means
    the artifact predates that field — those must not be conflated.
    """
    if not receipt:
        return None
    sources: dict[str, str] = {}

    if "meters_per_unit" in receipt:
        mpu = receipt["meters_per_unit"]
        sources["meters_per_unit"] = "receipt.meters_per_unit (explicit)"
    else:
        units = str(receipt.get("units", ""))
        if units.startswith("meters"):
            if job_mpu is None:
                return None  # metres, but nothing says how many per scene unit
            mpu = job_mpu
            sources["meters_per_unit"] = (
                "receipt.units=='meters' + meta.json meters_per_unit "
                "(receipt predates the meters_per_unit field)"
            )
        elif "scene-unit" in units:
            mpu = None
            sources["meters_per_unit"] = "receipt.units=='scene-units (uncalibrated)'"
        else:
            return None

    up_axis = receipt.get("up_axis")
    if up_axis:
        sources["up_axis"] = "receipt.up_axis (explicit)"
    else:
        up_axis = "Y"
        sources["up_axis"] = (
            "exporter convention: twin_finish.py and object_texture.py ALWAYS "
            "write Y-up glTF (receipt predates the up_axis field)"
        )
    return {"meters_per_unit": mpu, "up_axis": str(up_axis).upper(), "sources": sources}


def _exact_transform(conv: dict) -> tuple[np.ndarray, dict]:
    """Invert our own export convention. No fitting, no reference geometry."""
    scale = float(conv["meters_per_unit"] or 1.0)
    linear = (R_YUP_INV if conv["up_axis"] == "Y" else np.eye(3)) / scale
    return _affine(linear), {
        "mode": "exact_recorded",
        "meters_per_unit": conv["meters_per_unit"],
        "up_axis": conv["up_axis"],
        "field_provenance": conv["sources"],
        "formula": "v_scene = (1 / meters_per_unit) * R_YUP.T @ v_export"
                   if conv["up_axis"] == "Y" else
                   "v_scene = (1 / meters_per_unit) * v_export",
    }


def _approx_transform(verts: np.ndarray, ref_lo: np.ndarray, ref_hi: np.ndarray,
                      ref_name: str, job_mpu: float | None,
                      yaw_step: float = 5.0) -> tuple[np.ndarray, dict]:
    """Similarity fit (yaw + uniform scale + translation) onto a reference bbox.

    For a candidate whose provenance we do not own. glTF is Y-up by
    specification, so that rotation is assumed first; what is genuinely fitted
    is one uniform scale, one translation, and a yaw about the capture frame's
    up axis (Z). The yaw is scored purely geometrically — how well the
    axis-aligned extents agree — so no photograph ever influences the
    alignment. An AABB cost cannot distinguish yaw from yaw+180 deg, which is
    stated in the returned provenance rather than papered over.
    """
    ref_ext = np.maximum(ref_hi - ref_lo, 1e-9)
    ref_centre = (ref_lo + ref_hi) / 2.0
    ref_diag = float(np.linalg.norm(ref_ext))

    best = None
    costs = []
    for yaw in np.arange(0.0, 360.0, yaw_step):
        rad = np.radians(yaw)
        cos_y, sin_y = np.cos(rad), np.sin(rad)
        rot_z = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
        linear = rot_z @ R_YUP_INV
        rotated = verts @ linear.T
        lo, hi = _bbox(rotated)
        ext = np.maximum(hi - lo, 1e-9)
        scale = ref_diag / float(np.linalg.norm(ext))
        cost = float(np.abs(np.log((scale * ext) / ref_ext)).sum())
        costs.append(cost)
        if best is None or cost < best["cost"]:
            best = {"cost": cost, "yaw": float(yaw), "linear": linear,
                    "scale": float(scale), "lo": lo, "hi": hi, "ext": ext}

    linear = best["scale"] * best["linear"]
    centre_after = best["scale"] * (best["lo"] + best["hi"]) / 2.0
    matrix = _affine(linear, ref_centre - centre_after)
    fitted_ext = best["scale"] * best["ext"]
    return matrix, {
        "mode": "approximate_bbox",
        "fit_reference": ref_name,
        "assumed_source_convention": "glTF Y-up (spec default) — assumed, not recorded",
        "yaw_deg": round(best["yaw"], 1),
        "yaw_step_deg": yaw_step,
        "yaw_cost": round(best["cost"], 4),
        # How much the cost actually varies across the sweep. A near-flat
        # landscape means the bbox does not constrain yaw at all (true of any
        # roughly axisymmetric object, e.g. a hydrant) — the chosen yaw is then
        # arbitrary and any feature whose position depends on it (a nozzle, a
        # cap) is unaligned. Reported so nobody reads yaw_deg as a measurement.
        "yaw_cost_spread": round(float(max(costs) - min(costs)), 4),
        "yaw_degeneracy": "an axis-aligned-bbox cost is identical at yaw and yaw+180 deg; "
                          "yaw is determined only up to a half turn, and not at all when "
                          "yaw_cost_spread is small",
        "uniform_scale": round(best["scale"], 6),
        # If the foreign file really is in metres, 1/scale should land on the
        # capture's own metre calibration. An independent cross-check of the
        # fit that costs nothing to report.
        "implied_meters_per_unit": round(1.0 / best["scale"], 4) if best["scale"] else None,
        "job_meters_per_unit": job_mpu,
        "fitted_extent_ratio": [round(float(x), 3) for x in (fitted_ext / ref_ext)],
        "caveat": "uniform scale + translation + bbox-fitted yaw; NOT a recorded transform",
    }


@dataclass
class Candidate:
    name: str
    path: Path
    provenance: str                    # capture-native | ours | foreign
    receipt: Path | None = None
    note: str = ""
    matrix: np.ndarray | None = None
    registration: dict = field(default_factory=dict)
    predicted: dict = field(default_factory=dict)
    gate: dict | None = None
    error: str | None = None


def _discover(mesh_dir: Path) -> list[Candidate]:
    """The three artifacts our own object lane produces, when they exist."""
    built_in = [
        Candidate("raw_tsdf", mesh_dir / "mesh.ply", "capture-native",
                  mesh_dir / "mesh.json",
                  "TSDF export writes the capture frame directly (run_mesh.sh -> "
                  "mesh_report.py); no registration needed"),
        Candidate("twin", mesh_dir / "twin.glb", "ours", mesh_dir / "twin_finish.json",
                  "twin_finish.py: colour transfer + decimate, Y-up GLB"),
        Candidate("textured", mesh_dir / "textured.glb", "ours",
                  mesh_dir / "object_texture.json",
                  "object_texture.py: crop + Poisson + UV atlas, Y-up GLB in metres"),
    ]
    return [c for c in built_in if c.path.is_file()]


def _register(cands: list[Candidate], ref: Candidate, job_mpu: float | None,
              yaw_step: float) -> None:
    """Plan every candidate's transform, then pick an alignment reference.

    Ordered deliberately: exactly-registered candidates first, because the
    similarity fit for a foreign candidate needs a same-frame reference and
    the honest one is the TIGHTEST exact candidate, not the raw TSDF. The raw
    TSDF's bbox spans the isolate region — object PLUS the ground around it —
    so fitting a foreign model to it would scale the model up to cover the
    ground as well.
    """
    for cand in cands:
        verts = _vertices(cand.path)
        cand.predicted["source_bbox"] = {
            "centre": [round(float(x), 5) for x in (verts.min(0) + verts.max(0)) / 2],
            "extent": [round(float(x), 5) for x in verts.max(0) - verts.min(0)],
        }
        if cand.provenance == "capture-native":
            cand.matrix = np.eye(4)
            cand.registration = {"mode": "identity", "why": cand.note}
        elif cand.provenance == "ours":
            conv = _our_convention(_read_json(cand.receipt) if cand.receipt else None, job_mpu)
            if conv is None:
                # Our own artifact, but its receipt cannot pin the convention.
                # Demote to the foreign path rather than assume a scale.
                cand.provenance = "foreign"
                cand.note += (" | receipt missing or unreadable — cannot recover the "
                              "recorded transform, demoted to approximate")
                continue
            cand.matrix, cand.registration = _exact_transform(conv)
        if cand.matrix is not None:
            reg = _apply(cand.matrix, verts)
            lo, hi = _bbox(reg)
            cand.predicted["registered_bbox"] = {
                "centre": [round(float(x), 5) for x in (lo + hi) / 2],
                "extent": [round(float(x), 5) for x in hi - lo],
            }

    exact = [c for c in cands if c.matrix is not None]
    align_ref = min(
        exact,
        key=lambda c: float(np.prod(np.asarray(c.predicted["registered_bbox"]["extent"]))),
    ) if exact else ref
    ext = np.asarray(align_ref.predicted["registered_bbox"]["extent"])
    centre = np.asarray(align_ref.predicted["registered_bbox"]["centre"])
    ref_lo, ref_hi = centre - ext / 2, centre + ext / 2
    ref_label = (f"{align_ref.name} (registered, tightest same-frame model of the object; "
                 f"the raw TSDF bbox also contains the surrounding ground)")

    for cand in cands:
        if cand.matrix is not None:
            continue
        verts = _vertices(cand.path)
        cand.matrix, cand.registration = _approx_transform(
            verts, ref_lo, ref_hi, ref_label, job_mpu, yaw_step)
        lo, hi = _bbox(_apply(cand.matrix, verts))
        cand.predicted["registered_bbox"] = {
            "centre": [round(float(x), 5) for x in (lo + hi) / 2],
            "extent": [round(float(x), 5) for x in hi - lo],
        }


def _run_gate(cand: Candidate, config: Path, out_dir: Path, cams: int,
              python: Path) -> None:
    cand_dir = out_dir / cand.name
    cand_dir.mkdir(parents=True, exist_ok=True)
    xform_path = cand_dir / "transform.json"
    xform_path.write_text(json.dumps({
        "matrix": cand.matrix.tolist(),
        "registration": cand.registration,
        "frame": "capture (nerfstudio dataparser, Z-up, scene units)",
    }, indent=2))

    # Clear the previous run's evidence FIRST. Otherwise a failed gate would
    # leave a stale mesh_gate.json to be picked up and reported as this run's
    # score, and strips from a different --cams would sit in the evidence dir
    # looking like they belong to these numbers.
    (cand_dir / "mesh_gate.json").unlink(missing_ok=True)
    for stale in cand_dir.glob("gate_cam*.jpg"):
        stale.unlink()

    cmd = [str(python), str(MESH_GATE), str(cand.path), str(config), str(cand_dir),
           "--cams", str(cams), "--transform", str(xform_path)]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report = _read_json(cand_dir / "mesh_gate.json")
    if proc.returncode != 0 or report is None:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-4:])
        cand.error = f"mesh_gate exit {proc.returncode}: {tail or 'no mesh_gate.json written'}"
        return
    report["seconds"] = round(time.time() - started, 1)
    cand.gate = report


def _silhouette_masks(cand_dir: Path, gate: dict) -> dict[int, np.ndarray]:
    """The rendered silhouette per camera, read back off the gate's own strips.

    Each strip is [photo | render] and the render is on a black background, so
    the right half thresholded is exactly the pixels the candidate covered.
    Reading the receipt back is deliberate: it verifies registration where it
    actually matters — in the image the score was computed from — instead of
    in the 3D bbox, where two candidates can agree and still cover different
    pixels.
    """
    masks: dict[int, np.ndarray] = {}
    for entry in gate.get("per_cam", []):
        strip = cand_dir / f"gate_cam{int(entry['cam']):03d}.jpg"
        if not strip.is_file():
            continue
        arr = np.asarray(Image.open(strip)).astype(np.int32)
        render = arr[:, arr.shape[1] // 2:]
        # Sum over RGB out of 765; a loose threshold absorbs JPEG ringing
        # around the silhouette without admitting the black background.
        masks[int(entry["cam"])] = render.sum(axis=2) > 60
    return masks


def _silhouette_iou(cand: dict[int, np.ndarray], ref: dict[int, np.ndarray]) -> dict:
    per_cam = {}
    for cam, mask in cand.items():
        other = ref.get(cam)
        if other is None or other.shape != mask.shape:
            continue
        union = int((mask | other).sum())
        per_cam[cam] = round(float((mask & other).sum() / union), 3) if union else 0.0
    return {
        "per_cam": per_cam,
        "median": round(float(np.median(list(per_cam.values()))), 3) if per_cam else None,
    }


def _residual(cand: Candidate, ref_centre: np.ndarray, ref_ext: np.ndarray) -> dict:
    """Residual from the bbox the RENDERER drew, cross-checked against the plan."""
    rendered = cand.gate["bbox_render_frame"]
    centre = np.asarray(rendered["centre"], dtype=float)
    extent = np.asarray(rendered["extent"], dtype=float)
    predicted = cand.predicted.get("registered_bbox")
    ref_diag = float(np.linalg.norm(ref_ext))

    drift = None
    if predicted:
        drift = float(np.linalg.norm(centre - np.asarray(predicted["centre"], dtype=float)))
    offset = float(np.linalg.norm(centre - ref_centre))
    ratios = extent / np.maximum(ref_ext, 1e-9)
    return {
        "measured_on": "mesh_gate bbox_render_frame (the geometry actually drawn)",
        "plan_vs_rendered_centre_drift": round(drift, 5) if drift is not None else None,
        "plan_verified": (drift is not None and drift <= BBOX_VERIFY_TOL_FRAC * ref_diag),
        "centre_offset_units": round(offset, 5),
        "centre_offset_frac_of_ref_diag": round(offset / ref_diag, 4) if ref_diag else None,
        "extent_ratio_vs_ref": [round(float(x), 3) for x in ratios],
        "max_extent_ratio": round(float(ratios.max()), 3),
    }


def _judge(cand: Candidate, residual: dict) -> tuple[str, str]:
    """exact | approximate | untrusted, with the reason stated."""
    if not residual["plan_verified"]:
        return "untrusted", (
            "the bbox mesh_gate rendered does not match the bbox this registration "
            f"predicted (centre drift {residual['plan_vs_rendered_centre_drift']} scene "
            "units) — the number would describe geometry other than the one planned")
    frac = residual["centre_offset_frac_of_ref_diag"]
    if frac is not None and frac > MAX_CENTRE_OFFSET_FRAC:
        return "untrusted", (
            f"registered centre sits {frac:.2f} of a reference diagonal away from the "
            "reference object — it is not being scored where the object is")
    if residual["max_extent_ratio"] > MAX_EXTENT_RATIO:
        return "untrusted", (
            f"registered extent is {residual['max_extent_ratio']:.2f}x the reference on "
            "one axis — the scale of the registration is wrong")
    iou = residual.get("silhouette_iou_vs_reference")
    if iou is not None and iou < MIN_SILHOUETTE_IOU:
        return "untrusted", (
            f"the rendered silhouette barely overlaps the reference's (median IoU {iou}) — "
            "whatever it is being scored against, it is not the same pixels")
    if cand.registration.get("mode") == "approximate_bbox":
        ratios = np.asarray(cand.registration["fitted_extent_ratio"], dtype=float)
        aniso = float(np.exp(np.abs(np.log(np.maximum(ratios, 1e-9))).max()))
        if aniso > MAX_FIT_ANISOTROPY:
            return "untrusted", (
                f"no single uniform scale fits the reference bbox (worst axis off by "
                f"{aniso:.2f}x) — the similarity fit is not describing the same shape")
        return "approximate", (
            "alignment was FITTED to the reference object's bbox (uniform scale + "
            f"translation + bbox-fitted yaw {cand.registration['yaw_deg']} deg, "
            "undetermined up to 180 deg), not recovered from a recorded transform")
    return "exact", cand.registration.get(
        "formula", cand.registration.get("why", "registered in the capture frame"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Score competing reconstructions in ONE frame.")
    ap.add_argument("job_dir")
    ap.add_argument("--object", required=True, help="object slug under <job>/_objects/")
    ap.add_argument("--cams", type=int, default=8)
    ap.add_argument("--candidates", nargs="*", default=[],
                    help="extra candidates as NAME=PATH (repeat); a NAME matching a built-in "
                         "overrides its path")
    ap.add_argument("--skip", nargs="*", default=[], help="candidate names to drop")
    ap.add_argument("--out-dir", default=None,
                    help="default <job>/_objects/<slug>/mesh/bakeoff")
    ap.add_argument("--config", default=None, help="override the splatfacto config.yml")
    ap.add_argument("--meters-per-unit", type=float, default=None,
                    help="override the job's metre calibration")
    ap.add_argument("--yaw-step", type=float, default=5.0)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used for mesh_gate.py subprocesses")
    args = ap.parse_args()

    job_dir = Path(args.job_dir).resolve()
    mesh_dir = job_dir / "_objects" / args.object / "mesh"
    if not mesh_dir.is_dir():
        print(f"FATAL: no object mesh dir at {mesh_dir}", file=sys.stderr)
        return 2

    if args.config:
        config = Path(args.config).resolve()
    else:
        found = sorted((job_dir / "processed").rglob("splatfacto/*/config.yml"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            print(f"FATAL: no splatfacto config.yml under {job_dir}/processed",
                  file=sys.stderr)
            return 2
        config = found[0].resolve()

    meta = _read_json(job_dir / "meta.json") or {}
    job_mpu = args.meters_per_unit if args.meters_per_unit is not None \
        else meta.get("meters_per_unit")

    cands = _discover(mesh_dir)
    by_name = {c.name: c for c in cands}
    for spec in args.candidates:
        name, sep, raw = spec.partition("=")
        if not sep:
            name, raw = Path(spec).stem, spec
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            print(f"FATAL: candidate {name!r} not found: {path}", file=sys.stderr)
            return 2
        if name in by_name:
            by_name[name].path = path
        else:
            cand = Candidate(name, path, "foreign", None,
                             "supplied on the command line; provenance unknown")
            cands.append(cand)
            by_name[name] = cand
    cands = [c for c in cands if c.name not in set(args.skip)]
    if not cands:
        print("FATAL: no candidates to score", file=sys.stderr)
        return 2

    ref = next((c for c in cands if c.provenance == "capture-native"), None)
    if ref is None:
        print("FATAL: no capture-frame reference mesh (mesh.ply) — there is nothing to "
              "register the other candidates against", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve() if args.out_dir else mesh_dir / "bakeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    _register(cands, ref, job_mpu, args.yaw_step)
    for cand in cands:
        print(f"[bakeoff] {cand.name}: {cand.registration.get('mode')} -> gate", flush=True)
        _run_gate(cand, config, out_dir, args.cams, Path(args.python))
        if cand.error:
            print(f"[bakeoff] {cand.name}: FAILED {cand.error}", file=sys.stderr, flush=True)

    if ref.gate is None:
        print(f"FATAL: the reference candidate did not score ({ref.error}); without it "
              "no residual can be measured", file=sys.stderr)
        return 1
    ref_centre = np.asarray(ref.gate["bbox_render_frame"]["centre"], dtype=float)
    ref_ext = np.asarray(ref.gate["bbox_render_frame"]["extent"], dtype=float)
    ref_masks = _silhouette_masks(out_dir / ref.name, ref.gate)

    # --- camera health -------------------------------------------------
    # A frame whose pose is inconsistent with the reconstruction drags EVERY
    # candidate down at the same camera. That shared dip is separable from a
    # candidate defect (which is not shared) and belongs to the capture, so it
    # is named and excluded from the paired comparison instead of quietly
    # entering four medians at once.
    psnr_by_cam = {c.name: {e["cam"]: e["psnr_covered"] for e in c.gate["per_cam"]
                            if "psnr_covered" in e} for c in cands if c.gate}
    cov_by_cam = {c.name: {e["cam"]: e["coverage"] for e in c.gate["per_cam"]}
                  for c in cands if c.gate}
    all_cams = sorted({cam for m in cov_by_cam.values() for cam in m})
    own_median = {n: float(np.median(list(v.values()))) for n, v in psnr_by_cam.items() if v}
    suspect_cams: dict[int, float | None] = {}
    for cam in all_cams:
        offsets = [psnr_by_cam[n][cam] - own_median[n]
                   for n in own_median if cam in psnr_by_cam[n]]
        if offsets and float(np.median(offsets)) <= SUSPECT_CAM_DB:
            suspect_cams[cam] = round(float(np.median(offsets)), 2)
        elif all(v.get(cam, 0.0) < COVERAGE_FLOOR for v in cov_by_cam.values()):
            suspect_cams[cam] = None   # nothing was visible to anyone here
    paired_cams = [cam for cam in all_cams if cam not in suspect_cams
                   and all(cam in m for m in psnr_by_cam.values())]

    def _paired(name: str, table: dict) -> float | None:
        vals = [table[name][cam] for cam in paired_cams if cam in table.get(name, {})]
        return round(float(np.median(vals)), 3) if vals else None

    entries = []
    for cand in cands:
        entry = {
            "name": cand.name,
            "source": str(cand.path),
            "source_bytes": cand.path.stat().st_size,
            "provenance": cand.provenance,
            "note": cand.note,
            "receipt": str(cand.receipt) if cand.receipt and cand.receipt.is_file() else None,
            "transform": cand.matrix.tolist() if cand.matrix is not None else None,
            "registration": cand.registration,
            "bbox": cand.predicted,
        }
        if cand.gate is None:
            entry.update({"trust": "failed", "why": cand.error, "scores": None})
            entries.append(entry)
            continue
        residual = _residual(cand, ref_centre, ref_ext)
        iou = _silhouette_iou(_silhouette_masks(out_dir / cand.name, cand.gate), ref_masks)
        residual["silhouette_iou_vs_reference"] = iou["median"]
        residual["silhouette_iou_per_cam"] = iou["per_cam"]
        trust, why = _judge(cand, residual)
        entry["bbox"]["rendered_bbox"] = cand.gate["bbox_render_frame"]
        entry.update({
            "registration_residual": residual,
            "trust": trust,
            "why": why,
            "scores": {
                # Paired = the same camera set for every candidate, suspect
                # frames removed. It is what the ranking uses; the all-camera
                # medians are kept beside it so the difference is visible.
                "median_psnr_paired": _paired(cand.name, psnr_by_cam),
                "median_coverage_paired": _paired(cand.name, cov_by_cam),
                "median_coverage": cand.gate["median_coverage"],
                "median_psnr": cand.gate["median_psnr"],
                "median_ssim": cand.gate["median_ssim"],
                "shading": cand.gate["shading"],
                "cams": cand.gate["cams"],
                "seconds": cand.gate["seconds"],
            },
            "per_cam": cand.gate["per_cam"],
            "artifacts": {
                "dir": str(out_dir / cand.name),
                "strips": cand.gate["artifacts"],
                "gate_report": "mesh_gate.json",
                "transform": "transform.json",
            },
        })
        entries.append(entry)

    def _rank_psnr(entry: dict) -> float | None:
        scores = entry.get("scores") or {}
        return scores.get("median_psnr_paired") or scores.get("median_psnr")

    rankable = [e for e in entries if e["trust"] == "exact" and _rank_psnr(e) is not None]
    rankable.sort(key=lambda e: (-_rank_psnr(e), -e["scores"]["median_coverage"]))

    caveats = [
        "median_psnr is computed over COVERED pixels only, so it is a per-object number; "
        "the PAIRED median (same cameras for every candidate) is the ranking metric.",
        "median_ssim is FULL-FRAME against a black-backed render. On a single small object "
        "it is dominated by background and is reported but NOT ranked on.",
    ]
    shadings = {e["scores"]["shading"] for e in entries if e.get("scores")}
    if len(shadings) > 1:
        caveats.append(
            f"candidates rendered with different colour sources ({sorted(shadings)}); an "
            "'untextured' candidate is drawn flat white and its PSNR reflects that, not "
            "geometry alone.")
    scored = [e for e in entries if e.get("scores")]
    if scored:
        vols = {e["name"]: float(np.prod(np.asarray(e["bbox"]["rendered_bbox"]["extent"])))
                for e in scored}
        if max(vols.values()) > 2.0 * min(vols.values()):
            caveats.append(
                "candidates enclose different amounts of the scene "
                f"(bbox volumes {json.dumps({k: round(v, 4) for k, v in vols.items()})}); "
                "median_coverage therefore measures how much is modelled, not how well — "
                "do not read it as a quality ranking.")

    if suspect_cams:
        caveats.append(
            f"camera(s) {sorted(suspect_cams)} penalised EVERY candidate at once "
            f"(shared median PSNR offset {json.dumps(suspect_cams)} dB, null = nothing "
            "visible). A shared-mode dip is a property of the capture — a frame whose "
            "pose is inconsistent with the reconstruction — not a candidate defect. Those "
            "frames are excluded from the paired comparison; their strips are kept.")

    ranked_names = {e["name"] for e in rankable}
    not_ranked = [{"name": e["name"], "trust": e["trust"], "why": e["why"],
                   "scores": e["scores"]}
                  for e in entries if e["name"] not in ranked_names]
    verdict = {
        "ranked_on": "median_psnr_paired over covered pixels, exact registrations only",
        "ranked": [{"name": e["name"], **e["scores"]} for e in rankable],
        "not_ranked": not_ranked,
        "caveats": caveats,
    }
    # A median gap of a few tenths of a dB is not a result. The paired cameras
    # let the winner be checked the honest way — camera by camera, same frame,
    # same photo — so a narrow median win that is not actually consistent
    # cannot be presented as one.
    if len(rankable) > 1:
        top_name = rankable[0]["name"]
        verdict["head_to_head"] = {
            other["name"]: {
                "winner_ahead_on_cams": sum(
                    1 for cam in paired_cams
                    if psnr_by_cam[top_name].get(cam, -np.inf)
                    > psnr_by_cam[other["name"]].get(cam, -np.inf)),
                "of_cams": len(paired_cams),
                "median_margin_db": round(_rank_psnr(rankable[0]) - _rank_psnr(other), 2),
            }
            for other in rankable[1:]
        }

    if rankable:
        top = rankable[0]
        runner = rankable[1] if len(rankable) > 1 else None
        reason = (f"{top['name']} scores {_rank_psnr(top)} dB median PSNR over covered "
                  f"pixels across {len(paired_cams)} paired cameras, at "
                  f"{top['scores']['median_coverage_paired']} coverage, rendered from "
                  f"{top['scores']['shading']}, registered by "
                  f"{top['registration'].get('mode')}")
        if runner:
            h2h = verdict["head_to_head"][runner["name"]]
            reason += (f"; next is {runner['name']} at {_rank_psnr(runner)} dB "
                       f"({h2h['median_margin_db']:+} dB median, winner ahead on "
                       f"{h2h['winner_ahead_on_cams']}/{h2h['of_cams']} paired cameras)")
            if h2h["winner_ahead_on_cams"] * 2 <= h2h["of_cams"]:
                reason += (" — the median win is NOT consistent camera-by-camera; treat "
                           "the top two as tied")
        verdict["winner"] = top["name"]
        verdict["reason"] = reason
    else:
        verdict["winner"] = None
        verdict["reason"] = ("no candidate had a trustworthy registration and a score — "
                             "nothing here is comparable")

    report = {
        "v": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "job_dir": str(job_dir),
        "object": args.object,
        "config": str(config),
        "meters_per_unit": job_mpu,
        "frame": "capture frame (nerfstudio dataparser cameras, Z-up, scene units)",
        "reference": ref.name,
        "cams": args.cams,
        "camera_check": {
            "cams_used": all_cams,
            "paired_cams": paired_cams,
            "suspect_cams": suspect_cams,
            "rule": f"a camera is suspect when the MEDIAN across candidates of "
                    f"(its PSNR - that candidate's own median PSNR) is <= {SUSPECT_CAM_DB} dB "
                    f"— a dip shared by every candidate — or when no candidate cleared the "
                    f"{COVERAGE_FLOOR} coverage floor there",
        },
        "candidates": entries,
        "verdict": verdict,
    }
    (out_dir / "bakeoff.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"paired cameras: {paired_cams}   suspect: {suspect_cams or 'none'}")
    print(f"{'candidate':12s} {'trust':12s} {'cov':>6s} {'psnr':>7s} {'ssim':>7s} "
          f"{'IoU':>5s} {'shading':14s} registration")
    for entry in entries:
        s = entry.get("scores") or {}
        res = entry.get("registration_residual") or {}
        print(f"{entry['name']:12s} {entry['trust']:12s} "
              f"{str(s.get('median_coverage_paired', '-')):>6s} "
              f"{str(s.get('median_psnr_paired', '-')):>7s} "
              f"{str(s.get('median_ssim', '-')):>7s} "
              f"{str(res.get('silhouette_iou_vs_reference', '-')):>5s} "
              f"{str(s.get('shading', '-')):14s} "
              f"{entry['registration'].get('mode', '-')}")
    print()
    print(f"winner: {verdict['winner']} — {verdict['reason']}")
    print(f"report: {out_dir / 'bakeoff.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
