"""Schema-driven parametric authoring lane: numbers in, clean editable solid out.

This is the generalisation of the one-off, hand-tuned ~/scripts/build-hydrant.py.
That script proved the method (a lathe of a MEASURED radius-vs-height profile,
plus features placed at MEASURED azimuths/heights, plus idealised hardware and
flat material roles); this module turns its PARAMETERS block into a validated
schema (parametric_schema.json) and its build code into an archetype
interpreter, so the next object does not need a new hand-written script.

The hard rule of the lane
-------------------------
A model proposes PARAMETERS. It never emits geometry and never emits code.
Everything downstream of the schema instance is deterministic: the same JSON
always builds the same mesh, the JSON is small enough to review in a diff, and
a wrong number is edited rather than re-prompted. Anything that would let a
model smuggle geometry in (expressions, code strings, free-form vertex lists)
is deliberately absent from the schema.

Three stages, each independently runnable
-----------------------------------------
  measure   object mesh (or splat)  ->  a starting schema instance
  validate  schema instance         ->  loud errors, never silent defaults
  build     schema instance         ->  Blender -> GLB + re-opened receipt

`measure` exists so the lane is usable with NO model in the loop: it derives
the axis, the grade offset, the radius-vs-height profile, the radial feature
lobes and the material palette straight off the capture. A vision model's job
is then the genuinely visual residual (which cap is which, how many flange
bolts, the form of a nozzle end) on top of numbers that are already real.

Frames and the trap that has already been paid for
--------------------------------------------------
Model frame: metres, +z up, z = 0 at GRADE (underside of the base flange),
azimuth CCW from +x. An object-lane mesh is NOT in that frame — it carries a
ground/sidewalk slab under the object, so every height read off it is too high
by the slab thickness (~0.12 m for the reference hydrant). Subtract it. Get it
wrong and the features land on the dome. `measure` detects the offset and
records it in provenance.grade_offset_m; `validate` independently refuses a
profile whose widest ring looks like a slab rather than a flange.

Running it
----------
  # derive a schema from a capture (any python3 with numpy)
  python3 parametric_build.py measure \
      --mesh <job>/_objects/<slug>/mesh/mesh.ply \
      --meters-per-unit 1.9226381324854986 --name fire-hydrant \
      --out /tmp/fire-hydrant.schema.json

  # validate on its own (cheap, no Blender)
  python3 parametric_build.py validate /tmp/fire-hydrant.schema.json

  # build (re-execs Blender headless automatically; matches the proven
  # blender_assemble.py pattern -- NOT blender-mcp, which needs a GUI instance)
  python3 parametric_build.py build /tmp/fire-hydrant.schema.json \
      --out /tmp/fire-hydrant.glb --receipt /tmp/fire-hydrant.receipt.json

  # re-open any GLB and report what is actually in it
  python3 parametric_build.py inspect /tmp/fire-hydrant.glb

An exporter's exit code proves nothing
--------------------------------------
`build` never trusts export_scene.gltf. It re-opens the written GLB, decodes
the JSON chunk, walks the node graph, applies every node transform and decodes
the POSITION and index accessors out of the binary chunk -- then asserts the
triangle count, the world extent, the material count and the flange diameter
against values PREDICTED FROM THE SCHEMA before the build ran. A mismatch is a
non-zero exit, not a warning. (Same failure class this program already caught
once: a writer that silently corrupted its output while returning success.)

Dependencies: `measure` needs numpy; `build` needs bpy; `validate` and
`inspect` need neither -- the JSON Schema validator here is hand-rolled
precisely because Blender's bundled Python has no `jsonschema`, and a
validation step that only runs outside the builder is a validation step that
will one day not run at all.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("parametric_schema.json")
SCHEMA_VERSION = "parametric-archetype/1"
BLENDER_BIN = Path(
    os.environ.get(
        "SPLATLAB_BLENDER_BIN",
        str(Path.home() / "tools" / "blender-4.5.11-linux-x64" / "blender"),
    )
)

EPS = 1e-9


class SchemaError(Exception):
    """Structural or semantic rejection. Carries every problem found, not just the first."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(f"  - {e}" for e in errors))


# =========================================================================== #
# 1. JSON Schema validator (self-contained subset)                            #
# =========================================================================== #
# Supports exactly the keywords parametric_schema.json uses. Anything the
# schema starts using that is not handled here raises -- an unknown keyword is
# treated as an authoring bug, never silently ignored, because a silently
# ignored constraint is worse than no constraint.

_HANDLED = {
    "$schema", "$id", "$ref", "$defs", "title", "description", "type", "const",
    "enum", "properties", "required", "additionalProperties", "minProperties",
    "maxProperties", "items", "minItems", "maxItems", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern",
    "oneOf",
}

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _resolve(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _validate_node(value: Any, schema: dict, root: dict, path: str, errs: list[str]) -> None:
    unknown = set(schema) - _HANDLED
    if unknown:
        raise ValueError(f"schema at {path}: unsupported keyword(s) {sorted(unknown)}")

    if "$ref" in schema:
        _validate_node(value, _resolve(schema["$ref"], root), root, path, errs)
        return

    if "oneOf" in schema:
        matched = 0
        for branch in schema["oneOf"]:
            sub: list[str] = []
            _validate_node(value, branch, root, path, sub)
            if not sub:
                matched += 1
        if matched != 1:
            errs.append(f"{path}: matches {matched} of the {len(schema['oneOf'])} allowed shapes "
                        f"(exactly 1 required); value={_brief(value)}")
            return

    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must be {schema['const']!r}, got {_brief(value)}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: must be one of {schema['enum']}, got {_brief(value)}")
        return

    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        for t in allowed:
            if t not in _TYPE_CHECKS:
                raise ValueError(f"schema at {path}: unknown type {t!r}")
        if not any(_TYPE_CHECKS[t](value) for t in allowed):
            errs.append(f"{path}: must be {'/'.join(allowed)}, got {type(value).__name__} "
                        f"{_brief(value)}")
            return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, ok, word in (
            ("minimum", lambda v, b: v >= b, ">="),
            ("maximum", lambda v, b: v <= b, "<="),
            ("exclusiveMinimum", lambda v, b: v > b, ">"),
            ("exclusiveMaximum", lambda v, b: v < b, "<"),
        ):
            if key in schema and not ok(value, schema[key]):
                errs.append(f"{path}: {value!r} violates {word} {schema[key]!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']} chars")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']} chars")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: {value!r} does not match /{schema['pattern']}/")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: needs at least {schema['minItems']} items, has {len(value)}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: at most {schema['maxItems']} items, has {len(value)}")
        if "items" in schema:
            for i, item in enumerate(value):
                _validate_node(item, schema["items"], root, f"{path}[{i}]", errs)

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errs.append(f"{path}.{key}: REQUIRED and absent — this lane never defaults a "
                            f"missing value ({_describe(props.get(key, {}), root)})")
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errs.append(f"{path}: needs at least {schema['minProperties']} entries")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            errs.append(f"{path}: at most {schema['maxProperties']} entries")
        extra_schema = schema.get("additionalProperties", True)
        for key, sub in value.items():
            if key in props:
                _validate_node(sub, props[key], root, f"{path}.{key}", errs)
            elif extra_schema is False:
                errs.append(f"{path}.{key}: unknown property (schema forbids extras) — "
                            f"known: {sorted(props)}")
            elif isinstance(extra_schema, dict):
                _validate_node(sub, extra_schema, root, f"{path}.{key}", errs)


def _brief(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe(schema: dict, root: dict) -> str:
    if "$ref" in schema:
        schema = _resolve(schema["$ref"], root)
    desc = schema.get("description", "")
    return (desc[:120] + "...") if len(desc) > 120 else (desc or "no description")


def load_schema() -> dict:
    if not SCHEMA_PATH.is_file():
        raise SchemaError([f"schema file missing: {SCHEMA_PATH}"])
    return json.loads(SCHEMA_PATH.read_text())


def validate_structure(instance: dict, schema: dict | None = None) -> None:
    """JSON-Schema pass. Raises SchemaError listing EVERY structural problem."""
    schema = schema or load_schema()
    errs: list[str] = []
    _validate_node(instance, schema, schema, "$", errs)
    if errs:
        raise SchemaError(errs)


# =========================================================================== #
# 2. Profile helpers                                                          #
# =========================================================================== #

def profile_points(instance: dict) -> list[tuple[float, float]]:
    return [(float(p["r"]), float(p["z"])) for p in instance["profile"]]


def profile_height(instance: dict) -> float:
    pts = profile_points(instance)
    return max(z for _, z in pts) - min(z for _, z in pts)


def profile_max_radius(instance: dict) -> float:
    return max(r for r, _ in profile_points(instance))


def body_radius_at(pts: list[tuple[float, float]], z: float) -> float:
    """Outer radius of the solid of revolution at height z.

    Takes the MAXIMUM over every profile segment spanning z, so a vertical
    segment (a flange rim) reports the rim, not whatever the interpolation
    happened to hit first.
    """
    best = 0.0
    for (r0, z0), (r1, z1) in zip(pts, pts[1:]):
        lo, hi = min(z0, z1), max(z0, z1)
        if lo - EPS <= z <= hi + EPS:
            t = 0.0 if abs(z1 - z0) < EPS else (z - z0) / (z1 - z0)
            best = max(best, r0 + (r1 - r0) * max(0.0, min(1.0, t)))
    if best == 0.0:  # above the apex or below the base: clamp to the nearest end
        zs = [z for _, z in pts]
        best = pts[0][0] if z < min(zs) else pts[-1][0]
    return best


def barrel_radius(instance: dict) -> float:
    """Representative body radius: the median profile radius over the middle of
    the object, used as the yardstick for 'is that widest ring a flange, or is
    it a ground slab someone forgot to subtract?'."""
    pts = profile_points(instance)
    z_lo, z_hi = min(z for _, z in pts), max(z for _, z in pts)
    span = z_hi - z_lo
    mid = sorted(r for r, z in pts if r > EPS and z_lo + 0.25 * span <= z <= z_lo + 0.85 * span)
    if not mid:
        mid = sorted(r for r, _ in pts if r > EPS)
    return mid[len(mid) // 2]


# =========================================================================== #
# 3. Semantic validation (everything JSON Schema structurally cannot say)     #
# =========================================================================== #

SLAB_RATIO = 2.5  # widest ring / barrel radius above which the "flange" is really a ground slab


def _material_usage(instance: dict) -> dict[str, int]:
    """How many primitive groups each declared role will actually receive."""
    used = {role: 0 for role in instance["materials"]}

    def bump(role: Any) -> None:
        if isinstance(role, str) and role in used:
            used[role] += 1

    # the lathe body always goes to slot 0's role
    slot0 = min(instance["materials"], key=lambda k: instance["materials"][k]["slot"])
    bump(slot0)
    for feat in instance["features"]:
        bump(feat["body_material"])
        if feat["type"] == "nozzle":
            bump(feat["cap_material"])
            bump(feat["nut_material"])
    if instance["flange_bolts"]:
        bump(instance["flange_bolts"]["material"])
    if instance["crown_nut"]:
        bump(instance["crown_nut"]["material"])
    return used


def validate_semantics(instance: dict) -> list[str]:
    """Cross-field checks. Raises SchemaError on anything that would build a
    wrong or broken solid; returns warnings for anything merely suspicious."""
    errs: list[str] = []
    warns: list[str] = []
    pts = profile_points(instance)

    # ---- profile shape -----------------------------------------------------
    if abs(pts[0][1]) > 1e-6:
        errs.append(f"profile[0].z = {pts[0][1]} but must be 0.0 — profile heights are measured "
                    f"from grade; a non-zero start usually means provenance.grade_offset_m was "
                    f"never subtracted")
    if pts[0][0] > EPS or pts[-1][0] > EPS:
        errs.append(f"profile must start and end on the axis (r=0) to close the solid; "
                    f"got r_first={pts[0][0]}, r_last={pts[-1][0]}")
    for i, ((r0, z0), (r1, z1)) in enumerate(zip(pts, pts[1:])):
        if z1 < z0 - EPS:
            errs.append(f"profile[{i + 1}].z = {z1} is below profile[{i}].z = {z0}; the profile "
                        f"must be non-decreasing in z (it is revolved, not swept)")
        if abs(z1 - z0) < EPS and abs(r1 - r0) < EPS:
            errs.append(f"profile[{i}] and profile[{i + 1}] are identical ({r0}, {z0}); "
                        f"duplicate control points create degenerate faces")
    if profile_height(instance) <= EPS:
        errs.append("profile has zero height")
    if profile_max_radius(instance) <= EPS:
        errs.append("profile has zero maximum radius")

    # ---- the grade-offset trap, checked independently of the measurer ------
    if not errs:
        widest, barrel = profile_max_radius(instance), barrel_radius(instance)
        if barrel > EPS and widest > SLAB_RATIO * barrel:
            errs.append(
                f"widest profile ring is r={widest:.4f} m against a barrel radius of "
                f"{barrel:.4f} m (ratio {widest / barrel:.1f} > {SLAB_RATIO}). That is a ground "
                f"slab, not a flange: subtract it via provenance.grade_offset_m and re-measure, "
                f"or every feature height in this instance is too high by the slab thickness")

    # ---- materials ---------------------------------------------------------
    slots = sorted(m["slot"] for m in instance["materials"].values())
    if slots != list(range(len(slots))):
        errs.append(f"material slots must be unique and contiguous from 0; got {slots}")
    usage = _material_usage(instance)
    for role, n in usage.items():
        if n == 0:
            errs.append(f"materials.{role}: declared but nothing references it. An unused role is "
                        f"silently dropped by the glTF exporter, so the built file would not match "
                        f"this schema — delete the role or give it geometry")
    for feat in instance["features"]:
        for key in ("body_material", "cap_material", "nut_material"):
            role = feat[key]
            if isinstance(role, str) and role not in instance["materials"]:
                errs.append(f"feature {feat['name']!r}.{key} = {role!r} is not a declared material")
    for block, key in ((instance["flange_bolts"], "flange_bolts"),
                       (instance["crown_nut"], "crown_nut")):
        if block and block["material"] not in instance["materials"]:
            errs.append(f"{key}.material = {block['material']!r} is not a declared material")

    # ---- features ----------------------------------------------------------
    z_top = max(z for _, z in pts)
    seen: set[str] = set()
    for feat in instance["features"]:
        name = feat["name"]
        if name in seen:
            errs.append(f"duplicate feature name {name!r}")
        seen.add(name)

        cap_fields = ("cap_radius_m", "cap_length_m", "nut_radius_m", "nut_length_m", "nut_sides")
        if feat["type"] == "nozzle":
            missing = [f for f in cap_fields if feat[f] is None] + \
                      [f for f in ("cap_material", "nut_material") if feat[f] is None]
            if missing:
                errs.append(f"feature {name!r} is a nozzle but {missing} are null; "
                            f"use type 'boss' for a bare stub")
        else:
            extra = [f for f in cap_fields if feat[f] is not None] + \
                    [f for f in ("cap_material", "nut_material") if feat[f] is not None]
            if extra:
                errs.append(f"feature {name!r} is a boss but {extra} are set; a boss has no cap "
                            f"or nut — null them explicitly")

        h = feat["height_m"]
        if h > z_top + EPS:
            errs.append(f"feature {name!r} sits at z={h} m, above the profile apex z={z_top} m")
            continue
        r_body = body_radius_at(pts, h)
        if feat["neck_radius_m"] >= r_body:
            errs.append(
                f"feature {name!r}: neck radius {feat['neck_radius_m']:.4f} m >= body radius "
                f"{r_body:.4f} m at z={h:.3f} m — the neck would swallow the body. The usual "
                f"cause is a height still in the SOURCE MESH frame (subtract "
                f"provenance.grade_offset_m), which puts a nozzle up on the narrow dome")
        elif feat["neck_radius_m"] > 0.75 * r_body:
            warns.append(f"feature {name!r}: neck is {feat['neck_radius_m'] / r_body:.0%} of the "
                         f"body radius at that height — check the lobe was not the whole barrel")
        if feat["inset_m"] >= r_body:
            errs.append(f"feature {name!r}: inset {feat['inset_m']} m >= body radius {r_body:.4f} m "
                        f"at that height; the neck would start past the axis")

    # ---- hardware ----------------------------------------------------------
    bolts = instance["flange_bolts"]
    if bolts:
        widest = profile_max_radius(instance)
        reach = bolts["circle_radius_m"] + bolts["head_radius_m"]
        if reach > widest + EPS:
            errs.append(f"flange_bolts reach r={reach:.4f} m but the widest profile ring is "
                        f"{widest:.4f} m — the heads would hang off the flange")
        if bolts["circle_radius_m"] <= bolts["head_radius_m"]:
            errs.append("flange_bolts.circle_radius_m must exceed head_radius_m or the ring "
                        "collapses through the axis")
        if bolts["seat_height_m"] > z_top:
            errs.append(f"flange_bolts.seat_height_m = {bolts['seat_height_m']} is above the apex")
        else:
            r_seat = body_radius_at(pts, bolts["seat_height_m"])
            if bolts["circle_radius_m"] + bolts["head_radius_m"] < r_seat:
                warns.append(f"flange_bolts sit entirely inside the body radius "
                             f"({r_seat:.4f} m) at their seat height — they will be invisible")

    nut = instance["crown_nut"]
    if nut:
        if nut["seat_inset_m"] >= nut["height_m"]:
            errs.append("crown_nut.seat_inset_m must be less than its height or the nut is buried")
        if nut["radius_m"] > profile_max_radius(instance):
            errs.append("crown_nut.radius_m is wider than the whole body")
        if "measure" in instance["provenance"]["method"]:
            warns.append("crown_nut is set on a measured instance: a scan normally fuses the "
                         "operating nut into the dome, so the profile already contains it and "
                         "the built object will be taller than the real one")

    if errs:
        raise SchemaError(errs)
    return warns


def validate(instance: dict) -> list[str]:
    validate_structure(instance)
    return validate_semantics(instance)


# =========================================================================== #
# 4. Predicted receipts — what the schema SAYS the build must produce         #
# =========================================================================== #
# Computed before Blender runs, then asserted against the re-opened GLB. If
# these two ever disagree the build is wrong, whatever the exporter returned.

def _tube_tris(segments: int, cap0: bool, cap1: bool) -> int:
    return 2 * segments + (segments - 2) * (int(cap0) + int(cap1))


def predict(instance: dict) -> dict:
    seg_l = instance["segments"]["lathe"]
    seg_f = instance["segments"]["feature"]
    pts = profile_points(instance)

    tris = 0
    for (r0, _), (r1, _) in zip(pts, pts[1:]):
        a, b = r0 > EPS, r1 > EPS
        tris += 2 * seg_l if (a and b) else (seg_l if (a or b) else 0)

    for feat in instance["features"]:
        tris += _tube_tris(seg_f, False, False)                       # neck
        if feat["collar_length_m"] > EPS and feat["collar_radius_scale"] > 1.0:
            tris += _tube_tris(seg_f, False, False)                   # collar ridge
        if feat["type"] == "nozzle":
            tris += _tube_tris(seg_f, True, True)                     # cap
            tris += _tube_tris(int(feat["nut_sides"]), True, True)    # cap nut

    bolts = instance["flange_bolts"]
    if bolts:
        tris += bolts["count"] * _tube_tris(bolts["sides"], True, True)
    nut = instance["crown_nut"]
    if nut:
        tris += _tube_tris(nut["sides"], True, True)

    z_top = max(z for _, z in pts)
    height = profile_height(instance)
    radius = profile_max_radius(instance)
    if nut:
        height = max(height, z_top - nut["seat_inset_m"] + nut["height_m"] - min(z for _, z in pts))
    for feat in instance["features"]:
        r_body = body_radius_at(pts, feat["height_m"])
        reach = feat["neck_length_m"] + (feat["cap_length_m"] or 0.0) + (feat["nut_length_m"] or 0.0)
        out = r_body - feat["inset_m"] + reach * math.cos(math.radians(feat["tilt_deg"]))
        radius = max(radius, out + max(feat["neck_radius_m"], feat["cap_radius_m"] or 0.0))
        top = feat["height_m"] + reach * math.sin(math.radians(feat["tilt_deg"])) \
            + max(feat["neck_radius_m"], feat["cap_radius_m"] or 0.0)
        height = max(height, top)

    return {
        "triangles": tris,
        "materials": len(instance["materials"]),
        "height_m": round(height, 6),
        "max_radius_m": round(radius, 6),
        "flange_diameter_m": round(2.0 * profile_max_radius(instance), 6),
        "profile_points": len(pts),
        "features": len(instance["features"]),
    }


# =========================================================================== #
# 5. PLY reader (vertices + colours, no third-party mesh library)             #
# =========================================================================== #

_PLY_DTYPE = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_vertices(path: Path):
    """Return (xyz float64 [N,3], rgb float64 [N,3] in 0..1 or None).

    Deliberately hand-rolled: this has to run inside Blender's Python, which
    has numpy but no trimesh/open3d.
    """
    import numpy as np

    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = None
        elements: list[dict] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == b"format":
                fmt = parts[1].decode()
            elif key == b"element":
                elements.append({"name": parts[1].decode(), "count": int(parts[2]), "props": [],
                                 "has_list": False})
            elif key == b"property":
                if parts[1] == b"list":
                    elements[-1]["has_list"] = True
                else:
                    elements[-1]["props"].append((parts[2].decode(), parts[1].decode()))
            elif key == b"end_header":
                break
        if fmt is None:
            raise ValueError(f"{path}: PLY header has no format line")
        idx = next((i for i, e in enumerate(elements) if e["name"] == "vertex"), None)
        if idx is None:
            raise ValueError(f"{path}: PLY has no 'vertex' element")
        vert = elements[idx]
        if vert["has_list"]:
            raise ValueError(f"{path}: 'vertex' element has a list property — unsupported")

        names = [n for n, _ in vert["props"]]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"{path}: vertex element is missing '{axis}'")

        if fmt == "ascii":
            for skipped in elements[:idx]:
                for _ in range(skipped["count"]):
                    fh.readline()
            rows = [fh.readline().split() for _ in range(vert["count"])]
            table = {n: np.array([float(r[i]) for r in rows]) for i, n in enumerate(names)}
        else:
            order = "<" if "little" in fmt else ">"
            dtype = np.dtype([(n, order + _PLY_DTYPE[t]) for n, t in vert["props"]])
            for skipped in elements[:idx]:
                if skipped["has_list"]:
                    raise ValueError(f"{path}: cannot skip list-bearing element "
                                     f"{skipped['name']!r} before 'vertex'")
                size = np.dtype([(n, order + _PLY_DTYPE[t]) for n, t in skipped["props"]]).itemsize
                fh.seek(size * skipped["count"], os.SEEK_CUR)
            raw = np.frombuffer(fh.read(dtype.itemsize * vert["count"]), dtype=dtype,
                                count=vert["count"])
            table = {n: raw[n].astype(np.float64) for n in names}

    xyz = np.column_stack([table["x"], table["y"], table["z"]])
    rgb = None
    for keys in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green",
                                                             "diffuse_blue")):
        if all(k in table for k in keys):
            rgb = np.column_stack([table[k] for k in keys])
            hi = float(rgb.max()) if rgb.size else 0.0
            rgb = rgb / (255.0 if hi > 1.001 else 1.0)
            break
    return xyz, rgb


# =========================================================================== #
# 6. Measurement — capture geometry -> a starting schema instance             #
# =========================================================================== #
# Every constant below is a named, documented estimator choice, not a magic
# number. They were picked by running the alternatives against the hand-tuned
# reference profile, not by taste.

BAND_M = 0.02          # height band for every per-slice statistic
AZ_BINS = 36           # 10-degree azimuth bins
BIN_PCT = 50.0         # per-azimuth-bin radius percentile -> that bin's surface radius
ACROSS_PCT = 35.0      # percentile ACROSS bins -> the axisymmetric body radius, lobes rejected
LOBE_PCT = 85.0        # per-bin percentile used to detect a protrusion
LOBE_EXCESS = 0.30     # a bin is "hot" at >30% over the body radius ...
LOBE_FLOOR_M = 0.012   # ... but never on less than 12 mm, so noise cannot invent a nozzle
LOBE_LINK_DEG = 30.0   # azimuth tolerance when chaining hot bands into one feature
LOBE_MIN_BANDS = 3     # a feature must survive 3 consecutive bands (>= 6 cm of height)
LOBE_KEEP_FRAC = 0.35  # keep lobes at least this prominent relative to the strongest one
SLAB_FACTOR = 2.0      # a band wider than 2x the barrel, low down, is ground not object
CAP_COLOR_MIN_D = 0.20 # minimum RGB distance from the body colour to earn its own material role


def _circle_fit(x, y):
    import numpy as np
    a = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    return float(cx), float(cy), float(math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))


def _trimmed_circle(x, y, keep=0.6, iters=4):
    """Least-squares circle with the worst 40% of residuals dropped each pass.

    Plain least squares is hopeless here: nozzles are big outward outliers and
    drag the fitted centre tens of millimetres off the real axis, which then
    smears every azimuth measurement downstream.
    """
    import numpy as np
    cx, cy, r = _circle_fit(x, y)
    keep_idx = np.arange(len(x))
    for _ in range(iters):
        d = np.abs(np.hypot(x - cx, y - cy) - r)
        keep_idx = np.argsort(d)[: max(8, int(len(d) * keep))]
        cx, cy, r = _circle_fit(x[keep_idx], y[keep_idx])
    res = float(np.sqrt(np.mean((np.hypot(x[keep_idx] - cx, y[keep_idx] - cy) - r) ** 2)))
    return cx, cy, r, res


def _rdp(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker on the (r, z) polyline: keep only the control
    points the scan actually justifies. Tolerance should track the source's own
    resolution (~half a TSDF voxel), not a wish for a small file."""
    if len(points) < 3:
        return list(points)
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm < EPS:
        dist = [math.hypot(px - x0, py - y0) for px, py in points]
    else:
        dist = [abs(dx * (py - y0) - dy * (px - x0)) / norm for px, py in points]
    i = max(range(len(points)), key=lambda k: dist[k])
    if dist[i] <= tol:
        return [points[0], points[-1]]
    return _rdp(points[: i + 1], tol)[:-1] + _rdp(points[i:], tol)


def _circ_mean(angles_deg, weights):
    import numpy as np
    a = np.radians(np.asarray(angles_deg, dtype=float))
    w = np.asarray(weights, dtype=float)
    return float((math.degrees(math.atan2(float((w * np.sin(a)).sum()),
                                          float((w * np.cos(a)).sum()))) + 360.0) % 360.0)


def _ang_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def measure(mesh_path: Path, name: str, meters_per_unit: float | None,
            up_axis: str = "auto", profile_tolerance: float = 0.002,
            max_features: int = 8, flange_bolt_count: int = 8,
            lathe_segments: int = 72, feature_segments: int = 24,
            albedo_gain: float = 1.0) -> dict:
    """Derive a schema instance from an object-lane mesh.

    The pipeline, in order, because each step depends on the last:
      1. scale to metres          2. pick the up axis and the base end
      3. robust axis centre       4. detect grade (the ground slab)
      5. radius-vs-height profile 6. radial lobes -> features
      7. vertex colours -> material roles
    """
    import numpy as np

    xyz, rgb = read_ply_vertices(mesh_path)
    if meters_per_unit:
        xyz = xyz * float(meters_per_unit)

    # -- 2. up axis: the axis the object is a COLUMN along, i.e. the one whose
    #       cross-sections are smallest relative to the object's size.
    #  The horizontal pair is always the CYCLIC successor pair (x->y,z / y->z,x
    #  / z->x,y) so that (h0, h1, up) stays right-handed; taking them in sorted
    #  order would silently mirror every azimuth when the up axis is y.
    horiz = {0: [1, 2], 1: [2, 0], 2: [0, 1]}
    if up_axis == "auto":
        scores = []
        for ax in range(3):
            others = horiz[ax]
            lo, hi = xyz[:, ax].min(), xyz[:, ax].max()
            if hi - lo < EPS:
                scores.append(float("inf"))
                continue
            radii = []
            for t in np.linspace(lo, hi, 21)[:-1]:
                sel = (xyz[:, ax] >= t) & (xyz[:, ax] < t + (hi - lo) / 20.0)
                if sel.sum() < 10:
                    continue
                c = np.median(xyz[sel][:, others], axis=0)
                radii.append(np.median(np.hypot(xyz[sel][:, others[0]] - c[0],
                                                xyz[sel][:, others[1]] - c[1])))
            scores.append(float(np.median(radii)) / (hi - lo) if radii else float("inf"))
        axis = int(np.argmin(scores))
        axis_note = f"auto ({'xyz'[axis]}); slenderness scores {[round(s, 3) for s in scores]}"
    else:
        axis = "xyz".index(up_axis)
        axis_note = f"forced {up_axis}"
    others = horiz[axis]

    # base end = the wide end (a hydrant stands on a slab/flange, not on its dome)
    h = xyz[:, axis] - xyz[:, axis].min()
    total = float(h.max())
    c0 = np.median(xyz[h < 0.15 * total][:, others], axis=0)
    c1 = np.median(xyz[h > 0.85 * total][:, others], axis=0)
    r_bottom = float(np.median(np.hypot(xyz[h < 0.15 * total][:, others[0]] - c0[0],
                                        xyz[h < 0.15 * total][:, others[1]] - c0[1])))
    r_top = float(np.median(np.hypot(xyz[h > 0.85 * total][:, others[0]] - c1[0],
                                     xyz[h > 0.85 * total][:, others[1]] - c1[1])))
    flipped = r_top > r_bottom
    xy = xyz[:, others].copy()
    if flipped:
        # turn the object over AND mirror one horizontal axis, or the frame goes
        # left-handed and every measured azimuth comes out reflected
        h = total - h
        xy[:, 1] = -xy[:, 1]

    # -- 3. axis centre: median of the per-band trimmed circle fits, over the
    #       better-fitting half of the bands (nozzle bands fit badly by design).
    fits = []
    for lo in np.arange(0.20 * total, 0.95 * total, BAND_M):
        sel = (h >= lo) & (h < lo + BAND_M)
        if sel.sum() >= 40:
            fits.append(_trimmed_circle(xy[sel][:, 0], xy[sel][:, 1]))
    if not fits:
        raise SchemaError([f"{mesh_path}: too few points to fit an axis "
                           f"({len(xyz)} vertices over {total:.3f} m)"])
    fits.sort(key=lambda f: f[3])
    best = fits[: max(3, len(fits) // 2)]
    cx = float(np.median([f[0] for f in best]))
    cy = float(np.median([f[1] for f in best]))

    r_all = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    az_all = (np.degrees(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)) + 360.0) % 360.0

    # -- 4. grade: scan the bottom 40%; the highest band still more than
    #       SLAB_FACTOR wider than the barrel is the top of the ground slab.
    mids = [float(np.percentile(r_all[(h >= lo) & (h < lo + BAND_M)], 50))
            for lo in np.arange(0.35 * total, 0.85 * total, BAND_M)
            if ((h >= lo) & (h < lo + BAND_M)).sum() >= 20]
    barrel_r = float(np.median(mids)) if mids else float(np.median(r_all))
    grade = 0.0
    for lo in np.arange(0.0, 0.40 * total, BAND_M):
        sel = (h >= lo) & (h < lo + BAND_M)
        if sel.sum() >= 20 and float(np.percentile(r_all[sel], 90)) > SLAB_FACTOR * barrel_r:
            grade = float(lo + BAND_M)

    # -- 5/6. per-band statistics: body radius, and hot azimuth bins
    def band_bins(sel, pct):
        rr, aa = r_all[sel], az_all[sel]
        idx = np.minimum((aa / (360.0 / AZ_BINS)).astype(int), AZ_BINS - 1)
        return {k: float(np.percentile(rr[idx == k], pct))
                for k in range(AZ_BINS) if int((idx == k).sum()) >= 3}

    raw_profile: list[tuple[float, float]] = []
    band_lobes: list[tuple[float, list[dict]]] = []
    for lo in np.arange(grade, total, BAND_M):
        sel = (h >= lo) & (h < lo + BAND_M)
        if int(sel.sum()) < 12:
            continue
        surf, outer = band_bins(sel, BIN_PCT), band_bins(sel, LOBE_PCT)
        if len(surf) < 8:
            continue
        base = float(np.percentile(list(surf.values()), ACROSS_PCT))
        z = float(lo + BAND_M / 2.0 - grade)
        raw_profile.append((base, z))

        thr = base + max(LOBE_EXCESS * base, LOBE_FLOOR_M)
        hot = [k for k in range(AZ_BINS) if k in outer and outer[k] > thr]
        groups, run = [], []
        for k in range(AZ_BINS):
            if k in hot:
                run.append(k)
            elif run:
                groups.append(run)
                run = []
        if run:
            groups.append(run)
        if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == AZ_BINS - 1:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        band_lobes.append((z, [
            {"az": _circ_mean([k * 10 + 5 for k in g], [outer[k] - base for k in g]),
             "span_deg": len(g) * (360.0 / AZ_BINS),
             "r_max": max(outer[k] for k in g),
             "base": base, "z": z, "bins": g}
            for g in groups]))

    if len(raw_profile) < 3:
        raise SchemaError([f"{mesh_path}: only {len(raw_profile)} usable height bands above grade "
                           f"— the mesh is too sparse or the up axis ({axis_note}) is wrong"])

    # Close the polyline on the axis at the REAL extremes (grade below, the
    # highest vertex above), not at the first/last band centre — a band centre
    # is half a band short at each end, which quietly shortens the object.
    profile = _rdp(raw_profile, profile_tolerance)
    profile = [(0.0, 0.0)] + list(profile) + [(0.0, round(total - grade, 6))]

    # chain hot bands into features
    chains: list[dict] = []
    for z, lobes in band_lobes:
        for lobe in lobes:
            hit = next((c for c in chains
                        if _ang_delta(c["az"], lobe["az"]) < LOBE_LINK_DEG
                        and abs(c["z_last"] - z) <= BAND_M * 1.5), None)
            if hit is None:
                chains.append({"az": lobe["az"], "items": [lobe], "z_last": z})
            else:
                hit["items"].append(lobe)
                hit["z_last"] = z
                hit["az"] = _circ_mean([o["az"] for o in hit["items"]],
                                       [o["r_max"] - o["base"] for o in hit["items"]])

    for c in chains:
        excess = [o["r_max"] - o["base"] for o in c["items"]]
        c["reach"] = float(max(excess))
        c["prominence"] = float(sum(excess))
        c["z"] = float(sum(e * o["z"] for e, o in zip(excess, c["items"])) / sum(excess))
        c["span_deg"] = float(np.median([o["span_deg"] for o in c["items"]]))
        c["base"] = float(np.median([o["base"] for o in c["items"]]))
        c["n_bands"] = len(c["items"])
    strong = max((c["prominence"] for c in chains), default=0.0)
    kept = [c for c in chains
            if c["n_bands"] >= LOBE_MIN_BANDS and c["prominence"] >= LOBE_KEEP_FRAC * strong]
    kept.sort(key=lambda c: -c["prominence"])
    kept = kept[:max_features]
    kept.sort(key=lambda c: (c["z"], c["az"]))

    # -- 7. colours: body from the barrel surface, one per feature from the lobe
    def sample_color(sel):
        if rgb is None or int(sel.sum()) < 20:
            return None
        return [float(v) for v in np.median(rgb[sel], axis=0)]

    body_sel = (h > grade + 0.20 * (total - grade)) & (h < grade + 0.45 * (total - grade)) & \
               (r_all < 1.15 * barrel_r)
    body_color = sample_color(body_sel) or [0.80, 0.80, 0.80]
    feat_colors = []
    for c in kept:
        zc = c["z"] + grade
        half = max(BAND_M, c["n_bands"] * BAND_M / 2.0)
        sel = (h >= zc - half) & (h <= zc + half) & (r_all > c["base"] * 1.3) & \
              (np.abs((az_all - c["az"] + 180.0) % 360.0 - 180.0) < c["span_deg"] / 2.0 + 10.0)
        feat_colors.append(sample_color(sel))

    dists = [math.dist(fc, body_color) if fc else 0.0 for fc in feat_colors]
    cap_thr = max(CAP_COLOR_MIN_D, 0.5 * max(dists, default=0.0))
    is_cap = [d >= cap_thr for d in dists]

    def gain(color):
        return [round(min(1.0, v * albedo_gain), 4) for v in color] + [1.0]

    materials = {"body": {"slot": 0, "base_color": gain(body_color), "roughness": 0.55,
                          "metallic": 0.0, "source": "measured"}}
    if any(is_cap):
        picked = [fc for fc, flag in zip(feat_colors, is_cap) if flag and fc]
        cap_color = [float(np.median([p[i] for p in picked])) for i in range(3)]
        materials["cap"] = {"slot": len(materials), "base_color": gain(cap_color),
                            "roughness": 0.45, "metallic": 0.0, "source": "measured"}
    needs_metal = bool(flange_bolt_count) or bool(kept)
    if needs_metal:
        materials["metal"] = {"slot": len(materials), "base_color": [0.38, 0.38, 0.395, 1.0],
                              "roughness": 0.40, "metallic": 0.9, "source": "assumed"}

    pts = [(r, z) for r, z in profile]
    features = []
    for i, (c, cap_flag, color) in enumerate(zip(kept, is_cap, feat_colors)):
        r_body = body_radius_at(pts, c["z"])
        neck_r = max(0.2 * r_body, min(0.75 * r_body,
                                       c["base"] * math.sin(math.radians(c["span_deg"] / 2.0))))
        reach = max(c["reach"], neck_r * 0.8)
        # tilt from the lobe's own points: how the protruding surface rises with radius
        sel = (h >= c["z"] + grade - c["n_bands"] * BAND_M / 2.0) & \
              (h <= c["z"] + grade + c["n_bands"] * BAND_M / 2.0) & \
              (r_all > c["base"] * 1.2) & \
              (np.abs((az_all - c["az"] + 180.0) % 360.0 - 180.0) < c["span_deg"] / 2.0)
        tilt = 0.0
        if int(sel.sum()) >= 30 and float(np.std(r_all[sel])) > 1e-4:
            slope = float(np.polyfit(r_all[sel], h[sel] - grade - c["z"], 1)[0])
            tilt = max(-45.0, min(45.0, math.degrees(math.atan(slope))))
        features.append({
            "name": f"feature_{i + 1}",
            "type": "nozzle",
            "azimuth_deg": round(c["az"] % 360.0, 2),
            "height_m": round(c["z"], 4),
            "tilt_deg": round(tilt, 2),
            "inset_m": round(0.33 * r_body, 4),
            "neck_radius_m": round(neck_r, 4),
            "neck_length_m": round(0.5 * reach, 4),
            "collar_radius_scale": 1.14,
            "collar_length_m": 0.012,
            "cap_radius_m": round(1.2 * neck_r, 4),
            "cap_length_m": round(0.5 * reach, 4),
            "nut_radius_m": round(0.6 * neck_r, 4),
            "nut_length_m": 0.016,
            "nut_sides": 6,
            "body_material": "body",
            "cap_material": "cap" if cap_flag else "body",
            "nut_material": "metal",
            "derivation": {
                "source": "measured",
                "notes": "azimuth/height/reach/tilt from radial lobe analysis; neck radius from "
                         "the lobe's angular width; cap and nut proportions are archetype "
                         "constants (photo-refinable)",
                "evidence": {
                    "bands": c["n_bands"], "span_deg": round(c["span_deg"], 1),
                    "reach_m": round(c["reach"], 4), "prominence": round(c["prominence"], 4),
                    "body_radius_m": round(c["base"], 4),
                    "median_rgb": [round(v, 3) for v in color] if color else None,
                    "color_distance_from_body": round(math.dist(color, body_color), 3)
                    if color else None,
                },
            },
        })

    flange_bolts = None
    if flange_bolt_count:
        flange_r = max(r for r, _ in pts)
        seat = 0.0
        cut = flange_r - 0.5 * (flange_r - barrel_r)
        for r, z in pts:
            if r >= cut:
                seat = z
        head_r = round(0.115 * flange_r, 4)
        # the ring is placed tangent to the body at the seat so the heads stand
        # proud of the barrel yet stay inside the measured rim; a TSDF scan of a
        # painted flange never resolves the heads themselves, hence source=assumed
        circle_r = min(body_radius_at(pts, seat) + head_r, flange_r - head_r)
        flange_bolts = {
            "count": int(flange_bolt_count),
            "circle_radius_m": round(circle_r, 4),
            "head_radius_m": head_r,
            "head_height_m": head_r,
            "seat_height_m": round(seat, 4),
            "sides": 6,
            "material": "metal",
            "derivation": {"source": "assumed",
                           "notes": f"count={flange_bolt_count} is an operator/photo value — the "
                                    f"scan resolves the flange but not individual heads. Ring "
                                    f"radius and seat height ARE measured off the profile; a "
                                    f"TSDF flange is blurred into the barrel fillet, so the "
                                    f"heads sit partly merged into it, as they do on the real "
                                    f"casting"},
        }

    notes = [
        f"up axis: {axis_note}" + (" (base end detected at the +axis end, heights flipped)"
                                   if flipped else ""),
        f"axis centre ({cx:.4f}, {cy:.4f}) m from {len(best)}/{len(fits)} trimmed circle fits",
        f"grade offset {grade:.4f} m detected as the highest band in the lower 40% wider than "
        f"{SLAB_FACTOR}x the barrel radius ({barrel_r:.4f} m); all heights below are above grade",
        f"profile: {len(raw_profile)} measured bands simplified to {len(profile)} control points "
        f"at {profile_tolerance * 1000:.1f} mm",
        f"{len(chains)} radial lobe chains found, {len(kept)} kept "
        f"(>= {LOBE_MIN_BANDS} bands and >= {LOBE_KEEP_FRAC:.0%} of the strongest)",
        "colours are SHADED albedo sampled off the scan, not paint chips; raise --albedo-gain "
        "for the painted look",
        "crown_nut is null: a scan fuses the operating nut into the dome, so the measured "
        "profile already contains it",
    ]
    if not any(is_cap):
        notes.append("no feature colour was far enough from the body colour to earn its own "
                     "material role — cap colours need the photo")

    return {
        "schema_version": SCHEMA_VERSION,
        "archetype": "lathe_with_features",
        "name": name,
        "units": "meters",
        "provenance": {
            "method": "parametric_build.py measure v1",
            "source": str(mesh_path),
            "meters_per_unit": float(meters_per_unit) if meters_per_unit else None,
            "grade_offset_m": round(grade, 4),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": notes,
            "measurement": {
                "vertices": int(len(xyz)), "up_axis": "xyz"[axis], "flipped": bool(flipped),
                "center": [round(cx, 5), round(cy, 5)],
                "source_height_m": round(total, 4), "grade_offset_m": round(grade, 4),
                "barrel_radius_m": round(barrel_r, 4),
                "band_m": BAND_M, "az_bins": AZ_BINS,
                "estimator": {"bin_pct": BIN_PCT, "across_pct": ACROSS_PCT, "lobe_pct": LOBE_PCT},
                "profile_tolerance_m": profile_tolerance,
                "body_rgb": [round(v, 4) for v in body_color],
                "lobe_chains_found": len(chains), "lobe_chains_kept": len(kept),
            },
        },
        "segments": {"lathe": int(lathe_segments), "feature": int(feature_segments)},
        "shade_auto_smooth_deg": 40.0,
        "materials": materials,
        "profile": [{"r": round(r, 4), "z": round(z, 4)} for r, z in profile],
        "features": features,
        "flange_bolts": flange_bolts,
        "crown_nut": None,
    }


# =========================================================================== #
# 7. Build (inside Blender)                                                   #
# =========================================================================== #

def _build_in_blender(instance: dict, out_glb: Path, out_blend: Path | None) -> dict:
    import bmesh
    import bpy
    from mathutils import Matrix, Vector

    seg_l = instance["segments"]["lathe"]
    seg_f = instance["segments"]["feature"]
    pts = profile_points(instance)
    name = instance["name"]

    def ring(bm, mat, r, z, segs, phase=0.0):
        return [bm.verts.new(mat @ Vector((r * math.cos(2 * math.pi * i / segs + phase),
                                           r * math.sin(2 * math.pi * i / segs + phase), z)))
                for i in range(segs)]

    def bridge(bm, a, b, segs):
        if len(a) == 1 and len(b) == 1:
            return
        if len(a) == 1:
            for i in range(segs):
                bm.faces.new((a[0], b[i], b[(i + 1) % segs]))
        elif len(b) == 1:
            for i in range(segs):
                bm.faces.new((a[i], a[(i + 1) % segs], b[0]))
        else:
            for i in range(segs):
                bm.faces.new((a[i], a[(i + 1) % segs], b[(i + 1) % segs], b[i]))

    def tube(bm, mat, r, z0, z1, segs, cap0=True, cap1=True, phase=0.0):
        a = ring(bm, mat, r, z0, segs, phase)
        b = ring(bm, mat, r, z1, segs, phase)
        bridge(bm, a, b, segs)
        if cap0:
            bm.faces.new(list(reversed(a)))
        if cap1:
            bm.faces.new(b)

    def feature_frame(az_deg, tilt_deg, height, radius_at, inset):
        az = math.radians(az_deg)
        origin = Vector((math.cos(az) * (radius_at - inset),
                         math.sin(az) * (radius_at - inset), height))
        rot = Matrix.Rotation(az, 4, "Z") @ Matrix.Rotation(math.radians(90.0 - tilt_deg), 4, "Y")
        return Matrix.Translation(origin) @ rot

    bpy.ops.wm.read_homefile(use_empty=True)
    identity = Matrix.Identity(4)
    roles = sorted(instance["materials"], key=lambda k: instance["materials"][k]["slot"])
    meshes = {role: bmesh.new() for role in roles}

    # body revolve
    body_bm = meshes[roles[0]]
    rings = [[body_bm.verts.new(Vector((0.0, 0.0, z)))] if r < EPS
             else ring(body_bm, identity, r, z, seg_l) for r, z in pts]
    for a, b in zip(rings, rings[1:]):
        bridge(body_bm, a, b, seg_l)

    for feat in instance["features"]:
        r_body = body_radius_at(pts, feat["height_m"])
        frame = feature_frame(feat["azimuth_deg"], feat["tilt_deg"], feat["height_m"],
                              r_body, feat["inset_m"])
        neck_end = feat["neck_length_m"]
        target = meshes[feat["body_material"]]
        tube(target, frame, feat["neck_radius_m"], 0.0, neck_end, seg_f, cap0=False, cap1=False)
        if feat["collar_length_m"] > EPS and feat["collar_radius_scale"] > 1.0:
            tube(target, frame, feat["neck_radius_m"] * feat["collar_radius_scale"],
                 neck_end - feat["collar_length_m"], neck_end, seg_f, cap0=False, cap1=False)
        if feat["type"] == "nozzle":
            cap_end = neck_end + feat["cap_length_m"]
            tube(meshes[feat["cap_material"]], frame, feat["cap_radius_m"], neck_end, cap_end,
                 seg_f)
            tube(meshes[feat["nut_material"]], frame, feat["nut_radius_m"], cap_end,
                 cap_end + feat["nut_length_m"], int(feat["nut_sides"]))

    bolts = instance["flange_bolts"]
    if bolts:
        for i in range(bolts["count"]):
            angle = 2 * math.pi * i / bolts["count"]
            offset = Matrix.Translation(Vector((math.cos(angle) * bolts["circle_radius_m"],
                                                math.sin(angle) * bolts["circle_radius_m"],
                                                bolts["seat_height_m"])))
            tube(meshes[bolts["material"]], offset, bolts["head_radius_m"], 0.0,
                 bolts["head_height_m"], bolts["sides"], phase=angle)

    nut = instance["crown_nut"]
    if nut:
        z0 = max(z for _, z in pts) - nut["seat_inset_m"]
        tube(meshes[nut["material"]], identity, nut["radius_m"], z0, z0 + nut["height_m"],
             nut["sides"])

    parts = []
    for role in roles:
        bm = meshes[role]
        if not bm.faces:
            raise SchemaError([f"materials.{role}: received no geometry during the build; "
                               f"the exporter would silently drop it"])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        spec = instance["materials"][role]
        data = bpy.data.meshes.new(f"{name}_{role}_mesh")
        bm.to_mesh(data)
        bm.free()
        obj = bpy.data.objects.new(f"{name}_{role}", data)
        bpy.context.scene.collection.objects.link(obj)
        mat = bpy.data.materials.new(f"{name}_{role}")
        mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = tuple(spec["base_color"])
        bsdf.inputs["Roughness"].default_value = spec["roughness"]
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = spec["metallic"]
        data.materials.append(mat)
        parts.append(obj)

    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]  # slot 0 owner joins first
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = f"{name}_mesh"

    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(instance["shade_auto_smooth_deg"]))
    except Exception:  # older/newer Blender: plain smooth is a cosmetic fallback, never fatal
        bpy.ops.object.shade_smooth()

    if out_blend:
        out_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB",
                              export_yup=True, export_apply=True, export_extras=True)

    world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    return {
        "blender_vertices": len(obj.data.vertices),
        "blender_polygons": len(obj.data.polygons),
        "blender_triangles": sum(len(p.vertices) - 2 for p in obj.data.polygons),
        "blender_materials": [m.name for m in obj.data.materials],
        "blender_height_m": round(max(p.z for p in world) - min(p.z for p in world), 6),
    }


# =========================================================================== #
# 8. GLB readback — the only evidence that counts                             #
# =========================================================================== #

_COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
              5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _mat_mul(a, b):
    return [sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4)]


def _trs_matrix(node) -> list[float]:
    if "matrix" in node:  # glTF matrices are column-major
        m = node["matrix"]
        return [m[c * 4 + r] for r in range(4) for c in range(4)]
    out = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
    t = node.get("translation", [0.0, 0.0, 0.0])
    out[3], out[7], out[11] = t
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        rot = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0,
               2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0,
               2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0,
               0.0, 0.0, 0.0, 1.0]
        out = _mat_mul(out, rot)
    if "scale" in node:
        sx, sy, sz = node["scale"]
        out = _mat_mul(out, [sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1])
    return out


def readback_glb(path: Path) -> dict:
    """Re-open a written GLB and report what is REALLY in it.

    Decodes the binary chunk itself — accessor min/max are exporter claims, and
    this whole function exists because exporter claims (and exit codes) are not
    evidence. Deliberately dependency-free so it runs anywhere.
    """
    raw = path.read_bytes()
    if len(raw) < 20:
        raise SchemaError([f"{path}: {len(raw)} bytes, not a GLB"])
    magic, _version, length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF":
        raise SchemaError([f"{path}: bad GLB magic {magic!r}"])
    if length != len(raw):
        raise SchemaError([f"{path}: header says {length} bytes, file is {len(raw)}"])

    offset, gltf, binary = 12, None, b""
    while offset < len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        body = raw[offset + 8: offset + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(body)
        elif chunk_type == 0x004E4942:
            binary = body
        offset += 8 + chunk_len + (-chunk_len % 4)
    if gltf is None:
        raise SchemaError([f"{path}: no JSON chunk"])

    def accessor(index: int) -> list[tuple]:
        acc = gltf["accessors"][index]
        fmt, size = _COMPONENT[acc["componentType"]]
        ncomp = _NCOMP[acc["type"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        stride = view.get("byteStride") or size * ncomp
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return [struct.unpack_from("<" + fmt * ncomp, binary, start + i * stride)
                for i in range(acc["count"])]

    triangles, verts_world, prim_count = 0, [], 0
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes") or [{"nodes": list(range(len(nodes)))}]
    roots = scenes[min(gltf.get("scene", 0), len(scenes) - 1)].get("nodes", [])
    stack = [(i, [1.0 if k % 5 == 0 else 0.0 for k in range(16)]) for i in roots]
    while stack:
        idx, parent = stack.pop()
        node = nodes[idx]
        world = _mat_mul(parent, _trs_matrix(node))
        for child in node.get("children", []):
            stack.append((child, world))
        if "mesh" not in node:
            continue
        for prim in gltf["meshes"][node["mesh"]].get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            prim_count += 1
            triangles += (len(accessor(prim["indices"])) if "indices" in prim
                          else len(accessor(prim["attributes"]["POSITION"]))) // 3
            for x, y, z in accessor(prim["attributes"]["POSITION"]):
                verts_world.append((world[0] * x + world[1] * y + world[2] * z + world[3],
                                    world[4] * x + world[5] * y + world[6] * z + world[7],
                                    world[8] * x + world[9] * y + world[10] * z + world[11]))
    if not verts_world:
        raise SchemaError([f"{path}: exported with zero triangle geometry"])

    xs = [p[0] for p in verts_world]
    ys = [p[1] for p in verts_world]
    zs = [p[2] for p in verts_world]
    # export_yup=True maps the model's +Z (up) onto glTF +Y.
    height = max(ys) - min(ys)
    base = min(ys)
    skirt = min(0.05, 0.08 * height)  # the "base band" the flange diameter is read over
    flange = 2.0 * max((math.hypot(x, z) for x, y, z in verts_world if y - base < skirt),
                       default=0.0)
    return {
        "file": str(path),
        "bytes": len(raw),
        "triangles": triangles,
        "primitives": prim_count,
        "vertices": len(verts_world),
        "materials": len(gltf.get("materials", [])),
        "material_names": [m.get("name", "") for m in gltf.get("materials", [])],
        "height_m": round(height, 6),
        "extent_m": [round(max(xs) - min(xs), 6), round(height, 6), round(max(zs) - min(zs), 6)],
        "max_radius_m": round(max(math.hypot(x, z) for x, _y, z in verts_world), 6),
        "flange_diameter_m": round(flange, 6),
        "height_axis": "gltf +Y (model +Z under export_yup)",
    }


def compare(predicted: dict, actual: dict, tol: float) -> tuple[list[str], list[str]]:
    """Structural receipts, not prose. Returns (failures, lines)."""
    checks = [
        ("triangles", predicted["triangles"], actual["triangles"], 0.0),
        ("materials", predicted["materials"], actual["materials"], 0.0),
        ("height_m", predicted["height_m"], actual["height_m"], tol),
        ("flange_diameter_m", predicted["flange_diameter_m"], actual["flange_diameter_m"], tol),
    ]
    failures, lines = [], []
    for key, want, got, allow in checks:
        if want == 0:
            ok, delta = got == 0, 0.0
        else:
            delta = (got - want) / want
            ok = abs(delta) <= allow + 1e-9
        lines.append(f"  {'ok ' if ok else 'BAD'}  {key:<20} predicted={want!r:<12} "
                     f"actual={got!r:<12} delta={delta:+.2%}")
        if not ok:
            failures.append(f"{key}: schema predicts {want!r}, exported GLB has {got!r} "
                            f"({delta:+.2%}, tolerance {allow:.0%})")
    return failures, lines


# =========================================================================== #
# 9. CLI                                                                      #
# =========================================================================== #

def _load_instance(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SchemaError([f"{path}: not valid JSON — {exc}"]) from exc


def _report_warnings(warnings: list[str]) -> None:
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)


def cmd_measure(args) -> int:
    instance = measure(
        Path(args.mesh).expanduser(), args.name, args.meters_per_unit,
        up_axis=args.up_axis, profile_tolerance=args.profile_tolerance,
        max_features=args.max_features, flange_bolt_count=args.flange_bolts,
        lathe_segments=args.lathe_segments, feature_segments=args.feature_segments,
        albedo_gain=args.albedo_gain,
    )
    warnings = validate(instance)  # a measurer that emits an invalid instance is a bug
    _report_warnings(warnings)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(instance, indent=2) + "\n")
    pred = predict(instance)
    summary = ", ".join("{}@{:.0f}deg/{:.2f}m".format(f["name"], f["azimuth_deg"], f["height_m"])
                        for f in instance["features"]) or "none"
    print(f"wrote {out}")
    print(f"  grade offset      : {instance['provenance']['grade_offset_m']} m "
          f"(subtracted from every height)")
    print(f"  profile           : {len(instance['profile'])} control points, "
          f"{pred['height_m']:.3f} m tall, widest dia {pred['flange_diameter_m']:.3f} m")
    print(f"  features          : {len(instance['features'])} ({summary})")
    print(f"  materials         : {list(instance['materials'])}")
    print(f"  predicted build   : {pred['triangles']} tris, {pred['materials']} materials")
    return 0


def cmd_validate(args) -> int:
    failed = 0
    for target in args.instance:
        path = Path(target).expanduser()
        try:
            warnings = validate(_load_instance(path))
        except SchemaError as exc:
            print(f"INVALID {path}\n{exc}", file=sys.stderr)
            failed += 1
            continue
        _report_warnings(warnings)
        print(f"VALID   {path} ({len(warnings)} warning(s))")
    return 1 if failed else 0


def cmd_build(args) -> int:
    path = Path(args.instance).expanduser()
    instance = _load_instance(path)
    _report_warnings(validate(instance))
    predicted = predict(instance)

    out_glb = Path(args.out).expanduser()
    out_blend = Path(args.blend).expanduser() if args.blend else None

    try:
        import bpy  # noqa: F401
        inside = True
    except ImportError:
        inside = False

    if not inside:
        if not BLENDER_BIN.is_file():
            raise SchemaError([f"Blender not found at {BLENDER_BIN}; set SPLATLAB_BLENDER_BIN"])
        cmd = [str(BLENDER_BIN), "--background", "--python", str(Path(__file__).resolve()), "--",
               "build", str(path), "--out", str(out_glb)]
        if out_blend:
            cmd += ["--blend", str(out_blend)]
        if args.receipt:
            cmd += ["--receipt", str(args.receipt)]
        if args.expect:
            cmd += ["--expect", str(args.expect)]
        cmd += ["--tolerance", str(args.tolerance)]
        print("[build] " + " ".join(cmd), flush=True)
        return subprocess.run(cmd, check=False).returncode

    blender_stats = _build_in_blender(instance, out_glb, out_blend)
    actual = readback_glb(out_glb)  # re-open: the exporter's exit code proves nothing

    failures, lines = compare(predicted, actual, args.tolerance)
    print(f"\n=== {instance['name']} — schema vs re-opened GLB ===")
    print("\n".join(lines))

    # Reference comparison is ADVISORY: a deliberate difference from a
    # hand-built precedent (more measured features, a coarser profile) is a
    # finding to explain, not a broken build. The hard gate above is the one
    # that says the exported file is the file the schema described.
    reference = None
    if args.expect:
        expected = json.loads(Path(args.expect).expanduser().read_text())
        reference = {"file": str(args.expect), "expected": expected, "deltas": {}}
        print(f"\n=== {instance['name']} — re-opened GLB vs reference "
              f"{Path(args.expect).name} (advisory, ±{args.tolerance:.0%}) ===")
        for key, want in expected.items():
            got = actual.get(key)
            if got is None or not isinstance(want, (int, float)):
                print(f"  --   {key:<20} reference={want!r} (no comparable readback value)")
                continue
            delta = (got - want) / want if want else 0.0
            reference["deltas"][key] = {"reference": want, "actual": got,
                                        "delta": round(delta, 6),
                                        "within_tolerance": abs(delta) <= args.tolerance + 1e-9}
            flag = "ok " if abs(delta) <= args.tolerance + 1e-9 else "OFF"
            print(f"  {flag}  {key:<20} reference={want!r:<12} actual={got!r:<12} "
                  f"delta={delta:+.2%}")

    receipt = {"instance": str(path), "predicted": predicted, "actual": actual,
               "blender": blender_stats, "reference": reference,
               "tolerance": args.tolerance, "failures": failures}
    if args.receipt:
        Path(args.receipt).expanduser().write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"\nreceipt -> {args.receipt}")

    if failures:
        print("\nFATAL: the exported GLB does not match what the schema predicts:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\nBUILD_OK  {out_glb}  ({actual['triangles']} tris, {actual['materials']} materials, "
          f"{actual['height_m']:.3f} m tall)")
    return 0


def cmd_inspect(args) -> int:
    print(json.dumps(readback_glb(Path(args.glb).expanduser()), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="parametric_build.py",
        description="Schema-driven parametric authoring: measure -> validate -> build -> verify.")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="derive a starting schema instance from an object mesh")
    m.add_argument("--mesh", required=True, help="object-lane mesh .ply (vertex colours used if present)")
    m.add_argument("--name", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--meters-per-unit", type=float, default=None,
                   help="scene-unit -> metre calibration; omit only if the mesh is already metric")
    m.add_argument("--up-axis", choices=["auto", "x", "y", "z"], default="auto")
    m.add_argument("--profile-tolerance", type=float, default=0.002,
                   help="polyline simplification tolerance in metres (~half a TSDF voxel)")
    m.add_argument("--max-features", type=int, default=8)
    m.add_argument("--flange-bolts", type=int, default=8,
                   help="bolt count (0 = none); a scan cannot resolve individual heads")
    m.add_argument("--lathe-segments", type=int, default=72)
    m.add_argument("--feature-segments", type=int, default=24)
    m.add_argument("--albedo-gain", type=float, default=1.0,
                   help="multiply measured (shaded) colours; 1.0 keeps them honest")
    m.set_defaults(func=cmd_measure)

    v = sub.add_parser("validate", help="structural + semantic validation, no Blender needed")
    v.add_argument("instance", nargs="+")
    v.set_defaults(func=cmd_validate)

    b = sub.add_parser("build", help="build in headless Blender and verify the exported GLB")
    b.add_argument("instance")
    b.add_argument("--out", required=True, help="output .glb")
    b.add_argument("--blend", default=None)
    b.add_argument("--receipt", default=None)
    b.add_argument("--expect", default=None,
                   help="JSON of reference values (triangles/height_m/...) to compare against")
    b.add_argument("--tolerance", type=float, default=0.05)
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("inspect", help="re-open a GLB and report its real contents")
    i.add_argument("glb")
    i.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SchemaError as exc:
        print(f"FATAL:\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
