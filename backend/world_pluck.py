"""The pluck doc: which backdrop-splat rows belong to each interactive prop.

The pluck mechanic renders a prop as the photoreal SPLAT until the player
grabs it — then the mesh takes over and the prop's gaussians vanish from the
backdrop. That needs one artifact: per-prop row lists IN THE ROW SPACE OF THE
SERVED BACKDROP SPLAT (`_preview/splat.ply` order — `langweb.ply` is row-
identical by construction, and `fmt=langweb` falls back server-side to raw
splat.ply in the same space; `fmt=web` is decimated above 1.2M gaussians and
must never be used for row addressing).

Mapping chain (measured on the garage: all 5 props' checkpoint indices land
on served rows, 0 dropped): `_scene/isolated/<slug>/object_indices.npz` holds
CHECKPOINT-order rows; `_langfield/ply_index_map.npy` maps ply_row->ckpt_row
(injective), so served rows = inverse(map)[indices]. When the map is absent
the fallback is byte-exact xyz matching against splat.ply — total-or-skip per
slug (>=99% matched or the slug is skipped WITH a reason): a partial silent
map would hide the wrong gaussians, and wrong is worse than none.

Shaped like world_interactions/world_placed: bounded validator, one write
gate, stdlib+numpy only. Staleness is identity-checked on read — a splat
edit or re-isolate must un-trust the doc loudly, never silently veil the
wrong rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import artifact_manifest as manifests
from langfield_align import read_ply_xyz

PLUCK_SCHEMA = "dev.splatlab.world-pluck/v1"
PLUCK_NAME = "pluck.json"
MAX_ELEMENTS = 256
XYZ_MATCH_MIN_FRAC = 0.99


class PluckError(ValueError):
    """A pluck doc that must not be built/trusted — with the remedy."""


def _ply_vertex_count(path: Path) -> int:
    with open(path, "rb") as handle:
        header = handle.read(65536)
    match = re.search(rb"element vertex (\d+)", header)
    if not match:
        raise PluckError(f"{path.name}: no vertex count in the PLY header")
    return int(match.group(1))


def _prop_slugs(world_dir: Path) -> list[str]:
    """Interactive candidates: manifest props when graded, else every built
    world.json element (ungraded worlds default everything to prop)."""
    manifest = manifests.read_json(world_dir / "world_manifest.json")
    if isinstance(manifest, dict) and manifest.get("elements"):
        return [e["slug"] for e in manifest["elements"]
                if isinstance(e, dict) and e.get("slug")
                and e.get("role") == "prop"]
    world = manifests.read_json(world_dir / "world.json") or {}
    return [e["slug"] for e in world.get("elements") or []
            if isinstance(e, dict) and e.get("slug") and e.get("built")]


def _rows_via_map(indices: np.ndarray, index_map: np.ndarray
                  ) -> tuple[np.ndarray, int]:
    n_ckpt = int(index_map.max()) + 1
    inverse = np.full(n_ckpt, -1, dtype=np.int64)
    inverse[index_map] = np.arange(len(index_map), dtype=np.int64)
    in_range = indices[(indices >= 0) & (indices < n_ckpt)]
    dropped = int(len(indices) - len(in_range))
    rows = inverse[in_range]
    dropped += int((rows < 0).sum())
    return np.sort(rows[rows >= 0]), dropped


def _splat_xyz_keys(splat_ply: Path) -> dict[bytes, int]:
    xyz = read_ply_xyz(splat_ply)
    raw = np.ascontiguousarray(xyz, dtype="<f4").tobytes()
    return {raw[i * 12:(i + 1) * 12]: i for i in range(len(xyz))}


def _rows_via_xyz(object_ply: Path, splat_keys: dict[bytes, int]
                  ) -> np.ndarray | None:
    """Byte-exact xyz match — TOTAL or None (never a partial silent map)."""
    xyz = read_ply_xyz(object_ply)
    raw = np.ascontiguousarray(xyz, dtype="<f4").tobytes()
    rows = [splat_keys.get(raw[i * 12:(i + 1) * 12]) for i in range(len(xyz))]
    matched = [r for r in rows if r is not None]
    if len(xyz) == 0 or len(matched) / len(xyz) < XYZ_MATCH_MIN_FRAC:
        return None
    return np.sort(np.asarray(matched, dtype=np.int64))


def build_pluck(job_dir: Path) -> dict[str, Any]:
    """The doc, or PluckError with the remedy. Never writes anything."""
    job_dir = Path(job_dir)
    world_dir = job_dir / "_world"
    lf_dir = job_dir / "_langfield"
    splat_ply = job_dir / "_preview" / "splat.ply"
    isolate_receipt = job_dir / "_scene" / "isolated" / "batch_isolate.json"

    if (lf_dir / "STALE").is_file():
        raise PluckError(
            "language field is STALE (splat edited since the lift) — "
            "POST /langfield/rebuild first, then rebuild pluck")
    if not isolate_receipt.is_file():
        raise PluckError("no _scene/isolated/batch_isolate.json — run the "
                         "isolate stage first")
    if not (world_dir / "world.json").is_file():
        raise PluckError("no _world/world.json — solidify the world first")
    if not splat_ply.is_file():
        raise PluckError("no _preview/splat.ply — nothing to address rows in")

    n_rows = _ply_vertex_count(splat_ply)
    langweb = job_dir / "_preview" / "langweb.ply"
    if langweb.is_file() and _ply_vertex_count(langweb) != n_rows:
        raise PluckError(
            "langweb.ply row count differs from splat.ply — regenerate the "
            "web-optimized artifacts (the backdrop would address wrong rows)")

    props = _prop_slugs(world_dir)
    if not props:
        raise PluckError("the world has no prop elements — nothing to pluck")

    index_map_path = lf_dir / "ply_index_map.npy"
    index_map = None
    splat_keys: dict[bytes, int] | None = None
    if index_map_path.is_file():
        index_map = np.load(index_map_path).astype(np.int64)
        method = "index-map"
    else:
        splat_keys = _splat_xyz_keys(splat_ply)
        method = "xyz-match"

    elements: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for slug in sorted(props):
        idx_path = job_dir / "_scene" / "isolated" / slug / "object_indices.npz"
        if not idx_path.is_file():
            skipped[slug] = "no-isolate-indices"
            continue
        indices = np.load(idx_path)["indices"].astype(np.int64)
        if index_map is not None:
            rows, dropped = _rows_via_map(indices, index_map)
        else:
            obj_ply = idx_path.with_name("object.ply")
            if not obj_ply.is_file():
                skipped[slug] = "no-object-ply-for-xyz-match"
                continue
            matched = _rows_via_xyz(obj_ply, splat_keys or {})
            if matched is None:
                skipped[slug] = ("xyz-match below "
                                 f"{XYZ_MATCH_MIN_FRAC:.0%} — refused rather "
                                 "than veil the wrong gaussians")
                continue
            rows, dropped = matched, 0
        if len(rows) and (rows[0] < 0 or rows[-1] >= n_rows):
            raise PluckError(f"{slug}: mapped rows escape [0, {n_rows}) — "
                             "the index map does not describe this splat")
        if len(np.unique(rows)) != len(rows):
            raise PluckError(f"{slug}: duplicate rows after mapping — "
                             "the index map is not injective here")
        if not len(rows):
            skipped[slug] = "no rows survived the export mapping"
            continue
        elements[slug] = {
            "rows": [int(r) for r in rows],
            "count": int(len(rows)),
            "ckpt_count": int(len(indices)),
            "dropped_by_export": int(dropped),
        }

    if not elements:
        raise PluckError(
            "no prop produced pluck rows — " + (
                "; ".join(f"{s}: {r}" for s, r in skipped.items())
                or "isolate produced nothing usable"))

    # Cross-slug disjointness: first-claim-wins upstream makes this
    # structural, so a violation means the inputs are inconsistent.
    all_rows = np.concatenate([np.asarray(e["rows"]) for e in elements.values()])
    if len(np.unique(all_rows)) != len(all_rows):
        raise PluckError("two props claim the same splat rows — re-run the "
                         "isolate stage (its claims should be disjoint)")

    built_from: dict[str, Any] = {
        "splat_ply": manifests.file_identity(splat_ply, include_sha256=False),
        "batch_isolate": manifests.file_identity(isolate_receipt,
                                                 include_sha256=False),
    }
    if index_map is not None:
        built_from["ply_index_map"] = manifests.file_identity(
            index_map_path, include_sha256=False)

    return {
        "schema": PLUCK_SCHEMA,
        "version": 1,
        "job_id": job_dir.name,
        "built_at": manifests.utc_now(),
        "backdrop_fmt": "langweb",
        "n_rows": n_rows,
        "method": method,
        "elements": elements,
        "skipped": skipped,
        "built_from": built_from,
    }


def validate_pluck(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != PLUCK_SCHEMA:
        raise PluckError("not a pluck document")
    n_rows = document.get("n_rows")
    if not isinstance(n_rows, int) or n_rows <= 0:
        raise PluckError("pluck n_rows must be a positive int")
    elements = document.get("elements")
    if not isinstance(elements, dict) or len(elements) > MAX_ELEMENTS:
        raise PluckError(f"pluck elements must be a dict of at most "
                         f"{MAX_ELEMENTS} slugs")
    for slug, entry in elements.items():
        rows = (entry or {}).get("rows")
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(r, int) or r < 0 or r >= n_rows
                       for r in rows)):
            raise PluckError(f"{slug}: rows must be ints in [0, {n_rows})")
        if (entry.get("count") != len(rows)):
            raise PluckError(f"{slug}: count does not match rows")
    return document


def write_pluck(world_dir: Path, document: dict[str, Any]) -> dict[str, Any]:
    validated = validate_pluck(document)
    manifests.atomic_write_json(Path(world_dir) / PLUCK_NAME, validated)
    return validated


def read_pluck(world_dir: Path, job_dir: Path
               ) -> tuple[dict[str, Any] | None, bool, list[str]]:
    """(doc, stale, reasons). Absent -> (None, False, []). A corrupt doc or a
    changed input is reported, never silently trusted — the walker refuses to
    veil on stale=True."""
    path = Path(world_dir) / PLUCK_NAME
    if not path.is_file():
        return None, False, []
    try:
        doc = validate_pluck(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        return None, True, [f"{PLUCK_NAME} unreadable: {exc}"]

    job_dir = Path(job_dir)
    reasons: list[str] = []
    sources = {
        "splat_ply": job_dir / "_preview" / "splat.ply",
        "batch_isolate": job_dir / "_scene" / "isolated" / "batch_isolate.json",
        "ply_index_map": job_dir / "_langfield" / "ply_index_map.npy",
    }
    for key, record in (doc.get("built_from") or {}).items():
        source = sources.get(key)
        if source is None or not isinstance(record, dict):
            continue
        if not manifests.same_file_identity(source, record):
            reasons.append(f"{key} changed since the pluck doc was built")
    if (job_dir / "_langfield" / "STALE").is_file():
        reasons.append("language field is STALE (splat edited)")
    return doc, bool(reasons), reasons
