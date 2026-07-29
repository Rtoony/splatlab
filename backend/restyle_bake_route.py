"""Bake a restyle permanently — and undo it. The W3 fantasy lane's landing.

The restyle document is a runtime overlay: the walker applies it, and every
export ships the as-captured albedo (grep `restyle` in export_route — it
appears nowhere). These routes close that gap:

  POST /jobs/{id}/world/restyle/bake         re-bake every restyled element's
                                             atlas via mesh/restyle_bake.py,
                                             land through the polish-lane
                                             discipline (versioned, receipted,
                                             audited), then CLEAR the overlay —
                                             the look is now the capture.
  POST /jobs/{id}/world/restyle/bake/revert  restore the versioned GLB+atlas
                                             pairs the bake recorded and bring
                                             the overlay document back.

What is baked: tint + material(+material_scale). Lighting is NEVER baked —
the atlas already carries the capture's light (double-lighting hazard) and a
preset is a three.js rig, not surface data; the overlay keeps its lighting
block through a bake, so the mood survives as the runtime rig it always was.

Geometry passes through untouched (the baker enforces it with a readback
face-count gate), so collision, navmesh and the geometry gates are unaffected;
only the live GLB + sidecar `_atlas.png` change — versioned together at the
same `-vNNNN` seq so revert restores an exact pre-bake tree. The bake itself
is all-or-nothing in staging; landing is a fast local replace loop. A crash
mid-landing leaves priors safe in `_world/versions/` and the overlay intact —
re-running the bake converges.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

import artifact_manifest as manifests
import opregistry
import splat_route
import world_restyle as wr
from operator_audit import audit_operator_event
from polish_route import _next_version_path
from world_interactions_route import _require_world, _restyle_payload

router = APIRouter()

RESTYLE_BAKE_SCHEMA = "dev.splatlab.restyle-bake-receipt/v1"
BAKE_RECEIPT_NAME = "restyle_bake.json"
RESTYLE_BAKE_SCRIPT = Path(__file__).resolve().parent / "mesh" / "restyle_bake.py"
LIGHTING_NOTE = ("lighting presets are a walker light rig; the atlas already "
                 "carries the capture's light, so a relight is never baked")


class RestyleBakeBody(BaseModel):
    """`units_per_metre` is the walker's one scale dial — 1.0 on calibrated
    (metre-baked) worlds; the storey-heuristic guess on uncalibrated ones. It
    converts the document's metres-per-tile exactly like the preview does."""
    units_per_metre: float = Field(1.0, gt=0.0, le=1000.0)


def _target_path(world_dir: Path, slug: str) -> Path:
    return (world_dir / "shell.glb" if slug == "shell"
            else world_dir / "elements" / f"{slug}.glb")


def _marker_path(world_dir: Path, slug: str) -> Path:
    return (world_dir / "shell.restyle.json" if slug == "shell"
            else world_dir / "elements" / f"{slug}.restyle.json")


def _atlas_path(target: Path) -> Path:
    return target.with_name(f"{target.stem}_atlas.png")


def _parse_report(stdout: bytes) -> dict[str, Any]:
    for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    return {}


@router.post("/jobs/{job_id}/world/restyle/bake")
async def bake_world_restyle(job_id: str, body: RestyleBakeBody,
                             request: Request) -> dict[str, Any]:
    """Make the current restyle permanent: rewrite each restyled element's
    atlas/GLB, version the priors, archive + clear the overlay document."""
    splat_route.require_heavy_work_admitted()
    _job_dir, world_dir, world_manifest = _require_world(job_id)

    doc = wr.read_restyle(world_dir, job_id)
    entries = {slug: entry for slug, entry in (doc.get("elements") or {}).items()
               if entry.get("tint") or entry.get("material")}
    if not entries:
        raise HTTPException(
            status_code=409,
            detail="Nothing to bake — no element carries a tint or material. "
                   f"(A {LIGHTING_NOTE}.)")
    missing = [slug for slug in sorted(entries)
               if not _target_path(world_dir, slug).is_file()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail="No built GLB for restyled element(s): "
                   f"{', '.join(missing)} — re-run solidify first")
    if not (splat_route.MESH_ENV_PYTHON.is_file()
            and RESTYLE_BAKE_SCRIPT.is_file()):
        raise HTTPException(status_code=400,
                            detail="Restyle-bake toolchain unavailable.")

    lock = splat_route._mesh_export_lock(job_id)
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A mesh build or export is running for this scene — retry "
                   "when it finishes")

    staging = world_dir / f".building-restyle-{uuid.uuid4().hex}"
    world = manifests.read_json(world_dir / "world.json") or {}
    uncalibrated = world.get("units") != "meters"
    landed: list[dict[str, Any]] = []
    async with lock:
        op_id = opregistry.start("restyle_bake", job_id, step="bake",
                                 detail=f"{len(entries)} element(s)")
        try:
            rc, out, stderr = await splat_route._run_capture_subprocess([
                str(splat_route.MESH_ENV_PYTHON), str(RESTYLE_BAKE_SCRIPT),
                str(_job_dir), "--staging-dir", str(staging),
                "--units-per-metre", str(body.units_per_metre),
            ])
            report = _parse_report(out)
            if rc != 0 or not report.get("ok"):
                errors = [f"{el.get('slug')}: {el.get('error')}"
                          for el in report.get("elements", [])
                          if not el.get("ok")]
                if not errors:
                    errors = [report.get("error") or "\n".join(
                        stderr.decode("utf-8", errors="replace").splitlines()[-4:])]
                raise HTTPException(
                    status_code=500,
                    detail=f"Restyle bake failed: {'; '.join(errors)[:1500]}")

            opregistry.update(op_id, step="land", progress=0.8)
            versions_dir = world_dir / "versions"
            for element in report["elements"]:
                slug = element["slug"]
                target = _target_path(world_dir, slug)
                supersedes = manifests.file_identity(target)
                version_glb = _next_version_path(versions_dir, slug)
                os.replace(target, version_glb)
                atlas = _atlas_path(target)
                if atlas.is_file():
                    # Same -vNNNN seq as the GLB, so revert restores the pair.
                    os.replace(atlas, version_glb.with_name(
                        f"{version_glb.stem}_atlas.png"))
                os.replace(staging / element["staged_glb"], target)
                os.replace(staging / element["staged_atlas"], atlas)

                applied = {key: element[key]
                           for key in ("tint", "material", "material_scale")
                           if element.get(key) is not None}
                manifests.atomic_write_json(_marker_path(world_dir, slug), {
                    "schema": RESTYLE_BAKE_SCHEMA,
                    "provenance": "restyled",
                    "applied": applied,
                    "verbs": element["verbs"],
                    "units_per_metre": body.units_per_metre,
                    "baked_at": manifests.utc_now(),
                    "supersedes": supersedes,
                    # Unlike polish's atlas_superseded: the fresh sidecar DOES
                    # describe the file — texture_coverage stays truthful.
                    "atlas_updated": True,
                    "versioned_as": f"versions/{version_glb.name}",
                    **manifests.file_identity(target),
                })
                landed.append({
                    "slug": slug,
                    "verbs": element["verbs"],
                    "applied": applied,
                    "covered_frac": element.get("covered_frac"),
                    "versioned_as": f"versions/{version_glb.name}",
                    "seconds": element.get("seconds"),
                })

            manifests.atomic_write_json(world_dir / BAKE_RECEIPT_NAME, {
                "schema": RESTYLE_BAKE_SCHEMA,
                "job_id": job_id,
                "op_id": op_id,
                "baked_at": manifests.utc_now(),
                "units_per_metre": body.units_per_metre,
                "uncalibrated": uncalibrated,
                "lighting_baked": False,
                "lighting_note": LIGHTING_NOTE,
                # The FULL pre-bake overlay — revert's restoration source.
                "archived_restyle": doc,
                "elements": landed,
            })

            # The baked look IS the capture now. Clear the element overlay so
            # the walker does not double-apply it; the lighting mood survives.
            wr.write_restyle(world_dir, {"job_id": job_id, "elements": {},
                                         "lighting": doc.get("lighting")})

            opregistry.finish(op_id, result={
                "elements": len(landed), "uncalibrated": uncalibrated})
        except HTTPException as exc:
            opregistry.finish(op_id, status=opregistry.FAILED,
                              error=str(exc.detail)[:2000])
            raise
        except BaseException as exc:
            opregistry.finish(op_id, status=opregistry.FAILED,
                              error=f"{type(exc).__name__}: {exc}"[:2000])
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    meta = splat_route._read_meta(job_id) or {}
    await audit_operator_event(
        request=request,
        title="Restyle baked into world",
        description=f"{job_id}: {len(landed)} element(s) re-baked "
                    f"({', '.join(sorted(entries))})",
        variant="success",
        action="splat.restyle_bake",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "elements": len(landed), "op_id": op_id},
    )
    return {
        "ok": True,
        "job_id": job_id,
        "op_id": op_id,
        "uncalibrated": uncalibrated,
        "lighting_baked": False,
        "elements": landed,
        "restyle": _restyle_payload(job_id, world_dir, world_manifest),
    }


@router.post("/jobs/{job_id}/world/restyle/bake/revert")
async def revert_world_restyle_bake(job_id: str,
                                    request: Request) -> dict[str, Any]:
    """Undo the most recent bake: restore its versioned GLB+atlas pairs
    (COPIED back — versions stay immutable history), delete its markers, and
    bring the archived overlay document back. The reverted state is itself
    versioned first, so a revert can be un-reverted by baking again."""
    _job_dir, world_dir, world_manifest = _require_world(job_id)
    receipt_path = world_dir / BAKE_RECEIPT_NAME
    if not receipt_path.is_file():
        raise HTTPException(status_code=404,
                            detail="No restyle bake to revert for this scene")
    receipt = manifests.read_json(receipt_path)
    if not isinstance(receipt, dict) or not receipt.get("elements"):
        raise HTTPException(status_code=500,
                            detail="Restyle-bake receipt is unreadable")

    lock = splat_route._mesh_export_lock(job_id)
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A mesh build or export is running for this scene — retry "
                   "when it finishes")

    versions_dir = world_dir / "versions"
    # Two passes: verify EVERYTHING first so a missing version cannot leave a
    # half-reverted world behind.
    plan: list[tuple[str, Path]] = []
    for element in receipt["elements"]:
        slug = str(element.get("slug") or "")
        version_glb = versions_dir / Path(str(element.get("versioned_as"))).name
        if not slug or not version_glb.is_file():
            raise HTTPException(
                status_code=409,
                detail=f"{slug or '?'}: recorded version "
                       f"{version_glb.name} is missing — cannot revert")
        plan.append((slug, version_glb))

    restored: list[str] = []
    async with lock:
        op_id = opregistry.start("restyle_bake_revert", job_id, step="revert",
                                 detail=f"{len(plan)} element(s)")
        try:
            for slug, version_glb in plan:
                target = _target_path(world_dir, slug)
                atlas = _atlas_path(target)
                if target.is_file():
                    current = _next_version_path(versions_dir, slug)
                    os.replace(target, current)
                    if atlas.is_file():
                        os.replace(atlas, current.with_name(
                            f"{current.stem}_atlas.png"))
                shutil.copy2(version_glb, target)
                version_atlas = version_glb.with_name(
                    f"{version_glb.stem}_atlas.png")
                if version_atlas.is_file():
                    shutil.copy2(version_atlas, atlas)
                else:
                    # The pre-bake tree had no sidecar for this element (a
                    # polished GLB embeds its texture) — restore that exactly.
                    atlas.unlink(missing_ok=True)
                _marker_path(world_dir, slug).unlink(missing_ok=True)
                restored.append(slug)

            archived = receipt.get("archived_restyle")
            if isinstance(archived, dict):
                try:
                    wr.write_restyle(world_dir, archived)
                except wr.RestyleError:
                    # The overlay is cosmetic; a stale archive must not block
                    # the geometry restore that already happened.
                    (world_dir / wr.RESTYLE_FILENAME).unlink(missing_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            receipt_path.rename(
                receipt_path.with_name(f"restyle_bake.reverted-{stamp}.json"))
            opregistry.finish(op_id, result={"elements": len(restored)})
        except HTTPException as exc:
            opregistry.finish(op_id, status=opregistry.FAILED,
                              error=str(exc.detail)[:2000])
            raise
        except BaseException as exc:
            opregistry.finish(op_id, status=opregistry.FAILED,
                              error=f"{type(exc).__name__}: {exc}"[:2000])
            raise

    meta = splat_route._read_meta(job_id) or {}
    await audit_operator_event(
        request=request,
        title="Restyle bake reverted",
        description=f"{job_id}: {len(restored)} element(s) restored "
                    f"({', '.join(restored)})",
        variant="warning",
        action="splat.restyle_bake_revert",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "elements": len(restored), "op_id": op_id},
    )
    return {
        "ok": True,
        "job_id": job_id,
        "op_id": op_id,
        "elements": restored,
        "restyle": _restyle_payload(job_id, world_dir, world_manifest),
    }
