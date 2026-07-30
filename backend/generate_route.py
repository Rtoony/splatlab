"""Propose-generated: per-object generative reconstruction as a CANDIDATE.

The third triage treatment. `auto_generate` (solidify) exists for pipeline
runs, but it auto-APPLIES; this lane never decides — a human proposes, the
gates speak, a human promotes (or discards), and every step is reversible.

  POST   /jobs/{id}/objects/{slug}/generate/propose   run SAM-3D under the
         GPU arbiter (unlike the solidify path — review finding: a 13 GB
         CUDA load must hold a lease like every other generative lane);
         candidate lands in _regen/objects/<slug>/, NOTHING is applied
  GET    /jobs/{id}/objects/{slug}/generate/candidate  the review payload
  GET    /jobs/{id}/generate/file?slug&fmt             preview/alignment/
         mesh/report for the review UI (no route existed for _regen/objects)
  POST   /jobs/{id}/objects/{slug}/generate/promote    the HITL apply:
         place_generated (transform+decimate+bake, mesh env) -> staged ->
         version the prior element GLB+atlas -> replace -> marker
  POST   /jobs/{id}/objects/{slug}/generate/revert     restore the exact
         versioned pair the marker recorded
  DELETE /jobs/{id}/objects/{slug}/generate/candidate  discard (candidates
         are regenerable; the audit log keeps the history)
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

import artifact_manifest as manifests
import glb_check
import gpu_arbiter
import polish_route
import splat_route
from operator_audit import audit_operator_event

router = APIRouter()

GENERATED_MARKER_SCHEMA = "dev.splatlab.generated-promotion/v1"
# SA-3DAO + MoGe peak well under this on the 5090; the lease is what keeps a
# concurrent TRELLIS/proxy run from OOMing the card, not a precise budget.
SAM3D_VRAM_MB = 22_000
GENERATE_TIMEOUT_S = 1800


def _candidate_dir(job_dir: Path, slug: str) -> Path:
    return job_dir / "_regen" / "objects" / slug


def _read_candidate(job_dir: Path, slug: str) -> dict[str, Any] | None:
    """Mirror of scene_solidify._generated_candidate's PLACED semantics
    (that module imports trimesh and cannot load host-side): a candidate is
    a generated GLB that knows where it goes."""
    cdir = _candidate_dir(job_dir, slug)
    glb = cdir / "generated_mesh.glb"
    report_path = cdir / "generate_report.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    placement = ((report.get("capture_frame_placement") or {})
                 .get("mesh_glb") or {})
    placed = bool(glb.is_file()
                  and placement.get("placement_resolved")
                  and placement.get("transform_4x4_generated_to_capture"))
    return {"report": report, "placed": placed, "glb": glb.is_file(),
            "dir": cdir}


def _require_job(job_id: str) -> tuple[dict[str, Any], Path]:
    return polish_route._require_job(job_id)


def _candidate_payload(job_id: str, slug: str,
                       cand: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for fmt, name in (("mesh", "generated_mesh.glb"),
                      ("preview", "preview.png"),
                      ("alignment", "alignment.png"),
                      ("input", "model_input_rgba.png"),
                      ("report", "generate_report.json")):
        if (cand["dir"] / name).is_file():
            files[fmt] = (f"/api/splat/jobs/{job_id}/generate/file"
                          f"?slug={slug}&fmt={fmt}")
    marker = None
    return {"job_id": job_id, "slug": slug, "placed": cand["placed"],
            "report": cand["report"], "files": files, "marker": marker}


@router.post("/jobs/{job_id}/objects/{slug}/generate/propose")
async def propose_generated(job_id: str, slug: str,
                            request: Request) -> dict[str, Any]:
    """Run the generator; land a candidate; apply NOTHING. The gates' verdict
    (mask IoU, placement) comes back verbatim — a refusal is a first-class
    answer, not an error."""
    splat_route.require_heavy_work_admitted()
    meta, job_dir = _require_job(job_id)
    slug = splat_route._object_slug(slug)
    inventory = job_dir / "_scene" / "inventory.json"
    if not inventory.is_file():
        raise HTTPException(status_code=409,
                            detail="No scene inventory — run the isolate/"
                                   "inventory stage before proposing")

    def operation() -> tuple[int, str]:
        cmd = [str(splat_route.MESH_ENV_PYTHON),
               str(splat_route.MESH_DIR / "object_generate.py"),
               str(job_dir), "--object", slug]
        # Own process group; a timeout must kill the CUDA grandchild too
        # (scene_solidify's review finding, mirrored).
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        try:
            _out, err = proc.communicate(timeout=GENERATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                _out, err = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                _out, err = proc.communicate()
            return -1, "generation timed out"
        return proc.returncode, (err or "")[-2000:]

    try:
        rc, err = await gpu_arbiter.run_sync_gpu_operation(
            lane="object-generate", operation_id=f"{job_id}/{slug}",
            vram_mb=SAM3D_VRAM_MB, func=operation)
    except gpu_arbiter.GPUArbiterUnavailable as exc:
        raise HTTPException(status_code=503,
                            detail=f"Generation blocked: {exc}") from exc

    cand = _read_candidate(job_dir, slug)
    await audit_operator_event(
        request=request,
        title="Generated candidate proposed",
        description=f"{job_id}/{slug}: rc={rc}, "
                    f"{'PLACED candidate' if cand and cand['placed'] else 'refused/unplaced'}",
        variant="success" if cand and cand["placed"] else "warning",
        action="splat.generate_propose",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "slug": slug, "rc": rc,
                  "placed": bool(cand and cand["placed"])},
    )
    if cand is None:
        raise HTTPException(
            status_code=422,
            detail=f"Generation produced no report (rc={rc}): {err[-400:]}")
    return {"ok": True, "rc": rc, **_candidate_payload(job_id, slug, cand)}


@router.get("/jobs/{job_id}/objects/{slug}/generate/candidate")
async def get_generated_candidate(job_id: str, slug: str) -> dict[str, Any]:
    _meta, job_dir = _require_job(job_id)
    slug = splat_route._object_slug(slug)
    cand = _read_candidate(job_dir, slug)
    if cand is None:
        raise HTTPException(status_code=404,
                            detail="No generated candidate for this object — "
                                   "POST .../generate/propose")
    payload = _candidate_payload(job_id, slug, cand)
    marker = (job_dir / "_world" / "elements" / f"{slug}.generated.json")
    if marker.is_file():
        try:
            payload["marker"] = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            payload["marker"] = {"error": "marker unreadable"}
    return payload


_FILE_FMTS = {"mesh": ("generated_mesh.glb", "model/gltf-binary"),
              "splat": ("generated_splat.ply", "application/octet-stream"),
              "preview": ("preview.png", "image/png"),
              "alignment": ("alignment.png", "image/png"),
              "input": ("model_input_rgba.png", "image/png"),
              "report": ("generate_report.json", "application/json")}


@router.get("/jobs/{job_id}/generate/file")
async def get_generate_file(job_id: str, slug: str, fmt: str) -> FileResponse:
    _meta, job_dir = _require_job(job_id)
    slug = splat_route._object_slug(slug)
    if fmt not in _FILE_FMTS:
        raise HTTPException(status_code=400,
                            detail=f"fmt must be one of {sorted(_FILE_FMTS)}")
    name, media = _FILE_FMTS[fmt]
    path = (_candidate_dir(job_dir, slug) / name).resolve()
    try:
        path.relative_to(job_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    # Candidates are rewritten in place by re-proposes — same rule as the
    # world/preview routes.
    return FileResponse(str(path), media_type=media, filename=path.name,
                        headers={"Cache-Control": "no-cache"})


@router.post("/jobs/{job_id}/objects/{slug}/generate/promote")
async def promote_generated(job_id: str, slug: str,
                            request: Request) -> dict[str, Any]:
    """The HITL apply. place_generated (mesh env) transforms + decimates +
    bakes the candidate into a staged element, then the prior GLB+atlas pair
    is VERSIONED and replaced — the restyle-bake landing pattern, with the
    exact version filenames recorded in the marker so revert is exact."""
    meta, job_dir = _require_job(job_id)
    slug = splat_route._object_slug(slug)
    world_dir = job_dir / "_world"
    target = world_dir / "elements" / f"{slug}.glb"
    if not target.is_file():
        raise HTTPException(
            status_code=404,
            detail="Promotion replaces a built element; this object has no "
                   "element GLB — run the solidify stage first")
    cand = _read_candidate(job_dir, slug)
    if cand is None or not cand["placed"]:
        raise HTTPException(
            status_code=409,
            detail="No PLACED candidate — propose first (a refused or "
                   "placement-unresolved candidate cannot be promoted)")
    marker_path = world_dir / "elements" / f"{slug}.generated.json"
    if marker_path.is_file():
        raise HTTPException(
            status_code=409,
            detail="Already promoted — revert first to promote a new candidate")

    world = manifests.read_json(world_dir / "world.json") or {}
    mpu = world.get("meters_per_unit") or (meta.get("meters_per_unit") or None)
    staged = world_dir / f".building-promote-{uuid.uuid4().hex}.glb"
    snippet = f"""
import json, sys
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from pathlib import Path
from mesh.scene_solidify import _generated_candidate, place_generated
gen = _generated_candidate(Path({str(job_dir)!r}), {slug!r})
assert gen is not None, "candidate vanished"
res = place_generated(gen, __import__("pathlib").Path({str(staged)!r}),
                      mpu={mpu!r}, faces=8000, tex=1024)
print("PLACE " + json.dumps(res))
"""
    rc, out_b, err_b = await splat_route._run_capture_subprocess(
        [str(splat_route.MESH_ENV_PYTHON), "-c", snippet])
    out = out_b.decode(errors="replace")
    line = next((l for l in out.splitlines() if l.startswith("PLACE ")), None)
    res = json.loads(line[6:]) if line else {}
    if rc != 0 or not res.get("ok") or not staged.is_file():
        staged.unlink(missing_ok=True)
        staged.with_name(staged.stem + "_atlas.png").unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="place_generated failed: "
                   f"{res.get('error') or err_b.decode(errors='replace')[-300:]}")
    glb_check.validate_glb(staged)

    lock = splat_route._mesh_export_lock(job_id)
    if lock.locked():
        staged.unlink(missing_ok=True)
        staged.with_name(staged.stem + "_atlas.png").unlink(missing_ok=True)
        raise HTTPException(status_code=409,
                            detail="A mesh build or export is running — retry")
    async with lock:
        versions = world_dir / "versions"
        prior_glb_version = polish_route._next_version_path(versions, slug)
        supersedes = manifests.file_identity(target)
        os.replace(target, prior_glb_version)
        atlas = target.with_name(f"{slug}_atlas.png")
        prior_atlas_version = None
        if atlas.is_file():
            prior_atlas_version = prior_glb_version.with_suffix(".png")
            os.replace(atlas, prior_atlas_version)
        os.replace(staged, target)
        staged_atlas = staged.with_name(staged.stem + "_atlas.png")
        if staged_atlas.is_file():
            os.replace(staged_atlas, atlas)
        marker = {
            "schema": GENERATED_MARKER_SCHEMA,
            "slug": slug,
            "promoted_at": manifests.utc_now(),
            "supersedes": supersedes,
            "prior_glb_version": prior_glb_version.name,
            "prior_atlas_version": (prior_atlas_version.name
                                    if prior_atlas_version else None),
            "candidate_iou": (cand["report"].get("mask_alignment_gate") or {})
                             .get("iou_vs_captured_object"),
            "faces": res.get("faces"),
            "provenance": "generative render-only (promoted by operator)",
            **manifests.file_identity(target),
        }
        manifests.atomic_write_json(marker_path, marker)
        # world.json stays honest about what the element now is.
        for el in world.get("elements") or []:
            if isinstance(el, dict) and el.get("slug") == slug:
                el["geometry_source"] = "generated"
                el["provenance"] = marker["provenance"]
        manifests.atomic_write_json(world_dir / "world.json", world)

    await audit_operator_event(
        request=request,
        title="Generated candidate promoted",
        description=f"{job_id}/{slug}: candidate promoted into the element "
                    f"({res.get('faces')} faces); prior versioned as "
                    f"{prior_glb_version.name}",
        variant="success",
        action="splat.generate_promote",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "slug": slug, "faces": res.get("faces")},
    )
    return {"ok": True, "slug": slug, "marker": marker,
            "url": f"/api/splat/jobs/{job_id}/world/file?name=elements/{slug}.glb"}


@router.post("/jobs/{job_id}/objects/{slug}/generate/revert")
async def revert_generated(job_id: str, slug: str,
                           request: Request) -> dict[str, Any]:
    meta, job_dir = _require_job(job_id)
    slug = splat_route._object_slug(slug)
    world_dir = job_dir / "_world"
    marker_path = world_dir / "elements" / f"{slug}.generated.json"
    if not marker_path.is_file():
        raise HTTPException(status_code=404,
                            detail="Nothing to revert — no promotion marker")
    marker = json.loads(marker_path.read_text())
    versions = world_dir / "versions"
    prior_glb = versions / str(marker.get("prior_glb_version") or "")
    if not prior_glb.is_file():
        raise HTTPException(status_code=409,
                            detail=f"Versioned original missing "
                                   f"({marker.get('prior_glb_version')}) — "
                                   "cannot revert exactly")
    target = world_dir / "elements" / f"{slug}.glb"
    lock = splat_route._mesh_export_lock(job_id)
    if lock.locked():
        raise HTTPException(status_code=409,
                            detail="A mesh build or export is running — retry")
    async with lock:
        os.replace(prior_glb, target)
        if marker.get("prior_atlas_version"):
            prior_atlas = versions / marker["prior_atlas_version"]
            if prior_atlas.is_file():
                os.replace(prior_atlas,
                           target.with_name(f"{slug}_atlas.png"))
        marker_path.unlink(missing_ok=True)
        world = manifests.read_json(world_dir / "world.json") or {}
        for el in world.get("elements") or []:
            if isinstance(el, dict) and el.get("slug") == slug:
                el["geometry_source"] = "gaussians"
                el.pop("provenance", None)
        manifests.atomic_write_json(world_dir / "world.json", world)

    await audit_operator_event(
        request=request,
        title="Generated promotion reverted",
        description=f"{job_id}/{slug}: captured element restored from "
                    f"{marker.get('prior_glb_version')}",
        variant="success",
        action="splat.generate_revert",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "slug": slug},
    )
    return {"ok": True, "slug": slug, "restored": marker.get("prior_glb_version")}


@router.delete("/jobs/{job_id}/objects/{slug}/generate/candidate")
async def discard_generated_candidate(job_id: str, slug: str,
                                      request: Request) -> dict[str, Any]:
    meta, job_dir = _require_job(job_id)
    slug = splat_route._object_slug(slug)
    marker = job_dir / "_world" / "elements" / f"{slug}.generated.json"
    if marker.is_file():
        raise HTTPException(status_code=409,
                            detail="Candidate is promoted — revert before "
                                   "discarding it")
    cdir = _candidate_dir(job_dir, slug)
    if not cdir.is_dir():
        raise HTTPException(status_code=404, detail="No candidate to discard")
    shutil.rmtree(cdir)
    await audit_operator_event(
        request=request,
        title="Generated candidate discarded",
        description=f"{job_id}/{slug}: candidate directory removed "
                    "(regenerable; audit history retained)",
        variant="success",
        action="splat.generate_discard",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "slug": slug},
    )
    return {"ok": True, "slug": slug}
