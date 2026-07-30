"""Set-dressing lands: the audited create/delete door for AUTHORED world
elements — the mirror image of polish.

Polish replaces a solidified artifact and refuses to invent one; placement
INVENTS one and refuses to replace. A slug that exists anywhere — the graded
manifest, world.json (including unbuilt instances), the placed registry, or
as a file on disk — is refused, so neither lane can ever shadow the other.

The entry lands in world_manifest.json through the pure registry merge
(world_placed.apply_placed_to_manifest), which buys the walker, affordances,
restyle, interactions and world_slugs in one append — and it SURVIVES
rebuilds because world_collision re-applies the registry every run (the
sticky-regrades pattern).

Frame gate: authored GLBs must carry world-frame-baked vertices under
IDENTITY node transforms (export from the Blender loop with
bake_world_transform=True). role=prop REFUSES node-TRS uploads (422) —
the walker's physics re-origin silently mis-places them; role=static only
warns, because the BVH collider bakes matrixWorld and tolerates it.

Crash-window ordering is GLB -> registry -> manifest, chosen because both
torn states heal: a GLB without a registry entry is invisible and
re-placeable after DELETE-by-hand of the file; a registry entry without a
manifest patch is healed by the next collision run's preservation hook.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile

import artifact_manifest as manifests
import glb_check
import polish_route
import splat_route
import world_interactions as wi
import world_placed as wp
from operator_audit import audit_operator_event

router = APIRouter()

PLACED_RECEIPT_SCHEMA = "dev.splatlab.placed-receipt/v1"


def _world_paths(job_dir: Path) -> tuple[Path, Path]:
    world_dir = job_dir / splat_route.WORLD_DIRNAME
    return world_dir, world_dir / "world_manifest.json"


def _read_registry_or_500(world_dir: Path, job_id: str) -> dict[str, Any]:
    try:
        return wp.read_placed(world_dir, job_id)
    except wp.PlacedError as exc:
        # Treating a corrupt registry as empty would orphan every prior
        # entry's record on the next write — refuse until it is repaired.
        raise HTTPException(
            status_code=500,
            detail=f"Placement registry is damaged: {exc}") from exc


def _patch_manifest(manifest_path: Path, manifest: dict[str, Any],
                    registry: dict[str, Any]) -> dict[str, Any]:
    merged, _conflicts = wp.apply_placed_to_manifest(manifest, registry)
    manifests.atomic_write_json(manifest_path, merged)
    return merged


@router.post("/jobs/{job_id}/world/elements")
async def place_world_element(
    job_id: str, file: UploadFile, request: Request,
    slug: str, role: str = "static", label: str = "",
) -> dict[str, Any]:
    """Land an authored GLB as a NEW world element (fantasy set-dressing).

    Refusal matrix, tree untouched on every path: bad job/slug 404, bad role
    400, un-solidified 409, ungraded 409, existing slug 409, bad GLB 400,
    oversize 413, busy lock 409, node-TRS prop 422, full registry 409."""
    meta, job_dir = polish_route._require_job(job_id)
    world_dir, manifest_path = _world_paths(job_dir)
    if not wp.SLUG_RE.match(slug or "") or slug == "shell":
        raise HTTPException(
            status_code=404,
            detail="Bad placement slug — lowercase kebab, at most 40 chars, "
                   "and never 'shell'")
    if role not in wp.ROLES:
        raise HTTPException(status_code=400,
                            detail=f"role must be one of {wp.ROLES}")
    if not (world_dir / "world.json").is_file():
        raise HTTPException(
            status_code=409,
            detail="No walkable world for this scene yet — run the world "
                   "solidify stage first; set-dressing needs a world to "
                   "stand in.")
    manifest = manifests.read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise HTTPException(
            status_code=409,
            detail="World is ungraded — run the collision stage before "
                   "placing assets (placement patches world_manifest.json).")
    registry = _read_registry_or_500(world_dir, job_id)
    if len(registry["elements"]) >= wp.MAX_PLACED:
        raise HTTPException(
            status_code=409,
            detail=f"Placement registry is full ({wp.MAX_PLACED} assets)")

    world = manifests.read_json(world_dir / "world.json") or {}
    taken = set(wi.world_slugs(manifest))
    taken |= {e.get("slug") for e in world.get("elements") or []
              if isinstance(e, dict)}
    taken |= {e["slug"] for e in registry["elements"]}
    target = world_dir / "elements" / f"{slug}.glb"
    if slug in taken or target.is_file():
        raise HTTPException(
            status_code=409,
            detail=f"Placement creates a new element; '{slug}' already "
                   "exists — polish replaces existing ones")

    filename = (file.filename or "").strip() or "upload.glb"
    tmp = world_dir / f".building-place-{uuid.uuid4().hex}.glb"
    try:
        written = await polish_route._stream_glb_to_tmp(file, tmp)
        summary = polish_route._validate_glb_or_400(tmp, filename)
        try:
            bounds = glb_check.position_bounds(tmp)
        except ValueError as exc:
            raise HTTPException(status_code=400,
                                detail=f"'{filename}': {exc}") from exc

        warnings: list[str] = []
        if not bounds["identity_transforms"]:
            if role in ("prop", "environment"):
                raise HTTPException(
                    status_code=422,
                    detail=f"{role} placement requires world-frame-baked "
                           "vertices under identity node transforms (props: "
                           "Rapier reads raw POSITION; environment: the "
                           "recorded AABB must match the walked-on collider) "
                           "— re-export from the Blender loop with "
                           "bake_world_transform "
                           f"(transformed nodes: {bounds['transformed_nodes']})")
            warnings.append(
                "GLB carries node transforms; statics tolerate them (the "
                "collider bakes matrixWorld) but the shared-frame contract "
                "prefers baked vertices")
        if world.get("units") != "meters":
            warnings.append(
                "world scale is a guess (uncalibrated) — a metre-modelled "
                "asset lands at a guessed size")
        shell_extent = (manifest.get("shell") or {}).get("extent")
        if isinstance(shell_extent, list) and len(shell_extent) == 3:
            centre = [(lo + hi) / 2 for lo, hi in
                      zip(bounds["aabb"]["min"], bounds["aabb"]["max"])]
            half_diag = sum((float(e) / 2) ** 2 for e in shell_extent) ** 0.5
            distance = sum(c * c for c in centre) ** 0.5
            # Environment pieces (terrain skirts, boundary walls) legitimately
            # sit at or past the shell edge — a wider band before crying wolf.
            band = 4.0 if role == "environment" else 1.5
            if half_diag > 0 and distance > half_diag * band:
                warnings.append(
                    f"asset centre is {distance:.1f} units from the origin — "
                    f"beyond ~{band}x the shell's half-diagonal "
                    f"({half_diag:.1f}); verify the placement "
                    "(the shared-frame rule)")

        lock = splat_route._mesh_export_lock(job_id)
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail="A mesh build or export is running for this scene — "
                       "retry when it finishes")
        async with lock:
            placed_at = manifests.utc_now()
            os.replace(tmp, target)
            entry = {
                "slug": slug, "label": label or slug, "role": role,
                "extent": bounds["extent"], "aabb": bounds["aabb"],
                "gltf": summary, "source_filename": filename,
                "uploaded_bytes": written, "placed_at": placed_at,
                "warnings": warnings,
                "file": manifests.file_identity(target),
            }
            registry["elements"].append(entry)
            validated = wp.write_placed(world_dir, registry)
            merged = _patch_manifest(manifest_path, manifest, validated)
            element = next(e for e in merged["elements"] if e["slug"] == slug)
            receipt = {
                "schema": PLACED_RECEIPT_SCHEMA,
                "provenance": "authored",
                "slug": slug, "role": role, "label": entry["label"],
                "source_filename": filename, "uploaded_bytes": written,
                "gltf": summary, "aabb": bounds["aabb"],
                "extent": bounds["extent"], "warnings": warnings,
                "uploaded_at": placed_at,
                "supersedes": None,
                **manifests.file_identity(target),
            }
            manifests.atomic_write_json(
                world_dir / "elements" / f"{slug}.placed.json", receipt)
    finally:
        tmp.unlink(missing_ok=True)

    await audit_operator_event(
        request=request,
        title="World asset placed",
        description=f"{job_id}/{slug}: {filename} ({written} bytes, {role})",
        variant="success",
        action="splat.world_place",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "slug": slug, "role": role,
                  "bytes": written},
    )
    return {
        "ok": True,
        "slug": slug,
        "url": f"/api/splat/jobs/{job_id}/world/file?name=elements/{slug}.glb",
        "element": element,
        "receipt": receipt,
        "warnings": warnings,
    }


@router.delete("/jobs/{job_id}/world/elements/{slug}")
async def remove_placed_world_element(
    job_id: str, slug: str, request: Request
) -> dict[str, Any]:
    """Remove a PLACED asset: GLB tombstoned to _world/versions/ (reversible
    — same shelf polish uses), registry entry dropped, manifest re-derived.

    Captured elements are refused: solidify rebuilds them, so removing one
    here would silently come back — only placed assets can be removed."""
    meta, job_dir = polish_route._require_job(job_id)
    world_dir, manifest_path = _world_paths(job_dir)
    if not wp.SLUG_RE.match(slug or "") or slug == "shell":
        raise HTTPException(status_code=404, detail="No such placed asset")
    registry = _read_registry_or_500(world_dir, job_id)
    target = world_dir / "elements" / f"{slug}.glb"
    if not any(e["slug"] == slug for e in registry["elements"]):
        manifest = manifests.read_json(manifest_path)
        known = wi.world_slugs(manifest) if isinstance(manifest, dict) else set()
        if slug in known or target.is_file():
            raise HTTPException(
                status_code=409,
                detail="Captured elements are rebuilt by solidify; only "
                       "placed assets can be removed")
        raise HTTPException(status_code=404,
                            detail=f"No placed asset '{slug}'")

    lock = splat_route._mesh_export_lock(job_id)
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A mesh build or export is running for this scene — "
                   "retry when it finishes")
    async with lock:
        tombstone = None
        supersedes = None
        if target.is_file():
            supersedes = manifests.file_identity(target)
            tombstone_path = polish_route._next_version_path(
                world_dir / "versions", slug)
            os.replace(target, tombstone_path)
            tombstone = f"versions/{tombstone_path.name}"
        registry["elements"] = [e for e in registry["elements"]
                                if e["slug"] != slug]
        validated = wp.write_placed(world_dir, registry)
        manifest = manifests.read_json(manifest_path)
        if isinstance(manifest, dict):
            _patch_manifest(manifest_path, manifest, validated)
        receipt = {
            "schema": PLACED_RECEIPT_SCHEMA,
            "provenance": "authored",
            "slug": slug,
            "removed_at": manifests.utc_now(),
            "tombstone": tombstone,
            "supersedes": supersedes,
        }
        manifests.atomic_write_json(
            world_dir / "elements" / f"{slug}.removed.json", receipt)
        (world_dir / "elements" / f"{slug}.placed.json").unlink(missing_ok=True)

    await audit_operator_event(
        request=request,
        title="World asset removed",
        description=f"{job_id}/{slug}: tombstoned to {tombstone or '(no file)'}",
        variant="warning",
        action="splat.world_place_remove",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "slug": slug},
    )
    return {"ok": True, "slug": slug, "tombstone": tombstone,
            "receipt": receipt}
