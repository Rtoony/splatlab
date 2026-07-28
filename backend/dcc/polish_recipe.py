"""One-call polish recipe: snapshot -> import element -> cleanup -> export.

The G1 return-leg run proved the loop as four hand-driven steps; this makes
it a typed, repeatable product feature. A recipe composes the EXISTING
blender_workflow actions — no new Blender surface, every step keeps its own
version + receipt — and returns one composite receipt naming them all.

Ingestion is deliberately NOT part of a recipe. The polish-upload route is
the one audited door back into the app (polish_route.py's module head), so
the recipe ends at a validated export plus the exact upload target; the CLI
(tools/polish-element.py) drives that last leg with the operator's own
credential."""

from __future__ import annotations

from typing import Any

from dcc import blender_workflow as workflow


def run_recipe(
    job_id: str,
    slug: str,
    *,
    merge_distance: float | None = 1e-4,
    decimate_ratio: float | None = None,
    min_component_frac: float | None = 0.02,
    shade_smooth: bool | None = True,
    note: str = "",
) -> dict[str, Any]:
    """Run the polish legs for one world element and return a composite receipt.

    Steps that mutate the scene each create an immutable version exactly as if
    driven by hand; a failure at any step raises BlenderWorkflowError with the
    completed step receipts preserved on the exception as `partial_receipts`.
    """
    tag = note or f"polish recipe: {slug}"
    steps: list[dict[str, Any]] = []

    def _step(name: str, receipt: dict[str, Any]) -> dict[str, Any]:
        steps.append({"step": name, "receipt": receipt})
        return receipt

    try:
        _step(
            "snapshot",
            workflow.run_action(job_id, "snapshot", note=f"{tag} (checkpoint)"),
        )
        imported = _step(
            "import_world_element",
            workflow.run_action(
                job_id, "import_world_element", {"slug": slug}, note=tag
            ),
        )
        object_name = (imported.get("result") or {}).get("object")
        if not object_name:
            raise workflow.BlenderWorkflowError(
                "import step did not report an imported object"
            )
        cleaned = _step(
            "cleanup_mesh",
            workflow.run_action(
                job_id,
                "cleanup_mesh",
                {
                    "object": object_name,
                    "merge_distance": merge_distance,
                    "decimate_ratio": decimate_ratio,
                    "min_component_frac": min_component_frac,
                    "shade_smooth": shade_smooth,
                },
                note=tag,
            ),
        )
        exported = _step(
            "export_glb",
            workflow.export_glb(
                job_id,
                base_version=cleaned["version"],
                object_name=object_name,
                note=tag,
            ),
        )
    except workflow.BlenderWorkflowError as exc:
        exc.partial_receipts = steps  # type: ignore[attr-defined]
        raise

    return {
        "schema": "dev.splatlab.polish-recipe-receipt/v1",
        "job_id": job_id,
        "slug": slug,
        "object": object_name,
        "params": {
            "merge_distance": merge_distance,
            "decimate_ratio": decimate_ratio,
            "min_component_frac": min_component_frac,
            "shade_smooth": shade_smooth,
        },
        "cleanup": cleaned.get("result"),
        "export": {
            "path": exported["output"]["path"],
            "sha256": exported["output"]["sha256"],
            "gltf": exported["gltf"],
        },
        "steps": steps,
        "upload": {
            "route": (
                f"/api/splat/jobs/{job_id}/world/elements/{slug}/polish"
            ),
            "note": "POST the exported GLB here (audited ingest door); "
            "tools/polish-element.py --upload drives it",
        },
    }
