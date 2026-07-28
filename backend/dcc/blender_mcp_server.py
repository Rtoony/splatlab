"""Loopback-only MCP server exposing SplatLab's typed Blender workflow."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from mcp.server import MCPServer

try:
    from mcp_types import ToolAnnotations
except ImportError:  # Compatibility with SDK packages that re-export protocol types.
    from mcp.types import ToolAnnotations

from dcc import blender_workflow as workflow
from dcc import mcp_auth


mcp = MCPServer("SplatLab Blender")
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
ADDITIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)


@mcp.tool(title="Inspect SplatLab job", annotations=READ_ONLY)
def inspect_job(job_id: str) -> dict[str, Any]:
    """Read provenance, units, dimensions, export state, and Blender versions."""
    return workflow.inspect_job(job_id)


@mcp.tool(title="Inspect Blender scene", annotations=READ_ONLY)
def inspect_blend(job_id: str, version: int | None = None) -> dict[str, Any]:
    """Inspect a .blend using headless Blender without saving any changes."""
    return workflow.run_action(job_id, "inspect", base_version=version)


@mcp.tool(title="List Blender versions", annotations=READ_ONLY)
def list_blender_versions(job_id: str) -> list[dict[str, Any]]:
    """List immutable scene versions and their operation receipts."""
    return workflow.list_versions(job_id)


@mcp.tool(title="Snapshot Blender scene", annotations=ADDITIVE)
def snapshot_blend(
    job_id: str, base_version: int | None = None, note: str = ""
) -> dict[str, Any]:
    """Save the current source as a new immutable version."""
    return workflow.run_action(job_id, "snapshot", base_version=base_version, note=note)


@mcp.tool(title="Toggle Blender collection", annotations=ADDITIVE)
def toggle_collection(
    job_id: str,
    collection: str,
    visible: bool,
    base_version: int | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Set one named collection's visibility and save a new version."""
    return workflow.run_action(
        job_id,
        "toggle_collection",
        {"collection": collection, "visible": visible},
        base_version=base_version,
        note=note,
    )


@mcp.tool(title="Transform Blender object", annotations=ADDITIVE)
def transform_object(
    job_id: str,
    object_name: str,
    location: list[float] | None = None,
    rotation_degrees: list[float] | None = None,
    scale: list[float] | None = None,
    base_version: int | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Apply an absolute typed transform to one object and save a new version."""
    return workflow.run_action(
        job_id,
        "transform_object",
        {
            "object": object_name,
            "location": location,
            "rotation_degrees": rotation_degrees,
            "scale": scale,
        },
        base_version=base_version,
        note=note,
    )


@mcp.tool(title="Import world element for polish", annotations=ADDITIVE)
def import_world_element(
    job_id: str, slug: str, base_version: int | None = None, note: str = ""
) -> dict[str, Any]:
    """Import a solidified world element GLB (or 'shell') into the workflow
    scene as polish_<slug> inside a Polish collection, saving a new version.

    The GLB path is resolved and containment-checked host-side; Blender never
    receives a free-form path."""
    return workflow.run_action(
        job_id,
        "import_world_element",
        {"slug": slug},
        base_version=base_version,
        note=note,
    )


@mcp.tool(title="Clean up mesh object", annotations=ADDITIVE)
def cleanup_mesh(
    job_id: str,
    object_name: str,
    merge_distance: float | None = None,
    decimate_ratio: float | None = None,
    min_component_frac: float | None = None,
    shade_smooth: bool | None = None,
    base_version: int | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Apply typed mesh cleanup to one object and save a new version.

    merge_distance welds near-coincident vertices (scene units, max 0.1);
    decimate_ratio reduces faces (0-1]; min_component_frac drops floating
    debris islands below that fraction of total faces (largest island is
    always kept); shade_smooth sets smooth shading. The receipt records
    faces/verts before and after plus components removed."""
    return workflow.run_action(
        job_id,
        "cleanup_mesh",
        {
            "object": object_name,
            "merge_distance": merge_distance,
            "decimate_ratio": decimate_ratio,
            "min_component_frac": min_component_frac,
            "shade_smooth": shade_smooth,
        },
        base_version=base_version,
        note=note,
    )


@mcp.tool(title="Export Blender scene as GLB", annotations=ADDITIVE)
def export_blend_glb(
    job_id: str,
    version: int | None = None,
    object_name: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Export a blend version as a validated GLB under _blender/exports/.

    Versions are untouched; the artifact is readback-validated (glb_check)
    before it lands. With object_name set, only that object is exported and
    the filename gains a slug suffix so whole-scene exports are preserved.
    Feeding it back into the app stays a separate, audited step via the
    polish-upload route."""
    return workflow.export_glb(
        job_id, base_version=version, object_name=object_name, note=note
    )


@mcp.tool(title="Restore Blender version", annotations=ADDITIVE)
def restore_blender_version(
    job_id: str, version: int, note: str = ""
) -> dict[str, Any]:
    """Restore by copying an old version forward; existing versions are untouched."""
    return workflow.restore_version(job_id, version, note)


@mcp.tool(title="Open Blender scene", annotations=ADDITIVE)
def open_blender(job_id: str, version: int | None = None) -> dict[str, Any]:
    """Open an approved SplatLab .blend in the attended Blender GUI."""
    return workflow.open_blend(job_id, version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9877)
    parser.add_argument(
        "--transport", choices=("streamable-http", "stdio"), default="streamable-http"
    )
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    # A misconfigured token is fatal here rather than a silent downgrade to no
    # auth -- the operator who set it must not be left believing it took effect.
    try:
        token = mcp_auth.configured_token()
    except mcp_auth.MCPAuthConfigError as exc:
        parser.error(str(exc))

    if token is None:
        print(
            f"[blender-mcp] listening on 127.0.0.1:{args.port}/mcp with loopback "
            f"trust only; set {mcp_auth.TOKEN_ENV} to require a bearer token",
            flush=True,
        )
        try:
            asyncio.run(
                mcp.run_streamable_http_async(
                    host="127.0.0.1",
                    port=args.port,
                    streamable_http_path="/mcp",
                    json_response=True,
                    stateless_http=True,
                )
            )
        except KeyboardInterrupt:
            pass
        return

    # Same construction run_streamable_http_async performs, with the auth gate
    # wrapped around the ASGI app before it is served.
    import uvicorn

    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        host="127.0.0.1",
    )
    print(
        f"[blender-mcp] listening on 127.0.0.1:{args.port}/mcp; bearer token required",
        flush=True,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            mcp_auth.bearer_token_middleware(app, token),
            host="127.0.0.1",
            port=args.port,
            log_level=mcp.settings.log_level.lower(),
        )
    )
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
