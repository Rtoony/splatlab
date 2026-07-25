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


if __name__ == "__main__":
    main()
