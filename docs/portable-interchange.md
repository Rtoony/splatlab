# Portable Splat Interchange

SplatLab keeps the canonical full-fidelity PLY as the source of truth. Every
derived format is disposable, checksum-addressed, and tied to the source PLY in
`_exports/manifest.json`. Rebuild exports after the canonical PLY changes.

## Format Selection

| Format | Intended use | Notes |
| --- | --- | --- |
| PLY | Lossless handoff, Blender, and Unreal renderer import | Canonical artifact; largest and most interoperable |
| SPZ v4 | Compact archival or transfer | Quantized; retain the PLY for future conversions |
| SOG | Compact WebGPU delivery | Best default for small and medium web scenes |
| Streamed SOG | Progressive large-scene delivery | Skipped below one million Gaussians unless explicitly forced |
| glTF/GLB | Standards-oriented interchange experiments | Uses `KHR_gaussian_splatting`; importer support is still uneven |
| Collision GLB | Navigation, physics, and selection | Derived geometry, never a replacement for the visual splat |

The API accepts a typed export request:

```http
POST /api/splat/jobs/JOB_ID/exports
Content-Type: application/json

{
  "formats": ["spz", "sog", "streamed-sog", "gltf"],
  "spz_version": 4,
  "force_streamed_sog": false,
  "overwrite": false
}
```

Conversions run as admitted host work and publish atomically. A failed rebuild
preserves the previous current artifact. Streamed-SOG chunk paths remain nested
when served and when copied into an Unreal bundle.

## Unreal 5.6 Handoff

Build portable exports first, then request a bundle:

```http
POST /api/splat/jobs/JOB_ID/unreal-bundle
Content-Type: application/json

{
  "include_zip": true,
  "include_canonical_ply": true,
  "include_survey": true,
  "require_current_exports": true
}
```

The bundle records checksums, provenance, coordinate axes, scale calibration,
and a renderer probe order of NanoGS, MLSLabsRenderer, then UnrealSplat. On the
Windows workstation, verify and stage the bundle with the scripts documented in
`integrations/unreal/README.md` before opening Unreal. Keep the visual splat,
collision mesh, and conventional geometry as separate child actors under one
alignment root.

Do not infer survey accuracy from render quality. If
`requires_operator_alignment` is true, scale and site alignment remain manual.

## Blender MCP

The restricted MCP server in `integrations/blender` exposes inspection,
immutable snapshots, collection visibility, absolute transforms, restore-forward,
and attended GUI opening. It binds to loopback, launches Blender with Python
auto-execution disabled, and does not accept arbitrary Python or filesystem
paths.

Use the isolated environment so the MCP SDK cannot change SplatLab's FastAPI
dependencies:

```bash
integrations/blender/run-mcp.sh
integrations/blender/run-mcp.sh --transport stdio
```

Every mutation writes a new `_blender/versions/scene-vNNNN.blend` and receipt.
Existing versions are never overwritten.

## Verification

Validate handoffs and produce storage reports without GPU work:

```bash
python research/benchmark.py exports \
  /path/to/JOB/_exports/manifest.json \
  --verify-files

python integrations/unreal/bundle_tool.py verify-bundle \
  /path/to/JOB-unreal-5.6.zip
```

Treat a checksum mismatch, stale source identity, unknown schema, path-policy
failure, or uncalibrated scale as a hard gate rather than a warning to ignore.
