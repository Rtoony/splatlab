# SplatLab Unreal Engine 5.6 Handoff

This directory is a Windows UE 5.6 staging workflow for SplatLab bundles. It
does not install plugins, start Unreal, modify the live SplatLab service, or run
GPU work. Raw handoffs remain outside `Content/` until an operator imports them.

## Renderer Decision

| Order | Renderer | Use | Important limits |
| --- | --- | --- | --- |
| 1 | NanoGS | Large static scenes and controllable VRAM budgets | PLY import; transparent splats can ghost with TSR; tile large captures |
| 2 | MLSLabsRenderer Lite | Sequencer, PLY sequences, and Shipping builds | Lite is beta; SOG support is advertised for the separate Pro edition |
| 3 | UnrealSplat | Compatibility experiments | Targets UE 5.5, practical scale is lower, transforms and SH support are incomplete |

The machine-readable record is `renderers.json`. `probe-renderers.ps1` reports
installed descriptors, versions, and hashes, then selects the first available
renderer in that order. It never downloads one.

## Windows Setup

Requirements are Unreal Engine 5.6, Visual Studio 2022 with the Unreal C++ tools,
and Python 3 available as `py -3` or `python`.

1. Copy `SplatLabUE56` to the Windows workstation.
2. Download a reviewed renderer release from its official repository.
3. Place its plugin directory under `SplatLabUE56/Plugins/`.
4. Generate a zipped Unreal handoff from SplatLab.
5. Verify and stage it before opening Unreal.

```powershell
.\probe-renderers.ps1 `
  -EngineRoot "C:\Program Files\Epic Games\UE_5.6"

.\verify-bundle.ps1 `
  -BundlePath "D:\Handoffs\capture-unreal-5.6.zip"

.\stage-bundle.ps1 `
  -BundlePath "D:\Handoffs\capture-unreal-5.6.zip" `
  -EngineRoot "C:\Program Files\Epic Games\UE_5.6"
```

Staging writes an immutable directory at
`SplatLabUE56/SplatLabImports/JOB_ID/MANIFEST_SHA_PREFIX`. A `current.json`
pointer identifies the selected version. Receipts live under
`Saved/SplatLab/Receipts`; these paths are intentionally ignored by Git.

Open `SplatLabUE56.uproject` after staging. The project has a minimal C++ target
so source plugins can compile. No Unreal binaries or generated content are
committed.

## Import Contract

Use the canonical `Gaussian/scene.ply` for NanoGS, MLSLabsRenderer, or
UnrealSplat. SPZ, SOG, streamed SOG, and `KHR_gaussian_splatting` GLB remain
portable interchange copies; do not assume a renderer can import all of them.

Create one parent actor named `SplatLabSceneRoot` and keep these children
separate:

- `GaussianRender` contains only visual splat actors and has collision disabled.
- `Collision` contains the generated collision mesh and is hidden in game.
- `ConventionalGeometry` contains optional survey, mesh, or GLB assets.

SplatLab sources are right-handed with `+Z` up and `+Y` forward. UE uses `+Z` up
and `+X` forward. Importer conventions differ, so verify an axis witness before
applying a renderer-specific correction. Put scale, axis correction, and site
alignment on `SplatLabSceneRoot`, not on individual children. If the manifest
marks scale as uncalibrated, do not infer real-world size.

For NanoGS, begin with frustum culling enabled and a conservative visible-splat
budget, then measure before increasing it. Spatially tile captures that remain
too large; this reduces both sorting work and residency. Test FXAA if nearly
transparent splats ghost under TSR.

## Unreal MCP Boundary

`mcp-policy.json` records the intended closed-world subset for the MIT-licensed
`ChiR24/Unreal_mcp` bridge: loopback only, typed inspect/import/transform/
screenshot operations, approved `/Game/SplatLab/Jobs` paths, and receipts for
mutations. Validate it with:

```powershell
.\validate-mcp-policy.ps1
```

The upstream server also exposes broad tools, including Python and system
control. The policy is therefore a contract, not an upstream enforcement
mechanism. Do not give an autonomous client direct access until a facade
enforces this allowlist. For supervised experiments, keep native Streamable HTTP
on `127.0.0.1`, leave non-loopback access disabled, and review every mutating
call.
