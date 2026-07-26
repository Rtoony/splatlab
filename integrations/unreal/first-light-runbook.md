# UE First Light — Hydrant on Triforce (click-by-click)

Everything below is already staged on Triforce at `G:\splatlab-ue\`:
the `SplatLabUE56` project, the NanoGS v1.0.3 plugin (prebuilt Win64
binaries) under `SplatLabUE56\Plugins\NanoGS`, and the verified hydrant
bundle staged immutably at
`G:\splatlab-ue\SplatLabUE56\SplatLabImports\splat_513e89171d\50a210ca9afe0019`
(14 files, sha256-verified on the machine; receipt printed 2026-07-26).

## 0. One-time prerequisites (from the AI PC)

```
! bash ~/scripts/triforce-ue-station-prep.sh --apply
```

Dry-run first if you want to re-read it. This enables RDP (NLA stays on) and
adds the missing Windows 11 SDK to the existing VS Build Tools — without the
SDK the project's C++ module cannot build. Takes several minutes; prints
receipts + rollback.

## 1. Connect

Any RDP client → `100.84.9.80` (Tailscale) or the Triforce LAN IP, user
`josh`. Expect a plain desktop — the box has been headless since June.

## 2. Open the project (first open compiles the stub module)

1. Double-click `G:\splatlab-ue\SplatLabUE56\SplatLabUE56.uproject`.
2. UE will say the `SplatLabUE56` module is missing and ask **"Would you like
   to rebuild them now?"** → **Yes**. The stub module is tiny but the 4-core
   i7-7700 still takes a few minutes the first time. (If this fails, the SDK
   step above did not complete — tell the agent, it can also run the build
   remotely: `Build.bat SplatLabUE56Editor Win64 Development
   -project="G:\splatlab-ue\SplatLabUE56\SplatLabUE56.uproject"`.)
3. When the editor opens: **Edit → Plugins**, search "Nano". Confirm **Nano
   Gaussian Splatting** is listed and **Enabled** (tick it + restart the
   editor if not).

## 3. Import the hydrant splat

1. Open the staged bundle folder in Explorer:
   `G:\splatlab-ue\SplatLabUE56\SplatLabImports\splat_513e89171d\50a210ca9afe0019`
2. Drag **`Gaussian\scene.ply`** (909,532 gaussians) into the editor's
   Content Browser. NanoGS registers the PLY importer.
   *If the drag does nothing*, use the NanoGS documentation at
   github.com/TimChen1383/NanoGaussianSplatting — the import surface is
   theirs, not ours; do not fight it blind.
3. Drop the imported asset into the level.

## 4. Scene hierarchy (the bundle contract)

1. Place an **Empty Actor** at the origin, name it `SplatLabSceneRoot`.
2. Parent the splat actor under it, renamed/grouped as `GaussianRender`
   (collision disabled on it).
3. Optional but worthwhile: drag
   `Objects\fire-hydrant\winner-textured.glb` (the bake-off winner,
   13.76 dB) in as ordinary static-mesh geometry under a
   `ConventionalGeometry` child — splat vs mesh side by side.
4. Leave `Collision` / `WorldGeometry` children empty — this capture has no
   world lane.

## 5. Axis witness + scale — BEFORE trusting anything

- The hydrant must stand **upright** with the blue port caps on the sides.
  If it lies on its side or is mirrored, note exactly which way — that is
  the importer's axis convention showing, and the correction belongs on
  `SplatLabSceneRoot`, never on children.
- Scale: the bundle is **calibrated at 192.26 cm per source unit**
  (`manifest.json → import_contract.centimeters_per_source_unit`). If the
  importer did not apply it, set uniform scale on `SplatLabSceneRoot` until
  the hydrant reads ~85 cm tall against a default 180 cm mannequin.
- Compare against `Receipts\thumb.webp` in the same staged folder.

## 6. NanoGS settings for the 2080 SUPER (8 GB)

Start with frustum culling **on** and a conservative visible-splat budget,
then raise it while watching VRAM (README: measure before increasing). If
near-transparent splats ghost under TSR, try FXAA.

## 7. Receipt

High-res screenshot (F9) or Win+Shift+S → save to
`G:\splatlab-ue\receipts\` (create it) — the agent will pull it back and
log the first light. Note the axis/scale findings from step 5 in one line;
they get folded into the bundle README for every future import.
