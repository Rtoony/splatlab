# SplatLab Research Intake

This directory turns the source review into a license-aware experiment queue.
It does not install dependencies, clone repositories, start services, or execute
GPU work.

## What Was Incorporated

The geometry-first pipeline described by Niantic is now reflected in SplatLab's
portable handoff: the canonical PLY remains the source of truth; exports carry
checksums, bounds, scale, axes, georeference, and provenance; collision is a
separate derived layer; SPZ v4, SOG, streamed SOG, and glTF are explicit
interchange artifacts.

The most useful immediate net-new evaluations are:

1. gsplat MCMC and memory-efficient rasterization settings, capability-detected
   against the installed Nerfstudio version and never made default without data.
2. DN-Splatter depth/normal supervision for low-texture surfaces and better
   collision meshes.
3. splatreg as a proposal-only SE(3)/Sim(3) alignment step before manual splat
   merging.
4. NanoGS in UE 5.6 for spatial LOD, frustum culling, compaction, and a bounded
   visible-splat budget.

Static inspection confirms the existing backend already has COLMAP 4.1 global
mapping, rig-constrained equirectangular solving, and automatic solver
escalation. It also confirms current Nerfstudio Splatfacto already enables
absolute-gradient densification. Those should be retained and regression
benchmarked rather than reimplemented.

The 3D-GIMP paper suggests a high-value independent editing prototype: perform
one generative inpaint on a selected key view and propagate it through
geometry-aware correspondences rather than repeatedly diffusing every view.
That should reduce model calls and multi-view hallucination drift.

GLAM-SLAM's sparse anchor grid, tracking/mapping separation, and scene
partitioning are useful architectural patterns for future long captures. Its
linked project page currently serves an unrelated template, its repository is
only a placeholder, and no code license is present, so it remains watch-only.

SubSplat is relevant only if SplatLab adopts a feedforward pixel-aligned model.
HyperGS represents video with 2D Gaussians rather than reconstructing a 3D
scene, so it is intentionally deprioritized. AnythingReality's incremental
mapping and semantic points of interest are worth monitoring, but no code was
available during this review.

## License Boundary

`sources.json` is the source and decision ledger. MVSAnywhere, ACE0, and MILo
are restricted research references. Their source, environments, weights, and
results must stay outside this repository and be labeled research-only. MILo
also inherits parts of the original Gaussian Splatting research license.

Apache, BSD, or MIT on a top-level repository does not automatically clear
model weights, datasets, native dependencies, or submodules. Every experiment
receipt identifies a full Git revision and dependency-lock hash.

`adapters.json` contains disabled, plan-only boundaries for six candidates.
There are deliberately no executable command templates. A future operator must
configure an external checkout through the named environment variable and
manually launch GPU experiments through `tools/splatlab-compute-gate.sh`.

## Validate

```bash
python research/benchmark.py validate-catalog
python research/benchmark.py validate-adapters
python research/benchmark.py probe-adapters
python research/capability_probe.py --probe-colmap-cli
```

`probe-adapters` only checks external roots and required files. It never imports
or runs upstream code.

Benchmark receipts follow `benchmark-schema.json`. Validate and compare them
with explicit metric direction and tolerance:

```bash
python research/benchmark.py validate-receipt receipt.json

python research/benchmark.py compare baseline.json candidate.json \
  --metric metrics.quality.psnr_db \
  --direction higher \
  --tolerance-percent 0.5
```

Portable-export storage and checksum reports can be generated without GPU work:

```bash
python research/benchmark.py exports \
  /path/to/JOB/_exports/manifest.json \
  --verify-files \
  --output /path/to/report.json
```

An experiment is not eligible for adoption unless a passing run records quality,
geometry where applicable, runtime, peak RAM/VRAM, Gaussian count, output size,
input identity, implementation revision, and dependency identity. Failed or
incomplete runs may omit unavailable quality metrics but must retain explicit
failure notes. Visual screenshots alone are not an acceptance metric.
