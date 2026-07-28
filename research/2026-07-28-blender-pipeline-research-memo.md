# SplatLab Blender-Pipeline Research Memo — 2026-07-28

**Scope:** Ranked adoption shortlist synthesized from 5 candidate dossiers + independent skeptic verification passes (all claims checked against the authoritative local stack facts, with structural receipts where the skeptic could obtain them).
**Standing constraint:** SplatLab is a paused hobby project. This memo authorizes nothing — spikes run only on RToony's explicit request. No training runs this cycle; inference-scale spikes only; no new mandatory backend dependencies; every candidate's license is recorded in the research ledger.

---

## Verdict table

| # | Candidate | Verdict | License | Code | Weights | One-line reason |
|---|-----------|---------|---------|------|---------|-----------------|
| 1 | **3DGS Render by KIRI Engine v5.0.0** | **SPIKE NOW** | Apache-2.0 (verified) | yes (GitHub, v5.0.0 release) | n/a (pure GN add-on) | Zero blockers; Blender 5.1.2 in-window; local PLYs are its native input; measurable via mesh_gate |
| 2 | **BlendSplat v0.6.0-core** | **SPIKE NOW** | CC-BY-4.0 (verified from clone) | yes (Codeberg) | n/a | Feasible, but 5.1-only assets refute the in-window studio-loop claim (pipeline pinned to 4.5.11); SH ≤ 2 vs our degree-3 splats |
| 3 | **UMI3D (SFC-Attn on TRELLIS.2)** | design-doc | none declared (backbone TRELLIS.2-4B is MIT) | no (placeholder repo) | n/a | Best conceptual fit for the SAM-3D geometry failure; blocked purely on code+license release |
| 4 | **GeoStereo (ACM MM 2026)** | reject | none (repo license=null; vendored NVIDIA NC) | yes | unverifiable (HF 401) | Four independent fatal blockers incl. sm_120 incompatibility and rectified-stereo input we cannot feed |
| 5 | **GenSplatCodec (arXiv 2607.24403)** | reject | unknown (paper only) | no | no | Structurally cannot serve the Spark export lane even if released (decoder = per-view diffusion U-Net) |

**Spike-now bar applied:** feasible_now=true from the skeptic AND code + weights + license confirmed. Only the two Blender add-ons clear it.

---

## 1. 3DGS Render by KIRI Engine v5.0.0 — SPIKE NOW (rank 1)

- **Repo:** https://github.com/Kiri-Innovation/3dgs-render-blender-addon — v5.0.0 release (June 5, 7 assets), `bl_info` version (5,0,0). Pure geometry-nodes add-on: no weights, no training, trivial VRAM.
- **License:** Apache-2.0, verified from the repo LICENSE — most permissive in the sweep.
- **Version window:** `bl_info` minimum (5,1,0); repo notes Blender **5.2 is UNSUPPORTED**. Local `/home/rtoony/Apps/blender-5.1.2` is exactly in-window (active `~/.config/blender/5.1` profile; registered splatlab-blender MCP on :9877). **Do not upgrade Blender for this**, and do not touch the 4.5.11 pipeline pin (`SPLATLAB_BLENDER_BIN`).
- **Data:** native input format already on disk — 14+ jobs with canonical 3DGS PLY at `/home/rtoony/projects/splatcli/outputs/3d/splat_*/_preview/splat.ply` (18M–441M). The torch-2.10 ns-export breakage is irrelevant: SplatLab's own export path (`backend/splat_route.py` `_preview_file_path`) produces the PLY.
- **Pipeline hook:** Edit mode exposes the splat as a real Blender mesh object with the full modifier stack → splat cleanup/cropping BEFORE TSDF/solidify + xatlas/k-NN bake. Render mode gives real-time/offline splat rendering. Blender-side only, opt-in; zero backend impact.

### 90-minute spike outline (bonsai job `splat_aea04ab3`)

| Time | Step |
|------|------|
| 0–10 | Register the GPU session with the VRAM arbiter (declare ~8 GB honestly — fleet history of unregistered sessions holding VRAM). Download the v5.0.0 release zip; install into Blender 5.1.2 via the :9877 MCP. Leave the 4.5.11-pinned pipeline untouched. |
| 10–25 | Burn down the two known risks FIRST: (a) headless/scripted driving of the N-Panel operators (bpy context overrides if needed); (b) does v5.0.0 re-export an **edited** splat back to PLY? If (b) fails, pivot the measurement now (fallback below). |
| 25–55 | Import `/home/rtoony/projects/splatcli/outputs/3d/splat_aea04ab3/_preview/splat.ply` (bonsai = the mesh_gate reference scene). Edit mode: bounded cleanup — crop floaters/background clutter outside object bounds. Export cleaned PLY. |
| 55–80 | In the `dn-splatter-probe` conda env: existing TSDF/solidify + xatlas unwrap + k-NN bake (`object_texture.py`) on the cleaned PLY; score with mesh_gate PSNR/SSIM vs held-out renders. |
| 80–90 | Decision gate + teardown: close the Blender session, release the arbiter claim, record Apache-2.0 in the ledger. |

**Decision gate (exact):** ADOPT if cleaned-input mesh_gate beats the bonsai baseline **17.64 dB PSNR / 0.234 SSIM on both metrics**, with world_gate prop checks intact (`has_uv`, `has_base_color_texture`, `largest_component_frac >= 0.80`). PARK at parity. REJECT on regression.
**Fallback metric** (if edited-splat re-export is unsupported): render-mode PSNR/SSIM from held-out dataparser cameras vs real photos, compared against Spark-viewer renders of the same views. Hydrant job `splat_513e89171d` is the backup scene if bonsai import misbehaves.

---

## 2. BlendSplat v0.6.0-core — SPIKE NOW, standalone only (rank 2)

- **Repo:** https://codeberg.org/soerensc/BlendSplat-Library. **License CC-BY-4.0** verified from a git clone's LICENSE file — the clone is the receipt because Codeberg's web views serve garbage to AI scrapers. Release `blendsplat_v0.6.0-core.zip` downloaded (765,578 bytes; 6 `.blend` GN assets; no weights).
- **Refuted dossier claim:** it does **not** compose with the :9877 studio loop / `export_apply` GN workflow in-window. The pipeline is pinned to Blender **4.5.11** (`blender_workflow.py:22-25`, `parametric_build.py:93-96`, `gate_p6f_assembly.sh:26`) and BlendSplat's assets are Blender **5.1-format** (v0501/BLENDER17 header) that 4.5.11 cannot open. Any spike runs standalone in 5.1.2. Also: the dossier's suggested `list_blender_versions` check is wrong — that tool lists per-job scene versions, not the app version.
- **Known handicaps:** SH degree ≤ 2 vs our canonical degree-3 PLYs (45 `f_rest` coeffs, verified) → PSNR will underreport view-dependent quality vs Spark. Camera glue from dataparser poses into a 5.1.2 camera is new code (main schedule risk; `mesh_gate.py` convention math is reusable). Dithered alpha is noisy at low samples; Cycles mesh shadows don't affect splats.
- **Spike shape (secondary):** standalone 5.1.2 + arbiter-registered session → install as opt-in asset library → import bonsai `splat.ply` via documented `splat.import` / `splat.attr_convert` nodes → camera glue (budget 30 min) → EEVEE render held-out views → PSNR/SSIM vs real photos in `dn-splatter-probe`.
- **Decision gate:** viable for studio-loop *rendering* only if PSNR is within ~1 dB of the Spark viewer on the same views (attribute the SH-degree-2 gap before judging). Regardless of score, **in-window adoption stays blocked until the pipeline moves off 4.5.11** — that migration is a separate decision, not part of this spike.

---

## 3. UMI3D — DESIGN DOC + watch (rank 3)

- **Why it matters:** highest conceptual fit in the sweep. It targets exactly the observed failure — distorted geometry from single-image conditioning in the SAM-3D "generated" prop lane — using an asset we already have (2–4 unconstrained, pose-free capture photos per prop from the SfM image set). Training-free (passes the no-training rule); ~24 GB VRAM (paper ran a single 4090) fits the 5090 under the arbiter.
- **Why not now:** https://github.com/quzefan/UMI3D verified 2026-07-28 as a placeholder (README + teaser; "code is still being organized. Stay tuned!" 2026-07-23). No LICENSE, no weights. Reimplementing SFC-Attn (per-layer TRELLIS.2 DiT cross-attention hooks, VRS routing, 3D k-NN smoothing, split texture/geometry schedules) is multi-day research work. Standing up bare TRELLIS.2-4B as a fallback measures the backbone, not the candidate — and is itself >90 min on this stack (4B download, CUDA 12.4 compile-time extensions, no stated sm_120 support, torch 2.10 breakage precedent).
- **Design-doc contents (when written):** TRELLIS.2-4B (MIT, HF `microsoft/TRELLIS.2-4B`) in an isolated conda env (dn-splatter-probe pattern; nvdiffrast/spconv-class deps stay banned from the backend) as an alternative prop generator, patched with SFC-Attn on code release. Side benefit to capture: TRELLIS.2 emits **PBR-textured meshes** that would natively pass world_gate's `has_uv` / `has_base_color_texture` checks the vertex-colored SAM-3D output fails — potentially bypassing the xatlas + k-NN bake for this lane. Caveats to carry: appearance-centric evaluation only (no Chamfer/F-score — the geometry-fix claim is qualitative), unstable routing on strongly symmetric objects, and porting SFC-Attn onto SAM-3D itself is undemonstrated (research-grade, not planned).
- **Re-check:** repo in ~2–3 weeks for code + license. Effort drops high → medium on release.

## 4. GeoStereo — REJECT (rank 4)

Four independent blockers, each individually fatal to a 90-minute measured spike, all CONFIRMED:

1. **No license** — GitHub API `license=null` (all rights reserved by default); fails the ledger's local-use gate. It also vendors FoundationStereo under the NVIDIA Source Code License-NC (research-only), which any use inherits.
2. **Cannot run on the 5090** — pins torch 2.4.1 + cu121 + flash-attn 2.8.3 + xformers 0.0.28.post1; Blackwell sm_120 needs CUDA ≥ 12.8 / torch ≥ 2.7, and cu121 cannot compile flash-attn for sm_120. A half-day env-port, not a spike.
3. **Weights unverifiable** — README's HF link 401s anonymously, and the check was calibrated (known-public stable-diffusion-2-1 also 401s from this environment), so downloadability is unproven either way; uncertainty counts against.
4. **Data we don't have** — it requires rectified left-right stereo pairs; SplatLab captures are monocular colmap/glomap orbits. A pseudo-stereo synthesis + de-rectification lane is real engineering, and convergent orbit baselines are off its Hypersim/IRS/KITTI stereo-rig training distribution. Even then, output normals live in the rectified-left frame and need re-rotation before dn-splatter supervision.

**The actionable takeaway:** the paper's entire delta over StableNormal is stereo conditioning we cannot naturally feed. If normal priors for the mesh-extraction lane are wanted this cycle, trial a **monocular** estimator (StableNormal / DSINE / Metric3D) — direct fit, cheaper, separate future candidate. Revisit GeoStereo only if (a) a permissive LICENSE lands, (b) weights are confirmed public, and (c) a half-day env-port + stereo-synthesis experiment is explicitly budgeted.

## 5. GenSplatCodec — REJECT, watch-only (rank 5)

- **Reality check:** paper is real (arXiv 2607.24403v1, Hu et al., submitted 2026-07-27 — 1 day old) but has no code, no weights, no license, and no third-party reimplementation. The three-stage training required to reproduce it is banned this cycle anyway.
- **Structural dead end for the stated target:** the bitstream is not a self-contained renderable splat — final quality only exists after a client-side VAE + one-step diffusion U-Net pass per view, which three.js/Spark can never run. It can never emit SPZ v4 / SOG / KHR_gaussian_splatting GLB. It also compresses its **own** feed-forward-predicted Gaussians from sparse 224–518 px views (DL3DV/RealEstate10K, ~20–23 dB streaming regime) with no ingest path for our dense-capture per-scene-optimized splatfacto/langfield splats — "measured result on local data" is undefined.
- **Salvage value (ideas, not adoption):** contribution-based RD pruning before SOG export, and G-PCC-centers + entropy-coded-attributes decoupling. A pruning spike on our own splats would be a **separate home-grown candidate** for a future cycle.
- **Watch:** the MediaX-SJTU group released code for 4DGC (CVPR'25) and 4DGCPro (NeurIPS'25); re-check their GitHub org in 1–2 months.

---

## Research ledger entries

| Candidate | License recorded | Status | Re-check |
|-----------|------------------|--------|----------|
| KIRI 3DGS Render v5.0.0 | Apache-2.0 (verified) | spike-ready; awaiting explicit go | — |
| BlendSplat v0.6.0-core | CC-BY-4.0 (verified, attribution-only) | spike-ready standalone; in-window blocked by 4.5.11 pin | on any Blender-pin migration |
| UMI3D | none declared (blocker); backbone MIT | design-doc + watch for code drop | ~2–3 weeks |
| GeoStereo | none (null) + NVIDIA NC vendored | rejected; conditions-to-revisit recorded | on LICENSE/weights change |
| GenSplatCodec | unknown (paper only) | rejected / watch-only | MediaX-SJTU org, 1–2 months |

## Cross-cutting notes

- **Blender version split is now a real fork:** both spike-now candidates require 5.1.x while the production studio loop is pinned to 4.5.11. Spikes stay standalone in `/home/rtoony/Apps/blender-5.1.2`; migrating the pipeline pin is a separate, deliberate decision with its own regression pass (`gate_p6f_assembly.sh` etc.), not a side effect of a spike.
- **GPU discipline:** every Blender 5.1.2 spike session registers with the VRAM arbiter with an honest declaration (fleet history: undeclared/underdeclared sessions caused real contention on 07-26).
- **Future home-grown candidates surfaced by this sweep:** (1) contribution-based splat pruning before SOG export (GenSplatCodec idea); (2) monocular normal-prior trial for the dn-splatter mesh lane (GeoStereo's cheaper substitute); (3) TRELLIS.2-4B as a standalone prop-lane backbone even before UMI3D code lands.

