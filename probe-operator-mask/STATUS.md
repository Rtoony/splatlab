# Operator-masking spike — fog root-cause A/B (2026-07-11)

**Goal**: flip the pool test-flight scene (`splat_7f3d29f3de`, verdict FOG 5/6 cams,
median shell 0.9972) to HEALTHY by masking corruption sources out of SfM + training.
Scored by the calibrated fog gate (`backend/health/fog_gate.py`) — no eyeballing.

## Evidence gathered (probe phase)
- Crops are SHARP, well-exposed, texture-rich; glomap registered 717/720. Frames are
  not the problem — reconstruction is.
- `probe_person_masks.py`: operator/persons cover only 0–4% of individual crops, BUT
  466/720 crops (65%) contain person pixels, and the stitched equirect
  (scratch equirect_probe.jpg) shows the operator smeared across the whole nadir
  ABOVE the crop_bottom=0.15 cut (head/hardhat/vest), plus a walking surveyor.
- The equirect shows BOTH lens seams as hard vertical discontinuities at x=0.25W /
  0.75W (bleachers visibly broken at the right seam) — v360 dfisheye parallax
  ghosting, fixed yaw, every frame.

## Hypothesis ledger
- **H1 seam ghosting** poisons multi-view consistency → arm S (seam-band masks ±3°,
  mean 1.5% masked, crops 1/3/6/7 carry the bands).
- **H2 moving people** (operator nadir smear + surveyor) → arm SP (seam ∪ person,
  mean 4.5%, max 17.8%). SP−S isolates the person effect; S−baseline isolates seam.
- **H3 crop intrinsics / stitch quality beyond masking** → only if S and SP both fail
  (next probes: wider seam band ±6°, Insta360 Studio optical-flow stitch (Entry B),
  colmap4 panorama_sfm rig lane).

## Method
`run_arm.sh <name> <masks>` replicates the pipeline glomap path standalone in
`arm_<name>/` (never touches the original job): masked colmap4 feature_extractor
(--ImageReader.mask_path) → sequential_matcher (overlap 16) → global_mapper →
ns-process-data --skip-colmap → inject per-frame mask_path into transforms.json
(splatfacto zeroes masked pixels from the loss, nerfstudio_dataparser.py:171) →
ns-train 7000 draft → fog gate. Mask semantics both tools: black=ignore.
Masks composed under both naming schemes by `compose_masks.py`
(crop-named for COLMAP, sequential for nerfstudio; lexical-sort mapping).

## Results
| arm | masks | verdict | median shell | median spread | registered |
|---|---|---|---|---|---|
| baseline | none | **FOG** 5/6 | 0.9972 | 1.317 | 717/720 |
| S | seam ±3° | **FOG** | 1.0 | 1.016 | 717/720 |
| SP | seam ∪ person | **FOG** | 1.0 | 1.015 | 717/720 |
| R | RIG-constrained SfM (panorama_sfm) | **UNCERTAIN** ← huge move | 0.2268 | 15.94 | 1080/1080 |
| RP | rig poses + person masks (retrain only) | **UNCERTAIN** | 0.2433 | 24.12 | (poses reused) |

**Arm RP note**: first gate pass scored the WRONG checkpoint (cp -al copied arm R's
trained splatfacto dir into arm_RP/processed; `find | head -1` grabbed it — identical
4-decimal medians was the tell). Rescored vs the fresh 100656 ckpt. Person masks helped
spread (15.9→24.1) not shell: masked regions are UNSUPERVISED, so junk gaussians still
form there (masking prevents fitting the person, not floaters in that space), and
featureless-sky views legitimately read as near-shell. Gate needs a 360-aware exemption
before it can grade rig scenes fairly — a supervised-pixels-only or sky-aware variant.

**Arm R (10:04)**: cocoon broken — shell 0.997→0.227, spread 1.0→15.9, and cam 777's
receipt is the FIRST recognizable insv reconstruction in the program (pool cover, deck,
building). Rig SfM ~10 min CPU wheel (binary path will be faster). Residual shell:
down views (operator = real near geometry) + up views (SKY: no parallax ⇒ legit
near-shell — **fog-gate blind spot on 360 rigs**, needs a sky-aware exemption before
the gate can grade rig scenes fairly). Arm RP adds person masks on R's poses.

**H1+H2 refuted** (arms S, SP): masking moved nothing. **H3 pinned by geometry probes:**
- camera-path bbox diag **1584** vs point-cloud diag **129** — trajectory 12× the scene
- same-frame 8-crop camera centers (truth: identical) solved **median 5.1 units apart**
  (p95 10.7) vs true frame-to-frame step **0.13** → the unrigged crop lane scatters poses;
  nerfstudio unit-box normalization then shrinks real local geometry to depth ~0.001-0.01
  = the fog fingerprint. THE COCOON IS REAL SCENERY AT COLLAPSED SCALE.
- Pattern check: all 3 insv scenes FOG, all 4 pinhole scenes HEALTHY — the insv crop
  lane has plausibly never produced a healthy reconstruction.

**Arm R** = colmap4 panorama_sfm.py (rig config: zero translation between per-pano
virtual cams, rig-verified matching, BA holds rig fixed; pycolmap 4.1.0 pip-installed
into colmap4 env, CPU wheel). If R flips HEALTHY → wire a rig-SfM lane into splat_route.

## Decision rule
- S flips HEALTHY → wire seam-band masking as a pipeline step (cheap, deterministic).
- SP flips but S doesn't → person masking is load-bearing → wire Mask R-CNN mask
  stage (29s/720 crops on the 5090).
- Neither flips → H3: escalate to stitch quality/intrinsics probes before building
  any masking feature.
