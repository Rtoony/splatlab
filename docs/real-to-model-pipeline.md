# Real images/video → independent 3D references → reviewed condo details

## Native surfaces checkpoint — September 9, 17:04 PDT

**Actual 2DGS execution now supersedes the staged/blocked checkpoint below.**
Open **http://127.0.0.1:32856/** for the native model-rendered eight-second tour,
live same-camera radiance/mesh switching, both Blender candidates and source-bound
Condo Lab handoffs. The private preview runs until approximately 23:03 PDT;
`data/spatial/condo-surfel-showcase-2026-09-09-07` persists afterward.

Three native runs finish. The source-anchored 6,000-step candidate has 407,131
surfels; reopening its native NPZ reproduces all 24 validation images byte-for-byte.
Both independent Blender scenes reopen with 72 packed photos and source cameras.
Their complete exported triangle sets preserve original arbitrary SfM coordinates
and winding. Corrected portable unlit vertex-color GLBs and schema-validated condo
manifests are separate derivatives, not accepted condo edits.

Surface coverage improves, but more training trades improved appearance for
worse reserved-point mean depth agreement. Keep the anchored 3,000-step geometry
candidate alongside the 6,000-step visual candidate. Neither establishes measured
scale, clean architecture or registration. The native pipeline now works at
experimental reference quality; the full condo-detailing objective is incomplete.

Final real-GPU browser proof, native replay, actual mesh downloads, two Blender
reopens and 142 focused Python tests pass. The temporary GPU override is removed
at 17:04; no new GPU work is authorized by this document. Read
`/home/rtoony/reports/2026-09-09-splatlab-condo-native-surfaces.md` for the complete
run commands, exact metrics, retained failures, resource handoff and next phase.

## Structural checkpoint — September 9, 15:51 PDT

Open **http://127.0.0.1:32853/** for 24 actual source/control/filtered/mask
comparisons and a separate saved/reopened Blender candidate. Local SAM3 and
cross-time sparse-track analysis execute, but no façade plane passes. Semantic
filtering removes clutter while reducing intended static-surface coverage from
89.36% to 79.74%; it is **not an architectural improvement**.

The next native 2DGS normal/distortion runner is staged, **not GPU-tested or run**.
Its CPU source/pose checks pass; launch requires a fresh bounded GPU window after
the previous 15:45 PDT start cutoff. Do not silently extend the 16:00 expiry.
Read `/home/rtoony/reports/2026-09-09-splatlab-condo-structural-study.md` for exact
evidence, limitations, proofs and the geometry-directed next step. The prior
appearance reconstruction below remains useful and unchanged.

## Actual reconstruction checkpoint — September 9, 14:18 PDT

**The input-only checkpoint below is superseded by real reconstruction.**
Open **http://127.0.0.1:32852/** on this workstation for the live Spark splat and
24 real held-out source/render/error comparisons. The six-hour loopback preview
starts approximately 14:17 PDT; files persist in
`data/spatial/condo-reconstruction-showcase-2026-09-09-02`.

- Actual held-out localization: **16/16 physical images**, frozen geometry/intrinsics.
- Training dataset: **72 train / 24 validation / 24 test** crops, timestamp-grouped.
- Completed appearance baseline: **2,000 iterations / 392,249 Gaussians**.
- Pixel-weighted validation: **20.0381 dB masked / 19.3906 dB full-image PSNR**.
  Four timestamps yield 24 overlapping crops; these are not 24 independent captures.
  Test appearance remains reserved. Scores do not establish surface accuracy.
- Actual RTX 5090 WebGL proof: navigation/parallax, reset, comparison images,
  mobile and idle rendering pause pass. Dependencies are local Spark 2.1/Three 0.183.2.
- Actual independent Blender save/reopen: **72 source cameras / 72 packed photos /
  1,949 sparse points**, with 376 on-image projection checks.
- Derived TSDF geometry is **not acceptable architecture**: 1.1 million triangles
  in 110,472 components. It remains an explicitly inferred diagnostic, not a solid,
  collider, measurement source or accepted condo replacement.

Current commands, failures, assets and structural next steps:
`/home/rtoony/reports/2026-09-09-splatlab-condo-reconstruction.md`.
The appearance-to-delivery path now works; registered architectural detailing does not.

## Current focus

The September 9 owner Goal replaces the unfinished Bonsai doorway objective.
Use the condo capture as the primary real-to-model demonstration. Finish studies
are a useful separate design workspace; saving a palette is not reconstruction.

The existing raw-video SfM pilot reconstructs 1,949 sparse points and estimates
24 training-image poses from 12 paired timestamps. This is actual triangulation,
but not a finished exterior model, dense surface, calibrated rig, or metric survey.

The new CPU input stage produces **120 rectilinear views from 40 existing raw
images**: three directions per physical lens/image. These are not 120 independent
photos, and there is no new capture baseline between crops of one image. Physical
lenses retain their distinct estimated centers. No provisional panorama stitch
is used as a calibrated reconstruction input.

## Repeatable input stage

From the SplatLab root:

```sh
timeout 180 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tools/prepare-fisheye-reference.py \
  --pilot data/spatial/condo-raw-pilot-2026-09-08-01 \
  --sfm data/spatial/condo-raw-sfm-2026-09-08-01 \
  --output data/spatial/condo-rectified-reference-NEW \
  --width 768 --fov 100
```

Choose an unused output directory. Inputs are hash checked before/after; original
media, training geometry, and the condo project are not modified. This stage
imports NumPy/Pillow only, with no CUDA/COLMAP/Blender invocation.

The bundle contains:

- `index.html`: offline gallery of actual rectified photos, labeled by source,
  physical lens, timestamp group, split, virtual direction and pose status.
- `camera-set.json`: all image/mask hashes, estimated pinhole intrinsics, source
  PTS, and known training camera-to-world transforms. Held-out poses are null.
- `training-transforms.json`: training views only, with explicit empty validation
  and test lists. Deliberately **not** auto-discoverable `transforms.json`: do not
  promote it for evaluation before localizing the held-out views and reviewing masks.
- `projection-masks/`: binary valid-remap regions, not semantic masks. Operator,
  sky, moving vehicles, reflections and shadows need a separate review.
- `sparse.ply`: byte-preserved existing sparse reconstruction, not newly generated
  geometry.
- `external-references.json`: independent camera/point-cloud handoff matching the
  condo contract, with arbitrary units and `registration: null`.
- `receipt.json`: source/artifact hashes and explicit incomplete-readiness flags.

## Projection contract

The implementation follows the installed COLMAP 4.1 `OPENCV_FISHEYE` model,
including all four radial terms, distinct focal axes, and principal point. It
restricts rays to 85 degrees from each physical optical axis; it does not
extrapolate the model behind a lens. Folded radial mappings are rejected. Bilinear
sampling uses COLMAP's edge-origin pixel coordinates, subtracting half a pixel
when indexing image arrays. See the version-pinned [camera implementation](https://github.com/colmap/colmap/blob/4.1.0/src/colmap/sensor/models.h)
and [SIFT coordinate conversion](https://github.com/colmap/colmap/blob/4.1.0/src/colmap/feature/sift.cc).

Virtual-camera rotation is composed with the inverse physical camera pose, then
camera Y/Z axes are flipped for [Nerfstudio/Blender camera conventions](https://docs.nerf.studio/quickstart/data_conventions.html).
The world itself is neither silently reoriented nor scaled. It is not yet the
condo's meter/Z-up coordinate frame. The current three horizontal directions do
not cover the complete fisheye hemisphere or whole building.

## Frozen-geometry held-out localization

`tools/localize-dual-fisheye-holdout.py` now passes the actual installed pycolmap
4.1 estimation path: 16/16 held-out images localize in 5.05 seconds. Its
correspondence helpers and direct-launch refusal are also tested.

The runner uses a consistent, separate copy of the training database, verifies
feature-index identity against reconstructed observations, extracts held-out SIFT
features, and matches against existing 3D tracks. Ambiguous correspondences and
duplicate 3D points do not count as extra evidence. Absolute pose refinement keeps
intrinsics and all training geometry fixed. There is no joint mapping or held-out
triangulation. Its pose-fit errors must not be presented as held-out appearance scores.

Even though this experiment selects CPU feature/pose operations, pycolmap is a
GPU-capable dependency. Use the existing compute wrapper, never a direct launch:

```sh
env SPLAT_GPU_MANUAL_VRAM_MB=1024 tools/splatlab-compute-gate.sh --run \
  timeout --kill-after=20 600 env CUDA_VISIBLE_DEVICES= \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  /home/rtoony/miniconda3/envs/colmap4/bin/python \
  tools/localize-dual-fisheye-holdout.py \
  --pilot data/spatial/condo-raw-pilot-2026-09-08-01 \
  --sfm data/spatial/condo-raw-sfm-2026-09-08-01 \
  --output data/spatial/condo-heldout-localization-NEW
```

This repeatable command is not standing launch authorization. Fresh owner GPU
permission enables the actual September 9 run through a bounded temporary
capability window, with the existing compute/resource/backup guards retained.

## Original milestones and current disposition

Steps 1–3 and the independent Blender delivery portion of step 5 now run. Step 4
produces a valid splat but an inadequate diagnostic mesh. Prioritize source-supported
facade/ground structure or one geometry-directed refinement, not another unbounded
appearance-only training run. Registration and owner-reviewed corrections remain open.

1. Run and inspect held-out localization. Retain failures instead of filling poses
   by interpolation, coarse GPS or inferred rigid lens separation.
2. Add reviewed masks and source-bound held-out pose ingestion to the rectification
   bundle. Keep every virtual crop in its parent timestamp split. Test the actual
   installed training dataparser; prevent automatic reshuffling/normalization drift.
3. Train one bounded exterior baseline; compare it with actual unseen validation
   images. Cap this at one baseline and one evidence-directed refinement before
   switching input or showing the result. No return to open-ended Bonsai diagnosis.
4. Export the result as a distinct splat/reference layer and a conservative surface
   candidate where image support is sufficient. Leave unknown/backside regions
   visibly incomplete rather than closing them with invented architectural truth.
5. Open a separate Blender review scene with source cameras and reference geometry.
   Register to the condo using reviewed controls; use the residuals and explicit
   scale evidence, not GPS alone. Preserve accepted model files and stable IDs.
6. Propose one bounded exterior detail correction with before/after evidence.
   Owner acceptance stays in Condo Lab. Repeat on suitable interior photos/video,
   rather than treating sparse exterior evidence as indoor geometry.

Success means a demonstrable source-to-3D asset and repeatable, tested Blender/condo
review handoff. Rectification, a photo atlas, a color study, or test counts alone
do not complete the Goal. The appearance and Blender file-delivery path now works;
usable structural geometry and registered condo detailing remain incomplete.

## Tests

### Independent-capture exclusions

`tools/capture-pilot-masks.py` prepares lossless decoded raw-lens images, runs
the installed local SAM3 checkpoint with fixed person/hand/sky prompts, then
produces source-bound binary feature masks and a training-only contact sheet.
Its three actions are `prepare`, `run`, and `review`. Inference requires a fresh
authorized lease and the shared compute gate; preparing and reviewing masks do
not launch CUDA. Never reuse another capture's masks just because dimensions match.

After a complete training-only camera model and successful frozen held-out
localization, pass `--raw-mask-review MASK_STUDY/mask-review.json` to
`tools/prepare-fisheye-training.py`. This optional input carries raw exclusions
through nearest-pixel rectification into every virtual crop, intersected with
projection/rim/nadir support. The builder verifies exact pilot identity, physical
image hashes, split identity, binary dimensions and mask hashes before and after
preparation. It rejects missing, duplicate, changed or cross-capture masks. The
default path remains available for the earlier geometric-mask-only experiment.

Review `training-mask-review.png` before training. Local predictions are not
ground truth; foliage, glass, flare, vehicles and shadows can remain. The training
receipt records whether semantic exclusions were actually used rather than
claiming that geometric masks are semantic masks.

The independent clip 007 camera pilot exposed sensitivity to the initial focal
guess: the default 0.3-width initializer produced two incomplete components. A
single retry at 0.268, informed only by the first training model's estimated focal
length, registered all 24 training views. All 16 reserved images then localized
against frozen geometry. Both attempts are retained. Do not blindly promote
0.268 to a device calibration or merge incomplete components by assumed rig/GPS.
Future capture profiles should make this measured-versus-estimated distinction
visible and stop before training when camera support is incomplete.

Focused tests also cover raw-mask propagation, identity/hash failures and original
split/camera preservation in `backend/tests/test_capture_pilot_masks.py` and
`backend/tests/test_fisheye_training.py`.

```sh
timeout 90 env PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 pytest -q \
  backend/tests/test_heldout_localization.py \
  backend/tests/test_fisheye_reference.py \
  backend/tests/test_rig_diagnostics.py \
  backend/tests/test_dual_fisheye_sfm.py \
  backend/tests/test_dual_fisheye_pilot.py
```

The initial input checkpoint has 70 passing tests. The expanded September 9
checkpoint has 140 focused Python tests plus three Node cases, and actual
localization, evaluation, GPU browser and Blender save/reopen receipts. No full
backend/frontend regression is claimed.
