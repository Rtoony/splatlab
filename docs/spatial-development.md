# Spatial workspace: first implementation tranche

Implemented 2026-09-06 against the interviewed next-phases proposal. This is a reviewable foundation, not completion of every phase's research and acceptance criteria.

The durable interview, findings, proposal and implementation ledger are in `/home/rtoony/reports/2026-09-06-splatlab-next-phases/`. No production restart, model installation, automatic GPU experiment or publication is part of this tranche.

## Enter an architectural walking preview

Once a revision containing an architectural connection is loaded, set the explicit total player height and radius. Use **Start at captured side** or **Start in extension** under that connection, then **Explore with WASD** to capture the mouse and walk. Esc releases the mouse. A start button explicitly repositions the player and faces the opposite endpoint; it does not automatically traverse the connection, apply the edit or resize the body.

Entry points come from that revision's pinned architectural navigation, never the current mutable study. The current collider must admit the stated body at the selected point. Missing navigation produces a connection-specific message; stale revisions, uncalibrated units, invalid body dimensions and isolated/draft inspection disable entry. Failed preflight preserves the current view; failed final admission restores the camera and stops walking. Start clearance is not whole-route acceptance.

The private `walking-entry-gpu-preview-01` evidence actually clicks both buttons at two body sizes before four rendered physical traversals. It passes the image/source audit with generation 48 unchanged. Full findings and honest instrumentation limits: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/walking-entry-2026-09-08.md`. This remains preview-only, not the pending apply/reload/undo proof or a live deployment.

## Connected-space floor recipes

Architectural preparation defaults to `nominal-floor/v1`. The explicit experimental `raised-floor-finish/v1` raises both authored floor slabs and their collision contribution by 16 mm without changing captured clipping, player dimensions or walking tolerances. Finished clear heights are 16 mm below the nominal frame dimensions. API preparation accepts `geometry_method`; the CLI accepts `--geometry-method`; the Studio's advanced dimensioned frame exposes “Authored floor recipe.” Adjusting a retained candidate preserves its method. A different recipe requires fresh sealed preparation and a new contained build, never reinterpretation of old evidence.

The September 8 private paired-floor candidate passes eleven local geometry gates and actual NVIDIA preview-only walking at both fixed body profiles. Its floors are clean in inspected views, but authored wall/jamb seams remain; the subsequent joined-boundary candidate addresses those seams. Neither is full apply/reload/undo acceptance. Historical floor evidence: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/paired-floor-rendered-findings-2026-09-08.md`. Keep the nominal default until the experimental recipe completes review.

The further opt-in `joined-floor-envelope/v1` retains the16mm floor finish, clipping volumes, protected reservations and physical player/navigation dimensions. It removes internal/coplanar box faces with a deterministic orthogonal union boundary on a1nm coordinate grid. It also extends the tunnel roof over the wall tops **within the existing reserved frame**: the earlier roof met those walls only along edges. This adds small authored corner-cap regions; it is not an exactly volume-preserving retessellation of the prior recipe. The exposed boundary supplies both visible geometry and collision, and its GLB uses flat per-face normals. A new `authored_room_watertight` gate supplements—not replaces—the existing gates. It requires a fresh preparation/build and real visual/walking review; old candidates remain readable under their original recipes.

Walking screenshots now require a same-frame `scene-visibility-ab/v1` control: ordinary render, hidden scene, exact restored render, without changing the camera, body or collider. The independent PNG audit requires significant scene contribution (at least 8 channel levels across 1% of pixels) and exact restoration rather than arbitrary color diversity: a flat authored wall can legitimately have very few colors. Each new walking trace retains three original images and six control images. Older proofs without these controls are historical evidence, not automatically upgraded to the new audit. Run `node tools/test-architectural-render-control.mjs` and the architectural Python proof tests for the synthetic control/cleanup regressions; these tests do not replace real GPU proof or visual review.

The joined-boundary candidate passes all 12 local gates and an actual NVIDIA preview with four traversals (outward and back at two fixed body sizes), 3,406 rendered frames and the independent image/artifact audit. A separate 15-frame appearance inspection confirms the old authored seams disappear while the inspected floors stay clean. That milestone is retained in `/home/rtoony/reports/2026-09-06-splatlab-next-phases/joined-room-envelope-2026-09-08.md`; it is not relabeled as evidence for subsequent rendering changes.

The subsequent fragment/depth repair uses the actual drawing-buffer viewport and homogeneous W for precise captured-fragment clipping, plus the existing Three/Spark logarithmic-depth path. Same-camera legacy/projection-only/corrected controls show ordinary 24-bit depth ties cause the isolated wall specks; the corrected renderer removes those components without enlarging the cut, moving camera clipping planes or changing collision. It passes a fresh NVIDIA walking preview (3,405 rendered frames, 17,029 physical positions) and independent audit. A small photographed floor/threshold band remains explicitly characterized. Named and unnamed protection decisions and this exact recipe's apply/reload/undo are still required; generation 48 is untouched. Details: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/fragment-depth-repair-2026-09-08.md`.

For the isolated rendering comparison, supply `--architectural-projection-comparison` with the existing private `--studio-inspect` / `SPATIAL_INSPECT_ARCHITECTURE_PROOF` workflow and a new output directory. The explicit option must reach the child and produce all 18 comparison frames, with exact old/new restorations; ordinary inspection is not accepted as this comparison. Controls temporarily omit logarithmic-depth writes only for diagnosis, restore material hooks and keep the active scene unchanged. They are not a user-facing alternate geometry or removal mode.

## Local capture and route workflow

`tools/capture-route.py` registers explicitly selected local recordings. Registration retains the full SHA-256 identity, both video streams, original media path, duration, complete extracted GPS records, available accuracy fields, clock assumptions and IMU extraction status. Originals are not modified or copied. The HTTP inspection endpoint only accepts Transfers entries, existing job inputs or already registered sources.

```bash
.venv/bin/python tools/capture-route.py inspect '/absolute/path/to/recording.insv'
.venv/bin/python tools/capture-route.py build CAPTURE_ID --start 30 --end 60 --interval 5 --width 2048 --budget 600
.venv/bin/python tools/capture-route.py resume ROUTE_ID
.venv/bin/python tools/capture-route.py prepare ROUTE_ID --section 45 --overlap 10 --nadir-exclusion 50
```

Runtime manifests and JPEGs live under `data/spatial/`, which is gitignored. `/routes` provides authenticated spherical viewing, GPS trajectory and time navigation, build/stop/resume controls, horizon pitch/roll settings and reconstruction preparation. GPS travel direction is not camera orientation. Clock verification is an explicit `PUT /api/splat/captures/{id}/clock` with video start UTC seconds, offset and uncertainty; it invalidates resume against an older route clock.

Extraction is CPU-only: both fisheye streams are mapped explicitly, decoding and encoding threads are bounded, and one selected JPEG is produced without transcoding the full movie. One cross-process route worker runs at a time; a conflicting build can be resumed later. A route checkpoints each frame, checksum-verifies retained outputs, retries failed frames without discarding successful ones, and pauses on stop/budget. Restart does not automatically restart processing. Default budget is one hour; the explicit maximum is eight hours.

### Accuracy boundaries

- The open-source stitch is generic dual-fisheye reprojection, not X5 optical calibration, optical-flow stitching or IMU stabilization.
- Viewpoint times are requested video seeks, not verified decoded presentation timestamps. Frame cadence and container timestamp offsets still need calibration.
- GPS interpolation never extrapolates and refuses gaps over two seconds. Missing positioning accuracy remains unknown; unverified clock alignment stays visibly approximate.
- A nadir exclusion cap removes observations, not people semantically. It can discard real ground and miss an operator away from the nadir. Images remain unchanged; masks use 255 to retain and 0 to exclude.
- Panoramas are observed viewpoints, not free-viewpoint 3D reconstruction. No route section is advertised as navigable 3D before it exists and passes review.

### Raw dual-fisheye pilot

For explicitly selected raw two-lens captures, `tools/prepare-dual-fisheye-pilot.py` provides a separate experimental path that does **not** use the generic panorama stitch. It retains original decoded frame indices and integer presentation timestamps/time bases, disables automatic rotation, scales without warping, checks both lenses have exactly matching container timestamps, and keeps their views together in train/validation/test splits. Matching container timestamps are not proof of hardware exposure synchronization. Intrinsics, rig extrinsics, UTC alignment and metric scale remain unknown.

Register the selected original using `capture-route.py inspect` first. Without `--output`, preparation prints a plan and does not decode. Choose a new output directory outside the original-media directory; partial runs retain diagnostic receipts and cannot be reused as completed reconstruction inputs. Originals and existing outputs are never overwritten. The default selection is twenty timestamp groups at thirty decoded-frame intervals, not an assumed exact one-second cadence.

```bash
.venv/bin/python tools/prepare-dual-fisheye-pilot.py CAPTURE_ID
env SPLAT_GPU_MANUAL_VRAM_MB=1024 tools/splatlab-compute-gate.sh --run \
  timeout --kill-after=20 720 .venv/bin/python tools/prepare-dual-fisheye-pilot.py CAPTURE_ID \
  --output data/spatial/selected-raw-pilot-01 --groups 20 --stride 30 --width 1536 --budget 600
```

After visual review, provide per-training-image grayscale feature masks under a separate root, e.g. `masks/lens-0/frame-000000.jpg.png`. A black pixel excludes features; the mask does not alter the source image. `tools/probe-dual-fisheye-sfm.py` verifies the complete pilot and masks, uses **only training images** for initial unconstrained COLMAP estimation, shares intrinsics per physical lens folder, and records components and same-timestamp pairs that register together. It targets the installed pycolmap 4.1 environment; it neither installs dependencies nor fabricates a fixed rig from nominal lens directions.

```bash
env SPLAT_GPU_MANUAL_VRAM_MB=1024 tools/splatlab-compute-gate.sh --run \
  timeout --kill-after=20 600 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/rtoony/miniconda3/envs/colmap4/bin/python tools/probe-dual-fisheye-sfm.py \
  data/spatial/selected-raw-pilot-01 --mask-root data/spatial/selected-raw-masks-01 \
  --output data/spatial/selected-raw-sfm-01 --focal-ratio 0.3
```

The focal ratio is an explicit numerical initializer, **not measured X5 calibration**. A completed sparse model still needs optical/rig plausibility, coverage and independently localized held-out views before a Gaussian-training decision. Physical raw lenses have different optical centers; virtual crops from one stitched panorama share a center and require a different rig treatment. The CLI is experimental preparation/probing, not yet a new `/routes` UI workflow. Coordinate heavy decoding and SfM with other workstation sessions even when using CPU-only settings.

## Reconstruction preparation

`POST /api/splat/routes/{id}/reconstruction/prepare` accepts `ReconstructionSpec`; `GET /api/splat/routes/{id}/reconstruction` returns the result. Preparation selects the sharpest eligible frame in each temporal window, screens clipping/sharpness, creates checksum-bound masks, partitions overlapping sections, samples a smaller semantic subset and creates separate train/validation/test **timestamp groups**. Every virtual view of a timestamp keeps its split across overlapping sections.

This is not a translation/coverage estimator. Those require solved cameras. Prepared sections are marked `planned` and `panorama-only`, not trained or evaluated. They do not launch COLMAP, semantic models or splat training.

The existing CPU rig renderer accepts the prepared plan:

```bash
/home/rtoony/miniconda3/envs/colmap4/bin/python backend/rig/render_rig.py \
  ROUTE_DIRECTORY RIG_OUTPUT --workers 2 --reconstruction-plan PLAN_JSON
```

It verifies panorama and mask checksums and combines projected exclusion masks with the rig ownership masks. After a separately admitted camera solve, preserve rig filenames and create a new Nerfstudio transforms document:

```bash
.venv/bin/python tools/capture-route.py split-transforms PLAN_JSON TRANSFORMS_JSON NEW_TRANSFORMS_JSON
```

Unknown timestamp names, missing splits and an existing output are refused. The original transforms file is not overwritten. Installed Nerfstudio supports explicit `train_filenames`, `val_filenames`, and `test_filenames`; training adoption of the prepared document remains an explicit step. Every GPU-capable solver/trainer command must use `tools/splatlab-compute-gate.sh --run ...`. No raw CUDA command is authorized by preparing a plan.

## Quality and edit safety

- Collision-shell reuse checks input identities, added/changed surface patches, calibration generation, output identity and failed/custom-source builds. Dependency snapshots precede long builds, so an intervening source change cannot be silently certified as current.
- Isolate receipts and pluck documents carry calibration generation and scale. Old or recalibrated claims must be re-isolated rather than relabeled current.
- New generated candidates record source/selection/scale and candidate identities. Unknown or changed lineage remains viewable but cannot promote; re-propose to obtain fresh evidence.
- Promotion retains a checksum-addressed generated-native master and generation report separately from the placed delivery artifact. Preview stays the default (8k triangles / 1k texture); optional balanced (32k / 2k) and detail (100k / 4k) budgets are explicit. Budgets are upper targets, not guarantees that an engine has that detail. Revert restores the prior element while retaining the master.

The legacy generation/promotion workflow still writes legacy artifacts. The creative studio below introduces a separate revision-pinned transaction; other viewers and export routes do not silently switch to it.

## Creative scene revisions

Open **Creative scene studio** from a scene's actions menu, or visit `/studio/JOB_ID`. A graded world is required. Initialization freezes the current viewer, mesh/splat assets, collision files, calibration, selection evidence and local metadata into `JOB/_studio/`. It does not modify the captured world, invoke a model or start Blender. Snapshot input is capped at 8 GiB; content-addressed copies are shared between revisions, not hardlinked to mutable originals.

The workflow is **propose → compare → review → apply → undo**:

- Place a versioned Blender export, remove an authored element, or replace an element with that export. The export must have an intact checksum receipt, embedded resources and identity transforms. Metre-authored placement requires a current metre world. The prose field records intent; it is not a text-to-geometry interpreter.
- A proposal names its base generation and complete candidate revision. Creating, viewing or saving it does not activate anything. Saved proposals survive page reload; stale ones remain inspectable but cannot apply.
- The viewer pins every asset URL to one immutable revision. Comparing preserves the camera across revision switches. Captured appearance and mesh/collision inspection are separate options. Failed captured appearance or missing required background collision prevents a successful removal preview.
- Apply requires an explicit review acknowledgement and compare-and-swap against the active generation. It checks source/calibration/selection dependencies and every artifact checksum, then atomically advances one pointer. Competing applies have one winner. Files and containing directories are fsynced; sudden-power-loss testing has not been performed.
- Undo copies an activated ancestor forward into a new revision, restoring its entire artifact map and state. It cannot activate a never-reviewed proposal through the undo endpoint. Old assets and revisions remain available.
- If legacy inputs change, **Preview fresh capture baseline** proposes a new snapshot for review. Applying it replaces studio edits, rather than silently moving them into a different frame. Previous revisions remain undoable. Refresh does not rebuild stale geometry or solve calibration: those still need their normal reconstruction workflow.

Captured removal requires fresh row selections and background collision evidence. Row masks, active semantic membership, visible elements, authored placement registry and collision inputs advance in the same revision. Raw observation labels and selection rows remain preserved as evidence and for undo. The photo-backed recovery below addresses supported portions of a planar region; it does not reconstruct every region exposed by hiding an object.

Studio activation affects **only revision-aware consumers**. The original world, legacy exports, global semantic search and generated-candidate promotion retain their existing behavior. Interaction/restyle documents are retained in snapshots, but this inspection viewer does not replay gameplay state or enable dynamic prop physics. Captured-wall cutting, complete background recovery, measured appearance acceptance, connected-room navigation, revision-aware export and generated-model adapters remain unfinished.

### API contract

All endpoints below are authenticated and live under `/api/splat/jobs/JOB_ID/studio`:

| Endpoint | Purpose |
| --- | --- |
| `POST /initialize` | Create an initial captured baseline; idempotent once active. |
| `GET /` | Active pointer, ancestry, saved proposals, versioned exports and stale-source status; omit the trailing slash in requests. |
| `POST /proposals` | `expected_generation`, `instruction`, typed `operation` (`place`, `remove`, `replace`). |
| `POST /refresh-proposal` | `expected_generation`; propose replacing studio state with the latest capture. |
| `GET /proposals/EDIT_ID` | Immutable proposal receipt. |
| `POST /proposals/EDIT_ID/apply` | `expected_generation`, `reviewed: true`; activate a reviewed proposal. |
| `POST /restore` | `expected_generation`, `revision_id`; restore an activated ancestor. |
| `GET /revisions/REVISION_ID` | Viewer, state and URLs pinned to that revision. |
| `GET /revisions/REVISION_ID/artifact?key=KEY` | Only an artifact belonging to that revision; no arbitrary filesystem reads. |

Conflict responses are HTTP 409; invalid request bodies are 422. Core storage is in `backend/scene_revisions.py`; authenticated routes are in `backend/scene_studio_route.py`.

### Private real-capture proof

`tools/prepare-scene-studio-proof.py` creates a private Bonsai-derived clone at `data/spatial/scene-proof/outputs/splat_c0ffee` and authors an adjacent 3 x 3 x 2.8 m room with CPU Blender. It refuses to overwrite an existing fixture. The original capture remains unchanged. This is an adjacent-room transaction fixture, not an opening cut through a captured wall.

The staged browser proof accepts `--studio-job`, `--outputs-root` and an explicitly inspected `--studio-generation`. It refuses a generation mismatch rather than blindly replaying mutations. It checks preview isolation, visible/collision membership, comparison, saved-proposal recovery, apply, exact-state undo, authentication and mobile overflow. It uses software WebGL with captured appearance disabled; screenshots do not certify photoreal splats, walkability or hardware FPS.

Evidence is retained under `/home/rtoony/reports/2026-09-06-splatlab-next-phases/scene-revision-proof/` and `scene-revision-refresh-proof/`. The subsequent background proofs advance the private fixture to generation 9 after undo. Inspect its active pointer before any additional mutation proof; do not replay a generation-0 command or overwrite its history.

## Photo-backed background recovery

The studio's **Recover observed background** panel fits a nearby approximately horizontal support surface and samples its source photographs. It is distinct from the existing splat-DC-color texture bake. A horizontal fit is not a semantic guarantee that the surface is the floor: a shelf, tabletop or other near-planar surface can also satisfy the fit.

`backend/reconstruction_evidence.py` reads retained COLMAP camera/image/point binaries without importing a model runtime. It checks camera-to-saved-pose agreement, recorded image resizing/intrinsics, calibration and sampled feature reprojections before exposing SfM points in studio Y-up metres. Raw COLMAP camera coordinates follow its documented convention; reconstruction transforms supply the conversion into the trained scene frame. [COLMAP output format](https://colmap.github.io/format.html)

Supported inputs currently require one unambiguous retained dataparser transform, one COLMAP binary sparse model, linked `colmap_im_id` values, and PINHOLE, SIMPLE_PINHOLE or OPENCV cameras. Fisheye/equirectangular adapters, ambiguous training runs and independently scaled saved datasets refuse rather than guess. Existing source photos and masks are read only. Contained mask symlinks are supported; links escaping the registered job are refused.

The bounded CPU recipe:

1. Fit an approximately horizontal plane to tracked, low-reprojection-error SfM features around the selected object's bounds. Gaussian centres are not used as surface measurements. The current inlier tolerance is 25 mm and maximum accepted fit RMS is 20 mm; these thresholds do not establish survey accuracy.
2. Require nearby tracked feature support for a texel, respect image masks and conservatively exclude rays intersecting captured object bounds. This is **not dense-depth visibility certification**; unmodelled occluders and specular surfaces remain important failure cases.
3. Sample at most 24 source images, require color agreement from at least two distinct groups with at least a 3° viewing-angle separation, and retain separate groups for up to four appearance checks. Explicit `timestamp_group` wins; otherwise shared image basenames preserve the current SplatLab rig convention. Other crop naming schemes need explicit grouping metadata.
4. Write real-photo RGB only in accepted cells. Unknown cells are transparent and omitted from geometry, not filled with nearest colors. The same retained mesh cells supply visual geometry and collision. Portable render/VR-only provenance tags survive GLB copies; these fitted artifacts are not survey outputs.
5. Retain the recipe/code identities, input checksums, used camera models/poses, support point IDs/coordinates and per-texel source-image membership. The candidate, atlas, coverage image and evidence NPZ are content-addressed. Applying a scene containing the recovery rechecks the source checksums; undo still restores the immutable prior bundle.

One CPU recovery worker is admitted across processes. Input binaries are capped at 512 MiB each, sparse points at 500,000, source cameras at 40 megapixels and texture resolution at 64/128/256. This work does not install or launch GPU models.

### Recovery endpoints and review

Under `/api/splat/jobs/JOB_ID/studio`:

- `POST /recoveries`: `expected_generation`, `selected_slug`, `texture_size` (default 128). Builds a candidate without activating it.
- `GET /recoveries`: retained recovery reports and stale state.
- `GET /recoveries/RECOVERY_ID/artifacts/NAME`: only an artifact belonging to that recovery.
- `POST /proposals` also accepts `operation.recovery_id` instead of `blender_export`. `place` previews the recovered cells without removing the object; `replace` combines them with captured removal and keeps the existing row/collision-evidence guard. Recovery and Blender sources are mutually exclusive.

**Candidate only (inspection)** and **Frame candidate** help inspect patches obscured by a coarse captured mesh. Isolation hides surrounding rendering, not geometry or collision. Walking and review/apply are disabled in isolation; return to context first. Neither isolation nor a successful transaction constitutes appearance acceptance.

The retained Bonsai-derived example selects `cardboard-box-2`: 4,086 support features, 8.98 mm plane-fit RMS, 24 source photos and four appearance-check views. The method supports **22.3%** of the proposed footprint and leaves **77.7% unresolved**. That is coverage under this conservative recipe, not proof that every unresolved pixel is inherently unobservable. A complete box-removal/floor-fill result is not claimed.

`tools/prove-background-recovery.py --expected-generation GENERATION` copies the retained reconstruction evidence into the existing private fixture and builds an unapplied candidate. It refuses differing pre-existing copies. The latest build receipt is `data/spatial/scene-proof/background-recovery-proof.json`. `tools/prove-spatial-ui.py --studio-recovery` exercises multi-angle inspection, contextual review, apply and exact undo on that private fixture. Screenshots/receipts are in the reports' `background-browser-proof/` and `background-inspection-proof/` directories. The source capture and live service remain unchanged.

## Typed architectural authoring

The paired captured-portal editor supports an explicitly authored cut/vestibule depth of 0.1–6 m. Longer depth is a design choice, not measured wall thickness. Its dimensioned guide, room transform and navigation proof use the same depth; named and unnamed foreground objects still need review. All existing floor, collision and preservation gates remain required. A source-tested dimension range is not proof that a particular captured-room placement is walkable.

The existing Blender MCP surface now has 18 tools, including:

- `create_wall`: dimensioned wall, origin at bottom centre, local X width and Z height.
- `cut_opening`: one bounded opening in an authored wall; preserves jambs and lintel and refuses captured collision meshes.
- `create_room`: room candidate with a front doorway, floor, side/back walls and optional ceiling, extending along local +Y from the doorway bottom centre.
- `assign_material`: bounded RGB, roughness and metallic parameters; material authoring, not captured-scene physical relighting.

Dimensions are Blender Z-up metres with scene `scale_length=1`; generated geometry is marked authored. Changes create immutable Blender versions with receipts. Architectural operations with an outdated explicit base version refuse. `restore_blender_version` copies an older version forward. Selective GLB export with baked transforms produces the existing world-placement frame; publication remains separate.

The real CPU Blender proof creates a wall and doorway, authors a room and material, validates an identity-transform Y-up GLB with expected dimensions, then restores the original version byte-for-byte. This synthetic proof does not demonstrate replacing a captured wall, avoiding splat ghosts, or navigating an extension attached to the storage-room capture.

## Multi-point geolocation

Feedback #7 is implemented as **Align matching points** in the existing Locate modal. Pick a point in the unrotated footprint, then its corresponding map point. Two or more pairs solve horizontal scale, heading and translation; three or more provide redundant-control residuals. Preview precedes apply. Each residual, RMS and maximum error are reported; residual agreement is not survey accuracy or an independent check point.

The solver uses WGS84 controls in a local ENU frame, limited to a 10 km footprint. Vertical values are preserved, not calibrated. Apply checks the expected calibration generation and refuses replacing stronger scale evidence without explicit force. Prior alignment metadata is retained under the job's `_geo/`. The footprint overlay respects non-centre anchors from the fitted transform.

Actual Civil 3D import verification, full 3D control/datum handling, measured LiDAR integration and an operator-facing alignment-restore action remain pending.

## Validation and staged review

```bash
cd backend
pytest -q --rootdir . tests
SPLATLAB_RUN_BLENDER_TESTS=1 pytest -q --rootdir . tests/test_blender_headless_integration.py -k architecture
cd ../frontend
npx tsc --noEmit
npm test
npx vite build --outDir /tmp/splatlab-spatial-ui-build
```

`tools/prove-spatial-ui.py` runs that build against an ephemeral loopback server with an in-memory test credential and startup recovery disabled. It does not restart the live app. Pass `--dist`, `--playwright-module` pointing to an existing local Playwright installation, `--route` and `--output`. The proof checks authentication refusal, real panorama loading, next-viewpoint navigation and mobile overflow. Its software WebGL screenshots are not a hardware FPS benchmark. Production can continue serving its previous build until a separate rollout decision.

## Captured removal: addressing, frames and collision

Captured removal now requires coordinate-verified selection evidence, not just a current receipt date. When an isolated object PLY is present, the pluck builder verifies its coordinates against the served splat rows through the export map. Malformed/non-injective maps are refused; sparse checkpoint identifiers do not trigger a dense inverse allocation. Coordinate PLY and index identities participate in freshness, and studio snapshots retain both. Legacy mappings without this evidence remain readable but cannot authorize a studio removal.

Collision receipts explicitly declare Y-up, unbaked scene units. The merged viewer manifest supplies `collision_shell.scale_to_world`, derived from the world's retained bake factor rather than the newest job calibration. The walker applies it to collision geometry and spawn coordinates. A current input fingerprint does not make a `NOT_WALKABLE`, `FAILED` or ungraded collider acceptable for removal. The revision API separately exposes the frozen raw-backdrop scale as `backdrop_meters_per_unit`.

Studio disables dynamic prop physics and explicitly merges the remaining prop meshes with the dedicated background collider. The original removal path removed the selected prop mesh but preserved a background solid that still contained that object. New removal proposals also require selection-aware local clearance, described below; fitted recovery geometry joins the same collider. This is static triangle collision, not dynamic-physics or navigation acceptance. Other walker callers retain their previous prop behavior unless they opt into `setStaticPropCollision(true)`.

The studio prepares Spark's masked splat data before reporting captured-preview readiness and disables MSAA for this splat-heavy view. A removal shown only in mesh mode cannot unlock the review checkbox. Candidate-only inspection still cannot substitute for full-context review.

`tools/prepare-captured-removal-proof.py --expected-generation N` prepares the **private** Bonsai-derived fixture using copied retained weights, instance memberships and reconstruction bounds. It neither trains nor activates a revision. Any subsequent isolation or mesh runner must still use `tools/splatlab-compute-gate.sh --run` and a bounded timeout. Re-materializing old memberships verifies addressing; it does not improve their semantic labels or completeness.

`tools/prove-spatial-ui.py --studio-removal` adds fixed captured-camera views, a short camera sequence, row-mask/collision/semantic checks, apply, reload and undo. Its software renderer is capped to 480×320 before each navigation, with a nonblank-pixel check. A private rollback checkpoint allows cleanup after a failed proof; such cleanup is recorded separately and must not be called a successful browser undo. `--studio-inspect` performs read-only rendering diagnosis. The removal helper refuses targets other than the dedicated private fixture.

Full-object selection recall, dense visibility, complete background synthesis, free-walking acceptance and captured-wall remodeling remain separate requirements. A verified row address or plausible support plane is not proof that the named physical object was completely removed or its hidden background recovered.

## Multi-view selection review and refinement

`backend/selection_reviews.py` prepares a CPU-only study against the active scene: source photos, verified resized intrinsics, raw/metre coordinate frames, current served rows and distinct camera groups. Four through eight views are required; cameras are separated by at least three degrees around the target. The retained camera optimizer must be `off`; optimized camera deltas and nonzero distortion refuse pending verified adapters. Preparation is capped at 512 MiB of splat data, two million scene rows, 200,000 selected rows and 960-pixel photograph long sides.

Under `/api/splat/jobs/{jobId}/studio/selection-reviews`, `GET` lists retained studies and staleness, and `POST` accepts `selected_slug` plus `expected_generation` to prepare a study without GPU work. `/{reviewId}/artifact?name=...` serves only checksum-verified immutable members. `/{reviewId}/rows?candidate=true` serves bounded refined rows only after sealing; default rows are the original selection. Inspection never activates membership or rebuilds collision.

Run the stages sequentially, from the repository root, substituting the chosen job and returned review ID:

```bash
.venv/bin/python tools/selection-review.py prepare JOB --selected-slug SLUG --expected-generation GENERATION
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --signal=TERM --kill-after=15s 240s \
  /home/rtoony/miniconda3/envs/langfield-spike/bin/python backend/mesh/selection_visibility.py JOB REVIEW_ID
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --signal=TERM --kill-after=15s 240s \
  .venv/bin/python tools/run-selection-masks.py JOB REVIEW_ID
.venv/bin/python tools/selection-review.py refine JOB --review-id REVIEW_ID
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --signal=TERM --kill-after=15s 240s \
  /home/rtoony/miniconda3/envs/langfield-spike/bin/python backend/mesh/selection_visibility.py JOB REVIEW_ID --candidate
.venv/bin/python tools/selection-review.py finalize JOB --review-id REVIEW_ID
```

These worker paths use already installed local environments and SAM3.1 weights. They do not install dependencies or schedule unattended work. Per-study file locks prohibit concurrent writers; recorded stages refuse overwriting. After a partial worker failure, prepare another study rather than rewriting sealed evidence. Input changes invalidate continuation. GPU workers retain the shared compute gate; preparation itself does not launch them.

The renderer measures alpha-composited selected Gaussian contribution and expected depth from the existing field. It is not an independent depth sensor. Visible-center voting requires alpha >= 0.85 and a 25 mm + 2% depth band. SAM association requires enough visible core and rejects ambiguous competing masks. Added rows need two distinct fit groups with at least 75% positive vote agreement, stay within original center bounds expanded 5 cm, and cannot take another instance's claimed rows. Core membership is preserved. Check views never vote, but are not held out from original Gaussian training.

The studio panel compares source photos, projected centers, current/refined visible contribution and predicted masks. Current/refined 3D isolation automatically frames the existing instance bounds, keeps normal removal state intact, and disables walking/review readiness. **Frame selected splats** returns to the close-up. Exiting restores surrounding geometry. Rows-only activation remains unavailable: the selection-specific local-clearance build below must join the same reviewed removal transaction. The old background collider is not valid evidence of selected-object clearance.

`tools/prove-spatial-ui.py --studio-selection` provides a read-only browser proof using sealed current evidence, two recorded cameras plus close-up framing, evidence-image decoding, inspection guards and unchanged active/collision state. It additionally renders the complement of each inspection mask with walking disabled: these are capture-only residual diagnostics, not admitted removals or repaired backgrounds. Two settled update/render passes and direct canvas PNG encoding allow exact baseline restoration without CSS-compositor effects. Use the same `--studio-job`, `--studio-generation`, private `--outputs-root`, staged `--dist`, local `--playwright-module` and report `--output` arguments as the other proofs. Software rendering remains 480×320; explicit `--browser-gpu` uses verified NVIDIA rendering at 960×640 through the compute gate.

### Experimental Gaussian-footprint refinement

Center pixels and expected-depth proximity miss Gaussians whose extended footprints contribute to an object. The opt-in contribution worker differentiates a zero-valued three-channel feature render with respect to each Gaussian's colors. Mask, complement and full-image channel sums yield integrated occlusion-weighted alpha mass per Gaussian; geometry, opacity and model weights are not optimized. The worker uses the installed `gsplat` renderer, checks nonnegative finite weights, per-row partitions and whole-image alpha conservation, and seals source-run identities, arrays, implementation hashes and memory/timing evidence.

After initial visibility and SAM masks, and **before** refining a fresh study:

```bash
tools/splatlab-compute-gate.sh --run timeout 180 \
  /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python \
  backend/mesh/selection_contribution.py JOB REVIEW_ID
.venv/bin/python tools/selection-review.py refine JOB --review-id REVIEW_ID --contribution-weighted
```

Then run candidate visibility and finalization as usual. Contributions need at least 0.5 visible alpha-pixels; at least 80% inside the associated mask is positive, at most 20% is negative, and mixed/tiny/occluded footprints abstain. Existing independent-fit-group votes, other-instance exclusion, core preservation and check-view exclusion remain. The default spatial margin stays **5 cm** for both methods. Explicit `--spatial-margin-m` accepts zero through 20 cm and records that independent experimental choice; it never relaxes collision gates. No alternative silently becomes the default selection method.

The UI shows the recipe plus two separate mass diagnostics: the fraction of baseline mask alpha belonging to selected rows, and the selected alpha fraction outside the mask. These are not binary mask-area coverage, newly exposed appearance, semantic ground truth or physical-object recall. Repeated same-capture experiments are exploratory even when check cameras never contribute votes. Final evidence retains all contribution arrays and refuses changed mask associations, camera splits or stage lineage.

During either current or refined 3D inspection, **Show remaining captured appearance** reveals the capture left behind by that selection. The renderer copies the existing removal mask and hides only the selected bounded row list, so it does not resurrect earlier removals or allocate an expanded complement row payload. Saved rows, collision and active revision stay unchanged. Authored meshes are hidden, walking and review/apply are disabled, and no missing background is generated. Exit inspection to return to normal contextual review.

The first 20 cm footprint candidate contains 710 rows and improves mask coverage, but its local collision worker correctly refuses insufficient clearance beside another named instance. The separately prepared 5 cm candidate contains 628 rows versus 469 from center/depth voting. Preserve both studies and their distinct outcomes; do not merge rows or lower the one-millimetre clearance minimum to admit the larger candidate. Actual results and remaining appearance limitations are recorded in `/home/rtoony/reports/2026-09-06-splatlab-next-phases/selection-footprint-findings.md`.

The retained box candidate adds 78 rows (391 → 469); inferred-mask coverage on two nonvoting checks changes 79.2% → 88.7% and 65.3% → 83.4%, with 8.0% and 8.4% outside-mask fractions. This is incomplete model agreement, not physical-object recall or a collision/removal acceptance. The installed SAM loader emits four missing-key warnings; checkpoint/config compatibility remains a quality-review item. See the reports' `selection-refinement-findings.md` for evidence identities and limitations.

## Selection-aware local collision and refined removal

The global background builder samples the retained full reconstruction mesh, so even a passing floor/solid result can contain objects excluded from its Gaussian input. Removing only a prop mesh is insufficient. `backend/selection_collision.py` and `backend/mesh/selection_collision_build.py` provide a separate revision-bound local edit rather than silently changing the legacy global builder's meaning.

Under `/api/splat/jobs/JOB_ID/studio`:

| Endpoint | Contract |
| --- | --- |
| `POST /selection-collisions` | CPU preparation only; `selection_review_id`, `expected_generation`, optional `voxel_m` (0.03–0.1). Requires a completed current selection study and graded, calibrated base solid. |
| `GET /selection-collisions` | Retained studies, recipes, counts, result summaries and stale state. No result means unstarted, running or interrupted, not a live-worker claim. |
| `GET /selection-collisions/COLLISION_ID/artifact?name=NAME` | Checksum-verified result artifact belonging to that build; no arbitrary filesystem access. |
| `POST /proposals` | Existing removal/replacement contract plus `operation.selection_collision_id`. Rows, frozen pluck document, local background collider, visibility and semantics join one preview revision. |

Prepare inputs and explicitly run the bounded worker only for the currently selected experiment:

```bash
.venv/bin/python tools/selection-collision.py JOB REVIEW_ID --expected-generation GENERATION
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --signal=TERM --kill-after=15s 240s \
  /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python \
  backend/mesh/selection_collision_build.py JOB COLLISION_ID
```

Use the returned identifier, not a historical proof ID. The worker uses already installed SciPy/Open3D/trimesh/Manifold in the existing geometry environment; normal API preparation does not import that runtime. Per-build locks prohibit concurrent writes, source checks refuse stale input, and existing partial geometry/result files refuse overwrite. After interruption, inspect the retained outputs and prepare a fresh build. Retired Gaussian-only diagnostic recipes cannot silently fall through as successful runs.

The local cutter is a convex hull of selected served Gaussian centers extruded down to a fitted protected-floor boundary. Adaptive padding avoids other named centers; the worker refuses overlap, less than 1 mm available clearance, floor intersection, an invalid base solid or a volume above one cubic metre. It uses the old reconstruction mesh for room bounds only, never for new occupancy. Painted/measured surface authority currently refuses until a dedicated adapter exists.

Eight mandatory gates cover the oriented closed solid, selected centers, interior samples, interior output vertices, sampled outside-surface preservation, positive bounded removed volume and unchanged floor/hole metrics. The result is **PASS_LOCAL_EDIT**, not navigation acceptance. `capture_partition.cleared_slugs` tracks actual local cuts. Other named meshes, Gaussian footprints, unknown content and semantic floor interpretation remain separate concerns; stored background-complement row arrays are not proof of whole-scene object exclusion. The navmesh records its background-only scope.

In studio, prepare and refresh retained collision studies, inspect their metrics, then **Propose refined removal**. Use captured context and **Collision wireframe** before the existing acknowledgement and **Apply together**. Comparison and undo retain the previous complete revision. New captured removals and saved pre-upgrade removal proposals refuse without selection-aware clearance. Historical undo stays available but does not retroactively certify old geometry. No separate row-only apply endpoint or automatic worker launch is added.

The private `--studio-refined` proof uses the other browser runner's explicit job/generation, private output root, staged build, local Playwright module and report directory arguments. It checks refined rows plus collider changes, other-instance visibility, fixed/sequence rendering, wireframe, no mutation during preview, apply/reload and browser undo. Its rollback checkpoint permits cleanup only on the matching disposable private fixture; failed attempts are not counted as successful browser undo.

Retained result: 469 selected centers and 813 interior samples clear; 54,070 outside samples have zero measured displacement. Combined collider triangles change 250,108 → 247,842. The adjacent upper-box mesh is open and still has eight selected centers within 5 mm, so this is not a whole merged-collider pass. That proof restores private generation **18** with exact state/artifact-map and pixel-identical matched-camera undo; the subsequent support-surface proof below restores generation **20**. Generation-16 study/build evidence is stale for another mutation. Read `selection-collision-findings.md` in the proposal reports for failures, identities and limits. Complete background appearance, real capsule traversal and broader phase acceptance remain open.

## Photo-linked support-surface selection

The background panel can now **Choose support plane from tracked photo points** instead of accepting the automatically strongest horizontal plane. Choose a captured object, expand the photo picker, then pick three to six well-spread tracked dots on the intended visible surface. Switch source photographs without losing selected anchors. Hovering a dot shows its retained point ID, world Y coordinate and reprojection error; a point-ID field supports keyboard/agent selection. No arbitrary image pixel is assigned an invented depth.

The picker displays actual retained COLMAP observations in the original photograph's verified resized pixel frame, not just projected Gaussian centers. Every displayed/picked anchor is individually checked against its linked track, positive camera depth and <=3 px reprojection. Existing source masks are respected. Photo decoding must finish before adding anchors; source/calibration/generation changes refuse a stale fit. Point IDs stay decimal strings so JavaScript cannot round 64-bit identifiers. At most 6,000 eligible dots appear per photograph.

Authenticated endpoints below share the `/api/splat/jobs/JOB_ID/studio` prefix:

| Endpoint | Contract |
| --- | --- |
| `GET /recovery-support` | `selected_slug`, `expected_generation`, optional `image_id`. Returns verified dots, photo identity, available cameras, current base and an evidence digest. Read-only; no model work. |
| `GET /recovery-support/IMAGE_ID/photo` | Same selection/generation plus the returned `photo_sha256`. Only a linked capture photograph can be served. Hash-bound URLs are private/cacheable; source mismatch refuses. |
| `POST /recoveries` | Existing fields plus optional `support_anchors` below. Omitting it retains automatic fitting. Returns a sealed, unapplied recovery; it does not remove an object or generate unknown pixels. |

```json
{
  "selected_slug": "cardboard-box-2",
  "expected_generation": 18,
  "texture_size": 128,
  "support_anchors": {
    "evidence_sha256": "DIGEST_FROM_CURRENT_INSPECTION",
    "points": [
      {"point_id": "POINT_ID", "image_id": 47, "photo_sha256": "PHOTO_DIGEST"},
      {"point_id": "ANOTHER_POINT_ID", "image_id": 280, "photo_sha256": "OTHER_PHOTO_DIGEST"},
      {"point_id": "THIRD_POINT_ID", "image_id": 280, "photo_sha256": "OTHER_PHOTO_DIGEST"}
    ]
  }
}
```

This is a shape example, not a replayable request: obtain the current generation, real decimal point IDs and SHA-256 values from inspection. A support plane uses three to six distinct, individually verified SfM points; duplicate/near-coincident points (<2 cm apart), insufficient spatial spread, non-horizontal or inconsistent anchors refuse. The anchor-defined least-squares plane is **not** replaced by a denser unrelated plane. Maximum anchor residual is 15 mm; at least 60 eligible nearby tracks must lie within 25 mm of that plane, with <=20 mm RMS. These are regularization checks, not survey tolerances.

The support search expands up to 2.5 m horizontally beyond the selected bounds, excluding captured object bounds. Receipts expose the anchor-to-target distance, point IDs/world coordinates/pixels/reprojection errors, chosen photo hashes, camera models/poses/intrinsics, sources and implementation hashes. The semantic class remains **unclassified support surface**: a human or agent choosing a visible floor does not prove the unseen surface or eliminate depth ambiguity.

Recovery recipe v2 ranks source cameras by tracked support near the **actual texture footprint plus its 8 cm sampling radius**, rather than by all distant plane inliers. It retains at most 24 texture sources, separate complete-group appearance checks, multi-view color agreement and absent geometry for unknown cells. The UI distinguishes total plane-support points from texture-local support points. Different planes/regions and source choices are not an equal-input model-quality comparison.

CPU inspection, photo reads and recovery remain serialized across processes. Read/build admission waits at most three seconds for short contention; it does not bypass the shared slot or enqueue persistent work. The observation query retries only brief admission contention, at most twice. Apply/revert stays in the shared reviewed transaction. Anchored recovery does not relax refined-removal collision requirements, authorize generated background, or label fitted geometry as measured.

`tools/prove-spatial-ui.py --studio-support` exercises the expanded photo picker and anchored recovery on the disposable private fixture. Use an explicitly inspected generation, private `--outputs-root`, staged `--dist`, existing `--playwright-module` and a new report directory. Its rollback guard is restricted to the matching private preview. Mesh-only support inspection is not a captured-removal or complete-background acceptance result.

The completed support proof uses four anchors across `DSCF5611.JPG` and `DSCF5844.JPG`, supports **22.9614%** of the footprint, adds **7,524** triangles and restores exact state/artifact maps at private generation **20**. Recovery `recovery_5bb7fb6bfb9a41ec8352edfa` binds that generation and matches all four browser-candidate artifacts byte-for-byte. The subsequent completion proof below restores generation **22**, making that recovery stale for another edit. Read `support-surface-findings.md` and `fresh-support-recovery-proof.json` in the proposal reports.

## Unknown-region material completion

The anchored recovery panel now exposes **Complete unknown support cells**. This is a provider-neutral creative-material import and review workflow, not an integrated diffusion model or proof of photoreal background reconstruction. It does not start a model, send reference images off-machine, or change the active scene during preparation/import.

1. Build a current recovery using three to six photo-linked support anchors. Automatic strongest-plane recovery remains inspectable but cannot authorize unseen-plane extension.
2. Describe the intended material, permitted inventions and uncertain boundaries, then **Prepare creative completion**. The receipt pins that intent, active generation, recovery, source photographs, calibration, original atlas and observation mask.
3. Download the observed atlas and coverage mask for your separately chosen generation workflow. Teal cells are observed; magenta cells are unknown. Generated output must align to the entire atlas UV layout. The diagnostic mask is not a material, and a generic attractive floor image is not necessarily UV-aligned.
4. Import one opaque, single-frame, square PNG/JPEG, 64–2048 pixels, at most 16 MiB. Record provider and model/source labels. These labels are uploader-reported, not independently verified. Material files retain their original bytes; GIF/WebP, transparency and invalid dimensions are refused.
5. **Preview completed support** places a candidate without hiding the captured object. **Propose removal + completion** additionally requires a current, passing selection-aware collision study for the same object. The shared review/apply/undo transaction remains authoritative; neither upload nor a filled grid constitutes approval.

### Representation and provenance

`backend/background_completion.py` builds two disjoint GLB primitive/material groups on the anchored grid. Observed cells retain the original recovery triangle positions, UV coordinates and embedded atlas bytes. Unknown cells receive newly extrapolated plane triangles and the imported opaque material. Imported pixels corresponding to observed cells are never rendered. The application does not composite over, resample or overwrite the captured atlas.

The source atlas can still have sparse or imperfect evidence, and its geometry is fitted, not directly measured. Extending that plane invents support geometry in previously unknown cells. Seams, wrong material boundaries, perspective misalignment, rug/object continuation and oblique-view coherence remain review concerns. Two-material occupancy is not a photorealism score, evidence-coverage improvement, survey accuracy or navigation verdict.

Each study retains `receipt.json`, original `atlas.png`, diagnostic `support.png` and `observations.npz`. A completed study also retains `result.json`, `candidate.glb`, original `generated.png`/`generated.jpg` and `cell-provenance.npz` with complementary observed/generated masks. GLB primitive/material extras distinguish captured-photo appearance from generated appearance, and fitted geometry from unobserved extension. Portable render/VR-only tags prevent these assets from silently becoming survey inputs. Prepared and imported files are content-addressed and pinned into the scene revision alongside the original recovery dependencies.

Preparation and import share the bounded CPU worker slot and scene write lock. Re-import cannot overwrite an existing candidate. A changed active generation, recovery, source identity or calibration requires new evidence. Historical undo restores its immutable snapshot rather than pretending stale studies can be reused for a fresh edit. A failed partial import may leave a retained candidate without a result; prepare a new study rather than overwriting forensic artifacts.

### Completion API

All paths below are relative to `/api/splat/jobs/{job_id}/studio` and use the existing authentication boundary.

| Endpoint | Contract |
| --- | --- |
| `GET /completions` | Retained studies, stale status, observed/invented fractions and optional result summary. |
| `POST /completions` | JSON `expected_generation`, `recovery_id`, `prompt` (1–2000 nonblank characters); no model launch. |
| `POST /completions/{id}/image` | Multipart `file`, `provider`, `model`; validates material and builds a review-only layered GLB. Source labels are 1–120 nonblank characters. |
| `GET /completions/{id}/artifact?name=atlas.png` | Registered, checksum-checked prepared artifact; no arbitrary path access. |
| `GET /completions/{id}/artifact?name=candidate.glb&result=true` | Registered result artifact. |
| `POST /proposals` | Existing contract with `operation.completion_id`; mutually exclusive with `recovery_id` and `blender_export`. Supports `place`/`replace`, not bare `remove`. |

The application environment needs no Torch, SciPy, Blender or external generator to prepare/import these candidates. Actual image inference is a separate, explicitly selected workflow subject to the existing compute/privacy rules. No model default or local-versus-cloud provider is promoted by this implementation.

`tools/prove-spatial-ui.py --studio-completion` tests the private anchored fixture with conspicuously striped **synthetic software-test material**. It checks retained atlas bytes, material ownership, stale-collision refusal, two-angle inspection, shared apply/reload/undo and mobile overflow. It is deliberately not a generative-model evaluation, captured-object removal or navigation acceptance. Read `completion-findings.md` in the development reports for the actual run status and current private generation before any further mutation.

The synthetic material proof imports **25,244 unknown-cell triangles** alongside **7,524 preserved observed triangles** and applies/reloads/undoes **20 → 21 → 22**. It verifies embedded source-image hashes, two-angle rendering, stale-collision refusal and exact state/artifact-map restoration. That invocation reports a mobile-overflow failure after successful undo; the read-only `--studio-completion-layout` follow-up diagnoses the captured-instance selector and checks its bounded-width correction without another scene mutation. Failed and follow-up receipts are retained separately. That checkpoint ends at `scene_4d11cadefaccfc5e8cb4fb65`, generation **22**. The later compound authored replacement below ends at generation **24**, revision `scene_a705529604585ecccfbe4b9e`, no active proposal. Actual model inference, convincing complete background and full navigation remain separate gates; prepare new evidence against the current pointer rather than replaying these historical IDs.

## Paired reconstruction experiments

`backend/reconstruction_comparison.py` and `tools/reconstruction-comparison.py` implement actual, isolated RGB reconstruction experiments with the installed Splatfacto/DN-Splatter/AGS-Mesh environment. These are private research runs, not new application defaults or automatic training jobs. Existing capture checkpoints and studio revisions remain unchanged.

Preparation preserves every supplied photo, scales intrinsics by exact resized dimensions, carries registered masks, and freezes source/file hashes and independent train/validation/test memberships. Frames with a shared `timestamp_group` stay together; explicit split lists must cover all frames without overlap. Panorama-derived inputs need registered whole-timestamp splitting. Do not strip timestamp lineage when adapting a new capture format.

Seed points need three valid independent training-photo observations. Colors come only from those observations, not from a checkpoint or pre-averaged full-capture PLY colors. Conflicting duplicate observations within one point/image pair are excluded. Existing SfM positions/poses can still incorporate all capture images: the stated scope is held-out **RGB optimization with shared known poses**, not independent pose-solving or measured geometric accuracy. Missing motion masks remain explicit, not silently treated as verified static capture.

### Running a comparison

Use the existing compute gate and isolated environment. A short `--iterations 64 --seconds 240` preparation is compatibility-only and evaluates validation images. Substantive runs use the frozen test split. The current AGS-Mesh normal weight starts only after step 7,000; a shorter run cannot evaluate that normal-regularization stage. The initial substantive recipe uses 10,000 iterations per method and a 1,200-second ceiling per arm.

```bash
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --kill-after=15s 240s \
  .venv/bin/python tools/reconstruction-comparison.py prepare \
  --job /home/rtoony/projects/splatcli/outputs/3d/splat_aea04ab3 \
  --iterations 10000 --seconds 1200 --downscale 2

tools/splatlab-compute-gate.sh --run /usr/bin/timeout --kill-after=15s 1230s \
  /usr/bin/env PYTHONNOUSERSITE=1 \
  /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python \
  tools/reconstruction-comparison.py train --comparison COMPARISON_DIRECTORY \
  --arm splatfacto
```

Run `--arm dn-splatter` and optionally `--arm ags-mesh` **sequentially** against that same directory and receipt. `--attempt 2` retains a second uniquely named arm after a failed attempt without overwriting the first; changes in preparation inputs/budgets require a new study. Changed harness/runtime/initialization cannot be passed off as a matched pair: rerun both required methods with the same configuration rather than comparing incompatible attempts.

```bash
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --kill-after=15s 120s \
  .venv/bin/python tools/reconstruction-comparison.py compare \
  --comparison COMPARISON_DIRECTORY --runs splatfacto dn-splatter ags-mesh
```

The comparator accepts two or three distinct completed methods including one Splatfacto baseline. It verifies equal registered budgets and actual iteration counts, identical initial positions, harness/packages/shared code, complete held-out image membership, exact reference-pixel hashes and retained artifact integrity. It reports candidate-minus-baseline deltas with `needs-review`, never an automatic promotion.

### Outputs and limits

Each arm retains its config, checkpoint, source/code hashes, timing, Gaussian count, peak Torch memory and process RSS, plus every held-out RGB render, undistorted reference image and depth array. Periodic training evaluation is disabled. Final metrics explicitly iterate the registered held-out dataset rather than accepting an empty evaluation loader. Normalized float image caching supports both research and baseline metrics; kernel/cache warmup is separated from measured optimization time.

Evaluation fully populates the lazy image/undistortion cache before selecting the first camera. This keeps intrinsics and reference pixels aligned on distorted captures; empty, mismatched or incomplete caches refuse evaluation. The first retained Bonsai study predates this defensive ordering fix but has zero distortion throughout; its exact original worker and comparison module are retained under `implementation-snapshot/` and match every arm's recorded source hashes.

The current metrics are uncorrected per-view PSNR, full-image SSIM when unmasked, opacity coverage and render time. Opacity is not verified geometric coverage. Geometric-reference error remains null when no independent dense reference exists. Native densification and surface constraints differ by method, so this is an equal-input/equal-iteration method comparison, not a fixed-Gaussian-count or isolated-loss ablation. A single seed, one capture and 10,000 steps do not establish a universally superior method or final convergence.

Read `/home/rtoony/reports/2026-09-06-splatlab-next-phases/reconstruction-comparison-findings.md` for retained pilot failures and current actual outcomes. Private study files live under `data/spatial/reconstruction-comparisons/`. Do not reuse the previously trained full-capture checkpoint to initialize a purported held-out benchmark.

### Reviewable exports

`tools/reconstruction-geometry-review.py` re-verifies the completed comparison and its registered final checkpoints, loads only tensor-safe checkpoint state, and creates a unique `review-*` directory. It exports every Gaussian in checkpoint order as full-SH float32 PLY, without opacity filtering, coefficient truncation or delivery quantization. A row-index map preserves checkpoint correspondence. Invalid shapes, nonfinite parameters, degenerate rotations and unsafe scales refuse export instead of silently dropping rows.

```bash
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --kill-after=15s 240s \
  /usr/bin/env PYTHONNOUSERSITE=1 \
  /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python \
  tools/reconstruction-geometry-review.py --comparison COMPARISON_DIRECTORY \
  --runs splatfacto dn-splatter ags-mesh
```

Open the resulting local `index.html` for all matched-camera reference/method images and PLY links. The static page uses no remote assets, script, server or scene mutations. `review.json` binds the comparison, source checkpoints, output hashes, parameter distributions and retained rendered-depth statistics. Its geometry error remains null: Gaussian extents/anisotropy and positive rendered depth are descriptive values, not measured surface accuracy, visibility confidence or a reliable floater detector. Coordinates remain in the normalized training frame, not established metres. PLY normal fields are zero placeholders; they must not become measured normals or survey evidence. External-viewer image parity and navigable mesh extraction are separate acceptance work.

`tools/prove-reconstruction-review.mjs` checks every retained image decodes, desktop/mobile overflow, page errors and unchanged review receipt using the already installed Playwright module. Run it through the existing bounded gate with `SPATIAL_PROOF_PLAYWRIGHT` pointing to that module; it installs nothing and preserves a separate `browser-proof` receipt.

## Inferred surface comparisons

`backend/reconstruction_surfaces.py` and `tools/reconstruction-surfaces.py` convert the completed comparison's model-predicted training-view depths into reviewable TSDF meshes. This is an explicit inferred-geometry lane, not a survey export or automatic replacement of a visible splat/collider. It uses the already installed Open3D and model environment; no dependency/model download is needed.

Preparation freezes shared crop, voxel, alpha/depth thresholds, coordinate normalization, trained-run identities and extraction-code hashes. All registered training views are fused; no validation/test image enters fusion or vertex coloring. Model config reconstruction must exactly match the registered training YAML before strict checkpoint loading. Runtime package/source and initial-coordinate checks refuse drift. Caching/undistortion precedes camera selection.

```bash
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --kill-after=15s 120s \
  .venv/bin/python tools/reconstruction-surfaces.py prepare \
  --comparison COMPARISON_DIRECTORY --runs splatfacto dn-splatter ags-mesh \
  --voxel 0.01 --seconds 600

tools/splatlab-compute-gate.sh --run /usr/bin/timeout --kill-after=15s 630s \
  /usr/bin/env PYTHONNOUSERSITE=1 \
  /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python \
  tools/reconstruction-surfaces.py build --surface-study SURFACE_STUDY_DIRECTORY \
  --arm splatfacto
```

Run the other registered arms sequentially with the identical surface study. The current recipe bounds raw output at six million triangles/four million vertices, without reducing the source voxel detail or modifying workstation limits. A failed arm retains its samples and failure receipt; successful fusion is separately sealed before mesh extraction. `tools/inspect-surface-fusion.py` can replay retained samples for diagnosis while the producer's frozen code still matches. A changed extraction recipe/code needs a fresh study, not an overwritten result.

```bash
tools/splatlab-compute-gate.sh --run /usr/bin/timeout --kill-after=15s 180s \
  .venv/bin/python tools/reconstruction-surfaces.py compare \
  --surface-study SURFACE_STUDY_DIRECTORY
```

The local `index.html` displays reference images, mesh colors and geometry normals for all held-out cameras. `comparison.json` reports topology, crop-ray coverage, crop-masked photo PSNR, model-depth self-consistency, runtime, memory and artifact size. These crop-masked mesh scores are not the earlier full-image splat scores. Missing mesh hits remain in coverage denominators. No metric is silently relabelled independent geometric accuracy; no successful extraction becomes an accepted collider or full-scene completion.

`surface.ply` preserves double-precision positions, all extracted components and derived RGB8 vertex color, with an in-file render/VR-only tag. `tools/verify-surface-delivery.py --surface-study SURFACE_STUDY_DIRECTORY`, through the gate and isolated environment, independently reloads those PLY files and re-raycasts every registered camera. It checks vertex/triangle counts, surface-hit membership, depth within 1e-5 training units, RGB within one byte and portable survey refusal. This verifies mesh file delivery, not Spark splat packing or moving-camera navigation.

`tools/prove-reconstruction-review.mjs` also supports the surface gallery and its `comparison.json` receipt, decoding all reference/mesh/normal images and checking desktop/mobile layout without mutating the study. It waits for eager image loading before decoding; a third argument such as `browser-proof-02` selects a bounded new attempt directory without overwriting a failed proof. Read `surface-comparison-findings.md` in the development reports for actual outcomes, retained failures and acceptance limits.

## Compound captured replacement

`POST /studio/proposals` accepts an optional `operation.background` with a Blender-backed or registered generated-object `replace`. Previously a replacement could add either the background patch or the replacement object, but not both in the same generation. The companion is a separate named element, not a merged mesh or a general-purpose batch of arbitrary operations:

```json
{
  "expected_generation": 22,
  "instruction": "Review the replacement and background together; preserve observed appearance.",
  "operation": {
    "kind": "replace",
    "selected_slug": "captured-object",
    "slug": "replacement-object",
    "blender_export": "scene-v0007-replacement.glb",
    "selection_collision_id": "collision_CURRENT_24_HEX_DIGITS",
    "background": {
      "slug": "repaired-background",
      "recovery_id": "recovery_CURRENT_24_HEX_DIGITS",
      "label": "Observed background; unknown cells absent"
    }
  }
}
```

This is a contract illustration, not a replayable request. Use actual current IDs/generation. A background may specify **exactly one** `recovery_id` or `completion_id`. It cannot contain a Blender export, path, nested background or another operation. Both assets need distinct, unused slugs. Source-object identity, calibration, current selection-aware clearance and recovery/completion freshness are checked together. A failure cannot publish a partial proposal or activate either half. Apply verifies the entire artifact bundle; undo restores both elements, row visibility, the collision partition and semantic state from one prior revision.

The Studio replacement form lists only current backgrounds for the selected captured object, labels observed-only versus invented material, and attaches current passing local clearance. Choosing no background is still possible and explicitly warns that gaps may remain. Candidate-only inspection shows both new elements, keeps walking/review approval disabled, and does not count as full-context acceptance. The contextual preview remains required for apply.

The typed Blender `transform_object` operation switches to XYZ mode when Euler angles are explicitly supplied. glTF imports can retain quaternion mode, where changing `rotation_euler` alone did not rotate their actual geometry. Inspection now reports effective local rotation and the active rotation mode. An opt-in Blender regression verifies a quaternion-backed rectangular mesh actually changes its exported bounds, rather than checking an inactive Euler property.

`tools/prepare-compound-replacement.py --expected-generation GENERATION --collision-id CURRENT_ID`, run through the compute gate with the installed `dn-splatter-probe` Python, prepares an explicitly **authored** library-chest experiment only on `splat_c0ffee`. It rebuilds the photo-linked recovery, retains a native master, uses typed Blender import/transform/export operations, and verifies every transformed triangle plus contact with the inferred support plane. It applies no revision, runs no image generator and fills no unknown texture. Blender may split/reorder vertices at shading seams; preservation compares face membership through exact master position identities and a bounded 20-micrometre export tolerance, not raw index counts.

`tools/prove-spatial-ui.py --studio-compound-receipt PREPARED_RECEIPT` extends the private refined-removal browser proof with the real replacement form, two-element isolation, moving-camera samples and apply/reload/undo. Supply the inspected `--studio-generation`, private `--outputs-root`, staged `--dist`, installed Playwright module and a new output directory through the compute gate. The rollback guard remains restricted to the matching disposable fixture. These software-rendered samples do not establish adult navigation, complete object recall, generative model quality or 1080p performance.

## Local generated-object masters and replacement

Under `/api/splat/jobs/{job}/studio`:

- `POST /generated-objects` freezes `{expected_generation, selection_review_id, recovery_id, image_id?, seed?}`. It requires finalized current multi-view selection and same-object photo-anchored support. Only accepted, untruncated **fit** views can condition generation. CPU preparation pins exact crop/alpha, photos/masks, target rows, scale, support, recipe and producer hashes; it starts no model.
- `GET /generated-objects` lists retained candidates and stale/result status. `GET /generated-objects/{id}?result=true` exposes the sealed full result; omit `result=true` for prepared inputs. Registered `/artifact?name=NAME&result=BOOL` serves hash-verified immutable blobs, not arbitrary paths.
- In the photo-linked recovery panel, choose a retained candidate to review the masked input and six photographic/master/delivery comparisons. Full native/placed mesh masters, native Gaussian master and model/placement/delivery receipt remain downloadable. Failed, unplaced and stale candidates cannot be proposed. Downloads of historical evidence remain available after undo.

From the repository root, with actual current values assigned:

```bash
.venv/bin/python tools/generated-object.py prepare "$JOB" \
  --review-id "$SELECTION_ID" --recovery-id "$RECOVERY_ID" \
  --expected-generation "$GENERATION"

tools/splatlab-compute-gate.sh --run \
  /usr/bin/timeout --signal=TERM --kill-after=15s 630s \
  /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python \
  tools/generated-object.py build "$JOB" --generated-object-id "$GENERATED_ID"
```

The installed SAM-3D worker runs in its existing separate model environment, with an allowlisted/offline environment and retained weights/config identities. The build refuses absent compute containment before heavy imports, rechecks policy, uses per-study exclusion, a 420-second model timeout and 600-second overall alarm, and terminates model process groups on failure. Keep the external timeout and existing compute/resource coordination. Never infer authorization for unattended generation from a prepared UI record.

All native generated mesh components and Gaussians remain masters. The fitted mesh is a separate artifact: fit-only silhouette initialization and bounded scale/yaw search constrain it to the inferred support; full-triangle mask agreement must reach the frozen admission threshold. Check masks never enter that optimization, but upstream SfM/support can share camera evidence: this is not a wholly independent validation split. A 32k-face QEM delivery is separate again. Render-only generative tags survive exported files; none becomes measured survey input. **The Gaussian decoder frame differs from the mesh frame. Use the separately verified native-frame derivative described next; the mesh transform alone is wrong. Gaussian room-layer transactions remain unimplemented.**

For a completed model stage with a still-current baseline, corrected placement/delivery can reuse verified native masters without repeating model inference:

```bash
.venv/bin/python tools/generated-object.py rederive "$JOB" \
  --generated-object-id "$PARENT_GENERATED_ID"
```

Run `build` on the **new** returned ID through the gate. The parent result, input identities and native masters remain immutable. Changed producer code requires a fresh preparation/derivation. This is not permission to bypass stale generation after an apply/undo; then fresh current evidence is required.

Proposal `operation.generated_object_id` is exclusive with a primary Blender/recovery/completion asset and supports `place` or `replace`. Generated replacement requires selection-aware clearance from the **same refined selection**. Optional `operation.background` contains one current same-object recovery/completion and a distinct unused slug. Studio's “Propose generated replacement + background” attaches the observed recovery and matching clearance automatically. Both assets enter one preview/apply/undo with row visibility, static collision, semantics, source/scale dependencies and full master snapshots. Generated assets remain visible against captured appearance; shared lit/unlit materials preserve vertex colors, and generated static/environment geometry contributes to the collider just like authored additions.

Independent export verification can inspect a historical candidate without changing the active scene:

```bash
tools/splatlab-compute-gate.sh --run \
  /usr/bin/timeout --signal=TERM --kill-after=15s 180s \
  /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python \
  tools/verify-generated-delivery.py --job "$JOB" \
  --generated-object-id "$GENERATED_ID" --output "$NEW_REVIEW_DIRECTORY"
```

This reloads both actual GLB exports, compares all retained fit/check cameras, verifies portable survey refusal and records strict hit/depth/RGB tolerances. It decodes explicitly tagged linear-float color exports before comparing their image-RGB references. It does not infer physical accuracy from master/delivery agreement. The compound private browser proof also accepts a receipt with `generated_object_id`, checks all 19 mesh review images separately from any nested Gaussian evidence, and retains rollback protection. Findings and known failures: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/generated-object-findings.md`.

## Native generated Gaussian frame review

The generated-object master and Gaussian decoder frames differ. The new **SAM-3D-specific** adapter traces the installed mesh-only export rotation and composes it with fitted mesh placement and calibrated world conversion. It transforms every Gaussian's center, scale and covariance orientation; it does not apply only a bbox translation or reinterpret centers as measured surface samples. Input remains strict tagged degree-zero, all-float, binary-little-endian SAM PLY. Higher SH, alternate SDK identities, nonuniform/reflected/sheared frames, invalid covariance and malformed files refuse.

For a retained placed generated object:

```bash
tools/splatlab-compute-gate.sh --run \
  /usr/bin/timeout --signal=TERM --kill-after=15s 210s \
  /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python \
  tools/generated-splat.py --job "$JOB" --generated-object-id "$GENERATED_ID"
```

The 180-second worker derives a new `gaussians_<24 hex>` study beneath that object's `gaussians/` directory, never overwrites an earlier study or starts the generation model, and preserves failed evidence. It freezes parent identities, installed SDK traces, producer identities and per-view tolerances. Actual GPU comparisons render the native asset/native camera and reloaded placed asset/world camera at six views, with depth in matching units. A model's historical parent may be reviewed after undo, but this is **not** permission to apply it to a stale scene. The worker itself never creates or activates a proposal.

The verified 572,960-row example uses ~191 MB peak Torch reserved memory plus driver overhead. Its known-small replay can use the existing per-command `SPLAT_GPU_MANUAL_VRAM_MB=4096` setting without changing coordinator/system policy; do not assume that reservation fits a new workload or model inference. Global compute, Redis, backup and VRAM guards still apply. The first run may compile the installed gsplat extension; cached timing is not comparable to cold JIT timing.

Read-only routes under `/api/splat/jobs/{job}/studio/generated-objects/{object}`:

- `GET /gaussians`: sealed studies, status and per-view metrics.
- `GET /gaussians/{id}`: complete immutable receipt and artifact identities.
- `GET /gaussians/{id}/artifact?name=NAME`: hash-verified registered files only, with private immutable caching.

The Studio native-review panel displays reference/native/placed/mesh/alpha images, downloads and an optional on-demand orbit viewer. The viewer imports the placed PLY directly in **Y-up metres**, without another capture-frame rotation, and switches between the full native splat and separate mesh delivery. Orbit inspection cannot approve a room revision; separate proposal buttons require current generated placement and, for replacement, matching selection clearance. Offscreen primary room rendering pauses to avoid competing with inspection, and resumes when visible.

Generated image-like vertex colors need an explicit display interpretation: glTF `COLOR_0` is linear, whereas the retained model-color bytes reproduce the model's image-space RGB. The `sam-image-srgb-to-gltf-linear-float/v1` export policy stores float32 linear colors with exact source RGB8/alpha round-trip. Original BIN geometry and original model-color exports remain retained; already-converted policy refuses a second conversion. New generated builds emit correctly encoded standard `placed-master.glb`/`delivery.glb` plus `*-model-colors.glb` copies. Historical Gaussian reviews derive a separate `appearance-delivery.glb`, or retain an already encoded delivery. This is not an automatic color-space conversion for arbitrary uploads, textures or measured albedo.

The private read-only browser proof adds `--studio-gaussian-object "$GENERATED_ID"` to `tools/prove-spatial-ui.py`, with the usual current `--studio-generation`, private `--outputs-root`, staged `--dist`, installed Playwright module and fresh `--output` through the compute gate. It decodes all 30 images, verifies full row import, four nonblank two-angle mesh/splat screenshots, canvas disposal, offscreen pause/onscreen resume, mobile width and unchanged active/parent receipts. It does not measure Spark packing fidelity, continuous-room navigation or 1080p performance. Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/generated-gaussian-findings.md`.

## Native appearance in reversible room revisions

A Studio `place` or `replace` operation may pair `generated_object_id` with `gaussians_id`. Native appearance is never a standalone collider. Admission requires the current parent selection/support/placement, exact native-master identity, verified SDK frame, complete camera comparisons within the original tolerances, and replay of **all** placed Gaussian parameters. The explicit color-interpreted mesh fallback must retain the fitted delivery geometry. Failed, historical, incomplete, mismatched or corrupted evidence refuses without changing the active scene.

The revision snapshots both `_world/elements/<slug>.ply` and `<slug>.glb`, all native/placed masters, preparation/result receipts and review artifacts. Its viewer entry declares `files.splat`, `files.glb` and `gaussian_appearance` with row count, `world-y-up-metres`, generated provenance and separate mesh-collision role. The existing authenticated revision artifact route serves both. Captured-row hiding, the selection-aware background collider, optional observed recovery and both generated representations enter one preview/apply transaction. Removing the generated element removes both visible/physical representations, not its retained evidence; undo restores the exact previous bundle and semantics.

The walker loads the complete PLY with an identity transform and refuses missing pairing metadata or a row-count mismatch. Spark handles native appearance separately from the mesh group; the unchanged mesh remains the collider and selection proxy. **Native generated appearance (mesh collision unchanged)** switches to/from the color-correct mesh without rebuilding collision or changing captured membership. Eye visibility and candidate isolation affect both representations; baked/restyled looks use the mesh. Revision cleanup disposes native buffers, stale asynchronous loads refuse attachment, and walker disposal releases the shared Spark renderer. This is static generated-environment support, not dynamic Gaussian deformation or physics-driven splat transforms.

After an undo, old generation-bound studies remain stale even if their source looks unchanged. Rebuild selection, support and clearance. A fresh preparation may explicitly reuse an earlier completed **native inference stage** only when conditioning PNG bytes, seed and model recipe match:

```bash
.venv/bin/python tools/generated-object.py prepare "$JOB" \
  --review-id "$CURRENT_SELECTION" --recovery-id "$CURRENT_SUPPORT" \
  --expected-generation "$GENERATION" --image-id "$FIT_CAMERA" \
  --reuse-generated-object-id "$RETAINED_GENERATED_ID"
```

Preparation starts no model. The normal gated `build` then verifies the immutable parent, retains its native mesh/PLY, and recomputes current placement, delivery and multi-view evidence. It does **not** copy an old pose or approve a historical selection. No reuse occurs silently; a changed conditioning image refuses. The current native derivative still needs the separate gated `generated-splat.py` check and normal human review/apply step.

The private compound browser proof accepts `gaussians_id`/`gaussians` in its prepared transaction receipt. It checks actual row import, identity frame, exclusive mesh/splat visibility, mesh-only switching with unchanged collision, two fixed and moving views, candidate isolation, apply/reload, paired removal/reload, paired undo and baseline undo. A rollback list is limited to exact private revisions/generations created by that proof. Software WebGL images and local inferred clearance do not establish full renderer parity, room navigation, physical accuracy or an FPS target.

If a private proof times out **after activation and owned cleanup**, `--studio-native-resume PREVIOUS_PROOF_DIRECTORY` can finish the remaining checks without inventing fresh authorization for stale studies. It requires the original `--studio-compound-receipt`, the exact current cleanup generation, unchanged baseline state and the previously activated revision in ancestry. It restores that historical revision through the UI, proposes a current paired removal, tests reload and both undos, and retains the failed first attempt. Its software-browser budget is 720 seconds; the outer compute-gated timeout must leave time for cleanup. This narrowly scoped proof option is not a production stale-proposal bypass or a general workload-timeout increase.

`--browser-gpu` uses the installed full Chromium with ANGLE/Vulkan instead of SwiftShader. It refuses execution outside the existing compute gate, saves `browser-renderer.json` and requires actual NVIDIA WebGL identity before scene mutation. The native proof waits for outstanding sorts and prepares an initial render plus subsequent update/render before matched-camera readback; awaiting `SparkRenderer.update()` alone does not guarantee a settled first display. Per-frame JSON retains camera, renderer and scene-splat transforms. No row decimation or looser image tolerance is used. These compact deterministic captures are not a continuous 1080p performance benchmark.

## Explicit overnight cloud experiments

`tools/cloud-experiment.py` is a private, short-lived experiment runner, not a public generation endpoint or a recurring authorization. One fixed ledger in `data/spatial/cloud-overnight-2026-09-07/spend.sqlite3` admits at most **$20 aggregate** under the operator's 2026-09-07 overnight grant. Admission ends at 13:40:25 UTC, before the 13:55:25 UTC workstation handoff. `initialize` is idempotent only for the same cap/deadline; it cannot increase, extend or reset an existing authorization. `status` shows retained conservative reservations separately from provider usage metadata.

Gemini, Z.ai and Kimi adapters use existing Vault injection in memory, fixed endpoints, bounded prompts/PNG crops, current model-specific token limits and a time-stamped rate card. Paid calls reserve twice the maximum quoted tokens plus $0.02 before submission. Failures and successes both retain their complete reservation; there are no automatic retries, routing fallbacks or tool/search charges. The ledger does not control unrelated account activity. Only explicitly staged input basenames are accepted by the CLI. New outputs retain prompt/reference hashes, usage, provider/model identity and private review-only artifacts; they never automatically modify a scene or become geometric evidence.

Prefer installed local models when adequate and higher-quality cloud models when they measurably help. Keep capture/intent/evidence/master/delivery contracts independent of a specific provider so future local models can replace cloud stages. Gemini 3.8 Flash, GLM-5.3/Flash and Kimi K3 provide reasoning/inspection; an image-output model is needed for generated texture, and neither text nor texture implies measured 3D geometry. Local SAM-3D remains the verified native mesh/Gaussian baseline. The existing provider-neutral completion import preserves observed cells independently of the generated image provider. New defaults require same-case visual/coherence evidence, not version numbers or a one-question model smoke test.

Read `/home/rtoony/reports/2026-09-06-splatlab-next-phases/cloud-model-and-budget-findings.md` and `extended-work-window.md` before using this runner. Do not reuse the grant after expiry, enable it on a public route, create a new account or repair provider billing without fresh direction.

Native Gemini image output may be **JPEG or PNG**, regardless of PNG input. The runner verifies the returned format/dimensions and preserves its original bytes, skipping thought-only content. `collect --request-id EXISTING_ID` validates a completed saved response against its ledger hash and extracts artifacts locally, with no Vault/API call or new reservation. It preserves the initial receipt and writes `collection.json`; it refuses changed retained artifacts instead of silently overwriting them.

New completion results include advisory `material_diagnostics`: nearest cell-center image-sRGB disagreement on observed texels, observed-side boundary disagreement, and protected versus generated-only boundary jumps. Studio displays these in expandable diagnostics. They do not modify pixels, classify hidden content as measured, or automatically select a model; real material edges can produce high jumps.

`--studio-completion-comparison COMPARISON_JSON` runs the private two-candidate browser proof with the normal generation, outputs-root, staged-dist and Playwright options. Its prepared manifest pins two imported completions plus fresh selection/clearance. It previews each through the UI, captures fixed/interpolated/isolated/collision views and tests one apply/reload/undo transaction. GPU mode still requires the compute gate. The 720-second inner budget needs an outer timeout with cleanup room. Direct canvas PNG encoding avoids page/compositor rounded-corner contamination while retaining exact pixel equality. Separate local artifact verification is required because the revision-detail API does not expose artifact maps.

The actual Flash/Pro floor comparison and retained failures are documented in `/home/rtoony/reports/2026-09-06-splatlab-next-phases/background-model-findings.md`. Same-case model generation and reversible scene integration work; residual captured content, source occluders, material seams, full removal and navigation remain separate acceptance gates.

## Optional inferred-depth background qualification

An original, current background recovery may be separately qualified against the captured Gaussian model. This is an agent-operated local evidence stage, not automatic semantic floor recognition, measured visibility or an image-generation request. Preserve the parent; each derivation is a new recovery. Existing source-view consensus, feature support and fit/check separation remain required.

```bash
.venv/bin/python tools/background-visibility.py prepare JOB RECOVERY_ID --expected-generation GENERATION
/home/rtoony/projects/splatlab/tools/splatlab-compute-gate.sh --run timeout 240 /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python backend/mesh/background_visibility_render.py JOB VISIBILITY_ID
.venv/bin/python tools/background-visibility.py derive JOB VISIBILITY_ID --expected-generation GENERATION
```

The worker requires alpha≥0.85 and expected optical-depth discrepancy≤0.025m+1% of plane depth at all four neighboring rendered pixels. Maximum render dimension is 960px. Prepared cameras normalize uniform scale and retain metric calibration; texture grid centers, policy and model/source identities are sealed. Each camera stores raw qualification evidence, not only a boolean verdict. Derivation rechecks hashes, exact plane/grid/camera membership and numerical decisions; the 28 camera artifacts plus preparation/result/grid travel with the new recovery. Stale, altered, repeatedly filtered or incomplete evidence is refused. Existing GPU policy, admission and resource controls still apply.

Studio labels depth-qualified entries and shows parent/new coverage, rejected/newly supported cells and paired delivered-PNG errors on identical check pixels. Reconsensus can admit a previously unsupported cell; do not assume a strict subset. Check views never provide texture. The model itself uses the capture, so these are not independent geometry measurements. Inferred depth or the fitted plane can be wrong; mixture depth and thin objects remain limitations. No unknown texture or geometry is invented by this step.

`tools/prove-spatial-ui.py --studio-visibility RECOVERY_ID` adds an unactivated private A/B proof to the normal staged-dist, private-job, generation, outputs-root, Playwright and output flags. GPU mode requires the compute gate; allow an outer 420-second timeout for the 360-second inner budget. It previews parent/qualified GLBs in captured context and exact matched isolated poses, verifies texture decoding and preview collision, and reloads the exact baseline without applying either proposal. Findings and commands are private development evidence, not deployment or asset acceptance.

## Read-only baseline performance

The private `--studio-performance --browser-gpu` proof uses the actual `WorldWalker.frame()` path, ordinary Spark automatic updates, fixed1920×1080/DPR1 rendering,60 warm-up frames and360 recorded animation-frame intervals per stationary/moving camera scenario. It retains displayed-row counts, start/middle/end poses, JS timings and optional disjoint-checked GPU timer queries. GPU queries cover the synchronous frame, not later asynchronous Spark preparation. Existing LoD/options remain unchanged; actual displayed counts are reported rather than inferred from loaded input size.

Use the existing compute gate and normal private-job, generation, outputs-root, staged-dist, Playwright and output flags; outer timeout420 seconds accommodates the360-second wrapper. This mode neither proposes nor activates an edit. It checks baseline/calibration identity and refuses changed active state. Results are scoped headless timing measurements, not universal FPS, end-user display latency, native-replacement performance or walking/collision acceptance. First findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/spatial-performance-findings.md`.

## Captured structural evidence

The Studio **Captured wall evidence** panel reviews retained original photographs, local wall masks and bidirectionally verified SfM observations. It does not cut captured geometry or activate a proposed wall. Prepare 4–12 distinct solved photo IDs, with every fourth photo reserved for checks before fitting:

```bash
.venv/bin/python tools/structural-surfaces.py prepare JOB --expected-generation GENERATION --image-id FIRST --image-id SECOND --image-id THIRD --image-id FOURTH
/home/rtoony/projects/splatlab/tools/splatlab-compute-gate.sh --run timeout 270 .venv/bin/python tools/structural-surfaces.py masks JOB --structure-id STRUCTURE_ID
.venv/bin/python tools/structural-surfaces.py fit JOB --structure-id STRUCTURE_ID
```

Preparation binds original source/camera/scale identities and immutable staged images to the active revision. Masks require the exact wall prompt, dimensions, scores and photo set. Verified tracks need two positive fit views, at least3° separation and a2:1 positive/negative vote ratio. Deterministic vertical fits reject insufficient or line-like support. Check semantics never establish membership. Outputs remain sealed and cannot be overwritten; prepare a new study for another attempt. Stale studies remain reviewable but cannot establish current geometry.

The read-only API is `GET /api/splat/jobs/{job_id}/studio/structural-surfaces`, with checked artifact access at `/{structure_id}/artifact?name=...&stage=receipt|masks|result`. No API request starts the GPU worker. `--studio-structure STRUCTURE_ID` selects the private structural UI proof in `tools/prove-spatial-ui.py`; use the usual staged dist, private job/generation, outputs-root, Playwright, GPU and output flags through the compute gate.

`structural_surfaces.projected_depth_comparison` checks candidate planes against inferred samples along the same camera rays, using projected intersections rather than a prior point's tangential coordinates. Returned signed optical-depth errors are in metres. Missing/parallel/behind-camera samples are excluded; a plane behind the prior can be occluded. These are inferred-depth consistency diagnostics, not independent measurements or automatic acceptance gates.

The first real study has only two multi-view wall points and no qualifying plane despite convincing semantic masks. Installed MoGe and MASt3R priors remain uncertain; no geometry is promoted. Findings, model/runtime identities, errors and the next typed portal/connected-room direction: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/captured-wall-findings.md`.

## Paired architectural portal prototype

`backend/architectural_edits.py` defines a bounded authored/inferred doorway frame, connected room and explicitly included captured-prop removals. This is not measured-wall reconstruction. CPU preparation binds the current generation, calibration, original sources, selection membership and immutable geometry. The Studio **Doorway and connected room** panel prepares studies without launching compute, shows retained local gates and previews only a fresh passing result through the normal proposal transaction.

The specification requires `origin` (world Y-up XYZ metres), `yaw_degrees` (zero outward +Z, positive toward +X), `opening_width`, `opening_height`, `cut_depth`, `room_width`, `room_depth`, `room_height` and `wall_thickness`. A doorway preserves a 1.5cm threshold lip. All dimensions must satisfy the bounded contract; unknown fields refuse rather than silently alter geometry.

After a fresh operator-authorized compute window, the manual sequence is:

```bash
.venv/bin/python tools/architectural-edits.py prepare JOB --spec SPEC_JSON --instruction 'Describe the authored doorway and connected room' --expected-generation GENERATION
SPLAT_GPU_MANUAL_VRAM_MB=1024 /home/rtoony/projects/splatlab/tools/splatlab-compute-gate.sh --run timeout 180 env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= /home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python backend/mesh/architectural_edit_build.py JOB ARCHITECTURE_ID
.venv/bin/python tools/architectural-edits.py inspect JOB --architecture-id ARCHITECTURE_ID
```

The second command is explicitly CPU-only but still uses containment, resource admission and backup exclusion. It is not a bypass or an automatic queue. Include `--replace-slug CAPTURED_PROP` during preparation only when that removal is explicitly intended; at most three captured props may be included, and the builder must prove their retained mesh vertices and selected centers fit the cut volume. Other named objects remain protected. Current code refuses refined captured-instance partitions, measured/painted surfaces and native generated-Gaussian layers until their adapters exist.

API: `POST /api/splat/jobs/{job_id}/studio/architectural-edits` accepts `spec`, `instruction`, `expected_generation` and optional `replaced_slugs`. `GET` lists retained studies; `/{architecture_id}/artifact?name=...&result=true|false` verifies artifact identity. The regular proposal endpoint accepts `kind: "extend-room"`, `architecture_id`, `slug` and optional `label`; it does not allow independent placement/removal fields to mix into this operation.

The separately retained result must pass the ten closed-solid, bounded-removal, portal, preservation, floor, capsule, protected-element and containment gates. New `cut-and-authored-room/v1` preparations additionally require `authored_room_interior_clear`, an exact retained-solid intersection with the room interior inset 0.1 mm from its surfaces. No captured intersection triangles may remain: appearance masking must not hide a collision obstacle. Historical `cut-only/v1` retains its ten-gate contract. Room, captured collider, clipping frames, row visibility and semantics activate and restore together. Later mesh placements may not obstruct the portal; native Gaussian placements refuse pending footprint clearance. Deleting only the connected room refuses: restore its paired revision.

New preparations use `shared-entry-footprint/v1`: the full 0.6 m centered approach fans into three interior lanes with at most 2 cm horizontal sample spacing. The retained 129-ray spherical footprint records effective support separately from the center ray; missing center floor, excessive lift or floor jumps refuse. The 1.70 m height / 0.22 m radius sampled body, 3 cm clearance margin, 8 cm floor tolerance and 4 cm adjacent-step limit remain explicit. Historical `parallel-center-ray/v1` receipts retain their original procedure and are not relabelled passing. Preparation, install, revision and browser checks bind the method and actual route/floor arrays. The UI explains the method and refuses unknown methods.

Spark clipping uses displayed per-fragment depth, and captured mesh materials receive a matching world-space volume discard. Original captured geometry/PLY files are not rewritten. Actual NVIDIA cut/unclipped/reclipped comparisons verify both legacy cut-only shader adapters. New `cut-and-authored-room/v1` source binds a second dimensioned room envelope, starting 5 mm above the floor without widening the vestibule or changing its 1.5 cm cut lip. Both adapters share the versioned bounds. Named selected Gaussian centers in both regions are protected; unnamed objects and full footprints still need review. Explicit authored-room lighting leaves captured appearance unlit and retains clipping across material changes. The new product recipe is privately built, not yet GPU-verified. Portable cut export, full Gaussian-footprint preservation, drag-axis handles and whole-scene navigation remain unimplemented. The worker's sampled corridor is not a global navmesh.

The September 7 renewed compute window permits current contained experiments. The fresh shared-entry candidate passes ten local gates and actual keyboard-driven rendered traversal of about 7.1 m in both directions at both 1.70m/0.22m and 2.02m/0.32m height/radius profiles. Its paired apply/reload/undo restores exact state, artifacts and fixed-view PNGs at generation48. A post-run audit bug makes the original command exit 1; the corrected independent re-audit passes on retained evidence. Room appearance and unnamed-object preservation still need work, so this is not goal completion. Never reuse/re-sign the now-stale generation46 receipt. Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/shared-entry-rendered-traversal-findings.md`.

For read-only visual diagnosis, `SPATIAL_INSPECT_ARCHITECTURE_PROOF=ABSOLUTE_RETAINED_PROOF_DIRECTORY` with the private proof runner's `--studio-inspect --browser-gpu` mode loads the retained historical preview directly without a new proposal or activation. It requires the exact private job/root, a fresh output directory and no mixed modes. Five in-memory variants compare original appearance, lighting, captured appearance disabled, added room-volume clipping, and lighting plus clipping. These variants do not modify the architectural recipe or authorize broader appearance removal; their saved images and unchanged active-state check are diagnostic evidence only. Launch through the existing compute gate.

The separate `--studio-architecture-preview` mode requires an explicit `--studio-architecture` ID. It performs shader/guide and fixed-body traversal checks, then dismisses the preview without activation and audits unchanged generation/state/artifacts. It writes `architectural-preview-proof.json` and `architectural-preview-audit.json`, not a full apply/reload/undo claim. On September 8 at 14:10 UTC this mode is source-ready but unexecuted: live typed policy denies SplatLab CUDA and backup exclusion blocks heavy work. Manifold coordinates CPU/software rendering through the Nexus inbox. Fresh GPU policy/window authorization and protected-object review remain separate prerequisites. Current findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/room-envelope-and-coordination-findings.md`.

Room-method worker source also records `metrics.capture_impact` with basis `captured-gaussian-centers/v1`: unique cut, room, overlap and union counts, partitioned into protected, explicitly included and unassigned centers. `geometry-probes.npz` retains each sorted membership as `appearance_NAME_rows`. Counts address the pinned baseline PLY, not object identities, newly visible content or exact per-fragment clipping. Unassigned centers may include unnamed furniture; they are not approved removals. The UI shows missing historical impact as unavailable and refuses inconsistent optional counts. This follow-up is source/unit-tested, not yet exercised by a fresh real worker or staged frontend build.

### Non-destructive positioning guide

The same panel now offers **Show positioning guide**, **Pick threshold in scene** and **Aim outward in scene**. An amber dashed box represents the proposed cut, cyan the room interior/outward direction, and green the bridge floor. These X-ray guides show through existing occluders and are explicitly not edited appearance, collision or measured structure. Numeric fields update the guide without preparation or activation.

Pick the nearest upward-facing collision surface for the threshold, then pick direction on that exact height plane; Esc finishes picking. Floor semantics are not inferred by a normal test: a table top can be upward-facing, so inspect height and dimensions before preparing. Invalid, behind-camera, parallel and excessively distant picks refuse without moving the draft. CSS viewport coordinates, not physical canvas pixels, define the camera ray.

Draft visibility binds job/revision/generation and excludes proposals, selection inspection, changed sources and non-metre scenes. Walking, interaction and live scale shortcuts are blocked while it is shown; free-camera inspection and ordinary unlocked typing remain available. Guides own their geometry/materials and pointer/keyboard listeners, and cleanup does not alter existing colliders, registered elements or revision artifacts. Actual NVIDIA guide render/hide and mobile layout now pass; complete two-click/touch placement acceptance remains pending. Original design: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/architectural-positioning-findings.md`; updated evidence is in the shared-entry findings.

### Explicit fixed-size Studio walking

Studio starts in **Free camera / inspection** mode. On a current metre-calibrated revision, **Explicit player dimensions** sets total height and radius; the default body is 2.02m tall with a 0.32m radius. Height includes the top sphere, unlike the internal eye-height parameter. Changing dimensions, calibration, collision geometry or revision stops walking and requires admission again. Other world viewers retain their legacy sizing behavior.

**Walking / collision** searches vertically below the current viewpoint for an upward-facing surface, checks support and the complete capsule against the actual collision BVH, and refuses an obstructed or unsupported start. It does not use the source-point height percentile as a ceiling or silently reduce the requested body. Position adjustment is limited to initial standing admission, not route steering. Respawn stays in inspection mode, and calibrated Studio scale cannot be changed with bracket shortcuts. **Explore** requests real mouse capture; errors explain how to retry. Escape explicitly releases controls and held movement keys.

`tools/prove-spatial-ui.py --studio-walking --browser-gpu` is a read-only private baseline proof. It requires the exact private `splat_c0ffee` outputs root, current generation, an installed Playwright module, staged build and new evidence directory. Launch through the compute gate with a 420-second outer timeout; the browser budget is 360 seconds. Browser API writes are rejected. The runner exercises explicit dimensions, real admission/refusal, retained architectural failure guidance, actual RAF-rendered keyboard movement, Escape, invalid dimensions, exact view restoration and 390px controls. An independent local audit rehashes sources/artifacts and validates four PNGs plus fixed-body movement samples.

The actual NVIDIA proof in `fixed-walking-gpu-proof-05` passes a **0.200m walk with the explicitly selected 1.70m-height / 0.22m-radius body**, 24 rendered samples, unchanged generation46 and byte-identical baseline/restored PNGs. This is **not** full-default-body doorway traversal, repaired-floor acceptance, an architectural transaction or completion of the connected-space goal. Earlier failed attempts remain retained.

### Paired architectural transaction evidence

After a fresh explicit compute window and successful current geometry build, `tools/prove-spatial-ui.py --studio-architecture ARCHITECTURE_ID --browser-gpu` runs the private architectural proof with the normal staged-dist, Playwright, generation, exact private outputs-root and fresh output-directory arguments. Other proof modes cannot be mixed into this one. The inner browser budget is 720 seconds; use the compute gate and an outer 780-second timeout for startup, local audit and cleanup. This runner applies and restores one private revision; it is not a read-only public-site check or operator acceptance.

The runner pins all input artifacts, verifies real Spark/mesh shader A/B, the room/clipping frame, two capsule profiles in both directions, and exact apply/reload/undo. Capsule traces use the actual solver at 120Hz/0.5m/s; they reject auto-shortening and teleport-based substitutes. Worker dimensions are 0.22m radius/1.70m total height; viewer-default dimensions are 0.32m radius/2.02m total height. These are explicit capsule extents, not an interchangeable eye-height label. The trace does not render every substep or establish navigation FPS.

`architectural-input-audit.json` freezes the baseline before mutation. `private-rollback-checkpoint.json` limits failed-run cleanup to owned revisions. Only after successful browser assertions does `architectural-restoration-audit.json` independently verify all original artifact identities, bound capture hashes, 13 saved PNGs and four trace files. Missing frames, wrong dimensions, blank views, altered hashes or contradictory pixel-undo claims fail the audit. A prior report cannot be reused because the output directory must be new.

CPU contract tests: `node --test tools/test-architectural-proof.mjs`; local artifact/CLI tests: `pytest -q backend/tests/test_architectural_proof.py`; actual synthetic capsule integration is included in `frontend/src/lib/world-walker.test.ts`. These have run without a GPU. The captured-scene browser runner has **not** run. Detailed scope, evidence and the fresh-run command: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/architectural-acceptance-harness-findings.md`.
