# Splat Lab — standalone app (extraction from portal)

> Source of truth for this build. Read before resuming; update after every step.

## Goal
Extract Splat Lab from the Nexus portal into its own standalone app at
**splatlab.roonytoony.dev**, with a fresh improved GUI. **PHASED** (RToony's call):
- Phase 1 (NOW): standalone app (own frontend + auth) reusing the proven splat
  backend in-place via a proxy. Zero risk to the GPU arbiter / TRELLIS / pipeline.
- Phase 2: extract the backend into its own service + cross-process GPU lock (Redis).
- Phase 3: retire /splat from the portal (redirect splat.roonytoony.dev -> splatlab).

## Architecture (Phase 1)
- `backend/main.py` — FastAPI on :3416. Auth (PORTAL_TOKEN -> signed
  `splatlab_session` cookie). Streaming reverse-proxy of `/api/*` and
  `/supersplat/*` -> portal 127.0.0.1:3300 with the portal bearer injected
  server-side. Serves the SPA from `frontend/dist`.
- Frontend: `frontend/` (Vite React TS) — TODO.
- Port 3416 (verified free). Tunnel nexus-ai. Token reused = PORTAL_TOKEN.

## Phase 1 — DONE & LIVE (2026-06-28), commit 8c6f204
- [x] Backend (backend/main.py): auth + streaming reverse-proxy to portal :3300.
      Verified: /healthz, login 303, proxied /api/splat/status (8 jobs), 401 unauth.
- [x] Frontend (frontend/, Vite/React): improved GUI shipped — stage timeline (Q1),
      GPU-queue banner (Q3), humanized stages (Q2), results gallery + featured
      viewer (M1), download-format menu (Q5), Transfers picker + refresh, quality
      presets. Routes: / and /view/:jobId. Lean (no portal hooks). Built clean.
- [x] Deployed: splatlab.service (systemd --user, vault-injected PORTAL_TOKEN,
      enabled+active on :3416). apps-registry/apps/splatlab.toml (published, protected;
      added to protected-hostnames.txt). cloudflared config generated+synced; DNS
      CNAME added. Manifest add-app + log.
- [x] Verified LIVE: https://splatlab.roonytoony.dev/healthz -> 200; root -> 303
      login; headless render of the public URL shows the full GUI. Portal /splat
      untouched and still serving.

## Phase 2 — IN PROGRESS
- [x] **Keystone: cross-process Redis GPU arbiter** (`backend/gpu_arbiter.py`).
      Drop-in for the portal's arbiter (HEAVY_GPU_LOCK async-with + .locked(),
      set/clear/holder_info, gpu_status/evict/acquire_gpu). Redis SET-NX lock +
      TTL(45s)+heartbeat(15s); holder in a Redis hash. **FAIL-OPEN**: Redis down →
      degrades to in-process asyncio.Lock + local holder (= old behavior), never
      deadlocks. redis-py installed in BOTH venvs (portal + splatlab).
      VERIFIED in isolation: 2-process mutual exclusion (no overlap); fail-open on
      dead Redis (no hang); locked()/holder_info correct. Test keys cleaned.
- [x] **Ported splat pipeline -> splatlab backend.** `backend/splat_route.py`
      (copy of portal splat.py; swapped to local `gpu_arbiter`, `operator_audit`
      stub, SPLAT_ROOT=/home/rtoony/projects/splatcli). `main.py` mounts the router
      at /api/splat (auth-gated via require_auth dep) + runs migrate/cleanup on
      lifespan; /api proxy REMOVED (splatlab owns it now), /supersplat still proxied.
      splatlab.service injects "Rtoony Portal" + "Infrastructure" (REDIS_PASSWORD).
      → The FRONTEND is implicitly cut over: it calls same-origin /api/splat which
        now hits splatlab's own backend (no code change needed).
- [x] **Ran a REAL job through splatlab's own backend** (splat_bf25300429, 128-frame
      processed dataset, 3000-iter): train->export->compress->webopt all completed;
      produced splat.ply(65M)+spz(4.3M)+web.ply(17.9M); **Redis GPU lock taken
      (holder lane=splat) during, released+cleared after**. Job visible in status,
      preview_available. (Smoke scene left in the gallery — RToony can delete.)
- [x] **Portal arbiter SWAPPED to Redis** (portal commit 5cdcb2c, pushed main).
      Backed up old -> server/lib/gpu_arbiter.py.bak.inprocess.* (one-cp rollback).
      Hardened per a 5-agent adversarial swarm (verdict fix-first): TTL 45->90s,
      cancel-safe __aenter__/__aexit__ (no local-lock leak), socket_timeout 0.5,
      broadened excepts. Re-verified: 2-proc mutual exclusion, fail-open, cancel-
      mid-acquire releases local lock.
      Post-swap gates ALL PASS: 87 vars (no stale-BW 0-var), REDIS_PASSWORD present,
      /api/3d/queue 200 (TRELLIS alive), and COORDINATION ENGAGED — external Redis
      lock flips the portal's gpu.locked to True (reads the shared lock, not per-proc).
- [x] Frontend cutover: done (splatlab calls its own /api/splat). All three lanes
      (splatlab-splat, portal-splat, portal-TRELLIS) now serialize on the 5090 via
      the shared Redis lock. **The Phase-1->2 coordination gap is CLOSED.**

## PHASE 2 COMPLETE (2026-06-29).
## PHASE 3 COMPLETE (2026-06-29, portal commit 716f3be pushed). EXTRACTION DONE.
- [x] splat.roonytoony.dev -> 307 splatlab (in auth_middleware, PRE-auth, so old
      bookmarks skip the portal login wall). portal /splat + /splat/view -> client
      redirect to splatlab (deep link preserved). Sidebar "Splat Lab" -> external.
      Launch Bay: splat="moved", splatlab=the studio. Verified all; portal+splatlab
      unaffected. Portal splat.py backend left DORMANT (still coordinated via the
      shared Redis arbiter); deleting it is optional cleanup.

## GUI backlog
- [x] Capture confidence (commit 78cd649): Customize iterations slider + live time
      estimate; preflight summary card; engine-ready gate on Create.
- [x] Retry-with-params: Re-run + ↑Quality (2x) on scene cards (re-POST /train with
      the job's params). Standard scenes faithful; 360 sub-params (images_per_equirect
      /crop_bottom/insv_fov) not persisted on SplatJob meta so a 360 re-run uses
      defaults — fine for standard, note for 360.
- [x] Scene pin + two-click delete on gallery cards (commit 477d2c8); per-scene
      color tint so cards are distinguishable; 30k-iters badge.
- [x] Cleanups: smoke scene splat_bf25300429 deleted; Launch Bay deduped (portal
      commit ecc8e13: projects.py hides superseded "splat", canonical card ->
      splatlab); redis declared in portal pyproject.
- [x] Real gallery thumbnails — DONE (different approach than the deferred ones).
      `backend/thumb.py`: a point-cloud projection of the splat's .ply, sampled by
      SEEKING (CPU-only, ~50ms even on millions of points), colored by SH-DC, drawn
      with Pillow, cached to _preview/thumb.webp. Endpoint GET /api/splat/jobs/{id}/
      thumbnail (auth-gated, off-thread). SceneCard shows it with the per-scene
      gradient+icon as the fallback (scenes without web.ply). Pillow added to venv +
      requirements. (Avoided the GPU-rasterizer / preserveDrawingBuffer paths.)
- [ ] Optional: delete portal dormant splat.py + splat*.tsx (harmless dead code,
      cross-referenced — risky to remove unattended).
- [ ] optional cleanup: delete the portal's now-dormant splat.py + splat*.tsx;
      declare `redis` in portal pyproject deps; dedupe the splat/splatlab Launch
      Bay cards; delete the splat_bf25300429 smoke scene if unwanted.
- [ ] Minor: declare `redis` in portal pyproject.toml deps (installed in venv now,
      not yet in the manifest — matters only on a clean rebuild).
- [ ] Phase 3: redirect splat.roonytoony.dev -> splatlab; remove the /splat page +
      nav entry from the portal (leave a redirect). Continue GUI backlog: capture-
      confidence (M3 preset sliders, M5 preflight validation), real gallery
      thumbnails, retry-with-params (M2, needs 360 sub-params persisted on SplatJob).

## Invariants (do NOT break)
- Do NOT touch the portal's gpu_arbiter / three_d.py / splat.py pipeline in Phase 1.
- Portal /splat keeps working until Phase 3.
- Reuse PORTAL_TOKEN; pull via vault (nexus-svc-inject), never write to disk.
- PROCESS MGMT: never broad-pkill `uvicorn backend.main:app` — many Nexus apps
  share that cmdline (nexus-vicinity :3404, etc.). Kill by exact port/cmdline only.

## CAPTURE RELIABILITY — "every capture just works" (2026-06-30, commit e560c41)
- [x] A1 registration GATE: splat_route.py — after `process`, ratio = registered
      (transforms.json frames) / extracted (processed/images); < MIN_REGISTRATION_RATIO
      (0.30) → fail fast pre-train with an actionable message (no GPU wasted).
      Additive/reversible/parse-safe; default COLMAP path byte-for-byte unchanged.
      Frontend: amber failure card surfaces the message + a "Retry with global SfM" button.
      VERIFIED fires on backyard (2/311=0.6%), passes good jobs (128/128).
- [x] A3 global-SfM rescue: opt-in `sfm_backend="glomap"` → glomap_sfm stage runs
      COLMAP 4.x feature_extractor + sequential_matcher + global_mapper, then
      ns-process-data --skip-colmap --colmap-model-path ../colmap/sparse/0 (RELATIVE).
      **DEP: conda env `colmap4` = COLMAP 4.1.0 built from source (CUDA 12.8, sm_120,
      -DCMAKE_CUDA_ARCHITECTURES=120). Binary: ~/miniconda3/envs/colmap4/bin/colmap.
      ISOLATED — the working `colmap` (3.11.1) + `splatops` envs are untouched.**
      Build needed 4 patches (CHOLMOD target, glog version macros, cuda_runtime.h
      include, Eigen config-mode) — see workflow output if rebuilding.
      PROVEN end-to-end: global_mapper 311/311 on backyard (vs 2); nerfstudio 1.1.5
      reads the 4.x model → transforms.json 311 frames; 4.x renamed the GPU flags
      (SiftExtraction→FeatureExtraction.use_gpu, SiftMatching→FeatureMatching.use_gpu).
- [x] **AUTO-FALLBACK (zero-click "just works")** — the A1 gate no longer just fails
      on low reg: it climbs the solver chain `SFM_ESCALATION = [colmap, glomap, mast3r]`
      automatically. `_maybe_escalate_sfm` rebuilds the next available solver's SfM
      pre-stage + a uniquely-named `reprocess<n>` and injects them into the live
      stages_planned ahead of train; the loop (enumerate over the live list) picks them
      up next. Only fails the job with guidance once the chain is exhausted. Manual
      "Retry with global SfM" button preserved; default COLMAP success path byte-for-byte
      unchanged (verified: colmap planner emits only `['process']`, no `--skip-colmap`).
      Loop-safety (no solver twice via sfm_tried; reroute cap = len(chain); equirect/
      dataset excluded via sfm_context=None) — 22/22 unit checks PASS.
- [x] **Phase B (pose-free MASt3R-SfM fallback) — WIRED & TESTED.** Terminal rung of the
      chain. `mast3r_sfm` stage runs the runner (`~/tools/mast3r-spike/run_mast3r_sfm.py`,
      ViT-Large dense matching → poses.npz/points3D.npz) then a DIRECT converter
      (`mast3r_to_nerfstudio.py`) that reproduces nerfstudio 1.1.5's colmap_to_json
      convention (proven identical to 4.4e-16) → writes transforms.json + images/ +
      sparse_pc.ply straight into processed_dir (NO ns-process-data). END-TO-END TESTED:
      39 backyard frames → 39/39 finite poses, 88.6s, 3.46GB peak → converter → full
      ns-train splatfacto 100-iter smoke EXIT=0 (seeded from the MASt3R ply, random_init=
      False). Coordinate gotcha handled (OpenCV c2w → OpenGL; world permute; applied_
      transform on the cloud too). 4 path constants env-overridable; `mast3r_available`
      True only if all 4 (env python + runner + converter + 2.6GB ckpt) exist —
      VERIFIED live True. **DEP: conda env `mast3r-spike` + checkpoint (2.6GB) at
      ~/tools/mast3r-spike/. CC-BY-NC-SA (non-commercial).**
- Review fixes folded in before commit (3-issue adversarial pass): (#1, ship-blocker)
  glomap `process` now `rm -rf processed_dir` so a colmap→glomap reroute can't measure
  a stale colmap/glomap mix; (#4) `mast3r_sfm` runs under HEAVY_GPU_LOCK (6GB reserve)
  so its ViT can't OOM the portal's TRELLIS lane (light colmap/glomap SfM stay lockless);
  (#3) reroute process uniquely named `reprocess<n>` → no duplicate stage-rail key / no
  false double-green. Review CONFIRMED safe: infinite-loop guards, mid-run list mutation,
  no false-escalation of good captures, default path + manual button intact.

## WAVE 1 — 360 fix + heatmap backend + edit-ops + Spark spike (2026-07-04, IN FLIGHT)
Master plan: ~/reports/splatlab-ultra-plan-2026-07-04/PLAN.md (RToony GO'd waves; UE
parked, replaced by survey/scale/benchmark design — see reports dir).
- [x] Housekeeping: 07-02 feedback+camera pass committed (e2a8409 XC-1 gpu_arbiter alert,
      6628092 feedback+camera). Tree was clean before wave-1 agents started.
- [x] 360 ROOT CAUSE (probe receipts in PLAN.md Appendix A): X4 .insv = TWO HEVC streams
      (one square fisheye per lens); ffmpeg -i read only stream 0 -> corrupt equirect ->
      2/624 (0.3%) registration on splat_ec1b984ffb. Fix VALIDATED manually:
      hstack both streams -> v360 dfisheye -> coherent panorama (scratchpad receipts).
- [x] BUILT + REVIEWED + FIXED + COMMITTED (workflow wf_aa28b8d5-7f4 + 2 fix agents;
      commits 9e565a3 360-fix / 48e701a langfield-heatmap-backend / 7174809 edit-ops /
      7d75ea0 spark-spike+supersplat-link; full suite 127 passed; adversarial reviews:
      backend SHIP w/ 4 findings fixed, edit-ops FIX_FIRST w/ all 10 fixed incl.
      dequantization blocker; splat-transform bumped 2.5.1->2.7.1 by the supersplat
      2.28.1 install — compress/webopt argv SMOKE-TESTED OK):
      A1 splat_route.py 360 fix (hstack compose, fail-loud, sanity gate, equirect
      matcher, glomap escalation, 360-param persistence) then langfield relevancy
      backend (langweb artifact + worker /relevancy + app proxy);
      A2 backend/edit_ops.py NEW (snapshots/versions, splat-transform ops, text-select
      delete/isolate/extract, MERGE scenes) — orchestrator mounts router in main.py;
      A3 frontend Spark spike /spark-test (fake-scalar heatmap via dyno worldModifier,
      nav prototype: reset/presets/pivot);
      A4 ~/projects/supersplat bump 2.27.4->2.28.1 (NODE_ENV-unset build gotcha);
      A5 survey/scale/benchmark DESIGN -> ~/reports/splatlab-survey-scale-design-2026-07-04/.
- [x] DEPLOYED 2026-07-04 ~17:04: splatlab.service (36 vars, healthz ok) +
      splatlab-langfield.service (worker /relevancy live in openapi). edit_ops
      router mounted (5 routes) + langfield STALE guard added to query/relevancy/
      inventory.
- [ ] LIVE 360 VALIDATION IN FLIGHT: job splat_98095cb055 = the SAME .insv, SAME
      params as failed splat_ec1b984ffb. Acceptance: registration >=30% (target
      >50%) + coherent render. Watcher bg6s98ixl.
- Deferred to next wave: "Edit in SuperSplat" deep-link + heatmap/nav UI on the real
  viewer (blocked on Spark spike verdict); portal dead-code deletion (0.5).

### Live 360 validation findings (2026-07-04 evening)
- splat_98095cb055 (the office .insv, SAME params as the original failure):
  stitch = hstack compose RAN (both HEVC streams mapped — receipt in job.log),
  sanity gate PASSED (no false-positive on the static-ish capture), sequential
  matcher used... registration STILL 2/624 (0.3%) -> ROOT CAUSE #2 (visual
  receipts in scratchpad/motion/): the clip is a SELFIE — operator holds the X4
  at arm's length facing himself; face/torso/arm dominate the sphere and move
  WITH the camera (dynamic occluder, camera-stable features) -> geometric
  verification rejects nearly all matches. NOT a pipeline bug; SfM physics.
  The new auto-escalation then fired (glomap_sfm rung, COLMAP 4.1 global_mapper,
  overlap 16) — mechanics receipt regardless of its verdict on doomed data.
- FOUND proper validation capture: ~/transfers/splatlab/VID_20260514_073947_00_002.insv
  (1.65GB, 106s, 3197 frames, dual-stream) = OUTDOOR POOL FACILITY WALKTHROUGH,
  camera overhead on stick, operator only at nadir (crop_bottom trims), textured
  concrete/buildings. Frames: scratchpad/may14/. This is the real acceptance run.
  (Also VID_20260514_064632_00_001.insv, 6.7GB/434s — same site, longer.)
- FOLLOW-UP FEATURE (high value, next wave): operator auto-masking for 360 —
  SAM2.1 person segmentation on fan-out crops -> COLMAP ImageReader.mask_path;
  would make selfie-style/visible-operator captures reconstructable. We already
  have SAM2.1 + the sam2 env on disk.
- UI guidance follow-up: 360 upload card should say "hold the camera OVERHEAD
  on a stick — if you're visible anywhere but straight down, the scan fails".

## CRASH POST-MORTEM + CPU LEASH (2026-07-04 evening)
- 17:36:53 splat_98095cb055 (office selfie clip) COMPLETED end-to-end: hstack
  stitch + glomap escalation -> trained, 1.92M gaussians, artifacts in
  _preview/ (splat.ply 454MB, web.ply 78MB). Quality UNVETTED — selfie data;
  eyeball in the viewer before judging. thumb.webp is 0 bytes (crash cut it).
- 17:37:22 the REAL acceptance run (May-14 pool walkthrough, 1.65GB) started as
  splat_fdac9edaab; the PC HARD-RESET within seconds of its stitch launching.
  Forensics: NOT VRAM/GPU (vram 31%, 56C, xid 0, gpu-watch clean at 17:35:37),
  NOT mains power (UPS event log silent), NOT kernel (no oops/pstore; journal
  tail lost). Firmware BERT record = CPER severity FATAL, section GUID
  81212A96-09ED-4996-9471-8D729C8E69ED (Firmware Error Record Reference /
  Intel CrashLog) -> CPU-domain hardware fatal error at the instant the
  all-core x264 encode launched (idle->250W package step; RAPL PL1=PL2=250W;
  ASUS ROG MAXIMUS Z890 HERO BIOS 3002, 285K ucode 0x121). The orphaned job
  was auto-marked failed on restart ("portal restarted while job was active").
- MITIGATION 1 (this commit): `_stitch_cpu_leash()` — taskset to half the
  cores (floor 4) + nice 10 on BOTH stitch paths. SPLAT_STITCH_CPUS overrides;
  0 disables. taskset/nice exec through -> job.pid still ffmpeg. 131 tests pass.
- MITIGATION 2 (system level): RAPL power-limit guard staged as
  ~/scripts/aipc-cpu-power-guard.sh (dry-run default; --apply caps PL1/PL2 +
  installs a persistent boot unit). BIOS checklist in the crash report.
- GATE: re-dispatch the pool-walkthrough acceptance run ONLY after the power
  guard is applied (app leash alone shrinks the transient but the fault is
  hardware-marginal).

## WAVE 2 START — acceptance run + Spark real-relevancy wiring (2026-07-04 late)
- Power guard APPLIED by RToony (RAPL PL1=125W/PL2=177W verified; boot unit enabled).
- Acceptance run DISPATCHED: `splat_75ebbcddde` (May-14 pool walkthrough,
  language_field=true — langfield stage queued last, will be the FIRST langfield
  scene on disk). Leash receipt in job.log: `taskset -c 0-11 nice -n 10 ffmpeg`;
  package 75C under load; the crash scenario now runs safely.
- spark-test upgraded (Wave 2.3 wiring): real language query → POST
  /langfield/relevancy → uint8 vector → RgbaArray → the SAME dyno modifier as the
  fake proof. Langfield scenes load fmt=langweb (index alignment with gauss_emb);
  FAIL-LOUD on any rows≠splats mismatch. End-to-end test unblocks the moment
  splat_75ebbcddde's langfield lands.

## WAVE 2.1-2.3 PROVEN END-TO-END ON REAL DATA (2026-07-04 evening)
- **Index mismatch ROOT-CAUSED + FIXED**: gauss_emb rows follow the CHECKPOINT,
  but ns-export FILTERS gaussians (Garden: 1,326,611 ckpt -> 1,321,833 ply;
  4,778 dropped) — so even langweb order could never match raw gauss_emb.
  Fix = `backend/langfield_align.py`: byte-exact float32 xyz hash map
  (ply row -> ckpt row), built+cached lazily per scene by the worker
  (`_langfield/ply_index_map.npy`), applied to /relevancy BEFORE quantization.
  Receipt: X-Count 1326611 -> 1321833 after fix; worker log "ply->ckpt map
  ready"; 100% of ply rows matched. Legacy scenes fixed retroactively, no
  retrain. 137 backend tests pass (6 new).
- **Live browser receipt (Garden, real GPU)**: "flower vase" -> 1,321,833 rows,
  420ms warm -> REAL per-splat heatmap tint on the Spark viewer + spotlight
  fade of low-relevancy splats. Spark verdict = PASS (Z-up correct, 1.3M splats
  crisp; fps unmeasurable headless — rAF throttled in background windows).
- **Spark gotcha (proven live)**: mutating a dyno uniform does NOT re-run the
  generator — spotlight/threshold flips were visual no-ops until
  `mesh.updateVersion()` after each uniform write.
- Garden langweb.ply backfilled (86MB vs 328MB raw fallback; 0.8s). TODO:
  backfill the other 5 mip360 langfield scenes the same one-liner way.
- **Portal clobber bug FIXED (portal commit 87632b6, NOT yet restarted)**: the
  portal's dormant splat.py ran cleanup_orphan_jobs() on every deploy and
  marked LIVE splatlab jobs failed ("portal restarted while job was active" at
  18:21:24 = portal ActiveEnterTimestamp, receipt). Startup hook removed;
  takes effect next portal restart (deferred — another session is deploying
  portal). splat_75ebbcddde meta hand-restored to running; its pipeline never
  actually stopped.

## SURVEY v1 SLICE + SPARK BETA ON THE VIEW PAGE (2026-07-04 night)
- **Scale calibration shipped end-to-end**: POST /jobs/{id}/scale stores
  meters_per_unit in meta (validated, null clears; 9 tests). Viewer measure
  tool: Spark raycast two-point pick -> markers+line -> scene units ->
  calibrate with known length (m/ft/in) -> real units everywhere after.
  LIVE RECEIPT (Garden): 0.5235 units = 1.524m = 5.00ft; meta shows
  meters_per_unit=2.9113 (NB: test calibration with a made-up 5ft reference —
  clear via {"meters_per_unit": null} or recalibrate on a real reference).
- **Spark beta viewer on /view/:jobId** (spark-scene-viewer.tsx, opt-in header
  toggle, sticky localStorage): real language heatmap + spotlight + measure.
  Classic viewer untouched/default; overlays/search-flyto stay classic until
  the full 2.4 cutover. Shared machinery extracted to lib/spark-heatmap.ts
  (spike page refactored onto it — one implementation).
- ezdxf 1.4.4 installed in backend/.venv (survey exports dep, per DESIGN.md).
- ⚠️ **LESSON (cost us the first acceptance run): `systemctl --user restart
  splatlab` SIGTERMs the WHOLE cgroup — start_new_session does NOT protect
  job subprocesses from systemd (KillMode=control-group). splat_75ebbcddde
  died mid-mapper ("Stage 'process' exited with code -15"). RULE: never
  restart splatlab.service with a job in flight. BACKLOG: job resume-on-start
  (rehydrate running meta + stage checkpoints) — codev candidate.
- Acceptance run RE-DISPATCHED: **splat_192e4223fb** (same params,
  language_field=true); leash verified (taskset 0-11, nice 10).

## OVERLAY v2 + DIMENSIONS (2026-07-04 late night, all browser-verified on Garden)
- **Multi-query language overlay**: up to 4 simultaneous searches, one color
  each (editable via color picker), packed into ONE RgbaArray (R/G/B/A
  channels) + mode-baked dyno modifier in lib/spark-heatmap.ts
  (buildOverlayModifier). Modes: Highlight (natural + colored matches),
  Isolate (only matches visible), Spotlight (colored + rest dimmed), Ramp
  (single-query scientific ramp: viridis/turbo/magma/grayscale). Live legend
  (bottom-right) tracks queries/colors/mode/threshold; per-query enable
  toggles + shared match-threshold slider are live uniforms (updateVersion).
  Receipts: "ball"@0.91 highlight = just the ball yellow on natural scene;
  isolate = table floats alone; 2-query legend (ball/wooden table).
  NOTE: relevancy bytes are PER-QUERY min-max normalized -> threshold is
  relative (default 0.75); absolute calibration = future work.
- **Dimensions**: unlimited two-point dimensions; draggable endpoints
  (pointer-capture, orbit paused during drag); floating midpoint labels
  (imperative DOM, projected per frame); list with per-dim delete + clear-all;
  sessionStorage persistence per scene; calibration binds to a selected
  dimension. Receipt: patio dim "3.039 m · 9.97 ft" label live.
- **Embedding-paint designed** (RToony's idea): sidecar override model
  (never mutate gauss_emb), query-select/sphere/brush rungs, worker apply +
  CRUD -> ~/reports/splatlab-embedding-paint-design-2026-07-04/DESIGN.md.
  P1 unblocked by today's langfield_align work.

## PAINT-THE-EMBEDDINGS SHIPPED (2026-07-04 night) — RToony's feature
- Backend COMPLETE + worker-verified live on Garden; frontend brush UI built.
  ⚠️ app proxy endpoints (select/sphere, overrides CRUD) need a splatlab.service
  restart — GATED until splat_192e4223fb finishes (no restarts mid-job). The
  worker side (:3417) is already live.
- Mechanism: sidecar overrides (backend/langfield_overrides.py — manifest json
  + per-record uint32 npy in _langfield/, EXPORTED-PLY order; gauss_emb.npz
  NEVER touched). Worker composes at scene load (assign/boost = blend toward
  label embedding — a zero/unseen row BECOMES the label, which is what makes
  abstract "liberal" labels work; suppress = remove projection). Worker
  endpoints: /select_sphere (GPU sphere test on resident positions),
  /overrides_add (guardrails: min 10 splats, ≤30% of scene unless force=true,
  bounds check), /overrides_delete; scene cache invalidated on mutation.
- EXACT-LABEL RECALL: /relevancy pins a painted region to max relevancy when
  the query names its label OR alias (X-Label-Hit header) — deterministic for
  labels SigLIP can't ground ("lucky orb" verified).
- LIVE RECEIPTS (worker-direct, Garden): sphere stroke @ ball focus r=0.12 ->
  2,229 rows; committed label "lucky orb" alias "the special thing" -> both
  queries X-Label-Hit:1; delete -> hit:0, files gone, manifest empty (full
  revert). 154 backend tests (8 new for guardrails/roundtrip).
- UI (beta viewer "Paint the field"): brush radius slider (meters when scale
  set), stroke preview in cyan w/ live count, UNDO per stroke + clear,
  "clip strokes to <query> matches" hygiene toggle, duplicate-label warning,
  Pin/Boost/Not-this ops, force-flow for oversized selections, painted-labels
  list with one-click revert.

## SCREENSHOT-DRIVEN FIX PASS (2026-07-04 late night, from RToony's 8 captures)
- **Percentile thresholds**: raw "match ≥ 0.75" replaced by "top X%" per query
  (cutoffForTopPercent histogram → per-channel cutoff uniforms). Root cause of
  the all-yellow Spotlight/Kitchen shots: relevancy is per-query min-max
  normalized so raw thresholds are meaningless across queries. Default top 2%.
- **Ramp mode honors enables**: tint channel + legend = first ENABLED query
  (was hardwired to channel 0 — RToony had ch0 disabled, got nothing + wrong
  legend). Enable toggles rebuild in tint mode.
- Stale paint 405 error cleared on mode/paint flips; stroke 404/405 now says
  "paint backend deploys on next splatlab restart (waiting for running job)".
- Legend raised above the Feedback FAB (was overlapped/truncated).
- Verified live on Garden: ball @ top-2% highlight, legend copy, dim persisted.
- Paintbrush 405 itself = the KNOWN deploy gate (old app process; endpoints
  land with the post-job restart). No code change needed.

## Test Flight + widescreen + segmentation program (2026-07-05)
- Program pack (plan for post-Fable executors): ~/reports/splatlab-360-sample-segment-plan-2026-07-05/
  (PLAN.md phased w/ acceptance gates, STATUS.md spine, evidence/ = 5-agent ultracode map).
- SHIPPED: Test Flight trim window (trim_start_s/trim_duration_s on /train; input-side
  -ss/-t on stitch; auto-centered; 400 on non-insv; meta now persists num_frames_target/
  sfm_backend/language_field/trim_*). 56/56 stitch tests. Frontend toggle (insv-only) +
  widescreen pass (max-w-[1880px], 2xl grid split, gallery 5-col, viewer 2xl:h-560).
- SHIPPED: ~/bin/splatlab-safe-restart — the ONLY sanctioned way to restart the service
  (the raw restart killed 2 real jobs on 07-04 = the "-15" cards).
- NEXT: Phase 3 segmentation (SfM-level join via colmap4 model_merger + bundle_adjuster,
  probe script first), train-resume via --load-dir. See the pack's PLAN.md.

## Langfield optional-stage bookkeeping fix (2026-07-05, from HANDOFF-PLAYBOOK PACKET 7)
**Bug**: langfield is a deliberately best-effort/opt-in stage — its failure correctly
never flips the job to `failed` (the splat itself is already done). But the pipeline loop
unconditionally appended `"langfield"` to `stages_completed` on both the non-zero-exit path
and the catch-all `except Exception` path, with no other record that it actually failed —
so a job's meta made a failed optional stage look identical to a successful one.

**Fix** (`backend/splat_route.py`, langfield stage only — compress/webopt share the
identical pattern but were left untouched, out of scope per the brief):
- New `_new_meta()` key `"stages_failed": []`, parallel to `stages_completed`.
- New helper `_record_stage_failure(job_id, stage, reason)` — read-patch-write, same
  pattern as the existing `stages_completed` append.
- Called from both langfield failure paths: `rc != 0` → `f"exit code {rc}"`; caught
  `Exception` → `f"error: {exc}"`. The "no config / toolchain unavailable" **skip** path is
  deliberately NOT recorded as a failure (it's a normal no-op, not a stage that ran and failed).
- `stages_completed` still gets `"langfield"` appended unconditionally either way — job
  semantics UNCHANGED: `final_status` still ends up `"completed"`, splat still "done".
- Also threaded `stages_failed` into the `audit_operator_event` metadata dict (the audit
  trail was hiding the same failure).
- Confirmed visible end-to-end: `grep -rn "stages_completed\|stages_failed"` showed every
  consumer — `_job_payload()` spreads `**meta` verbatim into every job API response (status
  list, stop, preview endpoints), so `stages_failed` reaches the frontend with zero extra
  plumbing. Updated `frontend/src/lib/contracts.ts` (`SplatJob.stages_failed?`) and
  `frontend/src/pages/splat.tsx` (gallery card: amber "field failed" badge + reason tooltip
  when `langfield_available` is false AND a `stages_failed` entry names `"langfield"`).

**Verification (real receipts, backend/):**
```
$ ~/.local/bin/pytest backend/tests/test_langfield_stage_bookkeeping.py -v
backend/tests/test_langfield_stage_bookkeeping.py::test_langfield_nonzero_exit_does_not_fail_job PASSED
backend/tests/test_langfield_stage_bookkeeping.py::test_langfield_exception_does_not_fail_job PASSED
backend/tests/test_langfield_stage_bookkeeping.py::test_langfield_success_leaves_stages_failed_empty PASSED
backend/tests/test_langfield_stage_bookkeeping.py::test_langfield_skipped_no_config_is_not_recorded_as_failure PASSED
4 passed in 0.15s

$ ~/.local/bin/pytest backend/tests/ -q
FAILED backend/tests/test_scale_calibration.py::test_scale_rejects_garbage[nan]
FAILED backend/tests/test_scale_calibration.py::test_scale_rejects_garbage[inf]
2 failed, 169 passed, 4 warnings in 2.19s
```
The 2 failures are PRE-EXISTING and unrelated (JSON `nan`/`inf` encoding in the scale
calibration endpoint) — confirmed identical (165 passed / 2 failed) on a `git stash` of this
diff before making any change. No regressions from this fix; 4 new tests, 0 net new failures.

Frontend typecheck (`npx tsc --noEmit` in `frontend/`): 43 pre-existing errors, byte-identical
before and after this diff (all in `splat-viewer.tsx` / `feedback.tsx` / `feedback-api.ts` /
`splat-view.tsx` — none in the two files this fix touched).

**Out-of-scope note**: `compress` and `webopt` (lines ~2437-2517 as of this commit) have the
IDENTICAL bug — best-effort, log-only on failure, unconditional `stages_completed` append,
no `stages_failed` record. Left untouched per the brief ("Do NOT touch the compress/webopt
stages... just note in your report"). Same `_record_stage_failure` helper would cover them
if/when someone picks that up.

## §1E "Promote to full build" SHIPPED (2026-07-05, Sonnet 5 swarm session, from the handoff pack)
Fixes the F2 "hybrid" trap: on a completed Test Flight (trimmed) scene, the old **Re-run**
button silently dropped the trim (job.input_path is always the original full .insv — trim is
stitch-time only, never re-sent) but kept the flight's draft `max_num_iterations` and fell back
to the request default `sfm_backend="colmap"` — a multi-hour run at draft quality on the doomed
84-min COLMAP rung, neither a test nor a full build.

- `frontend/src/lib/contracts.ts`: exposed `num_frames_target`, `sfm_backend`, `trim_start_s`,
  `trim_duration_s` on `SplatJob` (backend already returned them via the meta spread; the type
  just didn't declare them).
- `frontend/src/pages/splat.tsx`: new `promoteToFullBuild(job)` — same insv input, trim omitted
  (full clip), `num_frames_target: 300` (the backend's §1D′ duration-aware rule overrides this
  for insv jobs once deployed), `sfm_backend: job.sfm_backend ?? "glomap"` (the rung the flight
  actually proved — every flight requests glomap directly, so the persisted value is reliable;
  no reroute history needs to be exposed), iterations from the **currently selected quality
  preset** (`iters`), `language_field` from the current toggle (not the flight's stored value).
  On scene cards where `trim_duration_s != null`, this one button replaces BOTH Re-run and
  ↑Quality — both call the same old `rerun()` and both inherit the identical hybrid bug on a
  trimmed scene, so leaving ↑Quality in place would leave the trap under a different label.
- `rerun()`/`retryGlomap()` (non-flight jobs) now also forward the scene's persisted
  `num_frames_target` (both) and `sfm_backend` (rerun only — retryGlomap's whole point is to
  override it to glomap) instead of leaving them unset and falling back to request defaults
  that could silently contradict how the scene was actually built.

**Verification:**
```
$ npx tsc --noEmit    (frontend/)
43 errors — byte-identical to the pre-existing baseline (splat-viewer.tsx/feedback.tsx/
feedback-api.ts/splat-view.tsx); zero in contracts.ts or splat.tsx.

$ npm run build       (frontend/)
✓ built in 2.31s — clean.
```
No component-test runner exists in this repo (no vitest/jest configured), so the gate's
"manual dispatch meta.json diff" path was done **without creating a live job** (no visible
gallery row, no audit event, no meta.json write) — imported `splat_route` directly and ran the
exact JSON.stringify(body) shape `promoteToFullBuild()` sends through `SplatTrainRequest` ->
`_plan_3d_job` -> `_new_meta`, using the real `VID_20260514_064632_first90s.insv` for duration:
```
1. SplatTrainRequest validated OK:
   trim_start_s=None trim_duration_s=None sfm_backend='glomap' num_frames_target=300
2. _plan_3d_job stages: ['stitch', 'glomap_sfm', 'process', 'train', 'export', 'compress', 'webopt']
   stitch argv: [...'-i', '.../VID_20260514_064632_first90s.insv', '-filter_complex', ...]
   (no -ss/-t in the argv — full clip, trim correctly dropped)
3. _new_meta persisted fields (would-be meta.json):
   num_frames_target = 300 / sfm_backend = 'glomap' / trim_start_s = None / trim_duration_s = None
OK: promoted payload validates, drops the trim, keeps glomap, no job/meta.json written.
```
Confirms: trim dropped, glomap rung kept (not colmap), stages plan correctly — proves the
exact defect (F2) is fixed without spending GPU time or creating visible state.

Deploy: frontend-only, `npm run build` already run above — **no service restart needed**.

Committed locally (not pushed): see git log.

## §1D′ duration-aware num_frames_target SHIPPED (2026-07-05, Sonnet 5 swarm session)
The pipeline's only proven-good 360 config is ~3.0fps equirect frame density
(splat_9da9dff4b2 @1.76fps: 599 posed, ZERO points vs splat_5177f8d99a @3.0fps: 1078/1080
registered, 105k points — same window, both directions confirmed). Test Flight already
computes `num_frames_target = 3 * trim_duration_s` client-side, but a full (non-flight)
insv run has no way to know the real clip duration — the UI can only hardcode a flat
guess (75), which is 0.7fps on a 106s clip and hits the exact same 0-point cliff on
anything longer than ~25s.

**Fix** (`backend/splat_route.py`, `_plan_3d_job`'s `is_insv` branch):
- Duration is now always probed (not just when a trim is requested) — `full_duration`.
- After trim resolution: `density_window_s = trim_duration if trim_duration is not None
  else full_duration`. When known and > 0: `req.num_frames_target =
  min(ceil(3.0 * density_window_s), 4000 // req.images_per_equirect)` — overrides
  whatever the client sent, self-capped so it can never trip the `/train` endpoint's
  existing `perspective_images > 4000` guard (backend/splat_route.py:3020, unchanged).
  Probe failure (no ffprobe / unreadable container) leaves the client's value alone
  rather than guessing — mirrors the layout-probe's existing fail-open policy.
- Test Flight is a **no-op** under this rule: same 3fps formula, same (trim) window,
  same result the client already sends (30s -> 90, matches exactly) — only full runs
  change behavior.
- `_new_meta` runs AFTER `_plan_3d_job` in the `/train` handler, so meta.json now shows
  the REAL computed value, not the raw client request — also fixes the SfM-escalation
  gate's rebuild path (`sfm_req`), which is captured after the override and so reroutes
  to a fallback solver using the correct density too.

**Verification:**
```
$ ~/.local/bin/pytest backend/tests/test_360_stitch.py -q
64 passed in 0.56s   (56 baseline + 5 new §1D′ cases + 3 already added post-langfield fix)

$ ~/.local/bin/pytest backend/tests/ -q
2 failed, 174 passed, 4 warnings in 2.16s
```
The 2 failures are the same PRE-EXISTING `test_scale_calibration.py` nan/inf cases (unrelated,
confirmed unchanged since the langfield commit). 5 new tests, 0 regressions:
- full run computes duration-aware target (80s -> 240, no cap needed)
- crop-count cap engages on a long clip (300s -> would be 900, capped to 500)
- cap scales with images_per_equirect (14-crop: 4000//14=285)
- Test Flight trim produces the SAME value the client already computes (30s -> 90 —
  proves this ships with zero behavior change for the already-correct lane)
- probe failure leaves the client's value alone (no crash, no guess)

Deploy: **DEPLOYED 2026-07-05 11:34 PDT** via `splatlab-safe-restart` (RToony's go-ahead,
post-Phase-3.1). First attempt showed healthz `token:false` (should be `true`) —
`nexus-svc-inject` had written 0 vars. Root cause: systemd user manager's global
`BW_SESSION` (`systemctl --user show-environment`) was STALE relative to the current
valid session in `/dev/shm/nexus_session` (confirmed: `bw` rejected it, prompted for
the master password) — this is the known `gotcha_stale_systemd_bw_session.md` pattern
("wrote 0 vars" is its exact signature). Fix: `systemctl --user set-environment
BW_SESSION=<value from /dev/shm/nexus_session>`, then re-ran `splatlab-safe-restart`.
Second attempt: `nexus-svc-inject: wrote 36 vars`, healthz `token:true`, service active.
§1D′ is now genuinely live for every new insv/equirect dispatch.

Committed locally (not pushed): see git log.

Committed locally (not pushed): see git log for the Problem/Fix/Verification/Risk message.

## Phase 3.1 segment-merge probe: GATE PASSED 2026-07-05 (Sonnet 5 swarm session)
`tools/probe-segment-merge.sh` proves Architecture A (SfM-level join) works, after 3
failed attempts that each surfaced a real, distinct problem — none of them fixed by
retrying, each fixed by understanding root cause:

**Attempt 1 — FAILED, non-architectural**: dev clip, SEG1=[0,40)/SEG2=[30,70) @3fps,
independent per-segment databases. seg1 succeeded (960/960, 4MB points); seg2 posed
all cameras but triangulated ZERO points — the [30,70)s window of that specific clip
lacks parallax somewhere past t=40s (operator likely held still). Same failure class
as G3 attempt 1. Fixed by switching to the pool clip
(`VID_20260514_073947_00_002.insv`) with SEG1=[15,55)/SEG2=[45,85), centered on the
window G3 already proved has strong parallax (`[30.837,75.837)`).

**Attempt 2 — per-segment SfM PASSED, model_merger CRASHED (real architecture bug)**:
both segments' independent SfM succeeded cleanly (959/960 @ 70k pts, 960/960 @ 113k
pts) — proves the per-segment SfM step is solid given real parallax. But
`model_merger` SIGABRT'd: `Check failed: src_images[i]->ImageId() ==
tgt_images[i]->ImageId()` (`estimators/alignment.cc:76`,
`ReconstructionAlignmentEstimator::Estimate`). Root cause, traced through
colmap4-src: model_merger's alignment estimator requires the SAME numeric ImageId for
a common-by-name image across both input models — true only when both models load
from ONE shared database (colmap's actual "merge disconnected sub-models of one run"
use case, doc/faq.rst:315), not two independently-run segments with independently
assigned IDs. `database_merger` is NOT a workaround: `Database::Merge`
(scene/database.cc:60) explicitly refuses to merge databases sharing any image name —
built for disjoint sets, the opposite of what an overlap join needs. **Fix**: one
shared database + one shared `feature_extractor`/`sequential_matcher` pass, then two
bounded `global_mapper --GlobalMapper.image_list_path <segN.txt>` calls (confirmed in
source — `option_manager.cc:1195` + `global_pipeline.cc:81-82` — this genuinely
restricts the DatabaseCache input, not a post-hoc filter). Script rewritten to this
design; `model_merger` succeeded (ratio 1.000) on the very next attempt.

**Attempt 3 — merge succeeded, bundle_adjuster DIVERGED (NO_CONVERGENCE, one runaway
point)**: registration ratio 1.000 (1679/1679), but post-BA mean reprojection error
was an astronomical garbage value (~1.2e149 px) — one degenerate correspondence
admitted at `model_merger`'s default `--max_reproj_error 64` (a loose RANSAC inlier
threshold for the alignment sim3, not a point-quality filter) ran away during BA.
Diagnosed and fixed WITHOUT re-running any SfM: reused the existing seg1/seg2 sparse
models on disk, tightened `model_merger --max_reproj_error` to 8, ran colmap's own
`point_filtering` (`--max_reproj_error 4 --min_track_len 2`) on the merged model
before `bundle_adjuster`. Pre-BA error 2.67px → 1.32px (tight merge) → 0.99px
(+filter) → 0.89px stable post-BA (verified by hand against the real attempt-2
models before landing in the script).

**Attempt 4 (FINAL, clean single-invocation run of the fully-fixed script) — GATE
PASS**:
```
seg1 registered: 959   seg2 registered: 960
union (distinct names, seg1|seg2): 1679
merged (post-BA) registered: 1679
registration ratio (merged/union): 1.000 (gate: >= 0.80)
mean reprojection error: 0.807224px (gate: <= 1.50px)
GATE: PASS (ratio OK, reproj OK)
```
Full log: `tools/probe-segment-merge-run4.log` (repo-root, gitignored via `*.log`).
Output artifacts: `tools/probe-segment-merge-output/` (gitignored).

**⚠️ Correction needed before Phase 3.2 implementation**: PLAN.md's Phase 3.2 §3.2 text
(and its Phase 3 preamble) describes independent per-segment databases — that's the
design attempt 2 disproved. Phase 3.2 must instead: one shared database per job,
`feature_extractor`+`sequential_matcher` run once over the full frame set,
per-segment `global_mapper --GlobalMapper.image_list_path` calls (this is still the
independently-checkpointable expensive step — the restart-survival property is
preserved), `model_merger --max_reproj_error 8` (not the default 64), then
`point_filtering --max_reproj_error 4 --min_track_len 2` before the final
`bundle_adjuster`. `tools/probe-segment-merge.sh` is the reference implementation for
all of this — Phase 3.2 should port its logic into `_plan_3d_job`/new pipeline stages,
not re-derive it.

Committed (script + this STATUS.md, NOT the gitignored output/log): see git log.

## Photo-capture reliability + survey polish pass (2026-07-05, RToony's call)
RToony deprioritized the 360-video segmentation work (Phase 3.2 on hold) in favor of
the photo/standard-capture path — "smaller, quicker, easier for a small site." Two
review agents audited the escalation chain and survey/measurement tools; RToony picked
the two quick fixes to land now (DXF export, escalation UX polish, and the two
dimension-bug fixes below are deferred/already-fixed — see below):

- **fix(splat) 4e1e4f3**: compress/webopt now use the existing `_record_stage_failure`
  helper (previously langfield-only) — a failed .spz/web.ply/langweb build is no longer
  indistinguishable from a success in job meta. 7 new tests.
- **test(scale) 0e01afe**: `test_scale_rejects_garbage[nan/inf]` — root cause was
  httpx's own request encoder refusing NaN/Infinity (RFC 8259 compliant), not the
  endpoint (which was always correct). Fixed by sending raw bytes for those two cases.
  **Full backend suite now 183/183 green — first fully-green run this program.**
- **fix(splat) a8662b4**: sparse ("Few Photos") jobs seeded `sfm_tried` as an empty set
  (not escalation-eligible), so a failed sparse job's error message fell back to
  claiming "Auto-fallback tried colmap" even though it ran mast3r-sparse and never
  touched COLMAP — directly misleading for the small-site/few-photos use case.
  Extracted `_seed_sfm_tried()`, 3 new tests.
- **fix(survey) 2ce5692**: scale calibration UI (`spark-scene-viewer.tsx`) — (1)
  `calibDim` no longer silently falls back to "the last dimension in the list" when
  nothing is explicitly picked (bit a user who deletes their calibration target); (2)
  recalibrating now requires a two-click confirm (same idiom as scene delete) since
  meters_per_unit is one scalar shared by every dimension's displayed length. Live-
  verified on the Garden scene (see commit for the verification detail — the browser
  automation tool's synthetic clicks didn't register on this specific button, a tool
  quirk; a real dispatched click confirmed both fixes end-to-end).

**Deferred (not started, real findings on record for later)**:
- DXF/LandXML export: `ezdxf` is installed and the scale endpoint's own docstring
  claims "measure/DXF/LandXML all hang off it," but no export code exists at all —
  dimensions are 100% client-side sessionStorage with no save/export path. Biggest
  gap for the civil-survey use case; real feature work, not a quick fix.
- Escalation UX polish: `sfm_tried`/`reroute_count` never reach the frontend (the
  "Retry with global SfM" button doesn't know if that solver was already exhausted);
  reroute reasons only appear in scrolling logs, not the stage rail; exhaustion
  guidance text is video-flavored regardless of actual capture type (photo vs video).

## CAPTURE COACH PHASE 0 — fog-fingerprint gate + calibration PASSED (2026-07-11)

**Goal**: score reconstruction health so fog scenes stop being discovered after hours
of GPU spend (the 07-10 root-cause finding, splat-geometry-health-gate memory). Plan:
~/.claude/plans/lets-brainstorm-my-next-functional-backus.md (Phases 0→2 + earned
enforcement). REPORT-ONLY per the metric-trust doctrine.

**Shipped (new files only, no server change, no restart)**:
- [x] `backend/health/fog_gate.py` — langfield-spike env; renders ED depth + RGB at 6
      spread training cameras (640px downscale); per-cam metrics over opaque px;
      writes `<job>/_health/fog.json` + side-by-side [RGB|turbo log-depth] receipts.
      Exit 0 = analysis ran (any verdict); non-zero = execution failure only.
- [x] `backend/health/run_health.sh` — run_langfield.sh clone minus SAM (env
      hardening: unset CPATH/LIBRARY_PATH, pin CUDA_HOME). `SPLAT_HEALTH_PYTHON` override.
- [x] `backend/health/backfill_fog.py` — stdlib CLI; REFUSES while any meta.json is
      starting/running or GPU free <6GB (--force); --write-meta patches meta["health"]
      (only safe because of that preflight); writes calibration report + summary.json.
- [x] `tools/gates/gate_p0_fog_calibration.sh` — executable acceptance gate.

**METRIC CHANGE (calibration finding)**: the raw 07-10 fingerprint (p95/p5 spread < 3)
failed on the MIXED selfie scene splat_98095cb055 — every camera has p5 pinned at the
near plane (cocoon contamination) but 3 cams punch through to real structure, inflating
p95 (spread up to 45 while still junk). Verdict now uses per-camera **shell fraction**
(share of opaque px with depth ≤ 0.03 = 3× near plane): fog cam = shell ≥ 50% @ acc ≥
.98; clean cam = shell ≤ 5% AND p50 ≥ 0.1; 2/3 camera majority (CAM_FRAC=0.66 — 0.67
rejects a legit 4/6). Spread still reported for context. All thresholds HEALTH_FOG_* env.

**GATE PASS (exit 0), full separation on graded scenes** (~4s/scene after JIT warm):
| scene | graded | verdict |
| splat_5177f8d99a | FOG (07-10) | FOG 6/6 |
| splat_98095cb055 | FOG (07-10) | FOG 4/6 (mixed — operator cocoon + real office) |
| splat_32d926d9 garden | HEALTHY | HEALTHY |
| kitchen/bonsai/counter | unlabeled | HEALTHY (matches langfield-verified geometry) |

**⚠️ FINDING — pool scene splat_192e4223fb is FOG**: depth pinned at the near plane
(spread 1.00, p50 0.0100) at ALL cameras; RGB receipt is a structureless smear. Its
"HEALTHY" label was an ungraded assumption (07-05 acceptance passed on 90% REGISTRATION
— registration ≠ reconstruction). Gate asserts only RToony-graded scenes; pool is a
pending-grade row. Receipts: ~/reports/2026-07-11-capture-coach-fog-calibration/index.md.

**Next (gated on RToony's receipt review)**: Phase 0.5 = wire `health` stage after
export (kill-switch SPLAT_HEALTH_GATE) + meta["health"] + SceneCard badge +
CaptureHealthCard; then Phase 1 capture probe, Phase 2 upload heuristics. Enforcement
stays opt-in-later per gate.

## CAPTURE COACH PHASE 0.5 — health stage WIRED + LIVE, report-only (2026-07-11)

RToony graded the Phase-0 calibration receipts ("receipts check out") → go.

**Backend (splat_route.py)**:
- [x] Constants + `_health_available()` (runner + langfield-spike python only — NOT
      `_langfield_available()`, which also demands sam2) + `_append_health_stage()`
      (kill-switch `SPLAT_HEALTH_GATE=0`; extracted as a helper so the guard is unit-testable).
- [x] `health` planned right after train/export, before compress/webopt/langfield;
      generative lane naturally excluded (early-returns before the append).
- [x] Runner branch cloned from the langfield best-effort contract: whole body wrapped,
      `_run_locked_stage` under HEAVY_GPU_LOCK (HEALTH_VRAM_MB=4000), verdict from
      `_health/fog.json` → `_patch_meta(health={"v":1,"fog":{...,"enforced":False}})`,
      failure = `_record_stage_failure` + continue, provably never flips final_status.
- [x] Receipt route `GET /jobs/{id}/health/receipt/{name}` (regex-guarded, webp/png).

**Frontend**: contracts.ts `health?` block (all-optional, old scenes deserialize
unchanged); STAGE_HUMAN/SHORT "Checking capture health"/"Health"; SceneCard verdict
pill (amber "likely fog" / green "healthy" / gray "unverified") sharing the bottom-right
corner with the searchable badge; `CaptureHealthCard` under the featured viewer —
verdict headline (report-only wording), reshoot coaching, per-camera receipt strip.

**Gates + deploy (all receipts real)**:
- `tools/gates/gate_p05_wiring.sh` → **PASS exit 0**: 7/7 pytest
  (test_health_stage_bookkeeping.py — non-fatal on failure/exception, meta persisted,
  plan guard + kill-switch), frontend build OK, live API serves 7 scenes with verdicts,
  receipt route returns image/webp with bearer auth.
- Deployed via `splatlab-safe-restart` (no jobs in flight); healthz OK.
- Backfill `--write-meta` patched all 7 calibration scenes → badges live in gallery.
- Live traversal proof: Test Flight `splat_7f3d29f3de` (pool clip, 30s trim, glomap)
  dispatched with `health` in stages_planned after export — verdict lands when it
  finishes (expected FOG per the Phase-0 finding on the full-clip scene).
- Gotcha (repeat offender): `python3 - <<'PY'` heredoc CLOBBERS a curl pipe into
  stdin — fetch inside the script. And never pipe a gate through `| tail` (masks exit).

**Enforcement stays OFF** (`enforced:false` everywhere). The flip
(`SPLAT_HEALTH_ENFORCE_FOG` skipping langfield/mesh) is a later, per-gate, revocable
opt-in after RToony grades real-run receipts. Next: Phase 1 capture probe, Phase 2
upload-time Tier-0 heuristics (see the capture-coach plan file).

## 360 FOG ROOT CAUSE + RIG LANE (2026-07-11, Capture Coach spin-off)

**ROOT CAUSE of every insv fog cocoon — pinned by probes, not vibes**
(full ledger: probe-operator-mask/STATUS.md):
- Masking arms (seam bands, person masks) moved NOTHING: FOG, shell 1.0.
- Geometry probes on the SfM output: camera-path bbox 1584 units vs point-cloud 129
  (12×); same-frame 8-crop camera centers (physically identical) solved median
  **5.1 units apart** vs true step 0.13 — the unrigged fan-out scatters poses, the
  trajectory explodes, normalization collapses real geometry to depth ~0.01 = the
  fog fingerprint. All 3 insv scenes FOG, all 4 pinhole scenes HEALTHY.
- **Arm R (colmap4 panorama_sfm rig)**: 1080/1080 registered, shell 0.997→0.23,
  first recognizable insv reconstruction (pool receipts in the arm_R/_health dir).

**Gate v2 (fog_gate.py)**: sky-pitch exemption (cams pitched >+20° up = no parallax,
near-shell is legal — env HEALTH_FOG_SKY_PITCH_DEG) + mask-aware stats (person-masked
px are unsupervised, excluded) + default 8 probe cams + pitch in every receipt label.
**Recalibrated: gate_p0 PASS exit 0** (graded verdicts unchanged; report at
~/reports/2026-07-11-capture-coach-fog-calibration-v2/). Arm RP under v2: honest
UNCERTAIN (draft 7k iters; real 30k + floater cleanup expected to improve).

**RIG LANE WIRED, OPT-IN (`sfm_backend="rig"`, equirect video only)**:
- `backend/rig/render_rig.py` (colmap4 env): sphere → 12 virtual views (4 yaw ×
  3 pitch, 90°) + per-pixel ownership masks + rig_config.json.
- `_rig_sfm_command`: ffmpeg STRIDE extract → render_rig → colmap4 feature_extractor
  (per-folder SIMPLE_PINHOLE + ownership masks, GPU) → **rig_configurator** →
  sequential_matcher (rig_verification + skip_same_frame + loop_detection) →
  global_mapper (refine_sensor_from_rig 0, focal/extra fixed) → guards → ns-process-data
  --skip-colmap. Stage name `rig_sfm` (frontend labels added).
- NOT in SFM_ESCALATION, NOT default — falls back to colmap silently on non-equirect/
  non-video/missing toolchain. Default-flip = RToony's call after graded real runs.
- NEW DEP: pycolmap 4.1.0 pip-installed in colmap4 env (render script needs only
  cv2/scipy/PIL; pycolmap used by the spike's panorama_sfm arm, kept for parity).
- Tests: test_rig_sfm_plan.py (+ suite green, 14/14). Live acceptance:
  splat_ff2b9dd395 dispatched via the pipeline with rig_sfm planned.

## RIG LANE LIVE ACCEPTANCE — PASS (2026-07-11 10:57)
- First flight splat_ff2b9dd395 FAILED in sequential_matcher (exit 134): rig-config
  camera_params was a comma STRING; colmap's parser iterates it as a JSON array →
  empty params → poisoned camera rows. Fixed b1a594a (array form).
- Retry **splat_3885b68e54 COMPLETED end-to-end**: stitch → rig_sfm → process → train
  → export → health → compress → webopt, stages_failed=[], **~11.5 min total** (vs
  ~14.5 min for the glomap flight — the rig lane is FASTER despite 12 views/frame).
- Health (gate v2, no person masks, draft 7k): UNCERTAIN — shell 0.555, spread 21.8,
  4 counted / 4 sky-exempt. Consistent with the spike arms; receipts in the gallery
  health card for RToony's grading.
- **Open for default-flip**: RToony grades the live receipts; then candidates =
  full-quality 30k run, person-mask training stage (masks proven, ~30s/720 crops),
  rig-first escalation for equirect. All opt-in until graded.

## DEFAULT-FLIP: 360 captures route to the rig lane (2026-07-11, RToony /goal)
- The problem: the rig fix only worked if you typed sfm_backend="rig" — a default
  insv job still took the fog-producing unrigged fan-out.
- Backend: SFM_ESCALATION = [rig, colmap, glomap, mast3r] with EQUIRECT_ONLY_SOLVERS
  guard (flat captures never route into rig); _plan_3d_job upgrades default-colmap
  equirect VIDEO to rig when rig_available; legacy rungs remain the A1-gate fallback.
- Frontend: flights no longer force glomap when the rig toolchain exists (that
  override would have bypassed the server flip); glomap kept when rig is missing.
- Tests: golden-snapshot helper pins _health_available False (goldens = SfM/stitch
  drift only; health has its own plan-guard tests). Suite 199/199.
- LIVE PROOF: splat_f4c9416afb dispatched with NO sfm_backend → planner routed it
  to rig_sfm (meta shows requested colmap default + rig_sfm planned).
- CONFIRMED 13:23: splat_f4c9416afb (default path, auto-routed) COMPLETED all 8
  stages, ~11.5 min — health 0/4 fog cams, median shell **0.014** (vs 0.997
  baseline / 0.555 first rig run), spread 8.96, 2 healthy cams. Best 360 result
  in the program; one cam short of formal HEALTHY. Grading datapoint for RToony.

## RESUME-ON-START SHIPPED (2026-07-11 evening, Lane B #1)
- `resume_orphan_jobs()` replaces mark-failed-only at startup: the NEWEST orphaned
  in-flight job is re-planned from its persisted meta (every request knob persists in
  _new_meta) and relaunched under the SAME job_id from stage 1 (stage scripts are
  self-cleaning). Guards: SPLAT_RESUME_ON_START=0 kill-switch, RESUME_MAX_AGE_HOURS
  (12, env), RESUME_MAX_RESTARTS=2 crash-loop cap, input-still-exists, one restart
  only (single-job GPU), any error → honest failed marker (never wedges startup).
  Older orphans still get the failed marker. restart_count/restarted_at in meta.
- Tests: test_resume_on_start.py (7 cases). Suite 205/205.
- ⚠️ NOT YET DEPLOYED — full pool run splat_7c369afbde in flight; deploy via
  splatlab-safe-restart after it completes.

## WORKSTATION SAFETY ENVELOPE (2026-07-11 crash follow-up)
- `splat_7c369afbde` triggered a second abrupt platform reset while its x264
  stitch used about 12 cores. Telemetry ruled out OOM, thermal, GPU, and storage
  exhaustion; the load exposed underlying platform instability.
- The reboot auto-deployed resume-on-start and relaunched the same job. Recovery
  is now opt-in only: `SPLAT_RESUME_ON_START=1`. Unset, invalid, and `0` values
  leave interrupted jobs stopped for manual review.
- `splatlab.service.d/60-safety-guard.conf` confines the full service tree to
  E-cores 8-15, a four-core average CPU quota, nice 10, low CPU weight,
  32G/48G memory thresholds, 8G swap, and 512 tasks. This covers stitch, COLMAP,
  training, export, and every descendant, not just ffmpeg.
- Raw Insta360 work now defaults to the existing 30-second Test Flight. Full
  capture builds require a deliberate toggle and staged promotion.
- The interrupted full run is preserved as failed with stitch complete. It must
  not be resumed until the hardware gate and bounded flight ladder in
  `~/reports/splatlab-safe-evaluation-2026-07-11/plan.md` pass.

## ROBUSTNESS WAVE (2026-07-18, RToony /plan: "hardware acceptance first, then robustness")

Gate: `tools/gates/gate_robustness_wave.sh` — **PASS exit 0** (43 wave tests, full suite
348, tsc 23-error baseline, build clean). Commits 5a26f54..f6cf297:
- [x] **Dirty-tree audit landed** (5a26f54 supervised compute-unlock backend/tools,
      b84f0ac gate-visibility UI) — audit caught + fixed a REAL TDZ render crash
      (createDisabled referenced startMutation before declaration, splat.tsx:170).
- [x] **glomap default for photo folders** (2f60824): _plan_3d_job upgrades default-colmap
      flat image dirs to glomap when colmap4 present (07-17 A/B: 29/29 vs 17/29).
      Video/equirect/sparse untouched; escalation keeps colmap as fallback rung.
- [x] **Escalation surfacing** (83f003c): meta gains sfm_start_solver (RESOLVED) +
      sfm_tried + reroute_count + structured sfm_reroutes; RerouteChips on stage rail +
      failed card; Retry-with-glomap disabled when already tried; _recapture_guidance()
      keys exhaustion advice to photo/video/360.
- [x] **Optional-stage bookkeeping** (4fee640): failed langfield/compress/webopt/health
      no longer ALSO append to stages_completed (stages_failed is the record; skips
      unchanged; final_status semantics untouched).
- [x] **Capture Coach Phase 1** (3d29117): backend/health/probe.py pre-train probe at the
      A1 gate (pass AND fail paths) — trajectory/cloud bbox ratio (the 07-11 12x
      fingerprint), map density, orbit/walkthrough shape → meta.health.probe
      (MERGES with fog; fog patch now merges too). Report-only per doctrine.
- [x] **Capture Coach Phase 2** (c8badfc): backend/health/precheck.py + POST
      /api/splat/precheck (NOT compute-gated) — pure-Pillow blur/exposure/static +
      density advisories at upload; amber advisory panel; Create NEVER disabled.
- [x] **Polish** (f6cf297): 360 overhead-stick tip on the upload card; insv full runs
      omit num_frames_target (server §1D′ rule is the single source of truth).
- **Pre-Flight-A**: MemTest 4-pass ACCEPTED 07-16 16:21 (operator photo clarification);
  staged `~/scripts/splatlab-preflight-a-run.sh` (dry-run default; --apply stops
  langfield/sam-video-lab aux units, runs tools/gpu-hardware-acceptance.py with the
  three boot-bound evidence records + all confirmations, verifies receipt+sidecar).
  MUST run from a plain terminal (script refuses inside an aipc-safe-run scope).
  STOP at PASS_PRE_FLIGHT_A — marker stays; Flight A is a separate explicit decision.

## LOCATE-IN-THE-WORLD SHIPPED (2026-07-15, RToony /goal — splatedit.app-inspired)
Pin any scene to real WGS84 coordinates (feature work only — GPU pause, maintenance
marker, and Flight A ladder all untouched; endpoints are metadata/CPU-only and
deliberately NOT behind require_heavy_work_admitted, same policy as /scale).
- **Backend (b3168ba)**: `geo_route.py` mounted like edit_ops. POST /jobs/{id}/geo →
  meta["geo"] = {lat, lon, alt_m, heading_deg (bearing of scene +Y), anchor_scene,
  source, set_at}; {"geo": null} clears. GET geo/suggest = GPS from photo EXIF
  (Pillow) / video tags (ffprobe ISO6709) / embedded tracks (exiftool -ee), fail-soft.
  GET geo/footprint[.webp] = transparent top-down plan projection (`geo_footprint.py`,
  thumb.py-style CPU sampling, cached _preview/footprint.{webp,json}) with exact
  scene-unit bounds. GET geo/export?fmt=geojson|kml. Scene→ENU transform documented in
  the module docstring. 21 new tests; suite 313/313.
- **Frontend (602d594)**: "Locate" header button on /view/:jobId → lazy Leaflet modal
  (Esri satellite + OSM, Nominatim search, "Use photo GPS", draggable anchor +
  footprint overlay rotated by heading / scaled by meters_per_unit, opacity). Save
  writes /geo; /scale only on deliberate adjustment (checkbox guard over an existing
  calibration). Emerald "located" gallery badge. tsc 23 errors == baseline.
- **Live receipts (splat_f4c9416afb)**: footprint 541×768 renders the pool deck;
  heading 405.5→45.5; geo round-trips /status; KML/GeoJSON correct; clear → geo null
  (no visible state left); unauth 401; suggest [] on the GPS-less pool .insv (23k
  embedded records, 0 GPS tags — camera GPS was off). Deployed via splatlab-safe-restart
  (no jobs in flight), healthz token:true.
- Backlog ideas (NOT started): compass/north arrow in the 3D viewer once located;
  geo-anchored DXF/LandXML export (ties into the deferred survey-export gap);
  batch-locate from a GPS-tagged capture at upload time.

## DIGITAL TWIN KERNEL — P0 through P6a (2026-07-21/22, backfilled 07-22)
Splat → simplified/georeferenced solids kernel; plan `~/.claude/plans/fuzzy-foraging-moore.md`
(approved 07-21). This entry condenses P0–P5 (all shipped+deployed 07-21) and closes out P6a
(07-22) — full detail lives in memory `digital-twin-kernel-program-2026-07-21.md` and
`~/reports/2026-07-21-twin-kernel-day-digest.md`; this is the durable on-repo summary that was
previously missing.

- **P0 (e3ad59d/a76202b)**: opt-in `mesh` stage, `POST /jobs/{id}/mesh` — champion TSDF recipe
  (`gs-mesh o3dtsdf --voxel-size 0.015 --sdf-trunc 0.045 --depth-trunc 6`, `TSDF_ALPHA_MIN=0.5`)
  ported from `dn-splatter-probe`. Garden reproduces 78.81% LCC.
- **P1 (30a3550/a2cc5d4)**: `POST /jobs/{id}/geo/contours` — ground extraction (Track C) → cdt
  `survey_to_surface` → real EX-CONT-MJR/MNR-layer contours. Semantic ground filter (P5a,
  174f498/bb9af02) via langfield relevancy; section/ISO receipts auto-generated.
  **GLB gotcha (658718f)**: open3d 0.19 `write_triangle_mesh(.glb)` silently corrupts + returns
  True — mesh_report.py writes GLB via trimesh + mandatory readback now.
- **P2 (4cc6705/d66f9e0)**: `POST /jobs/{id}/geo/export` — probe-derived grid calibration (never
  trust convergence-sign conventions, measure instead) → site.dxf / surface.xml (LandXML) /
  site.geojson.
- **P3**: blender-mcp wired (official ahujasid upstream) — Blender 4.5.11 LTS installed
  separately (system 4.0.2 can't GPU-render Blackwell/5090). Cockpit launcher `blender-cockpit`.
- **P5b/P5c (9edda81/b4c7891)**: `POST /jobs/{id}/objects {"query": "..."}` — name an object,
  get isolated splat + mesh + a TripoSplat-generated proxy ICP-registered onto it. Proven on the
  garden table (icp_fitness 1.0) and, 07-21 field capture, RToony's **fire hydrant**
  (Santa Rosa, 49 Canon T3i CR2s, all 9 stages green from one POST, object mesh 100% LCC).
- **WS1/WS2 gates (215630d/2fb5ad7)**: `mesh_gate.py` (PSNR/SSIM/coverage vs source photos) +
  `mesh_completeness.py` (solid-gaussian distances) — quality receipts, not narrative.
- **Fix wave (3c9a90b)**: 14 confirmed findings from a 22-agent autonomous review closed in one
  commit (config-poisoning, guard hoisting, cached-mesh 500s, subset-ckpt leaks, stale-langfield
  guard, atomic contour staging). 407 tests at close.
- **ETH3D validation (840b121 + solidify-probe)**: laser-truth harness proves render-health gates
  (fog/acc) CANNOT certify mesh geometry — courtyard splat HEALTHY yet mesh vs laser = 130-171cm
  median error (sparse-view regime). Track B (depth-regularized extractors) blocked on a
  dn-splatter fork `--load-depths` compat bug — needs a proper fork-compat pass, not inline
  patches. Garden-class dense captures (RToony's actual capture spec) are a different, better
  regime.

### P6 — Scene Regeneration Lane (SAM3/TRELLIS-class), opened 07-22
Plan `~/.claude/plans/snazzy-gathering-dahl.md` (approved 07-22): enumerate + batch-isolate +
batch-proxy + assemble a FULL scene (not just one named object), all-local (SAM 3 + optional
TRELLIS.2). Doctrine: regenerated scenes are plausible-not-faithful — render/VR lane only,
mechanically enforced (quarantine dir + manifest + in-file tags + a refusal gate the survey lane
must fail against).

- **Step 0 de-risk spike: PROVEN, GO for P6b** (`~/tools/scene-regen-spike/STATUS.md`). The one
  algorithmic bet — SAM 3 text-prompted masks lift cleanly to per-gaussian instance sets via the
  existing PASS-B depth-gated lift + cross-view majority vote — holds with recipe
  `--min-views 2 --vote-frac 0.3`. Garden "table"/"flower vase" both PASS (vase lift actually
  *fixes* a known P5b langfield-query miss). Hydrant misses the pre-registered IoU floor
  (0.4994 vs 0.5) but root-caused as **reference contamination** (P5b's spatial expansion swept
  in ground-disc/shadow debris the SAM mask correctly excludes), not a lift failure — precision
  vs the reference = 1.000.
- **P6a SHIPPED + gate CLOSED (bdce4d3, then closed out same evening)**: `backend/mesh/
  provenance.py` (stdlib-only tag + quarantine rules + fail-loud survey-refusal, importable from
  every env), `proxy_register.py` retrofit (writes `transform_4x4` + `crop_camera_id` +
  `crop_box` + in-file PLY tag — needs `--crop-json` from `object_crop.py`'s new crop.json
  side-file), `scene_manifest.py` (schema + atomic writer + fail-loud validator), `sam3_doctor.py`
  preflight, `tools/gates/gate_p6a_scene_rails.sh`. 427 backend tests (20 new).
  ⚠️ **Gotcha**: the commit landed while `splatlab.service` was already running (started 16:49,
  commit at 20:13) — `proxy_register.py`/`object_crop.py` picked up the new code immediately
  (subprocess-invoked, read fresh each call) but `splat_route.py`'s new `--crop-json` wiring did
  NOT take effect until `splatlab-safe-restart` ran (it's imported once at process start). A
  post-commit rebuild without the restart silently produced a proxy.json missing
  `crop_camera_id` — always restart after a `splat_route.py`-touching deploy, not just when the
  gate says so.
  **Runtime close-out**: pinned `splat_513e89171d` (hydrant) + `splat_32d926d9` (garden) as
  scene-lane sources; restarted the service; rebuilt the hydrant `fire-hydrant` proxy (icp_fitness
  held at 1.0, `crop_camera_id: 8`). `bash tools/gates/gate_p6a_scene_rails.sh` → **GATE_P6A: PASS**
  (8/8 checks) — first real structural receipt that P6a actually holds, not just compiles.
- **P6b SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-22 night)**: `POST /jobs/{id}/scene/
  inventory` — enumerates every object in a scene, not just one named query. New:
  `backend/mesh/scene_views.py` (K evenly-spaced views, adapted from the spike's select_views.py),
  `backend/mesh/noun_consolidate.py` (clean/dedupe candidate nouns via SigLIP-cosine + curated
  STUFF_TERMS ground/vegetation split — pure-stdlib classify_stuff/clean_nouns, heavy SigLIP import
  isolated inside siglip_dedupe() so the module imports with zero ML deps), `backend/mesh/
  scene_sam3_masks.py` (one SAM3 model load, loops every "thing" noun × view), `backend/mesh/
  instance_lift.py` (the productionized multi-noun PASS-B lift+vote, conservation bookkeeping,
  optional regression vs known `_objects/*/object_indices.npz`, receipts). Route mirrors `/objects`
  (lock, arbiter lanes `scene-views`/`scene-sam3`/`scene-lift`, 409 contracts, sam3_doctor preflight
  BEFORE any GPU work). `nouns` body field is an explicit override that skips Qwen3-VL/langfield
  auto-sourcing entirely — the cheap HITL safety valve. Real captured gaussians only, no generative
  tag needed (P6d proxies are where that applies). 442 tests (15 new). `tools/gates/
  gate_p6b_instance_inventory.sh` — PASS.
  **Live run on garden (`splat_32d926d9`, auto-sourced, no explicit nouns)**: Qwen3-VL proposed
  "round wooden table", "flower vase", "blue ball" (langfield-vocab worker was inactive this run —
  fail-soft, VL-only sourcing). Table: 38,327 members, 8/8 views, regression IoU 0.7656 vs the
  known reference (recall 0.8115/precision 0.9312) — matches the Step 0 spike's 0.764. **Flower
  vase: 1,198 members — EXACT match to the Step 0 spike's number**, strong cross-validation that
  production faithfully reproduces the proven mechanism. "Blue ball" (a real object under the
  table, confirmed in the crop receipt) correctly VETOED rather than hallucinated — SAM3/lift
  declined an unconfident detection instead of fabricating one. Conservation held (39,525 claimed,
  0 overlap). Receipts: `~/projects/splatcli/outputs/3d/splat_32d926d9/_scene/` (overlay +
  per-instance crops) — RToony's eyeball grade is the next checkpoint per the HITL doctrine.
- **P6c SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-23 night)**: `POST /jobs/{id}/scene/
  isolate` — materializes P6b's already-determined instances into Blender-ready per-instance
  `object.ply` + a `background.ply` complement. New `backend/mesh/batch_isolate.py`: reuses
  P6b's SAM3-lifted membership directly (NOT a re-run of the older per-query DBSCAN
  `object_isolate.py` clustering — that's exactly the mechanism P6b was built to supersede,
  proven on the vase case) with first-claim-wins bookkeeping across instances (largest first,
  P6b's own size-sorted order), and the `write_splat_ply`/`sanity_sum_ok` convention lifted
  verbatim from the proven `~/tools/langfield-isolate-probe/isolate_export.py`. Never aborts on
  a thin instance (`SKIPPED:too-few-members-after-dedup`, batch continues). Receipt: a REAL
  gsplat RGB render with claimed gaussians' opacity zeroed (not a photo overlay) — genuinely
  shows the scene with instances removed. 446 tests (4 new). `tools/gates/
  gate_p6c_batch_isolate.sh` — PASS.
  **Live run on garden**: both instances built clean (table 38,327, vase 1,198, zero overlap,
  `sanity_sum_ok: true`, 39,525 claimed / 1,287,086 background of 1,326,611 total).
  ⚠️ **Honest finding from the background-removed receipt**: the ball (never an instance,
  correctly excluded) stays fully intact, and walls/grass/hedge render clean — but the table
  leaves a **ghostly translucent disc** rather than a clean hole, and part of the vase structure
  is still faintly visible. SAM3's tight masks are high-precision but don't claim every gaussian
  contributing to an object's visual footprint (unlike the old expand-based DBSCAN, which
  over-captures on purpose) — some residual low-opacity gaussians near the object's location
  remain. Not a bug; a real limitation to weigh for P6e/P6f (a hole-fill/backdrop-completion
  probe is already parked in P6x if graded unacceptable). Receipts:
  `~/Downloads/splatlab-scene-isolate-garden-background*.png`.

## HYBRID RECALL-EXPANSION — `--recall-expand` on batch_isolate.py (2026-07-23, autonomous day session)
RToony's question after seeing the ghost-disc finding: combine SAM3's precision with the
older DBSCAN+expand approach's recall? Probed in `~/tools/hybrid-recall-probe/` (full
writeup + both scene-class verdicts there). **Answer: yes, but scene-class-dependent —
shipped as opt-in, not a default.**
- **Mechanism**: SAM3 core (trusted identity) → KDTree radius query around the core's own
  members (shell dilation, radius = `dilation_mult` × the scene's typical nearest-neighbor
  spacing — scale-invariant, not object-size-based) → kept only if SigLIP relevancy to the
  instance's own label clears `rel_floor` (reuses `gauss_emb.npz`, no model reload for the
  lookup). Reuses `object_isolate.py`'s proven relevancy math.
- **PROVEN on garden** (multi-object, 971 dense views): `dilation_mult=10, rel_floor=0.30`
  — table core 38,327→41,273 (+7.7%), the large translucent ghost disc from the original
  P6c run substantially cleared; grass/pavement/hedge/ball stayed clean (no debris pulled
  in). Live-verified through the production route (`{"recall_expand": true}`), receipts:
  `~/Downloads/splatlab-p6c-recall-expand-garden-final*.png`.
- **PROVEN INSUFFICIENT on hydrant** (single-object, 49 tight close-up photos): core-only
  removal (14,295 members) left the hydrant visually almost fully intact; `dilation_mult=10`
  barely moved it (verified NOT a code bug — opacity zeroing measurably applied, `opac.sum()`
  dropped by 11,236.8). A tight close-up capture has far more locally-redundant/overlapping
  gaussians contributing to an object's visual mass than either SAM3's mask or a modest
  dilation reaches — a deeper, different problem than garden's ghost-disc, left unsolved
  and explicitly flagged rather than force-fit.
- **Separate honest finding (garden vase)**: the dried flower/frond sticking out of the vase
  was never claimed by SAM3's "flower vase" core at all (a masking-granularity question, not
  a recall-expansion bug) — stays visible in both before/after as real background content.
- Shipped: `batch_isolate.py --recall-expand --gauss-emb <path> [--dilation-mult 10]
  [--rel-floor 0.30]`; route body `{"recall_expand": bool, "dilation_mult": float,
  "rel_floor": float}` on `POST /scene/isolate` (needs a built language field). 448 tests
  (2 new). `gate_p6c_batch_isolate.sh` extended, still PASS.
- **Next**: P6d — batch proxy + gated registration (loop unchanged P5c per instance under ONE
  20GB TripoSplat lease; `SKIPPED:<reason>` degrades to `provenance:captured`, never aborts).
  P6e (ground/environment), P6f (assembly+Blender+contamination gate) follow in order, each
  behind its own gate/receipt/HITL checkpoint per the approved plan.

## P6d SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-23, autonomous day session)
`POST /jobs/{id}/scene/proxy` — loops the unchanged, already-proven P5c crop→TripoSplat→ICP
chain over every P6c-built instance. New `backend/mesh/proxy_triptych.py` (capture crop |
generated proxy preview | registered overlay — top+front orthographic scatter, deliberately
simple matplotlib math over a full perspective render, to stay robust). Route phases: (A)
crops CPU-only, per instance, before any GPU; (B) TripoSplat generation for ALL instances
under ONE shared 20GB lease (not one lease per instance); (C) ICP-register + triptych,
CPU, per instance. Per-instance `SKIPPED:crop-failed|generation-failed|registration-failed`
never aborts the batch. 453 tests (5 new). `gate_p6d_batch_proxy.sh`: verifies every built
element carries registration numerics AND the in-file generative PLY tag (P6a provenance
rails) — PASS.
⚠️ **Gotcha**: `object_crop.py` needs an `object.json` with a bbox, which P6c's
`batch_isolate.py` didn't write until this same session — added it (writes bbox from the
ACTUAL final claimed points, not P6b's pre-dedup bbox). First live P6d attempt on garden
0-for-2 SKIPPED:crop-failed because `/scene/isolate` hadn't been re-run since that fix landed
— object.json genuinely didn't exist on disk yet. Re-ran P6c, then P6d succeeded. Lesson:
after ANY change to a script another phase's route depends on, re-run the UPSTREAM phase
before testing the downstream one, not just restart the service.
**Live-verified on garden**: both instances built, **icp_fitness 1.0 on both** (table
rmse 0.0232, vase rmse 0.0099). Triptych receipts show real, informative signal: the
table's proxy is a near-perfect top-down silhouette match (thin captured-blue rim visible
at the disc edge) with minor leg-shape difference in the front view; the vase's captured
point cloud is visibly more diffuse/scattered than its clean regenerated proxy (consistent
with recall-expand's high candidate-acceptance rate on that instance, noted honestly, not
hidden). Also honestly notable: TripoSplat regenerates whatever the crop photo shows, so
the "table" proxy includes the vase sitting on it too (crops include visible context, not
just the tightly-masked object) — a real characteristic of crop-based generation worth
knowing before P6f assembly. Receipts: `~/Downloads/splatlab-p6d-proxy-triptych-*.png`.
- **Next**: P6e — ground + environment (`semantic_ground.py` wired into the scene lane,
  persisted `ground_gaussians.npz` → TIN → `ground_mesh.glb`). Then P6f (assembly + Blender +
  contamination gate). A comprehensive multi-agent adversarial review pass over everything
  shipped today (hybrid recall-expand, P6d) plus a fresh look at P6a-c is still queued before
  end of day, per the approved autonomous-run plan.

## P6e SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-23, autonomous day session)
`POST /jobs/{id}/scene/ground` — real captured ground gaussians → a scene-unit TIN →
splat-colored Y-up GLB. Render/VR lane only (`provenance:ground-derived`), deliberately NOT
the survey lane (`POST /geo/contours`/`ground_extract.py`, which needs a real CRS + geo
anchor this scene doesn't have — garden has scale but no anchor). New: `backend/mesh/
ground_mesh_build.py` (reuses `ground_extract.py`'s proven cell-bin + 15th-percentile-z +
spike-rejection + largest-connected-component algorithm VERBATIM, just staying in scene
units instead of transforming to ENU/CRS — no geo/scale requirement at all), `backend/
mesh/ground_mesh_receipt.py` (top+oblique renders, same Open3D EGL `OffscreenRenderer`
pattern as `mesh_report.py`, `defaultUnlit` shader to read the GLB's baked vertex colors).
Reused as-is: `semantic_ground.py` (already wired for P5a/geo_route, langfield-spike env),
`twin_finish.py` (6-NN color transfer from the scene's existing `_preview/splat.ply` — no
new export needed). 458 tests (5 new). `gate_p6e_ground.sh` — PASS.
**Live-verified on garden, worked cleanly first try** (no debugging detour, unlike P6d):
8,578 ground points, 17,139 triangles, real extent 9.25m × 11.51m footprint (0.95m
vertical range — correctly flat terrain). Receipts show genuinely legible, correctly
colored ground (green grass, tan pavement patch) in a real, irregular garden-boundary
shape. Honest finding: one small pyramidal spike artifact visible near center in the
oblique view — a likely single outlier cell the spike-rejection pass didn't catch: noted,
not chased further (timeboxed). Receipts: `~/Downloads/splatlab-p6e-ground-*.png`.

## COMPREHENSIVE REVIEW + FIX WAVE (2026-07-23, autonomous day session, 44-agent workflow)
5 parallel reviewers (correctness/security/resource-safety/test-quality/doctrine) over the full
`840b121..HEAD` diff (P6a close-out through P6e, 3,644 lines) → adversarial refute-by-default
verification (3 lenses per finding). **13/13 raw findings survived verification** — all real.
9 fixed this session (low-risk, well-scoped, matched "never break working things"); 4 left for
RToony's call (design-decision territory: DELETE/reset semantics for `/scene/*` artifact trees).

**Fixed:**
- **`scene_inventory` (P6b) now merges `meta["scene"]` instead of overwriting it** — this is the
  ONE route users are told to re-POST (the documented HITL correction flow), and it was silently
  erasing `isolate`/`proxy`/`ground` summaries every re-call even though the on-disk artifacts
  stayed untouched. **Live-verified on garden**: re-POSTing `/scene/inventory` gave `inventory` a
  fresh `built_at` while `isolate`/`proxy`/`ground` kept their original timestamps and data intact.
- `scene_inventory`'s `_work` scratch (frames/masks, can be multi-GB) now cleans up on EVERY exit
  (try/finally), not just the two success paths — every failure branch used to orphan it.
- `SceneInventoryBody.nouns: []` (explicit "there's nothing here") no longer collapses to the
  `None`/absent case and silently re-triggers Qwen3-VL + langfield auto-sourcing.
- `scene_inventory`'s zero-things short-circuit now still calls `audit_operator_event` — it used
  to return from inside the lock before reaching the audit call, silently dropping the record of
  a real GPU-time operation from the audit trail.
- `scene/isolate`'s `recall_expand=true` path now runs `_langfield_stale_guard()` before spending
  any GPU lease (it only checked `gauss_emb.is_file()`, which is true even when STALE) — restores
  the exact "preflight before any GPU work" doctrine this same diff's own comments cite as already
  fixed elsewhere, that `recall_expand` had quietly reintroduced.
- `scene/isolate`'s GPU-lease budget bumps to 10GB when `recall_expand=true` (was flat 6GB, didn't
  account for the extra SigLIP2 model load the flag adds on top of the checkpoint+render pass).
- **New `backend/mesh/slugify.py`** — the ONE noun→slug function, replacing two
  character-for-character-identical copies in `scene_sam3_masks.py`/`instance_lift.py` (pinned
  behavior-identical via a direct comparison test). **New slug-collision guard in
  `noun_consolidate.py`** (`dedupe_slugs`): two nouns differing only in punctuation (e.g. "Fire
  Hydrant" / "Fire-Hydrant") used to both reduce to slug `fire-hydrant` and silently clobber each
  other's SAM3 masks/instance files across a noun boundary — now the second is vetoed with a clear
  reason instead.
- `proxy_triptych.py`'s `_overlay_tmp.png` scratch file now cleans up on every exit (try/finally)
  instead of only after a fully successful run.
- Test-quality gaps closed: the ground route's `semantic_thresh`/`cell_units` now assert they
  reach `ground_mesh_build.py`'s argv (previously untested — a field-rename bug would've passed
  the whole suite); added coverage for the fail-soft `_langfield_worker_inventory() -> None` path
  STATUS.md already documented happening in production.
- 468 backend tests (up from 458; 10 new, all 5 gates re-verified PASS after a live re-run).

**Left for RToony (design-decision territory, not silently decided):** none of `/scene/isolate`,
`/scene/proxy`, or `/scene/ground` clear their prior output directory before a re-run with
different tuning knobs (e.g. raising `min_members`) — instances that stop qualifying leave stale
`object.ply`/proxy artifacts on disk indefinitely. No DELETE/reset route exists for any `/scene/*`
tree today. Fixing this means deciding what "re-POST with different params" should MEAN (wipe and
rebuild? keep both? explicit reset endpoint?) — a real product-contract call, not a bug fix.

## P6f NOT STARTED — out of scope for today's approved autonomous-run plan
Assembly + Blender + contamination gate is the final P6 phase but was explicitly out of scope for
today (the approved plan stopped at P6e + review + digest). Natural next session.
