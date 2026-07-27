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

## P6f SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-23 night) — the fidelity dial
`POST /jobs/{id}/scene/assemble` — the final P6 phase: assembles P6b-e's outputs into one
`scene.blend`/`scene.glb`. Built around RToony's explicit framing that this tool deliberately
straddles "digital twin" (perfect fidelity) and "reimagined 3D scenario" (creative AI
interpretation) — **that choice is now a real, explicit, user-controlled dial, not an implicit
default**: `mode: "faithful"|"styled"` (default) + an optional `overrides: {slug: "captured"|
"proxy"}` map layered on top (e.g. "styled scene but keep the hydrant real"). Every element
records WHY it ended up where it did (`selection: {mode, chosen, available, reason}`) — a
`faithful`-mode deliberate exclusion and an upstream proxy failure are otherwise indistinguishable
months later. An override naming an unbuilt/unknown slug is a hard fail (a deliberate creative
choice, not a best-effort batch op) — live-verified against real garden data before any Blender
work: `faithful`/`styled`/mixed-override modes all resolved correctly.

**OSS research done first** (RToony asked): Gaussian Grouping (ECCV 2024, Apache-2.0) validates
the identity-consistent-grouping direction conceptually but needs training-from-scratch in its
own pipeline, not nerfstudio — not adoptable now, worth remembering if the hydrant-class recall
problem is ever revisited from scratch. OpenUSD 26.03 (shipped THIS MONTH) added native 3DGS
support and its `purpose`/variant-set system is architecturally the industry answer to the same
fidelity duality — but tooling is a converter-script-only, no Blender/glTF-grade ecosystem yet.
Watch, don't build on, either.

**New**: `backend/mesh/scene_assemble.py` (pure-stdlib fidelity-dial resolution, no GPU/Blender),
`backend/mesh/blender_assemble.py` (headless build). **Retrofit** (additive): `scene_manifest.py`
gained an optional `selection` field on `add_element`.
- **Real bug found and fixed by testing against the actual Blender binary, not assumed**: a
  vertex-only (zero-face) point-cloud mesh is SILENTLY DROPPED by Blender's own glTF exporter
  ("Mesh has no primitives and will be omitted") — every splat element would have vanished from
  `scene.glb` while the script still reported success. Fixed by baking real triangulated geometry
  (Geometry Nodes Instance-on-Points → tiny cube → Realize Instances) before export instead of
  relying on a points-only representation.
- Per-element `try`/`except` lives INSIDE the one headless Blender process (the only place the
  "kill one instance, scene still completes, element flagged" requirement can be enforced, since
  the whole manifest builds in one `blender --background` call, unlike P6c/d's one-subprocess-
  per-element pattern) — **chaos-tested live**: a missing file for one element still produced a
  complete scene with the other 3 elements built and the bad one flagged with a clear reason.
- glTF export passes `export_extras=True` explicitly, followed by a **mandatory readback**
  (independently re-parse the written GLB's own JSON chunk) — same discipline as the P0 GLB fix
  (a writer that silently corrupts output while returning success). This readback IS the
  contamination gate: a `captured`/`ground-derived` node must NOT carry the generative tag, every
  `proxy` node MUST — verified by hand against the real exported GLB, not just the script's own
  self-report.
- Resumability = full idempotent rebuild from the manifest, never incremental `.blend` patching
  (matches how P6b-e are already safe to re-POST; sidesteps Blender's node-name collision
  suffixing on partial re-imports).
- The one mandatory HITL stays separate on purpose: `POST /scene/assemble/approve` is the only
  way `state` becomes `"approved"` — never automatic. Live-verified: build → `state: "built"` →
  approve call → `state: "approved"`.
- **Real bug found via the live run, not caught by mocked tests**: the route never actually
  patched the manifest's `state` from `"building"` to `"built"` after a successful assembly — the
  unit test's mock had accidentally hardcoded `"built"` directly, papering over the gap. Fixed
  (route now promotes state explicitly) and the test fixture corrected to write `"building"`
  (matching the real script's `new_manifest()` default) so the suite itself would catch a
  regression here going forward.
- 486 backend tests (18 new across 2 test files). `gate_p6f_assembly.sh` — PASS, including an
  independent glTF-extras re-verification (not trusting the app's own report).
- **A/B receipt** (the fidelity dial, literally visible): `faithful` vs `styled` render of garden
  from the same camera. Honest finding: the quick receipt-render script's first pass used a naive
  min/max bbox for camera framing and got blown out by far-field background outliers (whole scene
  reduced to a tiny fleck) — fixed with the same robust-percentile bbox `blender-receipt-views.py`
  already proved necessary for exactly this. The resulting renders show real, correctly-composed
  geometry with a genuine (if visually subtle at this render quality) difference between modes —
  the point-cloud-as-tiny-cubes visualization technique itself is a rough first pass, not tuned
  for real image quality, and would benefit from more work if a polished render is wanted later.
  The pipeline correctness (manifest, tags, gates) does not depend on this receipt looking good.
  Receipt: `~/Downloads/splatlab-p6f-ab-faithful-vs-styled.png`.

**Live-verified on garden end to end**: `styled` mode assembled all 4 elements (background,
table+proxy, vase+proxy, ground) in ~4s, contamination gate passed, approved. A separate
`faithful`-mode build (scratch location, not clobbering the approved styled build — the open
re-run-semantics question from the review is still genuinely open, deliberately not resolved
here) also assembled cleanly for the A/B comparison.

- **Not done, deliberately**: adding new generative capability (e.g. "place an object that was
  never captured") — P6f only chooses among representations of what P6b-e already produced. Pure
  imagination is a future P6x-style capability, kept separate on purpose.
- This closes out the entire P6 Scene Regeneration Lane (P6a through P6f) as originally scoped.

## P6 CONTROL PANEL (GUI) SHIPPED (2026-07-23, RToony /plan: "word salad" -> GUI-first plan)

RToony: "the central space to control and understand what tools we are making is important for
my ongoing understanding and future reference" — P6a-f had been REST-only all day (six endpoints,
zero frontend). This closes that gap: a full front-end control panel for the whole scene-regen
lane, no backend changes (all six routes/reports were already exactly what the UI needed).

**New**: `frontend/src/components/scene-regen.tsx` (`SceneRegenModal`) — a lazy-loaded full-screen
modal mirroring `geo-locate.tsx`'s exact mount pattern (`job`/`onClose` props, `lazy(() =>
import(...))` + `<Suspense>` in `splat-view.tsx`). Six `useMutation`s (inventory/isolate/proxy/
ground/assemble/approve) live in the modal shell and share one `busy` flag that disables every
trigger across all five stage tabs at once — lifted to the parent (not per-panel) specifically so
`busy` survives switching tabs mid-build. A NEW tab strip, not `StageRail` — deliberate, matches
the plan's reasoning: `StageRail` is contractually tied to one job's auto-advancing single-process
`stage` field; P6 is six independently-triggered, freely re-runnable endpoints with no unifying
stage.
- **Data model**: `SplatSceneSummary` (+5 detail-report interfaces) added to `contracts.ts`,
  5 `fetchScene*()` GETs added to `api.ts` (`.../file?fmt=report` convention, matching every other
  scene-file route already shipped). Each panel prefers its own mutation's just-returned report
  over the polled GET (`mutation.data ?? reportQuery.data`) — shows the real result immediately
  instead of waiting on the `["status"]` poll to update `job.scene` before the on-demand query
  even enables. Caught live: without this, the Assemble panel's Approve button stayed wrongly
  disabled right after a passing build.
- **Live-verified the entire API contract layer against garden** (curl + `PORTAL_TOKEN`, not
  guessed from source reading alone — no interactive browser tool available in this environment,
  so this is the strongest verification achievable here; the app's own login-cookie auth is what
  the real browser uses for both JSON and `<img>` tag requests, same as the existing thumbnail
  pattern already in production). All 5 `fmt=report` GETs, all 6 image/model file endpoints (crop,
  overlay, isolate receipt, proxy triptych, ground top/oblique, assemble glb/blend) returned real
  200s with correct content-types against garden's actual P6a-f output.
- **Two real, if minor, type inaccuracies found by that live check and fixed**: (1)
  `SceneIsolateReport.job_id` was declared required but the GET path (which serves
  `batch_isolate.json` verbatim — `batch_isolate.py` never stamps it with `job_id`, only the POST
  route's in-memory response does) never actually has it; now `job_id?: string`. (2)
  `SceneInventoryInstance.regression` can be wire-`null` (not just absent) for an instance with no
  reference match — the panel's truthiness check (`inst.regression && ...`) already handled `null`
  correctly at runtime, but the type now says so honestly (`| null`).
- **Fidelity-dial UI**: segmented `faithful`/`styled` control + a per-object override table (one
  row per isolate-built instance; proxy option disabled+tooltipped with its skip reason when no
  proxy was built for that slug) — lifted to the modal shell so it survives tab switches, per plan.
  After a build, each row also shows its live `selection.reason` from the manifest. Approve gate
  is its own visually-separated bordered block (amber "not approved yet" vs. emerald "approved"),
  matching the one-mandatory-HITL doctrine P6f was built to enforce.
- **Staleness banner**: no server-side cascade invalidation exists (re-running inventory doesn't
  mark isolate/proxy/ground/assemble stale) — the UI is the only place that catches it, by
  comparing `built_at` across `job.scene.*`. Live-verified this actually fires correctly: garden's
  real data has `inventory.built_at` (14:31) newer than `isolate.built_at` (13:37) from an
  in-session re-run earlier today, and the Isolate panel correctly shows "Built from an earlier
  inventory — re-run isolate…".
- **Gallery integration**: `SceneCard` gets one more badge (amber "scene assembled" for
  `state:"built"`, emerald "scene approved" for `"approved"`); `DownloadMenu` gets `Scene .glb`/
  `Scene .blend` entries gated on `state==="approved"` specifically (not `"built"`) — surfacing an
  unapproved draft as a top-level gallery download would quietly defeat the one mandatory HITL
  gate. A preview download of the built-but-unapproved GLB/blend stays inside the Assemble panel.
- Error surface bypasses `api.ts`'s `apiRequest()` (which throws the raw response body, often
  FastAPI's `{"detail": "..."}` JSON) in favor of a local `postJSON()` that parses `.detail` —
  matches `spark-scene-viewer.tsx`'s existing precedent. The three known 409/503 causes
  (`require_heavy_work_admitted()`'s maintenance/backup interlock, `_mesh_export_lock`'s "already
  running", and each route's own prerequisite check) already write distinct, human-readable
  `detail` strings server-side — no client-side remapping needed, just don't mangle them into a
  JSON blob.
- **Verification**: `npx tsc --noEmit` — exactly 43 errors both before and after, none in the new
  files (matches the established pre-existing baseline). `npm run build` clean (one transient
  esbuild crash mid-session from the same session-scope task-pressure class already diagnosed for
  `gate_p6a_scene_rails.sh` earlier today — not a code issue, retry succeeded with identical output
  hashes). 486 backend tests still pass (confirm-untouched baseline; no backend changes this pass).
  Live API-contract verification against garden as described above. **Not done**: an actual
  interactive-browser click-through — no browser automation tool is available in this environment,
  so DOM rendering/interaction itself is unverified beyond the build succeeding and the contract
  layer matching exactly. RToony should click through it once against garden before trusting it.
- **Accepted for v1, not fixed**: no progress signal for the multi-minute blocking POSTs (proxy
  especially) — an indeterminate spinner + "don't close this tab" copy, honestly, per plan. A
  refreshed/closed tab mid-request can't distinguish "still running" from "crashed" until the next
  poll.
- **Deferred, not started**: Phase 2 (Reference CAD/DXF alignment) — scoped in the plan, explicitly
  not picked up until Phase 1 ships and gets real use.

## VIEWER RENDER-LOOP PAUSE-ON-HIDDEN (2026-07-23, RToony live-test feedback)

RToony, testing P6 live at splatlab.roonytoony.dev: reported a suspected "resource hog animation"
on visiting a scene. Real, confirmed: `viewer.start()` (`@mkkellogg/gaussian-splats-3d`, the
classic viewer) and the hand-rolled `animate()` loop (`spark-scene-viewer.tsx`, Spark beta) both
run an unconditional `requestAnimationFrame` loop for as long as `/view/:jobId` stays open — even
for a 100% static scene nobody's touching. Not a bug (real-time WebGL splat rendering needs a
continuous loop by nature), but a legitimate, low-risk improvement: browsers already throttle a
*backgrounded* tab's RAF to ~1Hz, but that's still real render work every second, forever.
- **Fix**: both viewers now pause on `document.visibilitychange` (`document.hidden`). Classic
  viewer uses the library's own `start()`/`stop()` pair (confirmed via source read: `stop()` just
  `cancelAnimationFrame`s the loop, no GPU buffer teardown — cheap, instant resume). Spark beta's
  own loop checks `document.hidden` at the top of `animate()` and skips the real work (still
  reschedules RAF so it resumes instantly on refocus).
- **Not done, deliberately**: true render-on-demand (only re-render when the camera actually
  moves, foreground-and-idle) — a materially bigger, riskier change (would need to audit every
  per-frame effect — highlight markers, camera overlay, dimension labels — for hidden
  continuous-update assumptions). RToony's own framing ("if not, nevermind") didn't call for it;
  revisit only if the pause-on-hidden fix proves insufficient.
- Verification: `tsc --noEmit` steady at the 43-error baseline, `npm run build` clean. No backend
  changes. Not yet live-clicked by RToony (report-and-fix cycle, not yet re-confirmed by him).

## OBJECT-LANE TWIN FINISH SHIPPED + LIVE-VERIFIED (2026-07-23 night, RToony /plan)

RToony wants to shoot small bounded objects (hydrant-class, ~5'×5') with 5-10x denser coverage
(250-500 photos vs the field-proof's 49) and get back "decent yet reduced voxel/3d face... more
recent and better detailed" — a clean, colored, art-directable Blender mesh, not a raw scan dump
and explicitly not the "PS2 graphics" blocky low-poly he first floated. Direct code + live-artifact
inspection found the geometry side (P5b `/objects` isolation → TSDF mesh) was already excellent
(hydrant: 100% LCC, fully connected) — the actual gap was entirely on the finishing side: the raw
object mesh shipped with **zero decimation and only pale TSDF-baked color** (confirmed: `mesh_
report.py`'s own `defaultLit`+sun-light receipt already shows faint color from TSDF integration
itself — "zero color" in the plan was slightly imprecise; "pale, unenhanced color" is accurate,
matching `twin_finish.py`'s own docstring). `twin_finish.py` (WS3 — 6-NN gaussian color transfer +
pymeshlab quadric decimation + Y-up vertex-colored GLB + mandatory readback) already solves exactly
this and was already shipped/proven on two other routes — it had just never been wired into
`/objects`.

- **New `ObjectIsolateBody` fields**: `finish: bool = False`, `finish_target_faces: int = 10_000`
  (1k-100k range, request-overridable). Color source is the object's OWN `object.ply` (not the
  whole-scene splat) — verified field-for-field compatible with `twin_finish.py`'s loader
  (`object_isolate.py` writes raw/pre-activation `f_dc`/`opacity`, matching what the loader
  decodes itself) — scoped, ~140x less data to KDTree-index than the whole scene, and structurally
  guarantees no background-gaussian color leak at boundary vertices.
- New block in `POST /jobs/{id}/objects`, after the existing mesh block: runs `twin_finish.py`
  (`mesh.ply` + `object.ply` → `mesh/twin.glb`), then best-effort colored receipts via
  `ground_mesh_receipt.py` reused **unmodified** (already fully generic on any Y-up vertex-colored
  GLB). New `_OBJECT_FILES` entries (`twin`/`twin-top`/`twin-oblique`) + `twin_glb_url` in the
  response. No changes needed to `twin_finish.py`, `object_isolate.py`, `checkpoint_subset.py`, or
  `mesh_report.py` — confirmed the subset-checkpoint's `finally: shutil.rmtree` cleanup (which runs
  before the new block) is a non-issue since `twin_finish.py` never touches the checkpoint, only
  plain mesh/splat file paths that already exist and are never deleted.
- **`finish_target_faces` default 10,000**: the scene-level 400k default would never even trigger
  decimation on an ~90k-tri object (silently a no-op) — 10k is a ~9x reduction against the
  hydrant's real 89,990 raw tris, in the conventional low/mid-poly hero-prop range, not a "PS2"
  cut (~500-2k) or a near-lossless pass-through.
- **Capture-density guidance**: extended the one existing, already-firing `_recapture_guidance()`
  photo-orbit string with a dense-single-object recommendation (250-500 photos, stage via
  `splatcli/inputs/<name>/` + `input_path` to skip the 2GB web-upload cap) — deliberately NOT a new
  precheck heuristic (Tier-0 has zero access to real-world scale pre-SfM; a wrong auto-fired
  advisory costs more trust than it's worth, per the metric-trust doctrine).
- **Verified**: smoke-tested `twin_finish.py` directly against the hydrant's real on-disk artifacts
  BEFORE touching any route code (cheapest possible proof of the one real risk) — exit 0, 89,990
  raw tris → exactly 10,000, genuine color (per-channel std ~46, confirmed programmatically, not
  eyeballed). 490 backend tests pass (486 baseline + 4 new: builds twin.glb with the right args/
  object.ply scoping, `finish=True` without `mesh` is a loud 400, `finish_target_faces` override
  lands in the command, a finish failure is a loud 500 that leaves the already-succeeded raw mesh
  artifacts intact). **Live end-to-end regression** against the real hydrant job through the actual
  GPU pipeline (`splatlab-safe-restart` deployed, 0 active jobs first): re-isolated + re-meshed +
  finished, numbers matched the smoke test exactly (6,919 verts / 10,000 faces / 19,659 solid
  gaussians / 231,708 GLB bytes), all three new file endpoints 200, `meta.json["objects"]
  ["fire-hydrant"]["twin"]` populated correctly. Receipts + both GLBs copied to `~/Downloads/
  splatlab-hydrant-twin*`/`splatlab-hydrant-raw-mesh.glb` for RToony's own Blender eyeball —
  the actual "does this look good" call is his, not scriptable.
- **Not done, deliberately** (per plan): no `mesh_gate.py` PSNR/SSIM wiring for objects (would be
  structurally misleading — scores against full uncropped photos, dominated by background pixels
  an object mesh never claims); no photo-count heuristic in precheck; no size-adaptive
  `finish_target_faces` formula; no 2DGS/SuGaR/MILo reconstruction swap (stays a documented future
  "watch" item). The actual dense-capture field test (250-500 photos, real subject) is RToony's own
  hands-on work — not scripted here.

## PRESENTATION TOOLKIT PHASE 1 SHIPPED (2026-07-24, RToony /plan) — crop-sphere + dimensions

RToony wants to turn SplatLab into something that produces real, shareable files: named the Spark
tools (Locate/Add dimension/Paint), an official site-section generator, a radius-sphere crop to
discard poorly-captured surrounding context, and a direct "am I asking too much?" about
semantic-driven mesh smoothing. Researched, planned as 3 phases; this ships Phase 1.

**Correction found by reading the code**: "Locate" isn't a Spark tool (it's the pre-existing
`GeoLocateModal`, unrelated). Of the two real Spark tools, Paint was already fully complete
(server-persisted). Add Dimension was fully functional but session-only (`sessionStorage`) — the
one real gap.

- **Crop-sphere tool** — the backend was already 100% done: `edit_ops.py`'s `CropSphereOp` maps
  directly onto `splat-transform -S`, whose own docs say exactly what was asked: "Remove Gaussians
  outside sphere." Zero backend changes. New UI in `spark-scene-viewer.tsx`: click-to-place center
  (reuses `raycastAt()`), a red wireframe sphere gizmo, a live "N of M splats would be removed"
  count computed client-side via `PackedSplats.forEachSplat` (debounced 120ms) and fed into the
  same overlay-tint machinery Paint already uses, a two-step confirm (reusing the in-file
  `recalibrateArmed` idiom), POST to the already-tested `/edit/apply`, and a minimal "Undo crop"
  button against `/edit/revert` using the apply response's `version_before`.
- **Real bug found and fixed as part of this work**: the mesh-load effect was keyed only on the
  preview URL string, which never changes after an edit even though the file on disk does — the
  viewer would keep showing stale geometry after a crop. Fixed with a `reloadNonce` query param
  bumped on every apply/revert.
- **Server-persisted dimensions**: new `backend/dimensions_route.py` (pure stdlib, no numpy —
  confirmed the main backend's own venv has neither, matching `edit_ops.py`'s documented
  constraint), mirroring `geo_route.py`'s "small CPU-only feature + own router" shape rather than
  `langfield_overrides.py` (wrong process, numpy/index-file id scheme this data doesn't need).
  `GET/POST/DELETE /jobs/{id}/dimensions` + `.../export?fmt=csv|json`. POST is a client-id-keyed
  upsert (not server-assigned) so drag-to-move reuses the same route. `dimensions.json` at the job
  root. Frontend: sessionStorage kept only as an instant-restore/offline fallback, server is
  authoritative; inline CSV download link next to "delete all," shipped now (not deferred) since
  it's what actually answers "generate a file I can share" for this tool.
- **Verified**: 499 backend tests pass (490 baseline + 9 new for `dimensions_route.py`). `tsc
  --noEmit` steady at the 43-error baseline, `npm run build` clean. Live: full dimensions
  CRUD+export round-trip via curl against the real hydrant job (add → list → CSV export with
  correct computed length → delete → confirm empty); confirmed the deployed bundle contains the
  new crop/dimensions strings at the exact served asset hash.
- **Not yet done**: an actual interactive click-through in a real browser (no browser automation
  tool available in this environment — same limitation as the P6 GUI phase). RToony should try
  placing/dragging dimensions across a refresh (proves server persistence) and running a real crop
  (proves the reload-nonce fix + undo) before trusting this fully.
- **Phase 2 (site sections) and Phase 3 (curvature-adaptive smoothing) not started** — fully
  scoped in the approved plan, queued next.

## PRESENTATION TOOLKIT PHASE 2 SHIPPED (2026-07-24) — official on-demand site sections

Closes the "make site sections an official tool" ask. Previously `surface_receipts.py` (two
principal-axis cross-sections + an isometric ground TIN) only ran as an always-on, best-effort
tail step buried inside the much heavier `/geo/contours` CAD/DXF pipeline, with zero frontend
trigger — the only UI was a passive download link that appeared IF someone had already called the
API directly.

- **`GroundSampleParams` refactor** in `geo_route.py`: factored `ContoursBody`'s shared
  ground-sampling fields (epsg/cell_m/max_slope_deg/spike_tol_m/semantic/semantic_thresh) into a
  base class, backward-compatible (all 35 pre-existing `test_contours_route.py` +
  `test_geo_route.py` tests pass unchanged, proving it). Extracted the semantic-AUTO-fallback
  decision tree + ground_extract.py subprocess call (previously inline in `build_ground_contours`)
  into a shared `_extract_ground_points()` helper, so the two routes that both need it
  (`/geo/contours` and the new `/geo/sections`) can't drift out of sync.
- **New `POST /jobs/{id}/geo/sections`**: a genuinely lighter route — only needs ground-extract +
  `surface_receipts.py` (no CDT venv, no DXF authoring), sharing `_mesh/geo/` and its lock with
  `/geo/export`/`/geo/contours` but promoting only the files it just produced (verified: running
  either route after the other leaves the other's exclusive files — `contours.dxf` etc. —
  untouched). Unlike `/geo/contours` (where these images are a best-effort bonus on top of the
  DXF), a `surface_receipts.py` failure here is a loud 500 — it's this route's entire deliverable.
  v1 keeps the existing fixed "two auto-picked principal-axis sections + two iso angles"
  algorithm — no new tunable knobs, matching this project's "ship the baseline" pattern.
- **Frontend**: extracted `ReceiptLightbox` out of `scene-regen.tsx` into its own shared
  `receipt-lightbox.tsx` (mechanical, its props were already fully generic) so both it and the
  new modal use the same component. New small single-purpose `site-sections.tsx` modal — one
  trigger button, two result thumbnails, the shared lightbox — deliberately NOT folded into
  `GeoLocateModal` (scoped to map/anchor concerns) or `SceneRegenModal` (a different lane, P6
  object decomposition). New "Sections" header button in `splat-view.tsx` between Locate and
  Scene, same visibility/styling convention as its siblings.
- **Verified**: 508 backend tests pass (499 baseline + 9 new for `test_sections_route.py`). `tsc
  --noEmit` steady at the 43-error baseline, `npm run build` clean (new `site-sections`/
  `receipt-lightbox` lazy chunks). **Live end-to-end, not just mocked**: set a temporary synthetic
  geo anchor on garden (already had scale+mesh+langfield, just missing an anchor), ran the real
  route — 260,036 ground gaussians identified out of 1,326,611 total, 1,065 ground points, ~29s —
  pulled the actual rendered PNG and visually confirmed it: the garden table's silhouette is
  clearly recognizable and the ground TIN correctly traces the real terrain including the
  pedestal notch. Cleared the synthetic anchor afterward (the real generated images were left in
  place — legitimate artifacts, not test scaffolding). Confirmed the deployed bundle serves the
  new code at the live asset hash. Receipts copied to `~/Downloads/splatlab-garden-sections.png` /
  `splatlab-garden-surface-iso.png`.
- **Not yet done**: an actual interactive click-through in a real browser (no browser automation
  tool available in this environment). RToony should try the "Sections" button on a scene that
  already has scale + a real Locate anchor set.

## PRESENTATION TOOLKIT PHASE 3 SHIPPED (2026-07-24) — curvature-adaptive mesh smoothing

The concrete, ship-now answer to "am I asking too much?" about semantic-driven surface smoothing.
Dedicated research this session found the semantic version (SAM-mask-driven region-specific mesh
edits) is real published work (Häne et al. 2017, PlanarGS NeurIPS 2025, MagicRoad July 2025) but
never shipped as a general, robust feature — the hard part is multi-view-consistent 2D→3D label
fusion, not labeling. The mature, no-semantics-needed answer: curvature-adaptive mesh denoising,
already one filter call away in `pymeshlab`, already a `twin_finish.py` dependency.

- **Filter**: `apply_coord_two_steps_smoothing`, confirmed directly against the installed
  pymeshlab (2025.7.post1) parameter surface. Feature-adaptive (`normalthr` degree threshold
  excludes sharp face-pairs from normal-averaging, so rims/bolt heads stay crisp) and
  shrink-resistant by construction (fits vertex positions to the smoothed normal field rather than
  averaging neighbor positions — the actual mechanism that makes plain Laplacian shrivel a mesh,
  which is why plain Laplacian was rejected). Defaults: `smooth_iterations=2`,
  `smooth_feature_deg=40°` (tighter than pymeshlab's own 60° default).
- **Pipeline-order bug caught by the mandated pre-wiring smoke test, not by RToony**: the approved
  plan called for smoothing AFTER decimation (cheaper, fewer verts). Built exactly as planned, then
  smoke-tested `twin_finish.py --smooth` against the real hydrant `mesh.ply`/`object.ply` per this
  session's own "smoke-test the real subprocess call before wiring the route" discipline — the
  rendered receipt showed clearly WORSE results than unsmoothed: blocky, faceted, crumpled patches.
  Root cause: `apply_coord_two_steps_smoothing` fits positions to a locally-averaged normal field,
  which needs enough face resolution to express a smoothly-varying field — the already-decimated
  10k-face mesh didn't have it. Fixed by reversing the order: smooth the full-resolution raw mesh
  (~47k verts/90k tris) first, then decimate the now-clean surface. Concrete before/after on the
  same hydrant mesh: surface-area change vs. the unsmoothed baseline went from **-6.47% (faceted,
  bad)** in the wrong order to **-3.56% (clean noise removal)** in the corrected order; verts/faces
  and extent stayed effectively unchanged in both. `transfer_attributes_per_vertex` (color
  re-projection from the pristine mesh-0 source) now fires whenever EITHER smoothing or decimation
  moved a vertex, not only on decimation.
- **Route wiring scoped to `/objects {"finish": true}` only, this pass** — not `/mesh` or
  `/scene/ground`. This is the one lane with a real, proven smoke-test artifact (the hydrant), and
  a single global feature-angle threshold is a safer bet on one isolated object than across a
  heterogeneous scene. `ObjectIsolateBody` gained `smooth`/`smooth_iterations`/
  `smooth_feature_deg`; `finish_cmd` in the `/objects` route passes them through as `--smooth
  --smooth-iterations N --smooth-feature-deg D` when requested. The script-level change is shared
  and cheap to extend to `/mesh`/`/scene/ground` later — a mechanical follow-up, not a redesign.
- **Verified**: 510 backend tests pass (508 baseline + 2 new wiring-correctness tests in
  `test_objects_route.py` — assert `--smooth`/`--smooth-iterations`/`--smooth-feature-deg` appear
  correctly when requested and are absent when not; the smoothing algorithm itself is proven by the
  hand-run smoke test above, not re-proven in the mocked-subprocess unit test). **Live end-to-end**:
  `POST /objects {"query": "fire hydrant", "mesh": true, "finish": true, "smooth": true,
  "smooth_iterations": 2, "smooth_feature_deg": 40.0}` against the real hydrant job through the
  full GPU pipeline (re-isolation + re-mesh + finish+smooth, 45.9s) returned
  `twin.smoothing: {"applied": true, "iterations": 2, "feature_deg": 40.0}`, verts=6921,
  faces=10000, extent=[0.47,0.47,0.47] — matching the smoke test exactly (reproducible). Pulled the
  live-route-produced GLB and both twin receipts (`fmt=twin`, `twin-top`, `twin-oblique` all 200
  OK) and visually confirmed: smooth cylindrical hydrant body, nozzle caps and rim edges still
  crisp, no faceting. Copied to `~/Downloads/splatlab-hydrant-twin-smoothed.glb` +
  `-oblique.png`/`-top.png`.
- **Not yet done**: RToony's own Blender/eyeball call on whether the smoothing defaults
  (2 iterations, 40°) look right on other object shapes beyond the hydrant — this was the one
  planned verification step explicitly deferred to his own hands-on judgment, same as the finish-
  stage verification earlier this session.
- **Explicitly deferred, not attempted**: semantic-mask-driven region-specific smoothing (genuine
  open research problem per this session's research, not a near-term deliverable) and RANSAC
  plane/cylinder primitive-fitting (mature for planes, confirmed unreliable for cylinders even in
  established libraries — directly relevant to the hydrant, so deliberately not attempted). Both
  remain documented "watch" items.

**All 3 phases of the Presentation & Editing Toolkit plan are now shipped.**

## AUTORESEARCH MARATHON — 4 workstreams built, real trials run (2026-07-24, RToony /plan)

RToony asked for a 6-8h automated test series using "the karpathy/autoresearch system" to compare
splat generation, twin-finish simplification, and DXF exports against source photos as ground
truth. Found RToony already runs a real local implementation of this pattern
(`~/projects/autoresearch-lab`, `splatlab-mesh-quality` project, 28 prior trials). Built 3 new
sibling autoresearch-lab projects + extended the existing one. **Actual wall-clock: 1h24m, not
6-8h** — flagged honestly below, not padded to look like more happened.

- **New `eval_splat_holdout.py`** (`~/tools/dn-splatter-probe/mesh-trial/`): fills a confirmed gap
  — no held-out-photo eval existed anywhere for the raw Gaussian splat. Uses nerfstudio's own
  first-party `eval_setup()` + `get_average_eval_image_metrics()` against the 10%-holdout test
  split nerfstudio already computes and silently discards every training run. Carries the same
  `torch.load(weights_only=False)` monkeypatch already proven in `mesh_gate.py`.
- **New `splatlab-splat-holdout` project**: `baseline-30k-reuse`/`fresh-7k`/`fresh-15k`/`fresh-50k`
  on hydrant + garden. Fresh trials write to an ISOLATED `trials/` dir, never the real job's
  `processed/splatfacto/` tree (a newer config.yml landing there could silently become "latest" for
  production lookups — checked and confirmed clean after the run).
- **New `splatlab-twin-finish-quality` project**: sweeps Phase 3's smoothing params against real
  photos via `mesh_gate.py` (unchanged) pointed at each trial's `twin.glb`. **Caught and fixed a
  real bug before any real trial ran**: `twin_finish.py` rotates its GLB output to glTF's Y-up
  convention (correct for Blender), but `mesh_gate.py`'s cameras expect the scene's native Z-up
  frame — first smoke test rendered the hydrant sideways. Fixed via `twin_gate_prep.py` (inverts
  the rotation before scoring only; production `/objects` output is untouched).
- **New `splatlab-dxf-quality` project**: sweeps `cell_m`/`max_slope_deg`/`spike_tol_m` via the
  real `ground_extract.py`/`contours_build.py` scripts directly. Set a clearly-labeled TEST geo
  anchor on garden (`source: manual`, placeholder lat/lon) since it had none — **flagged here for
  RToony to replace with a real anchor**.
- **Extended `splatlab-mesh-quality`** with 4 strategies from its own documented "Frontier"
  backlog (lambda sweep, dn-Poisson+outlier-removal, cropbox-to-reference) — all using stock
  `gs-mesh` CLI flags, zero `mesh_trial.sh` changes needed.
- **Incident, self-caught and fixed mid-run**: `lambda-lo-s012` thrashed `scene_5177f8d99a`'s
  memory envelope under the `splatlab.slice` cgroup (8G swap cap maxed, 83-90% iowait, ~26min
  negligible progress) — killed by hand, root-caused (forces a fresh full finetune, a combination
  never exercised on this scene before), scoped the 2 lambda strategies to the other scene, and
  rebuilt the marathon script to loop per-strategy (was per-project, which hid the stall entirely)
  with a 1h per-strategy runaway cap. Relaunched clean; finished in 1h24m with zero further issues.

### Real findings (measured, not narrated)
- **Splat holdout (garden/hydrant, PSNR/SSIM/LPIPS vs real held-out photos)**: existing 30k-iter
  checkpoints are already near-optimal — garden 30k ssim=0.8426 vs 50k ssim=0.8447 (~flat); hydrant
  30k ssim=0.6411 vs 50k ssim=0.6381 (no gain). 7k is measurably worse on both (garden 0.7741,
  hydrant 0.6513 — note hydrant's 7k number is *slightly above* its own 30k baseline, within noise
  at n=4 eval images). **Validates the current 30k default; no iteration-count win found.**
- **Twin-finish smoothing (SSIM vs real photos, 3 objects)**: no-smooth baseline scored *highest*
  (0.069, only `accepted=True`); all smoothed variants scored 0.060-0.069, with more aggressive
  decimation (5000 faces) hurting most. **This does not contradict Phase 3's shipped defaults** —
  it measures a different axis. Phase 3's own eyeball check judged geometric cleanliness (less
  faceting); this metric measures per-pixel photo similarity, which decimation/smoothing can
  slightly hurt even when the surface looks visually cleaner. Both are real, legitimate, and
  different questions.
- **Mesh-quality (TSDF trunc sweep)**: `tsdf-s020-reuse` (looser truncation, 0.20 vs the prior
  0.12 champion) is the new numeric champion on `scene_5177f8d99a` — 90.92% LCC @ 75% coverage vs
  the previous 55.3%/74% at trunc=0.12, in this run. **Important honesty check**: pulled the actual
  receipt renders for both — visually, BOTH are heavily spiky/jagged with TSDF fusion noise from
  every camera angle tried (top, interior). The LCC×coverage metric confirms "one big connected,
  room-sized blob," not "a clean, presentable mesh." Neither the old nor new best config produces
  something visually clean — a real, humbling limitation of the current v4 scoring metric, not
  a new regression. `scene_98095cb055`'s fragmentation problem remains open (27% LCC even at the
  new champion config; `cropbox-ref-s012-reuse` didn't fix it either, 0.2732).
- **DXF ground-sampling**: all 7 param configs produced non-degenerate, audit-clean output (no new
  warnings beyond the constant "watermarked provisional" flag every trial gets). `cell_m` has a
  large real effect on point density the score doesn't capture (0.1→10,783 points/615 contours,
  0.5→631 points/92 contours) — the per-trial score formula only varies with audit-warning count,
  which didn't change across configs; a real, honestly-noted scoring-design gap for this project.

### Honest gaps / what's NOT done
- Ran **1h24m of the requested 6-8h** — the infrastructure is real and reusable, but this was a
  single pass through modest grids, not a deep multi-hour search. Scaling up (larger grids, more
  seeds per fable5 anti-pattern #6 on single-seed noise, extending `--smooth` to `/mesh`/
  `/scene/ground`, a semantic-ground axis for W4) is queued, not done.
- `splatlab-dxf-quality`'s score doesn't discriminate parameter sensitivity (see above) — would
  need a real stability metric (e.g. contour-length variance across configs), not just audit
  warnings, to rank ground-sampling settings meaningfully.
- Pre-existing, unrelated bug found in passing: `nexus-notify --title/--message` (used by both the
  existing `overnight_runner.sh` and this session's new marathon script, copied verbatim) doesn't
  match the current `nexus-notify` CLI (Slack-only, positional message, no `--title` flag) — the
  completion digest silently failed to send. Not fixed (out of scope tonight), flagged here.
- All 4 autoresearch-lab projects + `generator.py`/`eval_splat_holdout.py`/marathon script changes
  are uncommitted in `~/projects/autoresearch-lab` (separate repo from splatlab) — not committed,
  per standing "only commit when asked" — RToony's call on whether/how to commit that repo.

## AUTORESEARCH RUN 2 (2026-07-24, same day, RToony back at the PC) — 1h48m, follow-up strategies

RToony explicitly did NOT want another 6-8h run while he was actively using the PC — asked to scale
up depth instead of duration, iterate in short bursts, and use the data to plan a longer sustained
12-24h run for tonight's ~12-24h absence. Added 10 new strategies across all 4 projects, each a
direct follow-up to Run 1's own findings (not a blind bigger grid). 49 total strategies (up from
39), finished in 1h48m (vs Run 1's 1h24m) with zero incidents — the per-strategy-loop fix and
memory-envelope scoping from Run 1 held.

- **W1 truncation curve extended past s020 — numbers climb, but it's still metric-gaming, not a
  real win.** `tsdf-s024-reuse` (0.6484) and `tsdf-s028-reuse` (0.7432) both now PASS the gate on
  BOTH scenes (up from s020's 0.4727, pass_rate 0.5 — only one scene passing). Pulled the actual
  renders for s028 on BOTH scenes before trusting this: **identical spiky, incoherent TSDF fusion
  noise as every prior "champion," on both `scene_5177f8d99a` and `scene_98095cb055`.** Looser
  truncation is fusing more noise into one bigger connected blob, satisfying LCC×coverage without
  producing anything visually usable. The v4 scoring metric (survivor of v1-v3's own documented
  gaming problems) is itself now gameable by truncation alone — **do not treat any of s020/s024/
  s028 as real recipe improvements.** Cleanup combined with s020 (`clean10/20-s020-reuse`,
  0.4727/0.4715) didn't change anything — cleanup doesn't touch this failure mode.
  scene_98095cb055's fragmentation problem remains genuinely unsolved.
- **W2 iteration curve now well-bracketed (7k/15k/20k/30k/40k/50k), and the two scenes tell
  different stories.** Garden (18 held-out images): clean, real diminishing-returns curve —
  7k=0.7738 → 15k=0.8311 → 20k=0.839 → 30k=0.8426 → 40k=0.8429 → 50k=0.8451. Flattens hard by
  15-20k; everything past that is a rounding error. Hydrant (4 held-out images): **not
  monotonic at all** — 0.6529/0.6547/0.6483/0.6411/0.6447/0.6365 across the same iteration range,
  a ~0.018 spread with no discernible trend. With only 4 eval images this is almost certainly
  measurement noise, not a real iteration-count effect — can't draw a hydrant-specific conclusion
  beyond "nothing catastrophic happens across this range." **Actionable**: 15-20k iterations
  captures nearly all the achievable held-out fidelity on richer captures; the current 30k default
  isn't wrong, just not obviously necessary either.
- **W3 smoothing gap is real but tiny, and lighter settings win.** `smooth1-f25-t20000` (1
  iteration, 25° feature threshold, 20k target faces) scored 0.0693 — the only smoothed variant to
  beat the no-smooth baseline's 0.0690, driven by a small flower-vase gain (0.049 vs 0.046) with
  fire-hydrant unchanged and round-wooden-table slightly lower. `smooth2-f15-t10000` tied baseline
  exactly (0.0690). **This is a noise-level difference at 3 objects, not a strong result** — but
  directionally, the lightest/most conservative smoothing settings are consistently the ones that
  don't hurt photo-fidelity, while heavier settings (more iterations, looser feature threshold,
  smaller target_faces) consistently cost a little. If Phase 3's defaults get revisited, lighter is
  the supported direction, not heavier.
- **W4 boundary strategies found real structural effects the score still doesn't capture.**
  `cellm-xloose` (cell_m=1.0): 184 points / 352 tris / 56 contours — a real, large drop from
  baseline's 2118/4214/169, genuinely approaching (not yet crossing) a too-sparse-to-trust regime.
  `spike-xtight` (spike_tol_m=0.1): 1718 points / 3418 tris / 105 contours — also real
  thinning (~19% fewer points, ~38% fewer contours than baseline), consistent with aggressive spike
  rejection discarding legitimate terrain variation as false positives. Neither triggered actual
  degeneracy (0 points, audit failure) — the boundary is further out than tonight's grid reached —
  but both confirm the score's blindness (still 0.85, identical to baseline) is a real scoring gap,
  not proof the params don't matter.

**Bottom line for planning the 12-24h overnight run**: the two workstreams worth real overnight
depth are W2 (splat-holdout — clean, trustworthy, real signal, room for a genuinely thorough sweep
including the previously-deferred SfM-backend comparison) and a redesigned W1 (the LCC×coverage
metric itself needs work before more truncation sweeping is worth running — a v5 metric or a
paired visual-receipt gate should probably be built before spending overnight compute chasing more
of the same gameable signal). W3's signal is real but small; W4 needs either a real stability
metric or to just keep pushing boundaries further out.

## PORTABLE PIPELINE + RESTRICTED BLENDER MCP SHIPPED (2026-07-24) — SPZ/SOG/GLB export, UE 5.6 handoff, research ledger

Delivered: checksummed export manifest + portable-format API (`backend/artifact_manifest.py`,
`backend/export_route.py` 1164 lines — SPZ v4, CPU SOG, streamed SOG, KHR_gaussian_splatting GLB,
collision, caching, stale-source detection, lock-guarded rebuild), wired into the live app at
`main.py` (`app.include_router(export_route.router, prefix="/api/splat", ...)`). Frontend DownloadMenu
(`frontend/src/pages/splat.tsx`) gained a "Portable pipeline" section (Build formats / Package UE 5.6),
new `SplatExportManifest`/`SplatUnrealBundle` types in `contracts.ts`, new
`fetchPortableExports`/`buildPortableExports`/`buildUnrealBundle` calls in `api.ts`.

**NEW restricted Blender MCP** (`backend/dcc/blender_mcp_server.py`, port **9877**, `127.0.0.1`
streamable-http): exactly **8** `@mcp.tool` functions (inspect_job, inspect_blend,
list_blender_versions, snapshot_blend, toggle_collection, transform_object, restore_blender_version,
open_blender) — no exec/Python/URL-fetch tool, no arbitrary code execution. Backed by
`blender_workflow.py`: atomic `.building-` → final staged writes, symlink rejection, env-allowlisted
subprocesses. **This is a separate, distinct server from the EXISTING general-purpose Blender MCP on
port 9876** (official `ahujasid/blender-mcp` addon, GUI-attended via `blender-cockpit`, documented
2026-07-21 P3 COCKPIT). Do not conflate the two: 9876 = general, GUI-attended, full addon surface;
9877 = restricted, 8-tool, headless, no code exec. Currently a script
(`integrations/blender/run-mcp.sh`) with its own `.venv` — no systemd unit today; would need a
`nexus-manifest.json` entry only if later turned into a persistent service (not done now).

Also shipped: `integrations/unreal/` — `bundle_tool.py` (symlink/zip-bomb guarded), verify/stage
PowerShell scripts, renderer probe table (NanoGS/MLSLabsRenderer/UnrealSplat),
`docs/portable-interchange.md` (format table + typed `POST /exports` API docs). `research/` ledger
(`README.md` "What Was Incorporated", `sources.json` with 17 license-reviewed candidates,
`benchmark.py`/`capability_probe.py` harness).

**Verified independently, not trusted from narrative:** `~/.local/bin/pytest` on the 8 new test files
→ **64 passed, 2 skipped, 0 failed** (skips are opt-in real-binary smoke tests, correctly not
exercised in a normal pass). `ruff check` on all new/modified `.py` files: clean. `tsc --noEmit`: 0 new
errors in the 3 touched frontend files (23 pre-existing errors elsewhere, unrelated). No GPU workload,
no backend restart, no UE launch/compile occurred during delivery or during this verification pass.
Windows UE 5.6 target workstation ("Triforce": i7-7700/64GB/RTX 2080 SUPER 8GB, Win10 Pro 22H2)
confirmed online and hardware-sufficient the same day, but Unreal Engine itself is not yet installed
there and remote access (SSH/WinRM/RDP) is not yet open — SMB only — so the UE-side half of this kit
remains unexercised end-to-end.

Two unrelated AUTORESEARCH MARATHON/RUN 2 sections landed in this same file from a separate,
still-in-progress session — left uncommitted on purpose, not swept into this delivery; RToony's call
on if/how to commit that content separately.

## PROFESSIONALIZATION WAVES 1-3 SHIPPED (2026-07-25) — hygiene, tokens/brand, IA shell + first real export proof

RToony approved the full professionalization plan (~/.claude/plans/dazzling-tumbling-horizon.md —
7 waves: GUI polish, native Edit mode wiring the existing 9 edit ops, Export Center, Spark
consolidation, self-hosted SuperSplat roundtrip). Decisions locked: converge on Spark then delete
classic; native-first editing with SuperSplat as escape hatch; tokens + à-la-carte Radix (no
shadcn); interleaved quick-wins sequencing. Waves 1-3 shipped today, all frontend deploys via dist
swap (no restarts except the one noted below):

- **Wave 1.1** (`b0f021b`): all 14 non-classic-viewer tsc errors fixed; flat ESLint
  (typescript-eslint + react-hooks, compiler-era rules demoted to warn) + Prettier + `npm run
  check`/`lint`. Remaining 9 tsc errors are all in splat-viewer.tsx (classic — deleted in wave 7);
  `check` stays red by design until then.
- **Wave 1.2** (`99d7b6d`): CSS-var design tokens (--surface/--ink/--accent*/--radius-*) mapped
  into Tailwind; radius drift collapsed to a 3-step scale; @fontsource self-hosted fonts (Google
  Fonts @import gone); favicon.svg + OG/theme-color; ALL user-visible naming standardized to
  "SplatLab" (incl. backend login page).
- **Wave 1.3** (`8b66845`): DownloadMenu extracted to components/gallery/download-menu.tsx and
  rendered on /view (export parity with home cards) + **first-ever real-data portable export run**
  (below).
- **Wave 2.1** (`e64df4c`..`11d8430`): components/ui/ primitive library — Radix Dialog/Tooltip/
  Tabs/DropdownMenu (radix-ui 1.6.7, React 19 OK) + hand-rolled ToastProvider + Skeleton +
  EmptyState; all three hand-built modals (site-sections, geo-locate, scene-regen) migrated onto
  Dialog (focus trap/Escape/aria); app-wide toasts replace the setTimeout div; spark panel inputs
  unified onto Input size="xs".
- **Wave 2.2** (`964cd77`, `3826975`): splat.tsx decomposed 1,458→689 lines by pure moves —
  lib/stage-meta.ts, lib/format.ts, components/create/*, components/jobs/*, components/gallery/*.
  Extraction verified byte-identical.
- **Wave 3** (`757f2a2`): AppShell nav (Scenes / New capture / ⋯→Feedback) wraps /, /new,
  /feedback; pages/splat.tsx DELETED, split into pages/scenes.tsx + pages/new-capture.tsx (query
  keys unchanged — in-flight jobs survive navigation); route ErrorBoundary, styled 404, first-run
  EmptyState, skeletons, status-feed Retry card. ⚠️ Behavior note: Promote-to-full-build now always
  uses fresh-load defaults (standard/no-langfield/no-mesh).

### ⚠️ FOUND LIVE: the 07-24 portable-export delivery was never running
The service last restarted 07-24 06:04; the export/UE/collision commits landed 18:00 — production
"Build formats" pointed at routes that didn't exist (SPA catch-all answered 405). Fixed with an
idle-window `splatlab-safe-restart` (no jobs in flight). **Deploy rule reaffirmed: any
backend/*.py change needs a safe-restart to go live — verify with /openapi.json, not git log.**

### First real-data export proof (splat_32d926d9, 1.32M gaussians, SH-3)
- **SPZ v4 ✅** 313MB→30MB in ~30s, sha256 verified against manifest `files[].sha256`.
- **GLB (KHR_gaussian_splatting) ✅** 317MB, valid glTF-2 header, sha256 verified.
- **UE 5.6 bundle ✅** built in 22s, 627MB zip, 14 entries (gaussian ply/spz/glb, twin/mesh glb,
  approved scene.glb/.blend, survey artifacts, receipts), zipfile integrity OK.
- **SOG ❌ REAL LIMIT**: CPU SOG at default sog_iterations=10 exceeds CONVERSION_TIMEOUT_S (60
  min) on 1.32M gaussians → `-1 conversion exceeded 3600 seconds`. Wave-6 Export Center must
  default SOG iterations low / warn on big scenes; consider GPU SOG or a bigger timeout as backend
  follow-up.
- **streamed-SOG ❌ UPSTREAM BUG**: splat-transform v2.7.1 `writeLod` throws "Missing lod
  assignment" with `--lod-chunk-count`/`--lod-chunk-extent` on this scene. Needs a minimal repro +
  upstream issue (or flag-order fix) before wave 6 exposes the knobs.
- Manifest gap for the UI: failed artifacts carry the log in `error`, no short `reason` — Export
  Center should render `error` tail when `reason` is absent.

Next: wave 4 (Spark feature port + /view workspace tabs), wave 5 (Edit mode), wave 6 (Export
Center), wave 7 (classic deletion + SuperSplat self-host, restart window 2).

## PROFESSIONALIZATION WAVES 4-7.2 SHIPPED (2026-07-25, same day) — Spark default, workspace tabs, Edit mode, Export Center, SuperSplat self-host

Continues the waves 1-3 section above. All shipped and LIVE; classic-viewer deletion (wave 7.1) is
the only remaining plan item — gated on RToony's live soak + explicit sign-off.

- **Wave 4.1** (`a4e7acd`): Spark viewer parity port — fly-to, camera-frusta overlay (DOM/SVG
  screen-projected like classic, so crop/paint picking can't double-fire), zoom-to/view-from-camera,
  WASD pan + arrow roll (OrbitControls/mkkellogg math mirrored), reset, trimmed shortcut legend.
  **Spark is now DEFAULT** (absent localStorage key ⇒ Spark; escape hatches: ⋯ menu toggle,
  `?viewer=classic` (unpersisted), `splatlab.sparkBeta=0`); classic remains the auto error-fallback.
  NOT click-tested headlessly — fly-to/frusta/WASD feel needs the live soak.
- **Wave 4.2** (`0b15ed6`): /view is a tabbed workspace — breadcrumb + status badge + one ⋯
  overflow menu (secondary path for every action, one release); tabs View | Measure | Objects |
  Edit | Export with per-mode toolbars. Viewer NEVER unmounts on tab switch (lanes overlay it).
  Spark tool panel = Measure-only via new `toolsVisible` prop; bottom drawer <1024px.
- **Wave 5.1** (`bdeab5e`): native Edit ops — crop BOX beside crop-sphere (shared predicate
  `previewRemoval()`, red-tint previews byte-identical for sphere; `-B` KEEPS inside, preview
  tints outside), floater cleanup / decimate (pct = KEPT) / batched T-R-S transforms in the Edit
  lane, all armed-confirm + verbatim {detail} errors + version_before undo. New `reloadToken`
  prop routes lane edits into the existing crop reload nonce (camera resets on reload — inherited).
- **Wave 5.2** (`00b8814`): restore-point timeline (GET /edit/versions) with per-version armed
  Restore; restore is itself undoable; 5-cap explained.
- **Wave 5.3** (`73b6e70` + `22553af`): Backend Batch A (restart window 1, taken 13:5x after
  0-jobs check): `langfield_stale` in _job_payload (same helper as the 409 guard — can never
  disagree), GET /api/splat/activity (arbiter holder + per-job lock flags editing/
  preview_exporting/meshing/exporting; restart-truthful by construction), GET /jobs/{id}/objects
  listing. Frontend truth layer: useActivity 4s poll, P6-modal "started elsewhere" banner (closes
  the STATUS.md:1304 gap), polled staleness replaces ALL local made-stale state, amber header
  badge, distinct "no field built" vs "field stale" gating copy.
- **Wave 5.4** (`bdff5ef`): semantic edit panel — text → threshold → delete|isolate|extract →
  armed confirm; real matched counts; extract links the derived scene. Request field is `text`;
  **extract does NOT mark the field stale** (only delete/isolate do — coded to source after my
  brief said otherwise). No client-side preview (deliberate scope cut).
- **Wave 6** (`3d5ab71`, `78d2e54`): Export Center (4 lanes: portable formats w/ knobs — SOG
  iterations DEFAULTS 2 everywhere incl. the DownloadMenu quick-build that previously posted {}
  ⇒ backend default 10, the exact config that timed out live; big-scene >5-iter warning; failed
  chips render the `error` log tail; collision surfaces as top-level manifest key w/ voxel_url +
  mesh_url — scene.voxel.bin only ships inside the UE bundle; post-hoc mesh + contours triggers
  with honest prerequisite gating) + Objects panel (GET listing + extract form w/ activity banner).
- **Backend follow-ups** (`e52eb5c`, `9e3bd31`): streamed-SOG "Missing lod assignment" was OUR
  bug, not upstream — lod-meta.json output requires LOD-tagged gaussians; `-l 0` injected after
  the input (repro: backend's exact shape fails in 0.04s, tagged succeeds; regression test pins
  the command). `/activity` gained the `surveying` flag (geo_route._GEO_EXPORT_LOCKS was the one
  missed busy-lane).
- **Wave 7.2** (`20537e6`, `e2940a1`, `020db47`): SuperSplat SELF-HOSTED — authed static serve of
  `SUPERSPLAT_DIST` (default ~/projects/supersplat/dist) at /supersplat/*, traversal-guarded,
  303→/login for unauthed humans (020db47 fixed the initial 401); **portal :3300 /supersplat proxy
  DELETED** (portal can now retire without breaking /view; PORTAL_ORIGIN survives only for
  healthz). POST /jobs/{id}/edit/upload: streamed multipart ingest of an externally-edited PLY as
  a first-class edit version (validate header → snapshot op="upload" → replace → regen → mark
  stale; 400/409/413 paths leave everything untouched). Edit lane "Import edited .ply" card +
  SuperSplat roundtrip guidance. **Restart window 2 taken + verified live** (303/200/200,
  edit/upload in openapi, proxy gone). nexus-manifest log entry added.

Tests: 540 → **567 passed / 2 skipped**. eslint 0 errors; tsc clean outside the classic viewer
(its 9 errors die with wave 7.1). Frontend deploys all dist-swap; two restart windows total,
both 0-jobs-verified via splatlab-safe-restart.

### Open / gated
- **Wave 7.1** (classic deletion + `tsc && vite build` gate + vitest smokes): NEEDS RToony's
  Spark soak verdict + sign-off. Until then `npm run check` stays red by design.
- Cloudflare ~100MB body cap applies to the new upload roundtrip through the public hostname —
  big edited PLYs need the LAN/Tailscale origin (same as the create lane).
- SuperSplat sw.js/PWA behavior under the new origin unverified until first live open.
- Multi-level LOD pyramids for streamed-SOG (current fix = single level 0, spatial chunking works).
- Collision/unreal_bundle have no per-artifact staleness (manifest-level only) — backend follow-up.

### Streamed-SOG fix PROVEN LIVE (2026-07-25 15:37) + throughput truth
`splat_30b75bc81f` (green-bottle, 289,115 gaussians), forced streamed-SOG at sog_iterations=2
through the live API post-restart: **status ready** — valid lod-meta.json (1 LOD level, chunk
tree with bounds), 9 files, 69MB PLY → 6.2MB, served through the relative-chunk route. The lane
went from instant "Missing lod assignment" crash to working artifacts.
**Throughput caveat (measured):** that build took ~44 minutes for 289k gaussians at iterations=2
(single chunk — whole scene fits one extent-16 chunk, so it's one big SOG encode). Extrapolating,
scenes ≳400k gaussians risk the 60-min CONVERSION_TIMEOUT_S even at low iterations. Streamed-SOG
is CORRECT now but SLOW — backend follow-ups if it matters: longer timeout for this format, GPU
SOG (`-g gpu`), or parallel per-chunk encoding. Export Center copy already warns builds can take
many minutes.

## WAVE 7.1 SHIPPED (2026-07-25) — classic viewer DELETED, builds type-checked. Program COMPLETE.

RToony signed off after the Spark soak ("proceed to the next wave"). `a5f43c9` (+ style commits
`013faaf`/`f91689c`: Spark tool panel + all three legends now SOLID #0a0f1a/kept hues, per his
call):
- DELETED: components/splat-viewer.tsx (mkkellogg classic), pages/spark-test.tsx (the 601-line
  spike route), mkkellogg.d.ts, @mkkellogg/gaussian-splats-3d dependency. Zero mkkellogg chunks
  in dist.
- Shared viewer prop types live in components/viewer-types.ts.
- /view renders Spark unconditionally; a viewer crash shows a recover card (no silent fallback);
  sparkBeta/localStorage/?viewer=classic/menu toggle all removed.
- Home featured pane = lazily-loaded Spark (3D chunk stays out of the landing bundle).
- `npm run build` = `tsc --noEmit && vite build` — gate PROVEN (injected type error → exit 2).
  tsc fully clean repo-wide; the grep-exclusion era is over. eslint 0 errors, vitest 9/9.

**The 7-wave professionalization program is COMPLETE.** Remaining ideas are backlog, not plan:
multi-level streamed-SOG pyramids / GPU SOG (throughput), per-artifact collision staleness,
SuperSplat save-back button (trigger: manual roundtrip proves annoying), merge-scenes GUI
(trigger: two geo-anchored scenes of one site), post-hoc langfield rebuild lane, light theme.

## EDIT TOOLS UNBROKEN + STEPPED PROGRESS (2026-07-25 evening) — first successful edits ever

RToony hit "Unsupported output file type: ...edit-tmp" on Clean up floaters (3× live 500s).
Root cause: `_edit_tmp_path` built temp names ending `.edit-tmp`; splat-transform v2.7.1
dispatches output format purely on suffix. FIVE sites shared it (apply + all three regens +
`.float-clean`); **no edit had ever landed on this machine** — the extension-blind test stub
kept 80 tests green against the wrong contract.

- **Fix** (`649b539`): temp shape `.edit-tmp.<token>.<real name>`; `.float-clean` same
  treatment. **Stub now enforces the extension contract** (falsification-proven: buggy helper +
  new stub fails the suite). Unit test on the helper.
- **Stepped progress** (`1c96bc9` + `a11815d`): in-memory EDIT_PROGRESS (begin/step/end at
  transaction boundaries; semantic adds leading "match"), additive `edit_progress` on
  GET /activity; frontend EditProgress strip (stage dots + human labels + 1s elapsed ticker +
  patience copy, indeterminate degrade), useActivity(fast) 2s poll during local edits,
  "Applying…" button text, server-truth "edit in progress (started elsewhere)" lock banner.
- **Crop lives in the Edit tab now** (`b948729`, RToony's call): `toolsVisible` →
  `panelSections: "measure"|"edit"|null` — Measure = search/paint/dims, Edit = crop sphere+box
  beside the op cards; leaving Edit disarms crop modes; <1024px drawer capped 30vh on edit.
- **Live proof (restart window taken, 0 jobs)**: keep-everything crop on the hydrant → HTTP 200
  in 8.7s, 0 warnings (all regens work), v1 snapshot; /activity showed apply(2/6)→compress(3/6)
  mid-flight. Revert v1 → 5 files restored. **Acceptance = RToony's real ask: Clean up floaters
  → 910,560 → 909,532 (−1,028) in 10.1s, GPU splat-edit lane visible as holder, restore point
  v3 left in place for undo.** Note: −1,028 is `-G` engine defaults — conservative; if streaks
  remain, the backend accepts tuning params the lane doesn't yet expose (backlog), and crop
  handles the rest.
- Tests 567 → **570 passed / 2 skipped**; frontend gates green (tsc-gated build, eslint 0,
  vitest 9/9).

## NAVIGABLE WORLD + CANDIDATE BAKE-OFF (2026-07-26) — splat → walkable three.js world, and generative candidates

13 commits on `object-calibration-staleness` (unpushed). Suite **626 passed, 2 skipped** throughout.

### What shipped
- **Texture bake** — `object_texture.py`: clean → simplify → xatlas unwrap → bake gaussian colour into a UV
  map. Decouples colour fidelity from face budget. Exposed as `texture:true` on POST /jobs/{id}/objects
  plus an Objects-panel control. Also takes a POINT CLOUD as geometry source (Poisson direct from
  gaussians) — the scene lane never meshes, so this is the normal path.
- **Whole-capture solidify** — `scene_solidify.py` (props + shell), `world_collision.py` (static/prop
  classification + CoACD `UCX_` hulls), `world_shell.py` (watertight walkable solid), `world_gate.py`
  (acceptance gates).
- **Walkable world** — `/world/:jobId`, three.js + three-mesh-bvh capsule collision, backend
  `/world/manifest` + `/world/file`. Renders the visual shell, COLLIDES against the solid.
- **Candidate bake-off** — `world_bakeoff.py` + `mesh_gate.py --transform`, registers candidates into one
  frame then scores against the real photos.
- **Parametric authoring** — `parametric_schema.json` + `parametric_build.py` (measure/validate/build).
- **Generative** — `object_generate.py` wrapping SAM 3D Objects (13 GB ckpts local).

### Findings that cost time — do not re-derive
- **Render geometry ≠ collision geometry.** Bonsai's TSDF shell is a lacy web (645 components, 67% floor
  continuity). NO decimation budget preserves connectivity (2,088 components even at 40%). The walkable
  solid must be voxelised separately — `world_shell.py`, 222k tris, watertight, floor continuity 1.0.
- **`/collision` was voxelising in the wrong frame AND uncropped.** Captures are Z-up, splat-transform is
  Y-up, so `--voxel-floor-fill` filled sideways; and no `--filter-box` meant 64×53×70 units of sky.
  Fixed: 6,826,202 tris → 138,556. splat-transform's contract is `input [ACTIONS] … output`, actions in
  order — the source must LEAD.
- **Open3D aborts the PROCESS** (uncatchable C++ terminate) on `enable_post_processing` for an image-less
  GLB, and on `np.asarray()` of an empty texture placeholder. Read the glTF JSON chunk first; call
  `tex.is_empty()` always.
- **pymeshlab screened Poisson never returns** (>240 s at depth 6 on 49k faces). Open3D: 1.5 s at depth 8.
- **xatlas is steeply superlinear**: 0.2 s @8k → 46 s @200k → 1564 s @400k. Measured 13× fix (chunked
  unwrap, 118 s) documented at the call site, NOT applied — it changes atlas layout.
- **Volume-based hull checks are meaningless** on density-trimmed Poisson meshes (not watertight). Use
  sampled surface coverage.
- **The object lane's `bbox_extent_m` is a MISLABEL** (fixed 2026-07-21). Older meshes carry scene units
  under that key. Bonsai is UNCALIBRATED.
- **PSNR ranks by alignment, not quality.** The operator-preferred generated hydrant scores LAST (9.83 dB
  vs 13.76) because its yaw is fitted. Only rank exactly-registered candidates. Fix = photometric pose
  refinement, not a different verdict.
- **DATA DEFECT: camera 0 of `splat_513e89171d` is badly posed** — ~6 dB shared dip across every
  candidate, object renders ~3× oversized. One bad frame in 45.

### Open
- Bonsai has no real scale (viewer guesses 0.9428 u/m from a 2.6 m storey; bicycle then reads 1.86 m).
- Visual shell still fails its own gates; chunked xatlas is the measured fix, pending a design call.
- `object_texture.py` reports coverage PRE-dilation (0.53) under the same name as the shipped atlas (0.98).
- 13 commits unpushed.
- **UE 5.6.1 installed on Triforce** (`G:\UE_5.6`, 25.76 GB, verified; Quixel Bridge + Fab plugin).
  Triforce: i7-7700 4c/8t, 64 GB, RTX 2080 SUPER 8 GB, Win10 19045, VS Build Tools 2022 present.
  C: only 27.8 GB free — keep engine work on G:. UE is a POLISH station; three.js stays the runtime.

## BLENDER+UE PIPELINE CONTINUATION WAVE (2026-07-26, second session) — UE first light staged, bundle catches up, polish round-trip

Plan: ~/.claude/plans/purring-coalescing-crane.md (approved). 7 commits eb11eea..850d727 on
`object-calibration-staleness`. Suite 626 → **666 passed / 5 skipped** (3 opt-in lanes run once
this session: mesh-env e2e ×2, real headless Blender ×1 — all green). Two safe-restart windows
taken (0-jobs verified), routes proven via /openapi.json.

- **B2 coverage honesty (eb11eea)**: `texture.coverage` now = the SHIPPED dilated atlas;
  `coverage_rasterized` = the pre-dilation fraction (was misfiled under one key: 0.53 vs 0.98).
  world_gate comparison reads the rasterized key for either report vintage. First-ever
  test_object_texture.py (12 tests incl. real-env e2e).
- **B1 bundle catch-up (080f7e6)**: UE bundle ships `World/` (shell, collision_shell, elements,
  UCX hulls w/ `collision_for`, combined per-prop `World/Props/<slug>.glb` — UCX binds only
  within ONE imported file; node-name preservation probe-proven) + `Objects/` (textured+atlas,
  bakeoff.json, **winner GLB with score metadata**; provenance: capture-native/ours →
  captured-derived, else generated=render-only; out-of-tree winners skipped visibly) + 4th
  actor child `WorldGeometry` + manifest `world` block + serve-time `world_current` flag.
  bundle_tool requires WorldGeometry only when world files present (skew-safe both ways).
  world_collision.py emits `UE_<slug>.glb` per prop (first CoACD-real file on next collision run).
- **A-lane: UE FIRST LIGHT STAGED ON TRIFORCE.** Preflight probed live (UE 5.6 CL-44394996 at
  G:\UE_5.6, MSVC 14.44, py 3.14, driver 566.03; **Windows SDK MISSING**, RDP disabled, also a
  UE_5.8 dir). `~/scripts/triforce-ue-station-prep.sh` (dry-run proven) = RToony's --apply gate
  for RDP+SDK. Staged on G:\splatlab-ue\: SplatLabUE56 project + PS scripts + **NanoGS v1.0.3
  prebuilt** (Plugins/NanoGS) + the REAL hydrant bundle (splat_513e89171d: exports built SPZ
  22.6MB/GLB 218MB, 398MB zip, byte-verified transfer) — probe/verify/stage receipts ALL GREEN
  on Triforce (staged_path SplatLabImports\splat_513e89171d\50a210ca9afe0019, 14 files, NanoGS
  selected; bundle carries Objects/fire-hydrant + winner-textured 13.76dB). Attended half =
  `integrations/unreal/first-light-runbook.md` (copy beside the kit on G:). Gotchas: Windows
  scp needs `-O` (sftp mode dies); CLIXML noise → grep -v; base64 -EncodedCommand for PS.
- **C1 polish round-trip (09949fd)**: `backend/glb_check.py` (stdlib GLB validation — Open3D
  BANNED here, hard-aborts on image-less GLBs) + `polish_route.py`:
  POST /jobs/{id}/objects/{slug}/polish + /world/elements/{slug}/polish (shell ok). edit/upload
  discipline: stream→validate→lock→version prior→atomic land→provenance receipt; every
  400/404/409/413 leaves the tree untouched (20 tests). Walker needs no change; world manifest
  gains additive `polished` marker; _OBJECT_FILES += polished/polish-receipt.
- **C2 UI (f1e3312)**: objects listing carries additive `bakeoff` verdict; "Walk world" button
  on /view when `world_available` (bonsai proven live); verdict line on object cards
  (formatBakeoffVerdict, vitest'd); Polished .glb in the downloads list.
  ⚠️ Route-decorator gotcha cost 12 tests: inserting a helper between @router.get and its
  handler registers the HELPER as the route (422s everywhere). Helpers go above the decorator.
- **D1 chunked xatlas (c7ac40d)**: `--unwrap-chunks N` (default 1 byte-identical; solidify
  passes 4 for the SHELL only). **Real bonsai shell @400k: unwrap 96s vs 1564s = 16×**, bake
  total 119s, shipped coverage 1.0, texture gate PASSES. Honest residual (scratch world_gate
  A/B, live tree untouched): shell_connectivity/floor_continuity still FAIL — 2,088 components,
  the EXACT recorded source-TSDF number → the blocker is source lacing, not unwrap cost; fix =
  bake onto the voxel-solidified shell (design call, backlog). Also seen: live world.json was
  rewritten 08:24 with shell:null (stale shell.glb on disk) and prop_integrity now 3/5 on the
  current tree — pre-existing drift, not this wave.
- **D2 typed export_glb (850d727)**: 9th allowlisted action on the restricted workflow
  (:9877 MCP tool `export_blend_glb`): zero free-form params, staged output, glb_check
  readback, receipt at _blender/exports/scene-vNNNN.{glb,json}. REAL headless Blender 4.5.11
  integration test run this session (2 passed). Loop closes: export_glb → polish upload.

**Awaiting RToony**: (1) `! bash ~/scripts/triforce-ue-station-prep.sh --apply` (RDP + Windows
SDK), then the attended first light per the runbook (screenshot receipt → G:\splatlab-ue\
receipts\); (2) branch is ~20 commits ahead, unpushed — push is his call; (3) visual-shell
voxel-bake design call.

## UE FIRST LIGHT ACHIEVED UNATTENDED (2026-07-26 ~11:55) — receipt ok:true
RToony ran prep --apply (RDP live; SDK add initially failed on MY Start-Process array-quoting
bug — path with spaces split into 3 tokens; fixed in the script, retried clean: Windows SDK
22621+26100 + NetFxSDK 4.8.1 installed). SplatLabUE56Editor + NanoGS compiled FROM SOURCE on
Triforce (UBT green, 419s; needed the NetFx SDK — Build Tools omits it). Headless first light:
`-run=pythonscript -nullrhi` imports WORK but actor spawn EXCEPTION_ACCESS_VIOLATIONs
(commandlet has no level-editor frame; recorded in firstlight.py docstring) → full-editor
`-ExecutePythonScript -RenderOffscreen` run SUCCEEDED: scene.scene + winner mesh imported,
FirstLight.umap assembled (SplatLabSceneRoot → GaussianRender + ConventionalGeometry), numeric
witness: dominant axis Z = UPRIGHT (NanoGS preserves +Z-up PLY, no axis correction — folded
into README); scale NOT auto-applied (bounds = whole cloud incl. background splats; operator
sets root scale). Receipt: G:\splatlab-ue\receipts\firstlight-splat_513e89171d.json. Vault:
`Nexus - Triforce Windows Login` (d550d669) added by RToony via vault-add-secret; verified
readable in-memory. Remaining attended: 30-second RDP eyeball of the FirstLight level + F9.

## BLENDER STUDIO LOOP — W1-W3 SHIPPED (2026-07-26 pm; plan purring-coalescing-crane v2)
- **W1 voxel shell (64423ba)**: scene_solidify `--shell-source voxel` + `--shell-only`
  (patches world.json in place). Bonsai: 41s bake, 120k faces → **world_gate shell gates
  FLIPPED to PASS** (1 component / 1.0 fraction / floor 0.94; was 2,088 components). Visual
  surface now coincides with the collider. prop_integrity still FAIL = pre-existing drift.
- **W2 polish UI (0af2e3f)**: PolishUploadZone (shared, .glb preflight/arm/progress) on
  object cards + per-row in the world Elements panel (incl. shell) with reloadNonce reload.
  All frontend gates green.
- **W3 Blender infra (b6979c5, 52b3c3f)**: `splatlab-blender-mcp.service` LIVE (user unit,
  loopback :9877; initialize handshake verified "SplatLab Blender") + registered as claude
  MCP `splatlab-blender` (✔ Connected; :9876 untouched) + manifest node/log +
  `~/bin/splatlab-blender` launcher (versions→_regen→GLB fallback; prints return path).
  **First-ever real _blender lane on splat_aea04ab3**: P6 assemble via API (6 built, gate
  ok) → snapshot v1 → transform v2 → export_glb 6 meshes/81.5MB validated. Caught+fixed:
  glTF export of GN-driven assembled scenes needs export_apply (readback gate flagged the
  zero-mesh export exactly as designed). Polish-upload leg of the loop deliberately awaits
  REAL polished content (W4) rather than landing test data in a live slot.
- **W4 pending RToony's capture** (scene w/ ground, 150-300 photos, taped distance, map
  pin — spec in the plan). Sequencing: derive (objects/P6/world) BEFORE destructive edits.

## SEMANTIC PAINT-TO-GENERATE PHASE — P1+P2 SHIPPED (2026-07-26 pm; plan purring-coalescing-crane v3)
Direction (RToony): crop → label → generative uses labels ("representative but faked").
Taxonomy fixed+extensible; first gen output = class-textured ground; testbed = bicycle.
- **P1 crop honesty (8b664db)**: "queued" lead step (host-lock waits visible — the recorded
  "takes forever" source), revert/upload rails, to_thread snapshots (event loop no longer
  freezes /activity), GLOBAL EditProgress under the tab row (survives tab switches/viewer
  teardown), off-by-one fixed (rail read done while running), warnings[] surfaced,
  bbox-derived slider ranges, placement-miss notice, honest preview-count copy. 668 tests.
- **P2 langfield rebuild (aa9c949)**: REALIGNMENT not re-lift — langfield_realign.py
  (exact-xyz map rebuild, total-or-nothing; painted records carried via xyz snapshots,
  now written on every paint commit; legacy via old-map chain) + POST /langfield/rebuild
  (edit-lock, rail, GPU lane only for one-time ckpt_xyz cache, worker /invalidate) + UI:
  stale banner + one-click Rebuild CTA in search AND crop surfaces, TEST PATTERN badge.
  681 tests. **LIVE PROOF on the hydrant** (stale from RToony's own edit session):
  rebuild in 16 s incl. first checkpoint load — 909,527 rows realigned, 1,610 edited-away
  rows detected, relevancy 200 w/ X-Count exact + hydrant found. Bit-exact xyz through
  splat-transform CONFIRMED on real edits (risk #1 retired).
- **Worker truth found**: splatlab-langfield.service binds :3425 (drop-in; :3417 code
  default is cerberus-studio's port), deliberately on-demand/disabled — language tools are
  down whenever nobody started it (`systemctl --user start splatlab-langfield`). Started
  this session with the new /invalidate + snapshot code.
- Open: P3 class-label painting → P4 class-textured ground/shell → bicycle E2E.

## PAINT-TO-GENERATE P3+P4 SHIPPED (2026-07-26 evening) — the loop is closed
- **P3 class painting (400aa3f)**: class_taxonomy.json (9 classes, each w/ generative
  meaning) + stdlib loader (extend-only job extras); class_labels.py store (overrides
  discipline + xyz snapshots DAY ONE); worker class_add/delete/map/summary; app routes
  (taxonomy validated before the worker; binary map w/ X-Class-Order); realign carries
  class records. Viewer: Label|Class brush toggle, palette chips, per-record delete,
  "Show class layer" via new direct-color modifier (no 4-channel limit). 692 tests.
- **P4 class-textured world (b5e2462)**: semantic_ground keeps class_rel [N,C] (+taxonomy
  queries, --live-map post-crop honesty, --class-labels = user paint ABSOLUTE precedence);
  ground_mesh_build per-cell vote → ground_class_cells.npz; class_textures.py (seeded
  procedural tiles; textures/<id>.png upgrade path) + ground_texture.py (planar-XY UV,
  world-space sampling, jittered boundaries, readback, provenance
  "captured-geometry/class-textured") FAIL-LOUD in /scene/ground (fmt=glb_classed/
  atlas_classed/report_classed); shell --class-map (xyz-keyed, composite post-dilation,
  per-class blend_capture) via scene_solidify's auto class-map. Survey rails proven
  untouched. 697 tests + probe-env e2e.
- **LIVE BICYCLE PROOF (machine half)**: POST /scene/ground on splat_3aaf8067 →
  10,689 cells auto-classed (grass 7,505 / dirt 1,360 / pavement 1,223 / gravel 601) →
  ground_classed.glb, texel fractions 61/17/13/9% — matches the scene's real lawn+path.
- **Awaiting RToony (attended half)**: paint pass on bicycle (Measure → Paint → Class:
  correct/override the auto classes, e.g. paint the gravel patch properly) → re-run
  /scene/ground (paint precedence) → optionally solidify --shell-source voxel for the
  classed shell + /world walk. Worker note: language tools need
  `systemctl --user start splatlab-langfield` (on-demand by design, :3425).

## SPARK EDITOR PROFESSIONALIZATION (2026-07-26 late; plan enchanted-shimmying-hollerith)
RToony: "polish the edit tools + make it a more professional software experience,
e.g. Esc should cancel paint mode." Frontend-only wave, `6081ee9..2fbb300`.
Backend untouched — no restart window, `npm run build` alone deployed it.

- **Baseline commit `6081ee9`**: the working tree already held an uncommitted first
  cut from 15:41 (log-scale sliders, class-coloured brush tint, first-cut Esc/[/]/Z).
  Landed alone as a rollback point before building on it. Also corrected a stale
  receipt: recent commits claimed "eslint 0 errors" but HEAD actually had 1 (unused
  `setStrokeBusy`) — proven by stashing the diff and re-running eslint.
- **`lib/viewer-shortcuts.ts` (`e9cb5da`)** — pure `resolveShortcut` / `escapeAction` /
  `stepRadius`. The input layer lives in ONE `useEffect([url])` and is unreachable
  from a test; there is no DOM test runner here. 21 tests, **falsification-proven**
  (inverted the Esc ladder + dropped the clamp → 3 failures; restored → green).
- **Esc is a ladder, and never eats work**: cancel a pending confirm → clear an
  unapplied placement → disarm. RToony's explicit call: an uncommitted paint
  selection is NEVER discarded by Esc. So the new **ToolHud** (top-centre, solid
  `#0a0f1a`) always shows `N splats selected · not committed [Discard]` even with no
  tool armed — that state was previously invisible-but-live.
- **Four disarm holes closed**: `measureArm` never disarmed paint (and `onClick`
  tests paint FIRST, so arming the ruler silently kept painting); leaving Measure
  left paint+ruler armed; the paint section could unmount on a stale langfield with
  `paintMode` still true; `pointercancel` was unhandled so a cancelled drag left
  `controls.enabled=false` **permanently** ("orbit broke"). Also: dblclick no longer
  places a crop centre AND yanks the orbit pivot; Ctrl+S reaches the browser again.
- **Keys**: `B`/`C`/`Shift+C`/`M` tools — and a tool key **switches to the tab that
  owns the tool**, so a shortcut can never arm something invisible. `[ ]` size
  (clamped to the slider's OWN bounds — one shared `brushBounds`/`cropBounds`/
  `boxBounds` definition), `Ctrl+Z`/`Ctrl+Shift+Z` (new redo stack, invalidated by
  any new stroke), `Enter` commits (arm-then-apply on destructive crops), `Del`
  discards, `1`–`9` pick a class (taxonomy is exactly 9), `?` help, `0` reset view,
  `H` hide chrome. Everything state-reading goes through a **ref trampoline** —
  a direct binding acts on whatever was true when the scene last reloaded.
- **ShortcutLegend rewritten**: it listed 3 camera lines and its comment claimed
  `F/G` + `=/-` bindings **that never existed in Spark**. Every row is now wired and
  tested. `Kbd` chip promoted out of world-view.
- **SizeControl**: log slider + a typed exact box — you cannot land on 0.37 m twice
  by dragging, and "same radius as last time" is a real need.
- **One progress rail (`2fbb300`)**: the page rail already covered everything but 8
  section-local rails were never removed → two identical rails at once. Deleting them
  naively would have regressed (the page rail was only instant for viewer crops), so
  `EditLane` now reports `onBusyChange` up and the page ORs it in; unmount reports
  false so a mid-op tab switch can't strand it. Tab also syncs to `?tab=`.
- **Gates**: eslint **1 error → 0**, vitest **14 → 35/35**, tsc-gated build green,
  backend suite untouched at 699 passed / 6 skipped. Deploy confirmed structurally
  (new UI strings present in the built bundle, serves HTTP 200).
- ⚠️ **NOT driven in a real browser** — the Chrome extension is disconnected and
  playwright has no chromium. The live pass is RToony's; see the plan file.
- ⚠️ **PARKED: `POST /jobs/{id}/duplicate`** stays uncommitted in `splat_route.py`
  (+ untracked `test_duplicate_route.py`). Before it ships, the hardlink set MUST be
  fixed: `_DUP_HARDLINK_LANGFIELD = {gauss_emb.npz, ckpt_xyz.npy}` shares inodes and
  `backend/langfield/langfield_v2.py:172` writes `gauss_emb.npz` with a plain
  `np.savez_compressed` (truncating, in-place) — a language-field **re-lift** on
  either copy would destroy **both** scenes' fields. (`langfield_realign.py` is safe:
  tmp+replace, so the *rebuild* path is fine; only re-lift is lethal.) Secondary:
  `_DUP_IGNORE` isn't applied to the `processed`/`colmap` hardlink branch. A
  real-copy duplicate is ~5.8 GB/scene on a disk at 87%.

## LIVE-VERIFIED IN A REAL BROWSER (2026-07-26, same night) — the wave above is now proven
RToony: "connect with playwright and install chromium. I want this feature for you."
Installed `~/.cache/ms-playwright` (641 MB, chromium-headless-shell 148 + ffmpeg).
New gate: **`tools/verify-editor-live.py`** — drives the real app with real
keystrokes/clicks and asserts against the real DOM.

- **Results: `splat_6b2e82e5` (486,960 gaussians, live langfield) 31/31 ·
  `splat_7f98469203` (14k) 23/23.** Proven, not asserted: the Esc LADDER with a real
  placement (#1 cleared the centre and STAYED in the tool, #2 disarmed); `C` from View
  switching to `tab=edit` and arming; `[`/`]` clamping (90 presses bottomed at 0.0021,
  40 topped at 4.23); paint → **Esc → "834 splats selected / NOT COMMITTED / Discard"**
  (RToony's rule, verified); Ctrl+Z 834→591 and Ctrl+Shift+Z 591→834; Del discarding;
  1–9 class pick; `?` card rows; `H`; Ctrl+S NOT swallowed; typing "0.42" in the radius
  box firing no tool keys and committing on Enter; zero console/page errors.
- ⚙️ **GPU, not SwiftShader.** Software raster is NOT viable here: a 487k-gaussian scene
  pushed CDP round trips to **15 s** and a 900k one wedged the page outright (two runs
  hung, one for 12 min). `--use-gl=angle --use-angle=gl` binds the RTX 5090 headlessly
  (verified `ANGLE (NVIDIA ... RTX 5090, OpenGL 4.5.0)`) and the same page is instant.
  A responsiveness probe now fails fast instead of hanging.
- 🐛 **Three real bugs the browser caught that tsc/eslint could not:**
  1. `860afb9` — every tool's lower bound is max/2000, so the small end of the brush,
     crop-radius and box-extent ranges all printed as a flat **"0.00"**. Adaptive
     `preciseSize()`; the floor now reads 0.0021.
  2. `d39e9c1` — **the Edit/Export/Objects lane had NO background at all.** It was
     `bg-surface/95`, but the colour tokens are hex-valued CSS vars
     (`surface: "var(--surface)"`), so Tailwind cannot build their opacity variants —
     the built CSS has **zero** `.bg-surface\/95` rules. Only a backdrop-blur was
     rendering and the splat read straight through the floaters/decimate copy. Now
     solid `bg-surface`.
  3. `d39e9c1` — the new HUD was anchored top-centre, directly on top of the "In this
     scene" legend. Moved to bottom-centre above the search pill (it reads as a status
     bar there anyway).
- ✅ **RESOLVED same night (`3936714`) — the token-opacity failure was systemic and is
  now fixed at the root.** `bg-surface/85` (app nav), `bg-accent/30` (disabled primary
  buttons) and every `border-accent/NN` were also emitting nothing; `border-accent`
  never compiled at all. Tokens migrated to channel triplets +
  `rgb(var(--x) / <alpha-value>)`; the two raw `var(--x)` consumers in `body` became
  `rgb(var(--x))` (a bare var now resolves to the literal "5 7 13" and paints nothing);
  `surface-raised` stays bare on purpose (baked-alpha overlay, no `<alpha-value>` form).
  Guarded by `src/lib/tokens.test.ts`, **falsification-proven** three ways (token back
  to hex / config back to a bare var / a token consumed bare → exactly one failure
  each). That test reads index.css with **fs, not `?raw`** — vitest stubs CSS imports
  to `""`, and the first attempt asserted against an empty string and passed for the
  wrong reason. Receipts: built CSS now has
  `.bg-surface\/85{background-color:rgb(var(--surface) / .85)}` and
  `.disabled\:bg-accent\/30:disabled{...}`; in a real browser the nav computes
  `rgba(5,7,13,0.85)` (was transparent) while body is still `rgb(5,7,13)` — unchanged.
  vitest 35 → 49, live editor gate 23/23, screenshot reviewed.
