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
