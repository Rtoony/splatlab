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
- [ ] Real gallery thumbnails — DEFERRED (splat-transform .webp GPU rasterizer too
      slow/finicky for a pipeline stage; ~2min+ and hung headless). Options: a
      client-side canvas snapshot cached per scene, or a lighter offline render.
- [ ] Minor cleanups: scene delete/pin buttons in splatlab gallery; delete portal
      dormant splat.py + splat*.tsx; dedupe splat/splatlab Launch Bay cards; declare
      redis in portal pyproject; delete splat_bf25300429 smoke scene.
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
