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
- [ ] RISKY: swap portal `server/lib/gpu_arbiter.py` -> the Redis-backed version
      (same API). Verify TRELLIS (three_d.py) still runs + now coordinates. Restart
      portal (session-fresh check first).
- [ ] Cutover splatlab frontend: /api/splat now hits splatlab's own backend (not
      proxied). Verify E2E + cross-process coordination (splat + TRELLIS serialize).
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
