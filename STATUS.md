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

## Done (with evidence)
- [x] Backend built + venv (.venv) + deps (fastapi/uvicorn/httpx). py_compile OK.
- [x] Verified live on :3416: /healthz ok; login 303; proxied /api/splat/status ->
      8 jobs + engines ready; /api/splat/transfers proxies; unauth -> 401. Test
      process stopped afterward (will run via systemd when deployed).

## In-flight / Next
- [ ] Frontend: fresh Vite app with the IMPROVED GUI. Build the plan's first batch
      (stage timeline Q1, GPU-queue banner Q3, humanized stages Q2, download-format
      menu Q5), then results gallery (M1) + capture-confidence (M3/M5). Reuse the
      portal's SplatViewer + ui primitives; write a NEW lean splat page (avoid
      portal-only hooks: useTaskBoard / toast system / action-feedback).
      Routes: / (splat), /view/:jobId (fullscreen viewer). Calls same-origin /api/splat.
- [ ] Deploy: splatlab.service (systemd --user, ExecStartPre=nexus-svc-inject for
      PORTAL_TOKEN), tunnel route + DNS (cloudflared tunnel route dns -f nexus-ai),
      apps-registry/apps/splatlab.toml, nexus-manifest update. PUBLIC-FACING.
- [ ] Verify E2E: headless render splatlab.roonytoony.dev; run a splat through it.

## Invariants (do NOT break)
- Do NOT touch the portal's gpu_arbiter / three_d.py / splat.py pipeline in Phase 1.
- Portal /splat keeps working until Phase 3.
- Reuse PORTAL_TOKEN; pull via vault (nexus-svc-inject), never write to disk.
- PROCESS MGMT: never broad-pkill `uvicorn backend.main:app` — many Nexus apps
  share that cmdline (nexus-vicinity :3404, etc.). Kill by exact port/cmdline only.
