"""Splat Lab — standalone app backend.

Phase 2 of carving Splat Lab out of the Nexus portal: this service now OWNS the
splat pipeline (splat_route.py, ported from the portal) and coordinates the 5090
with the portal's TRELLIS lane through a cross-process Redis lock (gpu_arbiter.py).
/supersplat is still proxied to the portal (SuperSplat editor is mounted there).

Auth: log in with the same PORTAL_TOKEN; a signed cookie gates the app. The
proxied calls inject the portal bearer server-side (never sent to the browser).
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import os
import sys
import time
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

# Make the ported splat route + its local gpu_arbiter / operator_audit importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splat_route  # noqa: E402  (imports gpu_arbiter + operator_audit from this dir)
import edit_ops  # noqa: E402  (scene editing: snapshots, splat-transform ops, semantic edits, merge)
import feedback  # noqa: E402  (small SQLite-backed in-app feedback loop)
import thumb as thumbgen  # noqa: E402  (scene thumbnail generator)

PORTAL_ORIGIN = os.environ.get("SPLATLAB_PORTAL_ORIGIN", "http://127.0.0.1:3300").rstrip("/")
PORTAL_TOKEN = os.environ.get("PORTAL_TOKEN", "")
COOKIE = "splatlab_session"
MAX_AGE = 60 * 60 * 24 * 14  # 14 days
DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Match the portal's splat startup hooks: rehydrate persisted jobs.
    with contextlib.suppress(Exception):
        splat_route.migrate_legacy_metas()
    with contextlib.suppress(Exception):
        # Resume-on-start: auto-restart the newest orphaned in-flight job
        # (SPLAT_RESUME_ON_START=0 restores the old mark-failed-only behavior).
        await splat_route.resume_orphan_jobs()
    with contextlib.suppress(Exception):
        feedback.init_db()
    yield


app = FastAPI(title="Splat Lab", lifespan=_lifespan)
_client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))


# ── auth ────────────────────────────────────────────────────────────────────
def _sign(ts: int) -> str:
    mac = hmac.new(PORTAL_TOKEN.encode(), str(ts).encode(), hashlib.sha256).hexdigest()
    return f"{ts}:{mac}"


def _valid_cookie(value: str) -> bool:
    if not value or ":" not in value or not PORTAL_TOKEN:
        return False
    ts_str, mac = value.rsplit(":", 1)
    try:
        if time.time() - int(ts_str) > MAX_AGE:
            return False
    except ValueError:
        return False
    expected = hmac.new(PORTAL_TOKEN.encode(), ts_str.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, expected)


def _authed(request: Request) -> bool:
    if _valid_cookie(request.cookies.get(COOKIE, "")):
        return True
    auth = request.headers.get("authorization", "")
    return bool(auth.startswith("Bearer ") and PORTAL_TOKEN and hmac.compare_digest(auth[7:], PORTAL_TOKEN))


def _login_html(error: str = "") -> str:
    msg = f'<p class="err">{error}</p>' if error else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Splat Lab — Sign in</title><style>
*{{box-sizing:border-box}}body{{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
background:radial-gradient(circle at 30% 20%,rgba(34,211,238,.12),transparent 40%),#05070d;
font-family:ui-sans-serif,system-ui,sans-serif;color:#e4e9f2}}
.card{{width:340px;padding:32px;border:1px solid rgba(255,255,255,.1);border-radius:24px;
background:rgba(10,16,28,.7);backdrop-filter:blur(8px)}}
h1{{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}}
p.sub{{margin:0 0 20px;color:#8b97ad;font-size:13px}}
input{{width:100%;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.12);
background:rgba(255,255,255,.04);color:#e4e9f2;font-size:14px}}
button{{width:100%;margin-top:12px;padding:10px;border:0;border-radius:12px;cursor:pointer;
background:#22d3ee;color:#04121a;font-weight:700;font-size:14px}}
.err{{color:#f87171;font-size:12px;margin:8px 0 0}}
.brand{{display:flex;align-items:center;gap:8px;margin-bottom:14px;color:#22d3ee;
font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.28em}}
</style></head><body><form class="card" method="post" action="/login">
<div class="brand">◆ Spatial Pipeline</div>
<h1>Splat Lab</h1><p class="sub">Sign in with your portal token.</p>
<input name="portal_token" type="password" placeholder="Portal token" autocomplete="current-password" autofocus required>
<button type="submit">Enter</button>{msg}</form></body></html>"""


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "splatlab", "portal_origin": PORTAL_ORIGIN, "token": bool(PORTAL_TOKEN)}


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return _login_html()


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    token = (form.get("portal_token") or "").strip()
    if not PORTAL_TOKEN or not hmac.compare_digest(token, PORTAL_TOKEN):
        return HTMLResponse(_login_html("Invalid token."), status_code=401)
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(COOKIE, _sign(int(time.time())), httponly=True, secure=True, samesite="lax", max_age=MAX_AGE, path="/")
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE, path="/")
    return resp


# ── streaming reverse-proxy to the portal splat backend ──────────────────────
async def _proxy(request: Request, path: str) -> StreamingResponse:
    url = httpx.URL(f"{PORTAL_ORIGIN}{path}", query=request.url.query.encode("utf-8"))
    drop = {b"host", b"authorization", b"cookie", b"content-length"}
    headers = [(k, v) for k, v in request.headers.raw if k.lower() not in drop]
    headers.append((b"authorization", f"Bearer {PORTAL_TOKEN}".encode()))
    upstream_req = _client.build_request(request.method, url, headers=headers, content=request.stream())
    upstream = await _client.send(upstream_req, stream=True)
    # aiter_raw + original headers keep content-encoding consistent.
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=upstream.headers,
        background=BackgroundTask(upstream.aclose),
    )


def require_auth(request: Request) -> None:
    if not _authed(request):
        raise HTTPException(status_code=401, detail="unauthorized")


# /api/splat is now OWNED here (the ported pipeline), gated by splatlab auth.
app.include_router(splat_route.router, prefix="/api/splat", dependencies=[Depends(require_auth)])

# Scene editing (destructive ops are snapshot-versioned) — same auth gate as the pipeline.
app.include_router(edit_ops.router, prefix="/api/splat", dependencies=[Depends(require_auth)])

# Feedback is Splatlab-native runtime data, also gated by the same signed cookie.
app.include_router(feedback.router, dependencies=[Depends(require_auth)])


@app.get("/api/splat/jobs/{job_id}/thumbnail", dependencies=[Depends(require_auth)])
async def splat_thumbnail(job_id: str):
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="not found")
    meta = splat_route._read_meta(job_id)
    output_dir = (
        Path(meta["output_dir"]) if meta and meta.get("output_dir") else splat_route.DEFAULT_3D_ROOT / job_id
    )
    # Prefer a REAL rendered hero view (scenes with a language field); fall back to the
    # cheap CPU point-cloud thumbnail when there's no field / the worker is unavailable.
    hero = await splat_route.ensure_hero_thumb(output_dir)
    if hero is not None:
        return FileResponse(str(hero), media_type="image/webp")
    preview_dir = output_dir / splat_route.PREVIEW_DIRNAME
    thumb = await asyncio.to_thread(thumbgen.get_or_make, preview_dir)
    if thumb is None:
        raise HTTPException(status_code=404, detail="thumbnail unavailable")
    return FileResponse(str(thumb), media_type="image/webp")


@app.api_route("/supersplat/{path:path}", methods=["GET"])
async def proxy_supersplat(path: str, request: Request):
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    return await _proxy(request, f"/supersplat/{path}")


# ── static SPA ───────────────────────────────────────────────────────────────
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")


@app.get("/{full_path:path}", response_class=HTMLResponse)
async def spa(full_path: str, request: Request):
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    index = DIST / "index.html"
    if index.is_file():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Splat Lab</h1><p>Frontend not built yet.</p>", status_code=200)
