"""Cross-process GPU arbitration for the RTX 5090.

Drop-in replacement for the portal's in-process arbiter: same public surface
(HEAVY_GPU_LOCK as an async context manager + .locked(), set_holder/clear_holder/
holder_info, and the orchestrator helpers gpu_status/evict/acquire_gpu) — but the
lock + holder are now backed by Redis so the splat lane (splatlab) and the TRELLIS
lane (portal) serialize on the card even though they live in different processes.

FAIL-OPEN by design: if Redis is unreachable (down, no password, lib missing) the
lock degrades to a plain in-process asyncio.Lock + local holder dict — i.e. exactly
the previous behavior. It must NEVER deadlock a working lane because Redis hiccuped.
A TTL + heartbeat means a crashed holder's lock auto-expires instead of wedging.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

try:
    import redis as _redislib
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - lib should be present, but never hard-fail
    _redislib = None

    class RedisError(Exception):
        pass


log = logging.getLogger(__name__)

ORCHESTRATOR_URL = "http://127.0.0.1:4001"
GPU_ACQUIRE_TIMEOUT_SEC = 60

LOCK_KEY = "nexus:gpu:heavy_lock"
HOLDER_KEY = "nexus:gpu:heavy_holder"
LOCK_TTL_MS = 90_000          # holder's lock auto-expires after this if not refreshed
HEARTBEAT_SEC = 15.0          # refresh interval while holding (TTL is 6x => wide margin)
ACQUIRE_POLL_SEC = 0.5        # how often to retry acquiring a contended lock
_CLIENT_RETRY_SEC = 30.0      # backoff before re-probing a down Redis

# release/refresh only if we still own the key (compare-and-act, atomic)
_RELEASE_LUA = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
_REFRESH_LUA = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"

_client = None
_client_down_at = 0.0
_holder: dict[str, Any] = {"lane": None, "job_id": None, "since": None}
# XC-1: edge-trigger so the Redis-degrade transition pages ONCE per outage (not on
# every 30s re-probe). Reset to False on a successful reconnect so recovery re-arms.
_degraded_alerted = False


def _notify_degraded(detail: str) -> None:
    """XC-1: best-effort WARN that cross-process GPU arbitration has degraded to an
    in-process lock (Redis unavailable). MUST NEVER raise or block the arbiter -
    fail-open is the whole contract of this module; the log.warning above is the
    durable receipt if this send is dropped. Fire-and-forget via nexus-notify; if it
    is missing or errors we swallow it. (Delivery during a locked-vault boot also
    depends on the nexus-notify hardcoded-ntfy floor - see finding NOTIFY-1.)"""
    try:
        notify = os.path.expanduser("~/bin/nexus-notify")
        if not (os.path.isfile(notify) and os.access(notify, os.X_OK)):
            notify = shutil.which("nexus-notify")
        if not notify:
            return
        msg = (
            "GPU arbiter DEGRADED: Redis unavailable -> in-process lock only. "
            "Cross-lane 5090 serialization (splat / TRELLIS / langfield) is DISABLED; "
            "two concurrent heavy GPU jobs can now co-run and OOM/corrupt. Detail: "
            + detail
        )
        subprocess.Popen(
            [notify, "--source=gpu-arbiter", "--severity=warn", "--quiet", msg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception:  # noqa: BLE001 - notify is best-effort; never break the lock
        pass


def _redis():
    """Return a live Redis client, or None (then everything fails open)."""
    global _client, _client_down_at, _degraded_alerted
    if _redislib is None:
        return None
    if _client is not None:
        return _client
    if time.monotonic() - _client_down_at < _CLIENT_RETRY_SEC:
        return None
    try:
        c = _redislib.Redis(
            host=os.environ.get("REDIS_HOST", "127.0.0.1"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            password=os.environ.get("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=0.5,
            socket_connect_timeout=0.5,
        )
        c.ping()
        _client = c
        _degraded_alerted = False  # XC-1: recovered - re-arm the degrade alert
        return c
    except Exception as e:  # noqa: BLE001 - any failure => fail open
        _client_down_at = time.monotonic()
        log.warning("gpu_arbiter: Redis unavailable, using in-process lock only: %s", e)
        if not _degraded_alerted:  # XC-1: page the transition to in-process-lock-only ONCE
            _notify_degraded(str(e))
            _degraded_alerted = True
        return None


class _CrossProcessLock:
    """async-with lock that coordinates via Redis, falling back to a local lock."""

    def __init__(self) -> None:
        self._local = asyncio.Lock()
        self._token: str | None = None
        self._hb: asyncio.Task | None = None

    async def __aenter__(self) -> "_CrossProcessLock":
        await self._local.acquire()  # in-process serialization + fail-open fallback
        # Everything past here can await (to_thread SET, sleep) and thus be
        # cancelled. Guard with BaseException so a CancelledError can never leak
        # the local lock (which would deadlock this lane forever).
        try:
            token = f"{os.getpid()}:{uuid.uuid4().hex}"
            r = _redis()
            if r is None:
                self._token = None  # local-only this round
                return self
            try:
                while True:
                    got = await asyncio.to_thread(r.set, LOCK_KEY, token, nx=True, px=LOCK_TTL_MS)
                    if got:
                        break
                    await asyncio.sleep(ACQUIRE_POLL_SEC)
                self._token = token
                self._hb = asyncio.create_task(self._heartbeat(token))
            except Exception as e:  # broadened: a bare OSError must not escape -> fail open
                log.warning("gpu_arbiter: Redis acquire failed, holding local lock only: %s", e)
                self._token = None
            return self
        except BaseException:
            self._local.release()
            raise

    async def __aexit__(self, *_exc: Any) -> None:
        try:
            hb, self._hb = self._hb, None
            if hb is not None:
                hb.cancel()
                try:
                    await hb
                except asyncio.CancelledError:
                    pass
            token, self._token = self._token, None
            if token is not None:
                r = _redis()
                if r is not None:
                    try:
                        await asyncio.to_thread(r.eval, _RELEASE_LUA, 1, LOCK_KEY, token)
                    except Exception:
                        pass
        finally:
            self._local.release()  # ALWAYS release, even if teardown above misbehaves

    async def _heartbeat(self, token: str) -> None:
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_SEC)
                r = _redis()
                if r is None:
                    return
                try:
                    await asyncio.to_thread(r.eval, _REFRESH_LUA, 1, LOCK_KEY, token, str(LOCK_TTL_MS))
                except RedisError:
                    return
        except asyncio.CancelledError:
            pass

    def locked(self) -> bool:
        r = _redis()
        if r is None:
            return self._local.locked()
        try:
            return bool(r.exists(LOCK_KEY))
        except RedisError:
            return self._local.locked()


HEAVY_GPU_LOCK = _CrossProcessLock()


def set_holder(lane: str, job_id: str) -> None:
    since = datetime.now(timezone.utc).isoformat()
    _holder.update(lane=lane, job_id=job_id, since=since)
    r = _redis()
    if r is not None:
        try:
            r.hset(HOLDER_KEY, mapping={"lane": lane, "job_id": job_id, "since": since})
            r.pexpire(HOLDER_KEY, LOCK_TTL_MS * 2)
        except RedisError:
            pass


def clear_holder() -> None:
    _holder.update(lane=None, job_id=None, since=None)
    r = _redis()
    if r is not None:
        try:
            r.delete(HOLDER_KEY)
        except RedisError:
            pass


def holder_info() -> dict[str, Any]:
    r = _redis()
    if r is not None:
        try:
            h = r.hgetall(HOLDER_KEY)
            return {
                "lane": h.get("lane"),
                "job_id": h.get("job_id"),
                "since": h.get("since"),
                "locked": bool(r.exists(LOCK_KEY)),
            }
        except RedisError:
            pass
    return {**_holder, "locked": HEAVY_GPU_LOCK.locked()}


# ── orchestrator helpers (already cross-process: HTTP to nexus-gpu-orchestrator) ──
async def gpu_status() -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{ORCHESTRATOR_URL}/v1/gpu/status")
            r.raise_for_status()
            return r.json()
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        log.warning("gpu orchestrator unreachable: %s", e)
        return None


async def evict(service_id: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{ORCHESTRATOR_URL}/v1/gpu/evict/{service_id}")
            r.raise_for_status()
            return True
    except httpx.HTTPError as e:
        log.warning("evict %s failed: %s", service_id, e)
        return False


async def acquire_gpu(needed_mb: int) -> tuple[bool, str]:
    """Ensure at least `needed_mb` of VRAM is free, evicting if necessary."""
    status = await gpu_status()
    if status is None:
        return True, "orchestrator unreachable; proceeding uncoordinated"
    if status.get("vram_free_mb", 0) >= needed_mb:
        return True, f"sufficient headroom ({status['vram_free_mb']} MB free)"
    resident = [s for s in status.get("services", []) if s.get("resident")]
    # NB: `or 0` INSIDE the negation — a service can report idle_sec=None, and
    # `-None` crashes the whole acquire path (bad operand type for unary -).
    resident.sort(key=lambda s: (s.get("priority", 99), -(s.get("idle_sec") or 0)))
    evicted: list[str] = []
    deadline = time.monotonic() + GPU_ACQUIRE_TIMEOUT_SEC
    for svc in resident:
        if time.monotonic() > deadline:
            break
        sid = svc["id"]
        if await evict(sid):
            evicted.append(sid)
        await asyncio.sleep(1.5)
        s2 = await gpu_status()
        if s2 and s2.get("vram_free_mb", 0) >= needed_mb:
            return True, f"evicted {evicted}; {s2['vram_free_mb']} MB free"
    final = await gpu_status()
    free_mb = final.get("vram_free_mb", 0) if final else "unknown"
    return False, f"could not free {needed_mb} MB after evicting {evicted}; only {free_mb} MB free"
