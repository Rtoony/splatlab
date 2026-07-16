"""Fail-closed, host-local GPU arbitration for the RTX 5090.

Every SplatLab process first takes the same flock lease, then the Redis lease
used by adjacent GPU services. The host lease remains authoritative when Redis
is unhealthy, but work is refused by default because an older Redis-only client
could otherwise overlap it. A deliberately configured, journal-audited emergency
override may bypass only an unavailable coordination service; maintenance and an
active backup remain hard stops.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import httpx

try:
    from . import maintenance_gate
except ImportError:  # pragma: no cover - direct backend module import
    import maintenance_gate

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
GPU_MAINTENANCE_REASON = os.environ.get("SPLAT_TRAINING_DISABLED_REASON", "").strip()
HOST_LOCK_ENV = "NEXUS_HEAVY_WORK_LOCK_PATH"
HOST_LOCK_NAME = "nexus-heavy-work.lock"
EMERGENCY_OVERRIDE_ENV = "SPLAT_GPU_EMERGENCY_OVERRIDE"
EMERGENCY_REASON_ENV = "SPLAT_GPU_EMERGENCY_OVERRIDE_REASON"
EMERGENCY_ACTOR_ENV = "SPLAT_GPU_EMERGENCY_OVERRIDE_ACTOR"
BACKUP_INTERLOCK_UNITS = (
    ("system", "restic-backup-core.service"),
    ("user", "restic-tier0-offsite.service"),
    ("user", "restic-tier0-offsite-cold.service"),
    ("user", "nexus-backup.service"),
    ("user", "backup-docker-services.service"),
    ("user", "vaultwarden-backup.service"),
    ("user", "vm300-databases-backup.service"),
)
BACKUP_INTERLOCK_BUSY_STATES = {
    "activating",
    "active",
    "reloading",
    "deactivating",
    "unknown",
}


class GPUArbiterUnavailable(RuntimeError):
    """Heavy GPU work cannot safely proceed under the coordination policy."""


_T = TypeVar("_T")


def current_maintenance_reason() -> str:
    return maintenance_gate.maintenance_reason(GPU_MAINTENANCE_REASON)


LOCK_KEY = "nexus:gpu:heavy_lock"
HOLDER_KEY = "nexus:gpu:heavy_holder"
LOCK_TTL_MS = 90_000  # holder's lock auto-expires after this if not refreshed
HEARTBEAT_SEC = 15.0  # refresh interval while holding (TTL is 6x => wide margin)
ACQUIRE_POLL_SEC = 0.5  # how often to retry acquiring a contended lock
_CLIENT_RETRY_SEC = 30.0  # backoff before re-probing a down Redis

# release/refresh only if we still own the key (compare-and-act, atomic)
_RELEASE_LUA = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
_REFRESH_LUA = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"

_client = None
_client_down_at = 0.0
_holder: dict[str, Any] = {"lane": None, "job_id": None, "since": None}
# XC-1: edge-trigger so the Redis-degrade transition pages ONCE per outage (not on
# every 30s re-probe). Reset to False on a successful reconnect so recovery re-arms.
_degraded_alerted = False


def host_lock_path() -> Path:
    """Return the one host-wide lock path shared with workstation backups."""
    configured = os.environ.get(HOST_LOCK_ENV, "").strip()
    if configured:
        return Path(configured)
    runtime = (
        os.environ.get("XDG_RUNTIME_DIR", "").strip() or f"/run/user/{os.getuid()}"
    )
    return Path(runtime) / HOST_LOCK_NAME


def _emergency_override() -> tuple[bool, str, str]:
    """Validate the explicit three-part break-glass configuration."""
    if os.environ.get(EMERGENCY_OVERRIDE_ENV, "").strip() != "1":
        return False, "", ""
    reason = os.environ.get(EMERGENCY_REASON_ENV, "").strip()
    actor = os.environ.get(EMERGENCY_ACTOR_ENV, "").strip()
    if not reason or not actor:
        log.error(
            "GPU emergency override ignored: %s=1 requires both %s and %s",
            EMERGENCY_OVERRIDE_ENV,
            EMERGENCY_REASON_ENV,
            EMERGENCY_ACTOR_ENV,
        )
        return False, "", ""
    return True, actor, reason


def _audit_emergency_override(
    component: str, *, operation_id: str | None = None
) -> None:
    enabled, actor, reason = _emergency_override()
    if not enabled:
        return
    # systemd captures this structured CRITICAL record in the persistent journal.
    log.critical(
        "GPU_EMERGENCY_OVERRIDE actor=%r component=%r operation_id=%r reason=%r pid=%d",
        actor,
        component,
        operation_id,
        reason,
        os.getpid(),
    )


def backup_interlock_state(scope: str, unit: str) -> str:
    command = ["systemctl"]
    if scope == "user":
        command.append("--user")
    command.extend(("show", unit, "--property=ActiveState", "--value"))
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.SubprocessError):
        log.exception("Could not query %s backup interlock unit %s", scope, unit)
        return "unknown"
    state = result.stdout.strip()
    if result.returncode != 0 or not state:
        log.error("Could not query %s backup interlock unit %s", scope, unit)
        return "unknown"
    return state


def backup_interlock_busy() -> tuple[bool, str, str]:
    for scope, unit in BACKUP_INTERLOCK_UNITS:
        state = backup_interlock_state(scope, unit)
        if state in BACKUP_INTERLOCK_BUSY_STATES:
            return True, unit, state
    return False, "", "inactive"


def require_backup_idle() -> None:
    busy, unit, state = backup_interlock_busy()
    if not busy:
        return
    if state == "unknown":
        raise GPUArbiterUnavailable(
            f"backup state for {unit} could not be verified; refusing heavy work"
        )
    raise GPUArbiterUnavailable(
        f"backup {unit} is {state}; refusing overlapping heavy work"
    )


def _notify_degraded(detail: str) -> None:
    """Best-effort edge alert for a Redis coordination outage."""
    try:
        notify = os.path.expanduser("~/bin/nexus-notify")
        if not (os.path.isfile(notify) and os.access(notify, os.X_OK)):
            notify = shutil.which("nexus-notify")
        if not notify:
            return
        msg = (
            "GPU arbiter BLOCKED: Redis coordination is unavailable. "
            "New SplatLab GPU work is fail-closed; inspect Redis before resuming. Detail: "
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


def _mark_redis_down(detail: str) -> None:
    global _client, _client_down_at, _degraded_alerted
    _client = None
    _client_down_at = time.monotonic()
    log.warning("gpu_arbiter: Redis unavailable; new GPU work is blocked: %s", detail)
    if not _degraded_alerted:
        _notify_degraded(detail)
        _degraded_alerted = True


def _redis():
    """Return a live Redis client, or None while its retry backoff is active."""
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
    except Exception as e:  # noqa: BLE001 - all coordination failures fail closed
        _mark_redis_down(str(e))
        return None


class _HostWorkLock:
    """One cancellation-safe host flock shared by GPU and CPU-heavy work."""

    def __init__(self) -> None:
        self._local = asyncio.Lock()
        self._fd: int | None = None
        self._owner: asyncio.Task[Any] | None = None

    async def acquire(self) -> None:
        current = asyncio.current_task()
        if current is not None and self._owner is current:
            raise GPUArbiterUnavailable(
                "nested host-work lease acquisition is not allowed"
            )
        await self._local.acquire()
        path = host_lock_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
            os.fchmod(fd, 0o600)
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fd = fd
                    self._owner = current
                    return
                except BlockingIOError:
                    await asyncio.sleep(ACQUIRE_POLL_SEC)
        except BaseException:
            if "fd" in locals():
                os.close(fd)
            self._local.release()
            raise

    def release(self) -> None:
        fd, self._fd = self._fd, None
        self._owner = None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)
        self._local.release()

    async def __aenter__(self) -> "_HostWorkLock":
        await self.acquire()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self.release()

    def locked(self) -> bool:
        if self._local.locked():
            return True
        path = host_lock_path()
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
            except BlockingIOError:
                return True
            finally:
                os.close(fd)
        except OSError:
            return True


HOST_WORK_LOCK = _HostWorkLock()


class _CrossProcessLock:
    """Cancellation-safe host-file + Redis lease."""

    def __init__(self) -> None:
        self._local = asyncio.Lock()
        self._token: str | None = None
        self._hb: asyncio.Task | None = None
        self._host_acquired = False
        self._coordination_lost = asyncio.Event()

    async def __aenter__(self) -> "_CrossProcessLock":
        await self._local.acquire()
        # Everything past here can await (to_thread SET, sleep) and thus be
        # cancelled. Guard with BaseException so a CancelledError can never leak
        # the local lock (which would deadlock this lane forever).
        try:
            reason = current_maintenance_reason()
            if reason:
                raise GPUArbiterUnavailable(f"GPU maintenance gate active: {reason}")
            self._coordination_lost = asyncio.Event()
            await HOST_WORK_LOCK.acquire()
            self._host_acquired = True
            token = f"{os.getpid()}:{uuid.uuid4().hex}"
            r = _redis()
            if r is None:
                enabled, _actor, _reason = _emergency_override()
                if not enabled:
                    raise GPUArbiterUnavailable("Redis GPU coordination is unavailable")
                _audit_emergency_override("redis-unavailable")
                self._token = None
                return self
            try:
                while True:
                    got = await asyncio.to_thread(
                        r.set, LOCK_KEY, token, nx=True, px=LOCK_TTL_MS
                    )
                    if got:
                        break
                    await asyncio.sleep(ACQUIRE_POLL_SEC)
                self._token = token
                self._hb = asyncio.create_task(self._heartbeat(token))
            except Exception as e:
                _mark_redis_down(str(e))
                enabled, _actor, _reason = _emergency_override()
                if not enabled:
                    raise GPUArbiterUnavailable(
                        f"Redis GPU lock acquisition failed: {e}"
                    ) from e
                _audit_emergency_override("redis-acquire-failed")
                self._token = None
            return self
        except BaseException:
            if self._host_acquired:
                self._host_acquired = False
                HOST_WORK_LOCK.release()
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
                        await asyncio.to_thread(
                            r.eval, _RELEASE_LUA, 1, LOCK_KEY, token
                        )
                    except Exception:
                        pass
        finally:
            if self._host_acquired:
                self._host_acquired = False
                HOST_WORK_LOCK.release()
            self._local.release()

    async def _heartbeat(self, token: str) -> None:
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_SEC)
                r = _redis()
                if r is None:
                    self._coordination_lost.set()
                    return
                try:
                    refreshed = await asyncio.to_thread(
                        r.eval, _REFRESH_LUA, 1, LOCK_KEY, token, str(LOCK_TTL_MS)
                    )
                    if not refreshed:
                        log.error("gpu_arbiter: Redis lease ownership was lost")
                        self._coordination_lost.set()
                        return
                except Exception as exc:  # noqa: BLE001 - a lost heartbeat is unsafe
                    _mark_redis_down(str(exc))
                    self._coordination_lost.set()
                    return
        except asyncio.CancelledError:
            pass

    def locked(self) -> bool:
        if self._local.locked():
            return True
        if HOST_WORK_LOCK.locked():
            return True
        r = _redis()
        if r is None:
            return False
        try:
            return bool(r.exists(LOCK_KEY))
        except RedisError:
            return True

    async def wait_coordination_lost(self) -> None:
        await self._coordination_lost.wait()


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
        reason = current_maintenance_reason()
        if reason:
            return False, f"GPU arbiter unavailable during maintenance: {reason}"
        enabled, _actor, _reason = _emergency_override()
        if not enabled:
            return (
                False,
                "GPU orchestrator unavailable; refusing unverified VRAM admission",
            )
        _audit_emergency_override("orchestrator-unavailable")
        return (
            True,
            "GPU orchestrator unavailable; explicit emergency override accepted",
        )
    reason = current_maintenance_reason()
    if reason:
        return False, f"GPU maintenance gate active: {reason}"
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
    return (
        False,
        f"could not free {needed_mb} MB after evicting {evicted}; only {free_mb} MB free",
    )


async def _finish_sync_task_before_cancelling(task: asyncio.Task[_T]) -> _T:
    """Do not let asyncio cancellation outlive an uninterruptible worker thread."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _run_sync_cancellation_safe(
    func: Callable[..., _T], *args: Any, **kwargs: Any
) -> _T:
    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    return await _finish_sync_task_before_cancelling(task)


async def run_gpu_operation(
    *,
    lane: str,
    operation_id: str,
    vram_mb: int,
    operation: Callable[[], Awaitable[_T]],
    status_callback: Callable[[str], None] | None = None,
) -> _T:
    """Run one complete CUDA operation under every admission and serialization gate.

    The operation owns the lease until it has actually stopped. If Redis lease
    refresh fails, the operation is cancelled and drained before either lock is
    released. Sync CUDA callers use run_sync_gpu_operation, which similarly waits
    for a worker thread to finish before propagating cancellation.
    """
    reason = current_maintenance_reason()
    if reason:
        raise GPUArbiterUnavailable(f"GPU maintenance gate active: {reason}")
    require_backup_idle()
    async with HEAVY_GPU_LOCK:
        # Backup scripts take this same host lock for their whole run. Rechecking
        # after acquisition closes the backup-versus-GPU admission race.
        require_backup_idle()
        reason = current_maintenance_reason()
        if reason:
            raise GPUArbiterUnavailable(f"GPU maintenance gate active: {reason}")
        set_holder(lane, operation_id)
        op_task: asyncio.Task[_T] | None = None
        lost_task: asyncio.Task[None] | None = None
        try:
            ok, detail = await acquire_gpu(vram_mb)
            if status_callback is not None:
                status_callback(detail)
            if not ok:
                raise GPUArbiterUnavailable(detail)
            reason = current_maintenance_reason()
            if reason:
                raise GPUArbiterUnavailable(f"GPU maintenance gate active: {reason}")
            op_task = asyncio.create_task(operation())
            lost_task = asyncio.create_task(HEAVY_GPU_LOCK.wait_coordination_lost())
            done, _pending = await asyncio.wait(
                {op_task, lost_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if op_task in done:
                return await op_task

            enabled, _actor, _reason = _emergency_override()
            if enabled:
                _audit_emergency_override(
                    "redis-heartbeat-lost", operation_id=operation_id
                )
                return await op_task

            op_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await op_task
            raise GPUArbiterUnavailable(
                "Redis GPU coordination was lost during the operation; work was stopped fail-closed"
            )
        except asyncio.CancelledError:
            if op_task is not None and not op_task.done():
                op_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await op_task
            raise
        finally:
            if lost_task is not None:
                lost_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await lost_task
            clear_holder()


async def run_host_operation(
    *,
    operation: Callable[[], Awaitable[_T]],
) -> _T:
    """Run CPU/disk-heavy mutation under the backup-shared host flock."""
    reason = current_maintenance_reason()
    if reason:
        raise GPUArbiterUnavailable(f"GPU maintenance gate active: {reason}")
    require_backup_idle()
    async with HOST_WORK_LOCK:
        # A backup may have started after the first state query but before flock.
        require_backup_idle()
        reason = current_maintenance_reason()
        if reason:
            raise GPUArbiterUnavailable(f"GPU maintenance gate active: {reason}")
        return await operation()


async def run_sync_gpu_operation(
    *,
    lane: str,
    operation_id: str,
    vram_mb: int,
    func: Callable[..., _T],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> _T:
    async def operation() -> _T:
        return await _run_sync_cancellation_safe(func, *args, **(kwargs or {}))

    return await run_gpu_operation(
        lane=lane,
        operation_id=operation_id,
        vram_mb=vram_mb,
        operation=operation,
        status_callback=status_callback,
    )
