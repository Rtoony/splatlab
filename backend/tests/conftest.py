from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_arbiter
import maintenance_gate
import splat_route


@pytest.fixture(autouse=True)
def isolate_hardware_maintenance_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend tests must not inherit the workstation's live maintenance marker."""
    monkeypatch.setattr(
        maintenance_gate,
        "MAINTENANCE_FILE",
        tmp_path / "absent-hardware-maintenance.conf",
    )
    monkeypatch.setattr(
        maintenance_gate,
        "SUPERVISED_UNLOCK_FILE",
        tmp_path / "absent-gpu-compute-unlock.json",
    )
    monkeypatch.setattr(
        maintenance_gate,
        "WATCHER_STATUS_FILE",
        tmp_path / "absent-gpu-health-watch-status.json",
    )


@pytest.fixture(autouse=True)
def isolate_backup_interlock(request: pytest.FixtureRequest,
                            monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend tests must not inherit the workstation's live backup state.

    gpu_arbiter refuses heavy work while any of seven backup units is running —
    correct in production, and fail-closed by design. But it means the suite's
    result depends on whether a backup happens to be active: with
    nexus-backup.service merely `activating`, 95 tests flipped from pass to a
    409 that had nothing to do with the code under test. A suite that fails for
    reasons unrelated to the change is a suite you stop trusting.

    Same treatment the maintenance marker already gets above. The interlock
    itself is untouched in production; only its answer is pinned here, and
    test_backup_interlock.py exercises the real logic directly.
    """
    # test_backup_interlock.py exercises the real logic — stubbing it there
    # would test the stub.
    if request.node.fspath.basename == "test_backup_interlock.py":
        return

    idle = (False, "", "inactive")
    monkeypatch.setattr(gpu_arbiter, "backup_interlock_busy", lambda: idle)
    monkeypatch.setattr(gpu_arbiter, "require_backup_idle", lambda: None)
    # splat_route carries its OWN copy of the interlock (the duplication the
    # 2026-07-27 audit flagged: gpu_arbiter.py:53-61 and splat_route.py:290-297
    # keep separate unit lists). Patching only one leaves every heavy route
    # still reading the live system.
    monkeypatch.setattr(splat_route, "_backup_interlock_busy", lambda: idle)


class _LeaseFakeRedis:
    """In-memory stand-in for the SHARED GPU-lease Redis.

    The arbiter's two Lua scripts are both compare-then-act on the caller's
    token; emulated by direct comparison. Only the methods gpu_arbiter
    actually calls exist — a new call site failing loudly here is a feature.
    """

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self.hashes: dict[str, dict] = {}

    @staticmethod
    def _b(value: object) -> bytes:
        return value if isinstance(value, bytes) else str(value).encode()

    def set(self, key: str, value: object, nx: bool = False, px: int | None = None):
        if nx and key in self.kv:
            return None
        self.kv[key] = self._b(value)
        return True

    def get(self, key: str):
        return self.kv.get(key)

    # Mirrors redis-py's Redis.eval (server-side Lua), NOT Python's eval():
    # the script text is only sniffed for which of the two known Lua bodies
    # it is; nothing is ever executed.
    def eval(self, script: str, numkeys: int, key: str, *args: object):
        token = self._b(args[0]) if args else None
        if self.kv.get(key) != token:
            return 0
        if "del" in script:
            del self.kv[key]
        return 1

    def exists(self, key: str) -> int:
        return 1 if key in self.kv or key in self.hashes else 0

    def pexpire(self, key: str, ms: int) -> int:
        return 1 if key in self.kv or key in self.hashes else 0

    def hset(self, key: str, mapping: dict):
        self.hashes.setdefault(key, {}).update(mapping)
        return len(mapping)

    def hgetall(self, key: str) -> dict:
        return dict(self.hashes.get(key, {}))

    def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            n += 1 if (self.kv.pop(k, None) is not None) else 0
            n += 1 if (self.hashes.pop(k, None) is not None) else 0
        return n


@pytest.fixture(autouse=True)
def isolate_gpu_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not touch the LIVE host flock or the shared Redis GPU lease.

    Found 2026-07-31: any test whose route acquires the GPU lease was blocking
    on the REAL lock while a real training job held it — the suite hung at
    test_edit_ops::test_revert_restores_prior_content for 400s+ whenever a
    splat job was running, and passed only on an idle box (the long-recorded
    "transient pytest hang" was this). Same philosophy as the two fixtures
    above: production logic untouched, only its environment pinned. Tests
    that monkeypatch gpu_arbiter._redis themselves override this (test-body
    patches apply after autouse setup).
    """
    monkeypatch.setenv(gpu_arbiter.HOST_LOCK_ENV, str(tmp_path / "test-heavy-work.lock"))
    fake = _LeaseFakeRedis()
    monkeypatch.setattr(gpu_arbiter, "_redis", lambda: fake)
