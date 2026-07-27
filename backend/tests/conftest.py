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
