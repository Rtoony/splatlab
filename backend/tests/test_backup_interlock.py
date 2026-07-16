"""System backup and SplatLab launch ordering cannot overlap."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_arbiter  # noqa: E402
import splat_route  # noqa: E402


EXPECTED_BACKUP_INTERLOCK_UNITS = {
    ("system", "restic-backup-core.service"),
    ("user", "restic-tier0-offsite.service"),
    ("user", "restic-tier0-offsite-cold.service"),
    ("user", "nexus-backup.service"),
    ("user", "backup-docker-services.service"),
    ("user", "vaultwarden-backup.service"),
    ("user", "vm300-databases-backup.service"),
}


def test_all_local_backups_are_in_both_interlock_inventories():
    assert set(gpu_arbiter.BACKUP_INTERLOCK_UNITS) == EXPECTED_BACKUP_INTERLOCK_UNITS
    assert set(splat_route.BACKUP_INTERLOCK_UNITS) == EXPECTED_BACKUP_INTERLOCK_UNITS



def test_hardware_maintenance_gate_rejects_before_interlock_and_planning(
    monkeypatch: pytest.MonkeyPatch,
):
    reason = "Persistent RTX 5090 PCIe AER requires physical remediation."
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", reason)
    monkeypatch.setattr(
        splat_route,
        "_backup_interlock_busy",
        lambda: pytest.fail("backup query must not run while training is disabled"),
    )
    monkeypatch.setattr(
        splat_route,
        "_engine_availability",
        lambda: pytest.fail("planning must not run while training is disabled"),
    )
    req = splat_route.SplatTrainRequest(mode="3d", input_path="/does/not/matter.mp4")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(splat_route.start_splat_training(SimpleNamespace(), req))

    assert exc.value.status_code == 409
    assert reason in exc.value.detail


def test_interlock_state_classification(monkeypatch: pytest.MonkeyPatch):
    for state in ("activating", "active", "reloading", "deactivating", "unknown"):
        monkeypatch.setattr(splat_route, "_backup_interlock_state", lambda *args, value=state: value)
        assert splat_route._backup_interlock_busy() == (True, "restic-backup-core.service", state)
    for state in ("inactive", "failed", "disabled"):
        monkeypatch.setattr(splat_route, "_backup_interlock_state", lambda *args, value=state: value)
        assert splat_route._backup_interlock_busy() == (False, "", "inactive")


def test_state_query_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        splat_route.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="no bus"),
    )
    assert splat_route._backup_interlock_state("system", "restic-backup-core.service") == "unknown"
    assert splat_route._backup_interlock_busy() == (True, "restic-backup-core.service", "unknown")


def test_interlock_queries_system_and_user_units(monkeypatch: pytest.MonkeyPatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        state = "active" if "restic-tier0-offsite-cold.service" in command else "inactive"
        return SimpleNamespace(returncode=0, stdout=f"{state}\n", stderr="")

    monkeypatch.setattr(splat_route.subprocess, "run", fake_run)

    assert splat_route._backup_interlock_busy() == (
        True,
        "restic-tier0-offsite-cold.service",
        "active",
    )
    assert commands == [
        ["systemctl", "show", "restic-backup-core.service", "--property=ActiveState", "--value"],
        ["systemctl", "--user", "show", "restic-tier0-offsite.service", "--property=ActiveState", "--value"],
        ["systemctl", "--user", "show", "restic-tier0-offsite-cold.service", "--property=ActiveState", "--value"],
    ]


def test_active_backup_rejects_before_pipeline_planning(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        splat_route,
        "_backup_interlock_busy",
        lambda: (True, "restic-tier0-offsite.service", "active"),
    )
    monkeypatch.setattr(
        splat_route,
        "_engine_availability",
        lambda: pytest.fail("pipeline planning must not run while a backup is active"),
    )
    req = splat_route.SplatTrainRequest(mode="3d", input_path="/does/not/matter.mp4")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(splat_route.start_splat_training(SimpleNamespace(), req))

    assert exc.value.status_code == 409
    assert "restic-tier0-offsite.service is active" in exc.value.detail.lower()


def test_backup_starting_during_planning_loses_to_job_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"video")
    outputs = tmp_path / "outputs"
    default_root = outputs / "3d"
    states = iter(
        (
            (False, "", "inactive"),
            (True, "restic-backup-core.service", "activating"),
        )
    )

    monkeypatch.setattr(splat_route, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", default_root)
    monkeypatch.setattr(splat_route, "JOBS", {})
    monkeypatch.setattr(splat_route, "_backup_interlock_busy", lambda: next(states))
    monkeypatch.setattr(splat_route, "_engine_availability", lambda: {})
    monkeypatch.setattr(
        splat_route,
        "_plan_3d_job",
        lambda req, availability, job_dir, resolved_input: (["train"], {"train": ["true"]}, None),
    )
    req = splat_route.SplatTrainRequest(mode="3d", input_path=str(input_path))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(splat_route.start_splat_training(SimpleNamespace(), req))

    assert exc.value.status_code == 409
    assert "restic-backup-core.service is activating" in exc.value.detail.lower()
    assert splat_route.JOBS == {}
    assert not list(default_root.glob("splat_*"))
