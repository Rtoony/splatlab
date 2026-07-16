from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maintenance_gate


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
