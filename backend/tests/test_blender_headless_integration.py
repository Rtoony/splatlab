"""Opt-in smoke test against the installed Blender binary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc import blender_workflow  # noqa: E402


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_BLENDER_TESTS") != "1",
    reason="set SPLATLAB_RUN_BLENDER_TESTS=1 for the real Blender smoke test",
)
def test_headless_snapshot_and_inspect(tmp_path: Path) -> None:
    blender = blender_workflow.BLENDER_BIN
    if not blender.is_file():
        pytest.skip("Blender binary is unavailable")

    source = tmp_path / "source.blend"
    create = subprocess.run(
        [
            str(blender),
            "--disable-autoexec",
            "--background",
            "--factory-startup",
            "--python-expr",
            (
                "import bpy; "
                "bpy.context.active_object.name='SplatLabCube'; "
                f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r}, check_existing=False)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=blender_workflow._sanitized_env(),
    )
    assert create.returncode == 0, create.stdout + create.stderr

    output = tmp_path / "version.blend"
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "schema": "dev.splatlab.blender-action/v1",
                "job_id": "splat_smoke01",
                "action": "transform_object",
                "params": {"object": "SplatLabCube", "location": [1, 2, 3]},
                "source_blend": str(source),
                "output_blend": str(output),
            }
        )
    )
    mutate = subprocess.run(
        [
            str(blender),
            "--disable-autoexec",
            "--background",
            str(source),
            "--python",
            str(blender_workflow.ACTION_SCRIPT),
            "--",
            str(request),
            str(response),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=blender_workflow._sanitized_env(),
    )
    assert mutate.returncode == 0, mutate.stdout + mutate.stderr
    assert output.is_file()
    result = json.loads(response.read_text())
    assert result["status"] == "ok"
    assert result["result"]["location"] == [1.0, 2.0, 3.0]
