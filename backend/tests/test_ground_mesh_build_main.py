"""ground_mesh_build.main() itself — the layer both existing test styles
skipped, which is exactly where the `cells` NameError shipped (6c41a9f): the
route tests mock the subprocess and the mesh-env e2e is opt-in, so a crash
that fired AFTER the ply write but BEFORE the report went unseen. Live
evidence: Courtyard (splat_5c1db781) had ground_mesh_raw.ply and no
ground_mesh_build.json, and every ground build since 07-27 exited non-zero.

open3d is stubbed here (its writer writes real bytes so the artifact ORDER
stays honest); scipy is REAL — the Delaunay TIN genuinely triangulates. The
try-import guard keeps this file valid in the mesh env too.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "mesh"))

try:
    import open3d  # noqa: F401  (real module in the mesh env)
except ImportError:
    class _FakeMesh:
        def compute_vertex_normals(self) -> None:
            return None

    _o3d = types.ModuleType("open3d")
    _o3d.geometry = types.SimpleNamespace(TriangleMesh=_FakeMesh)
    _o3d.utility = types.SimpleNamespace(Vector3dVector=lambda a: a,
                                         Vector3iVector=lambda a: a)
    _o3d.io = types.SimpleNamespace(
        write_triangle_mesh=lambda path, mesh: Path(path).write_bytes(b"ply-stub"))
    sys.modules["open3d"] = _o3d

import ground_mesh_build  # noqa: E402


def _flat_ground(n_side=12, cell=0.1, per_cell=5, z=0.0, seed=0):
    """The test_world_numerical_cores recipe: a grid of occupied cells."""
    rng = np.random.default_rng(seed)
    points = []
    for i in range(n_side):
        for j in range(n_side):
            centre = ((i + 0.5) * cell, (j + 0.5) * cell)
            points.extend([[centre[0] + rng.normal(0, cell / 8),
                            centre[1] + rng.normal(0, cell / 8),
                            z + rng.normal(0, 0.001)] for _ in range(per_cell)])
    return np.array(points)


def _npz(tmp_path: Path, *, with_classes: bool = True) -> Path:
    pts = _flat_ground()  # 12x12 cells x 5 pts = 720 (clears the 500 FATAL)
    n = len(pts)
    payload = {
        "xyz": pts.astype(np.float32),
        "rel": np.full(n, 0.9, np.float32),
        "seen": np.ones(n, bool),
    }
    if with_classes:
        class_rel = np.zeros((n, 2), np.float16)
        class_rel[:, 1] = 0.9
        payload["class_rel"] = class_rel
        payload["class_ids"] = np.array(["grass", "pavement"])
    path = tmp_path / "ground_gaussians.npz"
    np.savez_compressed(path, **payload)
    return path


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, npz: Path,
         *extra: str) -> tuple[int, Path]:
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "ground_mesh_build.py", str(npz), str(out), "--cell-units", "0.1",
        *extra,
    ])
    return ground_mesh_build.main(), out


def test_main_writes_the_report_after_the_ply(monkeypatch, tmp_path) -> None:
    rc, out = _run(monkeypatch, tmp_path, _npz(tmp_path))
    assert rc == 0
    assert (out / "ground_mesh_raw.ply").is_file()
    # THE regression: before the fix, a NameError fired between the ply write
    # and this report, so the build "failed" with its artifacts half-landed.
    report = json.loads((out / "ground_mesh_build.json").read_text())
    assert report["ground_points"] == 144
    assert report["triangles"] > 0
    # Class leg ran too: every cell voted pavement.
    assert report["class_cells"] == {"pavement": 144}
    assert (out / "ground_class_cells.npz").is_file()


def test_report_ledger_is_coherent(monkeypatch, tmp_path) -> None:
    rc, out = _run(monkeypatch, tmp_path, _npz(tmp_path))
    assert rc == 0
    report = json.loads((out / "ground_mesh_build.json").read_text())
    # cells_with_data is the post-bin/pre-filter count; the filters' drops
    # must account exactly for what reached the TIN.
    assert (report["cells_with_data"]
            - report["cells_spike_rejected"]
            - report["cells_disconnected_dropped"]) == report["ground_points"]
    assert report["cells_with_data"] == 144


def test_coverage_floor_refuses_before_any_artifact(
    monkeypatch, tmp_path, capsys
) -> None:
    rc, out = _run(monkeypatch, tmp_path, _npz(tmp_path),
                   "--min-ground-points", "500")
    assert rc == 1
    assert "only 144 ground cells" in capsys.readouterr().err
    # The refusal must precede every artifact — this ordering is what lets
    # the prepare ladder's skip-if-exists resume cleanly after tuning.
    assert not (out / "ground_mesh_raw.ply").exists()
    assert not (out / "ground_mesh_build.json").exists()


def test_legacy_npz_without_classes_builds_classless(monkeypatch, tmp_path) -> None:
    rc, out = _run(monkeypatch, tmp_path, _npz(tmp_path, with_classes=False))
    assert rc == 0
    report = json.loads((out / "ground_mesh_build.json").read_text())
    assert report["class_cells"] is None
    assert not (out / "ground_class_cells.npz").exists()


def test_painted_field_flows_to_the_report(monkeypatch, tmp_path) -> None:
    """A painted detached island survives the whole pipeline and the report
    carries the authority ledger (PW-A3)."""
    main_pts = _flat_ground()                                  # 12x12 block
    island_pts = _flat_ground(n_side=2) + np.array([9.0, 9.0, 0.0])
    pts = np.vstack([main_pts, island_pts])
    n = len(pts)
    painted = np.zeros(n, bool)
    painted[len(main_pts):] = True
    path = tmp_path / "ground_gaussians.npz"
    np.savez_compressed(path, xyz=pts.astype(np.float32),
                        rel=np.full(n, 0.9, np.float32),
                        seen=np.ones(n, bool), painted=painted)

    rc, out = _run(monkeypatch, tmp_path, path)
    assert rc == 0
    report = json.loads((out / "ground_mesh_build.json").read_text())
    assert report["ground_points"] == 144 + 4
    assert report["cells_disconnected_dropped"] == 0
    assert report["painted_cells"] == 4
    assert report["painted_rescued_component"] == 4
    # Ledger still coherent with the rescues in play.
    assert (report["cells_with_data"]
            - report["cells_spike_rejected"]
            - report["cells_disconnected_dropped"]) == report["ground_points"]
