"""P6f fidelity-dial resolution (scene_assemble.py): pure stdlib, directly
unit-testable with no GPU/Blender. Already independently proven against real
garden data in a live run (see STATUS.md P6f entry); these tests pin the
exact rules so a future edit can't silently change them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))
import scene_assemble as sa  # noqa: E402


def _mk_scene(tmp_path: Path, *, table_proxy_built=True, vase_proxy_built=False) -> Path:
    scene_dir = tmp_path / "_scene"
    isolated = scene_dir / "isolated"
    proxied = scene_dir / "proxied"
    isolated.mkdir(parents=True)
    proxied.mkdir(parents=True)
    (isolated / "background.ply").write_bytes(b"ply")
    for slug in ("table", "vase"):
        (isolated / slug).mkdir()
        (isolated / slug / "object.ply").write_bytes(b"ply")
    (isolated / "batch_isolate.json").write_text(json.dumps({
        "instances": [
            {"slug": "table", "label": "round wooden table", "status": "built"},
            {"slug": "vase", "label": "flower vase", "status": "built"},
        ],
    }))
    proxy_instances = []
    if table_proxy_built:
        (proxied / "table").mkdir()
        (proxied / "table" / "proxy.ply").write_bytes(b"ply")
        proxy_instances.append({"slug": "table", "status": "built", "icp_fitness": 1.0,
                                "icp_rmse": 0.02, "total_scale": 0.6,
                                "transform_4x4": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    else:
        proxy_instances.append({"slug": "table", "status": "SKIPPED:generation-failed"})
    if vase_proxy_built:
        (proxied / "vase").mkdir()
        (proxied / "vase" / "proxy.ply").write_bytes(b"ply")
        proxy_instances.append({"slug": "vase", "status": "built", "icp_fitness": 0.9,
                                "icp_rmse": 0.01, "total_scale": 0.3,
                                "transform_4x4": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    else:
        proxy_instances.append({"slug": "vase", "status": "SKIPPED:generation-failed"})
    (proxied / "batch_proxy.json").write_text(json.dumps({"instances": proxy_instances}))
    return scene_dir


def test_faithful_mode_excludes_all_proxies(tmp_path):
    scene_dir = _mk_scene(tmp_path, table_proxy_built=True, vase_proxy_built=True)
    elements = sa.resolve_elements(scene_dir, "faithful", {})
    by_slug = {e["slug"]: e for e in elements}
    assert by_slug["table"]["provenance"] == "captured"
    assert by_slug["table"]["selection"]["reason"] == "faithful-excludes-proxy"
    assert by_slug["vase"]["provenance"] == "captured"


def test_styled_mode_prefers_proxy_falls_back_on_upstream_skip(tmp_path):
    scene_dir = _mk_scene(tmp_path, table_proxy_built=True, vase_proxy_built=False)
    elements = sa.resolve_elements(scene_dir, "styled", {})
    by_slug = {e["slug"]: e for e in elements}
    assert by_slug["table"]["provenance"] == "proxy"
    assert by_slug["table"]["selection"]["reason"] == "styled-mode"
    assert by_slug["vase"]["provenance"] == "captured"
    assert by_slug["vase"]["selection"]["reason"] == "upstream-skip:SKIPPED:generation-failed"


def test_override_keeps_real_object_in_styled_scene(tmp_path):
    scene_dir = _mk_scene(tmp_path, table_proxy_built=True, vase_proxy_built=True)
    elements = sa.resolve_elements(scene_dir, "styled", {"table": "captured"})
    by_slug = {e["slug"]: e for e in elements}
    assert by_slug["table"]["provenance"] == "captured"
    assert by_slug["table"]["selection"]["reason"] == "override:captured"
    assert by_slug["vase"]["provenance"] == "proxy"  # unaffected by the override


def test_override_requesting_unbuilt_proxy_fails_loud(tmp_path):
    scene_dir = _mk_scene(tmp_path, table_proxy_built=False, vase_proxy_built=False)
    with pytest.raises(sa.AssembleError, match="is available"):
        sa.resolve_elements(scene_dir, "styled", {"table": "proxy"})


def test_override_naming_unknown_slug_fails_loud(tmp_path):
    scene_dir = _mk_scene(tmp_path)
    with pytest.raises(sa.AssembleError, match="unknown slug"):
        sa.resolve_elements(scene_dir, "styled", {"nonexistent": "captured"})


def test_background_and_ground_are_never_overridable(tmp_path):
    scene_dir = _mk_scene(tmp_path)
    ground_dir = scene_dir / "ground"
    ground_dir.mkdir()
    (ground_dir / "ground_mesh.glb").write_bytes(b"glb")
    elements = sa.resolve_elements(scene_dir, "styled", {})
    by_slug = {e["slug"]: e for e in elements}
    assert by_slug["background"]["provenance"] == "captured"
    assert by_slug["ground"]["provenance"] == "ground-derived"


def test_ground_element_omitted_when_not_built(tmp_path):
    scene_dir = _mk_scene(tmp_path)  # no ground/ dir created
    elements = sa.resolve_elements(scene_dir, "styled", {})
    assert "ground" not in {e["slug"] for e in elements}


def test_missing_background_fails_loud(tmp_path):
    scene_dir = tmp_path / "_scene"
    (scene_dir / "isolated").mkdir(parents=True)
    (scene_dir / "isolated" / "batch_isolate.json").write_text(json.dumps({"instances": []}))
    with pytest.raises(sa.AssembleError, match="background"):
        sa.resolve_elements(scene_dir, "styled", {})
