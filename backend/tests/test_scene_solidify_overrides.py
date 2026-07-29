"""scene_solidify --element-source: fail-loud up-front validation and the
selection seam (override beats --prefer-generated), tested against the real
main() with the heavy legs monkeypatched at their function seams.

trimesh is stubbed with the standard guard (scene_solidify imports it at
module level; none of these paths touch it)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))

for _name in ("trimesh",):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)

import object_texture  # noqa: E402  (real path passed as --object-texture)
import scene_solidify  # noqa: E402

OBJECT_TEXTURE = Path(object_texture.__file__)


def _mk_job(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    isolated = job / "_scene" / "isolated" / "crate"
    isolated.mkdir(parents=True)
    (isolated / "object.ply").write_bytes(b"ply")
    (job / "_scene" / "inventory.json").write_text(json.dumps(
        {"instances": [{"slug": "crate", "label": "crate"}]}))
    (job / "meta.json").write_text(json.dumps({"meters_per_unit": 1.0}))
    return job


def _argv(job: Path, *extra: str) -> list[str]:
    return ["scene_solidify.py", str(job), "--skip-shell",
            "--python", sys.executable,
            "--object-texture", str(OBJECT_TEXTURE), *extra]


def _ok_texture(*_a, **_k):
    return {"ok": True, "seconds": 0.1,
            "report": {"faces": 12, "extent": [1, 1, 1],
                       "texture": {"baked": True}}}


def test_unknown_override_slug_is_fatal_before_any_build(
    tmp_path, monkeypatch, capsys
) -> None:
    job = _mk_job(tmp_path)
    built: list = []
    monkeypatch.setattr(scene_solidify, "run_object_texture",
                        lambda *a, **k: built.append(1) or _ok_texture())
    monkeypatch.setattr(sys, "argv", _argv(
        job, "--element-source", '{"ghost": "captured"}'))
    assert scene_solidify.main() == 1
    err = capsys.readouterr().err
    assert "ghost" in err and "known: crate" in err
    assert built == []  # refused before any element build
    assert not (job / "_world" / "world.json").exists()


def test_generated_override_without_a_candidate_is_fatal(
    tmp_path, monkeypatch, capsys
) -> None:
    job = _mk_job(tmp_path)
    monkeypatch.setattr(scene_solidify, "run_object_texture",
                        lambda *a, **k: _ok_texture())
    monkeypatch.setattr(sys, "argv", _argv(
        job, "--element-source", '{"crate": "generated"}'))
    assert scene_solidify.main() == 1
    assert "no PLACED candidate" in capsys.readouterr().err


def test_bad_override_json_and_values_are_fatal(tmp_path, monkeypatch, capsys) -> None:
    job = _mk_job(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(job, "--element-source", "{not json"))
    assert scene_solidify.main() == 1
    assert "not valid JSON" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", _argv(
        job, "--element-source", '{"crate": "proxy"}'))
    assert scene_solidify.main() == 1
    assert '"captured" or "generated"' in capsys.readouterr().err


def test_captured_override_beats_prefer_generated(tmp_path, monkeypatch) -> None:
    job = _mk_job(tmp_path)
    monkeypatch.setattr(scene_solidify, "_generated_candidate",
                        lambda job_dir, slug: {"report": {}, "iou": 0.9})
    monkeypatch.setattr(scene_solidify, "place_generated",
                        lambda *a, **k: pytest.fail(
                            "captured override must never place generated"))
    monkeypatch.setattr(scene_solidify, "run_object_texture",
                        lambda *a, **k: _ok_texture())
    monkeypatch.setattr(sys, "argv", _argv(
        job, "--prefer-generated",
        "--element-source", '{"crate": "captured"}'))
    assert scene_solidify.main() == 0
    doc = json.loads((job / "_world" / "world.json").read_text())
    (entry,) = doc["elements"]
    assert entry["selection"] == {"requested": "captured",
                                  "chosen": "captured",
                                  "reason": "override:captured"}
    assert entry["geometry_source"] == "gaussians"


def test_generated_override_places_the_candidate(tmp_path, monkeypatch) -> None:
    job = _mk_job(tmp_path)
    monkeypatch.setattr(scene_solidify, "_generated_candidate",
                        lambda job_dir, slug: {"report": {"note": "gen"},
                                               "iou": 0.9})
    monkeypatch.setattr(scene_solidify, "place_generated",
                        lambda gen, out, **k: {"ok": True, "seconds": 0.2,
                                               "faces": 34, "extent": [1, 1, 1],
                                               "texture": True})
    monkeypatch.setattr(scene_solidify, "_write_generated_sidecar",
                        lambda *a, **k: None)
    monkeypatch.setattr(scene_solidify, "run_object_texture",
                        lambda *a, **k: pytest.fail(
                            "generated override must not bake the captured mesh"))
    monkeypatch.setattr(sys, "argv", _argv(
        job, "--element-source", '{"crate": "generated"}'))
    assert scene_solidify.main() == 0
    doc = json.loads((job / "_world" / "world.json").read_text())
    (entry,) = doc["elements"]
    assert entry["selection"] == {"requested": "generated",
                                  "chosen": "generated",
                                  "reason": "override:generated"}
    assert entry["provenance"] == "generative render-only"


def test_no_overrides_default_selection_is_recorded(tmp_path, monkeypatch) -> None:
    job = _mk_job(tmp_path)
    monkeypatch.setattr(scene_solidify, "run_object_texture",
                        lambda *a, **k: _ok_texture())
    monkeypatch.setattr(sys, "argv", _argv(job))
    assert scene_solidify.main() == 0
    doc = json.loads((job / "_world" / "world.json").read_text())
    (entry,) = doc["elements"]
    assert entry["selection"] == {"requested": None,
                                  "chosen": "captured",
                                  "reason": "captured-default"}
