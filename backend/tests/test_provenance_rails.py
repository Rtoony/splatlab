"""P6a provenance rails: the generative tag survives in-file (PLY header), the
quarantine path rules hold, the survey-lane guard refuses tagged/quarantined
inputs loudly, and scene_manifest round-trips with fail-loud validation.
Pure-stdlib modules under test — no subprocess, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))
import provenance  # noqa: E402
import scene_manifest  # noqa: E402


def _write_ply(path: Path, comments: list[str]) -> Path:
    header = ["ply", "format ascii 1.0"]
    header += [f"comment {c}" for c in comments]
    header += ["element vertex 1", "property float x", "property float y",
               "property float z", "end_header", "0 0 0", ""]
    path.write_text("\n".join(header))
    return path


class TestPlyTag:
    def test_tagged_ply_detected(self, tmp_path: Path):
        p = _write_ply(tmp_path / "a.ply", [provenance.GENERATIVE_TAG])
        assert provenance.ply_is_generative(p)
        assert provenance.ply_header_comments(p) == [provenance.GENERATIVE_TAG]

    def test_untagged_ply_clean(self, tmp_path: Path):
        p = _write_ply(tmp_path / "b.ply", ["SplatLab object isolate (P5b)"])
        assert not provenance.ply_is_generative(p)

    def test_non_ply_and_missing_are_clean(self, tmp_path: Path):
        txt = tmp_path / "c.txt"
        txt.write_text("not a ply")
        assert not provenance.ply_is_generative(txt)
        assert not provenance.ply_is_generative(tmp_path / "missing.ply")


class TestPathRules:
    @pytest.mark.parametrize("p,expected", [
        ("/jobs/splat_x/_regen/elements/table/proxy.ply", True),
        ("/jobs/splat_x/_regen/scene.glb", True),
        ("/jobs/splat_x/_objects/fire-hydrant/proxy.ply", True),
        ("/jobs/splat_x/_objects/fire-hydrant/proxy_preview.webp", True),
        ("/jobs/splat_x/_objects/fire-hydrant/object.ply", False),
        ("/jobs/splat_x/_mesh/mesh.ply", False),
    ])
    def test_quarantine_paths(self, p, expected):
        assert provenance.path_is_generative(p) is expected


class TestSurveyGuard:
    def test_refuses_tagged_file_outside_quarantine(self, tmp_path: Path):
        # The tag must survive a copy OUT of the quarantine tree.
        p = _write_ply(tmp_path / "innocent-name.ply", [provenance.GENERATIVE_TAG])
        with pytest.raises(provenance.GenerativeInputRefused, match="survey"):
            provenance.assert_not_generative(p, lane="survey")

    def test_refuses_quarantine_path_without_reading(self, tmp_path: Path):
        with pytest.raises(provenance.GenerativeInputRefused):
            provenance.assert_not_generative(
                tmp_path / "_regen" / "does-not-even-exist.ply", lane="survey")

    def test_allows_captured_mesh(self, tmp_path: Path):
        p = _write_ply(tmp_path / "mesh.ply", [])
        provenance.assert_not_generative(p, lane="survey")  # no raise


class TestSceneManifest:
    def _built(self):
        m = scene_manifest.new_manifest("splat_test01", "meters", 2.3537)
        scene_manifest.add_element(
            m, slug="table", provenance="proxy",
            files={"ply": "elements/table/proxy.ply"},
            transform_4x4=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            registration={"icp_fitness": 1.0})
        scene_manifest.add_element(
            m, slug="hedge", provenance="captured",
            files={"ply": "elements/hedge/object.ply"})
        scene_manifest.add_element(
            m, slug="ground", provenance="ground-derived",
            files={"glb": "ground_mesh.glb"})
        return m

    def test_round_trip(self, tmp_path: Path):
        m = self._built()
        out = scene_manifest.write_manifest(tmp_path, m)
        assert out.name == scene_manifest.MANIFEST_NAME
        back = scene_manifest.read_manifest(tmp_path)
        assert [e["slug"] for e in back["elements"]] == ["table", "hedge", "ground"]
        assert back["doctrine"] == provenance.GENERATIVE_TAG

    def test_meters_requires_scale(self):
        with pytest.raises(scene_manifest.ManifestError, match="meters_per_unit"):
            scene_manifest.new_manifest("splat_x", "meters", None)

    def test_scene_units_allowed_without_scale(self, tmp_path: Path):
        m = scene_manifest.new_manifest("splat_x", "scene-units", None)
        scene_manifest.write_manifest(tmp_path, m)

    def test_proxy_without_transform_fails(self, tmp_path: Path):
        m = scene_manifest.new_manifest("splat_x", "scene-units", None)
        scene_manifest.add_element(m, slug="t", provenance="proxy", files={"ply": "p.ply"})
        with pytest.raises(scene_manifest.ManifestError, match="transform_4x4"):
            scene_manifest.write_manifest(tmp_path, m)

    def test_skipped_proxy_allowed_without_transform(self, tmp_path: Path):
        m = scene_manifest.new_manifest("splat_x", "scene-units", None)
        scene_manifest.add_element(m, slug="t", provenance="proxy",
                                   files={}, skipped="generation-failed")
        scene_manifest.write_manifest(tmp_path, m)

    def test_bad_provenance_fails(self, tmp_path: Path):
        m = scene_manifest.new_manifest("splat_x", "scene-units", None)
        scene_manifest.add_element(m, slug="t", provenance="hallucinated", files={"a": "b"})
        with pytest.raises(scene_manifest.ManifestError, match="provenance"):
            scene_manifest.write_manifest(tmp_path, m)

    def test_duplicate_slug_fails(self, tmp_path: Path):
        m = scene_manifest.new_manifest("splat_x", "scene-units", None)
        for _ in range(2):
            scene_manifest.add_element(m, slug="t", provenance="captured", files={"a": "b"})
        with pytest.raises(scene_manifest.ManifestError, match="duplicate"):
            scene_manifest.write_manifest(tmp_path, m)

    def test_altered_doctrine_fails(self, tmp_path: Path):
        m = self._built()
        m["doctrine"] = "trust me"
        with pytest.raises(scene_manifest.ManifestError, match="doctrine"):
            scene_manifest.write_manifest(tmp_path, m)
