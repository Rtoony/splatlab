import json
import pytest

import artifact_dependencies as dependencies
import artifact_manifest as manifests


def test_raw_collision_uses_the_world_bake_not_the_latest_capture_scale():
    report = {"geometry_frame": {"axis": "y-up", "units": "scene-units", "meters_per_unit": 9}}
    world = {"units": "meters", "meters_per_unit": 1, "calibrated_from_meters_per_unit": 0.94975}
    assert dependencies.collision_scale_to_world(world, report) == 0.94975
    assert dependencies.collision_scale_to_world(world, {}) == 1
    assert dependencies.collision_scale_to_world({"units": "scene-units"}, report) == 1
    for scale in (None, True, 0, -1, float("nan")):
        with pytest.raises(ValueError, match="scale"):
            dependencies.collision_scale_to_world({**world, "calibrated_from_meters_per_unit": scale}, report)


def test_shell_reuse_tracks_scale_patch_addition_and_source(tmp_path):
    world = tmp_path / "_world"
    world.mkdir()
    (world / "collision_shell.glb").write_bytes(b"geometry")
    (tmp_path / "meta.json").write_text(json.dumps({"scale_generation": 1, "meters_per_unit": 2}))
    receipt = world / "collision_shell.json"
    assert not dependencies.shell_is_current(tmp_path)


    manifests.atomic_write_json(receipt, {"verdict": "PASS", "dependencies": dependencies.shell_dependencies(tmp_path), "output": manifests.file_identity(world / "collision_shell.glb")})
    assert dependencies.shell_is_current(tmp_path)
    patches = tmp_path / "_scene" / "surfaces"
    patches.mkdir(parents=True)
    (patches / "patch_wall.ply").write_bytes(b"wall")
    assert not dependencies.shell_is_current(tmp_path)
    manifests.atomic_write_json(receipt, {"verdict": "PASS", "dependencies": dependencies.shell_dependencies(tmp_path), "output": manifests.file_identity(world / "collision_shell.glb")})
    assert dependencies.shell_is_current(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"scale_generation": 2, "meters_per_unit": 3}))
    assert not dependencies.shell_is_current(tmp_path)


@pytest.mark.parametrize("verdict", [None, "FAILED", "NOT_WALKABLE"])
def test_current_inputs_do_not_make_a_failed_collision_candidate_acceptable(tmp_path, verdict):
    world = tmp_path / "_world"
    world.mkdir()
    (world / "collision_shell.glb").write_bytes(b"geometry")
    manifests.atomic_write_json(world / "collision_shell.json", {
        "verdict": verdict, "dependencies": dependencies.shell_dependencies(tmp_path),
        "output": manifests.file_identity(world / "collision_shell.glb"),
    })
    assert not dependencies.shell_is_current(tmp_path)
