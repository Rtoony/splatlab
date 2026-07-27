"""The voxel shell's coordinate frame.

The bug this pins: `scene_solidify.build_shell` with `--shell-source voxel` fed
`collision_shell.glb` — which world_shell.py writes in the Y-up three.js frame —
straight into object_texture.py, which assumes CAPTURE-frame input. That did two
wrong things at once:

  * object_texture.py:863-866 applies the capture->Y-up rotation
    unconditionally (only the SCALE depends on meters_per_unit), so the shell
    was rotated a second time and ended up 90 degrees off the props and off the
    collision solid it is meant to coincide with;
  * colour is sampled from the capture-frame splat, so every texel was read
    from the wrong neighbourhood — the hazard object_texture.py:460 warns about
    — which is why voxel shells came out blotchy.

Measured on splat_3aaf8067 before the fix: rotating collision_shell.glb -90
degrees about X raised its bbox IoU with shell.glb from 0.349 to 0.733.

The TSDF path feeds capture-frame geometry and was always correct, so this only
ever affected --shell-source voxel.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))

# scene_solidify imports trimesh at module level; the frame maths under test is
# pure numpy, so a stub keeps this runnable in the app's test interpreter.
try:
    import trimesh  # noqa: F401
    # `import trimesh` succeeding is NOT proof it is real: test_world_numerical_cores
    # inserts a types.ModuleType stub into sys.modules, and whether this file sees
    # it depends on collection order. Probe for something only the real package has.
    _REAL_TRIMESH = hasattr(trimesh, "creation")
except ImportError:
    _REAL_TRIMESH = False
    sys.modules["trimesh"] = types.ModuleType("trimesh")

import scene_solidify as ss  # noqa: E402


# The conversion world_shell.py applies: capture -> Y-up.
def _capture_to_yup(v: np.ndarray) -> np.ndarray:
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


# The conversion object_texture.py applies on export, unconditionally.
_OBJECT_TEXTURE_EXPORT = _capture_to_yup


def _inverse(v: np.ndarray) -> np.ndarray:
    """What voxel_to_capture_frame does, isolated from any mesh I/O."""
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


CAPTURE = np.array([
    [1.0, 2.0, 3.0],
    [-1.0, 0.5, 0.0],
    [0.0, 0.0, 0.0],
    [4.0, -2.0, 7.5],
])


def test_the_inverse_actually_inverts_the_yup_conversion():
    assert _inverse(_capture_to_yup(CAPTURE)) == pytest.approx(CAPTURE)


def test_the_inverse_is_a_proper_rotation_not_a_reflection():
    """A reflection would mirror the shell — every face would wind backwards."""
    basis = np.eye(3)
    matrix = _inverse(basis).T
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_the_round_trip_preserves_lengths():
    lengths = np.linalg.norm(CAPTURE, axis=1)
    assert np.linalg.norm(_inverse(CAPTURE), axis=1) == pytest.approx(lengths)


def test_a_yup_mesh_passed_through_unfixed_lands_rotated_ninety_degrees():
    """The bug, reproduced: a Y-up input exported by object_texture ends up with
    its thin axis in Z instead of Y — exactly what was measured on disk."""
    flat_capture = np.array([[3.0, 4.0, 0.1], [-3.0, -4.0, -0.1]])   # thin in z
    already_yup = _capture_to_yup(flat_capture)                      # thin in y

    def thin_axis(v):
        return int(np.argmin(v.max(axis=0) - v.min(axis=0)))

    assert thin_axis(already_yup) == 1, "world_shell hands over a Y-up mesh"
    # Unfixed: object_texture rotates it AGAIN.
    assert thin_axis(_OBJECT_TEXTURE_EXPORT(already_yup)) == 2, "lands thin in Z"


def test_the_fix_restores_the_intended_frame():
    """With the inverse applied first, the export lands Y-up as intended."""
    flat_capture = np.array([[3.0, 4.0, 0.1], [-3.0, -4.0, -0.1]])
    already_yup = _capture_to_yup(flat_capture)

    exported = _OBJECT_TEXTURE_EXPORT(_inverse(already_yup))

    assert exported == pytest.approx(already_yup), "back to where it started"
    thin = int(np.argmin(exported.max(axis=0) - exported.min(axis=0)))
    assert thin == 1, "thin in Y, matching the props and the collision solid"


def test_object_texture_still_rotates_unconditionally():
    """The assumption this fix rests on. If object_texture ever gains its own
    frame flag, the pre-rotation here becomes a double-fix — fail loudly then."""
    source = (Path(__file__).resolve().parents[1] / "mesh" / "object_texture.py").read_text()
    export = source[source.index("# ---- export: scale to metres, Y-up"):][:400]

    assert "yup = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)" in export
    assert "scale = args.meters_per_unit or 1.0" in export, "only SCALE is conditional"


def test_build_shell_pre_rotates_only_the_voxel_source():
    """The TSDF path feeds capture-frame geometry and must stay untouched."""
    source = (Path(__file__).resolve().parents[1] / "mesh" / "scene_solidify.py").read_text()
    voxel_branch = source[source.index('if args.shell_source == "voxel":'):]
    voxel_branch = voxel_branch[:voxel_branch.index("else:")]

    assert "voxel_to_capture_frame" in voxel_branch
    calls = source.count("voxel_to_capture_frame(") - source.count("def voxel_to_capture_frame(")
    assert calls == 1, "called exactly once, and only in the voxel branch"


@pytest.mark.skipif(not _REAL_TRIMESH, reason="trimesh not in this interpreter")
def test_voxel_to_capture_frame_round_trips_a_real_mesh(tmp_path):
    import trimesh as tm

    box = tm.creation.box(extents=(4.0, 6.0, 1.0))
    src = tmp_path / "yup.glb"
    box.export(str(src))

    out = ss.voxel_to_capture_frame(src, tmp_path / "capture.ply")
    back = tm.load(str(out), force="mesh", process=False)

    original = np.asarray(box.vertices)
    assert np.asarray(back.vertices) == pytest.approx(_inverse(original))
    assert len(back.faces) == len(box.faces), "no geometry lost"


# ---------------------------------------------------------------------------
# world_gate measures the mesh the player actually stands on
# ---------------------------------------------------------------------------

def test_walkability_gates_prefer_the_collision_solid():
    """shell_connectivity and floor_continuity ask WALKABILITY questions, so
    they must be asked of the mesh the player collides with.

    world_shell.py exists because render geometry != collision geometry (the
    TSDF shell is lace). Measuring shell.glb asked those questions of the wrong
    artifact: a visual shell with holes scored unwalkable even though the player
    collides against a watertight solid and cannot fall anywhere. That is what
    made --drop-unobserved look like a regression when it was not.
    """
    source = (Path(__file__).resolve().parents[1] / "mesh" / "world_gate.py").read_text()
    block = source[source.index("collision_glb = world /"):]
    block = block[:block.index("gates[\"prop_integrity\"]")] if "gates[\"prop_integrity\"]" in block else block[:2000]

    assert 'collision_glb = world / "collision_shell.glb"' in block
    assert "gate_shell_connectivity(walk_mesh)" in block
    assert "gate_floor_continuity(walk_mesh" in block
    # A gate that will not say what it looked at is not evidence.
    assert '"measured_mesh"' in block


def test_the_gates_still_fall_back_to_the_visual_shell():
    """Worlds built before the collision-shell stage must still be gradeable."""
    source = (Path(__file__).resolve().parents[1] / "mesh" / "world_gate.py").read_text()
    block = source[source.index("collision_glb = world /"):][:1200]

    assert "walk_mesh, walk_source = shell_mesh, shell_glb.name" in block
    assert "if collision_glb.is_file():" in block


# ---------------------------------------------------------------------------
# mesh_gate reports whether the mesh encloses the cameras that score it
# ---------------------------------------------------------------------------

def test_mesh_gate_reports_camera_containment():
    """A watertight shell over an object-centric orbit capture encloses the
    camera ring, so every render looks at its own interior walls. That scores
    like noise and is invisible in PSNR alone — measured on splat_3aaf8067,
    175/175 cameras inside, 11.96 dB, and four different knobs moved it 0.23 dB
    total. The gate has to say it out loud."""
    source = (Path(__file__).resolve().parents[1] / "mesh" / "mesh_gate.py").read_text()

    assert '"cameras_inside_mesh_bbox"' in source
    assert '"encloses_capture"' in source
    assert "cam_pos = cams.camera_to_worlds[:, :3, 3].numpy()" in source
    # Must be measured in the frame the renderer saw, i.e. after --transform.
    assert "lo/hi are measured AFTER --transform" in source


def test_camera_containment_is_a_simple_bbox_test():
    """The maths, isolated — the gate itself needs open3d and real cameras."""
    lo, hi = np.array([-6.7, -6.4, -1.5]), np.array([3.8, 6.2, 5.9])
    cameras = np.array([[-0.88, -0.98, -0.27], [0.89, 1.0, 0.30], [0.0, 0.0, 0.0]])
    inside = np.all((cameras >= lo) & (cameras <= hi), axis=1)
    assert inside.mean() == 1.0, "the shell case: every camera enclosed"

    ground_lo, ground_hi = np.array([-2.1, -2.4, -0.5]), np.array([2.1, 2.4, 0.17])
    outside = np.all((cameras >= ground_lo) & (cameras <= ground_hi), axis=1)
    assert outside.mean() < 1.0, "the ground case: cameras look at it from outside"
