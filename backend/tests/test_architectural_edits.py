import hashlib
import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import architectural_edits as architecture
import architectural_navigation as navigation
import architectural_volumes as volumes
import artifact_manifest as manifests
import reconstruction_evidence as evidence
import scene_revisions as scenes
import scene_studio_route
import selection_reviews as selections
from test_background_recovery import observed_scene
from test_selection_reviews import selected_scene
from test_place_route import _authored_glb


def spec():
    return {"origin": [3., -1., 4.], "yaw_degrees": 35., "opening_width": 1.1, "opening_height": 2.1, "cut_depth": .6,
            "room_width": 3., "room_depth": 3., "room_height": 2.7, "wall_thickness": .15}


@pytest.mark.parametrize("coordinate_frame", ["world-y-up-metres", "capture-y-up-scene-units"])
def test_export_metadata_preserves_the_literal_survey_exclusion_tag(coordinate_frame):
    from mesh.provenance import GENERATIVE_TAG, GLTF_EXTRAS_KEY
    metadata = architecture.geometry_metadata("architecture_" + "a" * 24, coordinate_frame, "Authored, not measured")
    assert metadata[GLTF_EXTRAS_KEY] == GENERATIVE_TAG
    assert metadata["splatlab_architecture"] == {
        "method": architecture.METHOD, "architecture_id": "architecture_" + "a" * 24,
        "frame": coordinate_frame, "scope": "Authored, not measured"}


def replace_active(job, pointer, mutate):
    revision = scenes.read_revision(job, pointer["revision_id"])
    revision.pop("revision_id")
    revision.pop("content_sha256")
    mutate(revision)
    changed = scenes.write_revision(job, revision)
    pointer = {**pointer, "revision_id": changed["revision_id"]}
    manifests.atomic_write_json(scenes.root(job) / "active.json", pointer)
    return pointer


@pytest.fixture
def architectural_scene(selected_scene):
    job, pointer = selected_scene
    mesh = job / "test-room.glb"
    mesh.write_bytes(_authored_glb(aabb_min=(8, 0, 8), aabb_max=(9, 1, 9)))
    mesh_identity = scenes.store_file(job, mesh)
    def mutate(revision):
        revision["artifacts"]["_world/collision_shell.glb"] = mesh_identity
        revision["artifacts"]["_world/elements/box.glb"] = mesh_identity
        revision["artifacts"]["_preview/langweb.ply"] = revision["artifacts"]["_preview/splat.ply"]
        revision["artifacts"]["_world/collision_shell.json"] = scenes.store_json(job, {
            "verdict": "PASS", "geometry_frame": {"axis": "y-up", "units": "scene-units"}, "probe": {"floor_level_y": -1.}})
        state = revision["state"]
        state["viewer"]["elements"][0].update(role="prop", files={"glb": "artifact:_world/elements/box.glb"})
        state["viewer"]["collision_shell"] = {"glb": "artifact:_world/collision_shell.glb", "report": "artifact:_world/collision_shell.json", "scale_to_world": 1}
        state["semantics"]["box"] = {"label": "box", "provenance": "observed", "active": True}
    pointer = replace_active(job, pointer, mutate)
    return job, pointer


def stage_result(job, receipt, passed=True, navigation_changes=None):
    output = architecture.directory(job, receipt["architecture_id"])
    for name in ("room.glb", "collision_shell.glb", "cut-volume.glb"):
        (output / name).write_bytes(_authored_glb(aabb_min=(0, 0, 0), aabb_max=(3, 3, 3)))
    for name in ("collision_shell.json", "navmesh.json"):
        manifests.atomic_write_json(output / name, {"method": "synthetic unit fixture, not actual geometry verification"})
    frame = architecture.frame(receipt["spec"])
    points, _ = navigation.local_routes(receipt["spec"], receipt["recipe"])
    gates = {key: passed for key in architecture.required_gates(receipt)}
    selected_method = navigation.method(receipt["recipe"])
    navigation_document = {"v": 2 if selected_method == navigation.ENTRY_METHOD else 1, "navigation_method": selected_method,
        "frame": "world-y-up-metres", "architecture_id": receipt["architecture_id"], "gates": gates,
        "route": (points @ frame["rotation"].T + frame["origin"]).tolist(), "floor_y": [float(frame["origin"][1])] * len(points),
        "center_floor_y": [float(frame["origin"][1])] * len(points), "capsule_radius_m": .22, "capsule_height_m": 1.7,
        "scope": "Synthetic unit fixture, not actual geometry verification"}
    if selected_method == navigation.LEGACY_METHOD:
        navigation_document.pop("navigation_method")
        navigation_document.pop("center_floor_y")
    navigation_document.update(navigation_changes or {})
    manifests.atomic_write_json(output / "navmesh.json", navigation_document)
    np.savez_compressed(output / "geometry-probes.npz", synthetic=np.array([True]))
    return selections.seal(job, output, "result.json", {"architecture_id": receipt["architecture_id"], "prepared_sha256": receipt["sha256"], "method": architecture.METHOD,
        **({"navigation_method": selected_method} if selected_method == navigation.ENTRY_METHOD else {}),
        "appearance_method": volumes.method(receipt["clip"]),
        "geometry_method": architecture.geometry_method(receipt),
        "metrics": {"room_triangles": len(architecture.room_geometry(receipt["spec"], architecture.geometry_method(receipt))[1]),
                    "floor_finish_m": architecture.floor_finish_m(receipt["spec"], architecture.geometry_method(receipt))},
        "verdict": "PASS_ARCHITECTURAL_EDIT" if passed else "FAIL_ARCHITECTURAL_EDIT", "gates": gates,
        "scope": "Synthetic receipt fixture only"}, [output / name for name in ("room.glb", "collision_shell.glb", "cut-volume.glb", "collision_shell.json", "navmesh.json", "geometry-probes.npz")])


def test_metric_portal_frame_and_room_have_one_coordinate_system():
    value = spec()
    frame = architecture.frame(value)
    local = np.array([[0., 0., 0.], [.55, 2.1, .3], [-.55, .015, -.3]])
    world = local @ frame["rotation"].T + frame["origin"]
    assert architecture.local_coordinates(world, value) == pytest.approx(local)
    assert world @ frame["world_to_portal"][:3, :3].T + frame["world_to_portal"][:3, 3] == pytest.approx(local)
    vertices, triangles = architecture.room_geometry(value)
    local_vertices = architecture.local_coordinates(vertices, value)
    assert len(triangles) == 144
    assert local_vertices[:, 1].min() == pytest.approx(-.15)
    assert local_vertices[:, 1].max() == pytest.approx(2.85)
    assert local_vertices[:, 2].min() == pytest.approx(-.6)
    volume = np.sum(np.einsum("ij,ij->i", vertices[triangles[:, 0]], np.cross(vertices[triangles[:, 1]], vertices[triangles[:, 2]]))) / 6
    assert volume > 0
    assert not ((local_vertices[:, 0] > -.54) & (local_vertices[:, 0] < .54)
                & (local_vertices[:, 1] > .02) & (local_vertices[:, 1] < 2.09) & (local_vertices[:, 2] < .38)).any()


@pytest.mark.parametrize("yaw", [0., 35., -80., 180.])
@pytest.mark.parametrize("depth", [.1, .6, 4., 6.])
def test_floor_finish_moves_only_both_slab_vertices_and_keeps_collision_frame(yaw, depth):
    value = spec() | {"yaw_degrees": yaw, "cut_depth": depth}
    original, faces = architecture.room_geometry(value)
    finished, finished_faces = architecture.room_geometry(value, architecture.FINISHED_GEOMETRY)
    changed = np.any(finished != original, axis=1)
    assert changed.sum() == 16
    assert np.array_equal(finished[:, [0, 2]], original[:, [0, 2]])
    assert np.array_equal(finished_faces, faces)
    assert finished[changed, 1] - original[changed, 1] == pytest.approx(np.full(16, .016))
    local = architecture.local_coordinates(finished[changed], value)
    assert np.unique(np.round(local[:, 1], 9)) == pytest.approx([-.134, .016])
    regions = architecture.protected_volume_bounds(value, architecture.FINISHED_GEOMETRY)
    assert len(regions) == 4
    for point in local:
        assert any(np.all(point >= lower - 1e-12) and np.all(point <= upper + 1e-12) for lower, upper in regions)
    assert architecture.frame(value)["lower"][1] == .015
    assert architecture.floor_finish_m(value, architecture.LEGACY_GEOMETRY) == 0.


def test_finished_floor_protection_covers_leading_lip_without_enlarging_full_height_approach():
    value = spec() | {"cut_depth": 4.}
    regions = architecture.protected_volume_bounds(value, architecture.FINISHED_GEOMETRY)
    def contained(point):
        return any(np.all(point > lower) and np.all(point < upper) for lower, upper in regions)
    assert contained(np.array([0., .01, -2.2]))
    assert not contained(np.array([0., .1, -2.2]))
    assert not contained(np.array([.7, .01, -2.2]))


@pytest.mark.parametrize("method", [None, False, [], {}, "raised-floor-finish/v2"])
def test_unknown_geometry_method_refuses_before_preparing(architectural_scene, method):
    job, pointer = architectural_scene
    with pytest.raises(evidence.EvidenceError, match="geometry method"):
        architecture.prepare(job, spec(), 0, "Unknown recipe", selected_geometry=method)
    assert scenes.active(job) == pointer


def test_legacy_receipt_without_geometry_method_still_verifies(architectural_scene):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Nominal floor")
    assert receipt.pop("geometry_method") == architecture.LEGACY_GEOMETRY
    receipt["sha256"] = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in receipt.items() if key != "sha256"})).hexdigest()
    manifests.atomic_write_json(architecture.directory(job, receipt["architecture_id"]) / "receipt.json", receipt)
    architecture.verify(job, receipt)
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("method", [None, architecture.LEGACY_GEOMETRY, "raised-floor-finish/v2", "missing"])
def test_resealed_result_cannot_substitute_another_floor_method(architectural_scene, method):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Finished floor", selected_geometry=architecture.FINISHED_GEOMETRY)
    result = stage_result(job, receipt)
    if method == "missing":
        result.pop("geometry_method")
    else:
        result["geometry_method"] = method
    result["sha256"] = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in result.items() if key != "sha256"})).hexdigest()
    manifests.atomic_write_json(architecture.directory(job, receipt["architecture_id"]) / "result.json", result)
    with pytest.raises(scenes.SceneRevisionError, match="geometry method"):
        scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, "New room", 0)
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("defect", ["method", "missing-method", "finish", "missing-finish"])
def test_finished_revision_keeps_visible_floor_recipe_paired(architectural_scene, defect):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Finished floor", selected_geometry=architecture.FINISHED_GEOMETRY)
    stage_result(job, receipt)
    proposal = scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, "New room", 0)
    preview = scenes.read_revision(job, proposal["preview_revision"])
    architecture.verify_revision(job, preview)
    entry = next(value for value in preview["state"]["viewer"]["elements"] if value["slug"] == "extension")
    assert entry["floor_finish_m"] == .016
    if defect == "method":
        entry["geometry_method"] = architecture.LEGACY_GEOMETRY
    elif defect == "missing-method":
        entry.pop("geometry_method")
    elif defect == "finish":
        entry["floor_finish_m"] = .03
    else:
        entry.pop("floor_finish_m")
    with pytest.raises(evidence.EvidenceError, match="floor finish"):
        architecture.verify_revision(job, preview)
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("yaw", [0., 35., 69.43, -132.17, 180.])
def test_room_master_uses_exact_scalar_transform_without_blas_rounding(yaw):
    value = spec() | {"origin": [1.87, -1.754, .88], "yaw_degrees": yaw}
    local, faces = architecture.room_geometry(value | {"origin": [0., 0., 0.], "yaw_degrees": 0.})
    cosine, sine = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    expected = np.array([[cosine * float(vertex[0]) + sine * float(vertex[2]) + 1.87,
                          float(vertex[1]) - 1.754,
                          -sine * float(vertex[0]) + cosine * float(vertex[2]) + .88] for vertex in local])
    actual, actual_faces = architecture.room_geometry(value)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual_faces, faces)


@pytest.mark.parametrize("method", [architecture.LEGACY_GEOMETRY, architecture.FINISHED_GEOMETRY, architecture.JOINED_GEOMETRY])
def test_room_master_single_ulp_change_refuses_even_with_matching_blob_identity(architectural_scene, method):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), pointer["generation"], "Authored room", selected_geometry=method)
    vertices, faces = architecture.room_geometry(receipt["spec"], architecture.geometry_method(receipt))
    vertices[0, 0] = np.nextafter(vertices[0, 0], np.inf)
    altered = job / "altered-room.npz"
    np.savez_compressed(altered, vertices=vertices, faces=faces)
    receipt["artifacts"]["authored-room.npz"] = scenes.store_file(job, altered)
    output = architecture.directory(job, receipt["architecture_id"])
    receipt["sha256"] = hashlib.sha256(scenes.canonical_bytes({key: item for key, item in receipt.items() if key != "sha256"})).hexdigest()
    manifests.atomic_write_json(output / "receipt.json", receipt)
    with pytest.raises(evidence.EvidenceError, match="room master differs"):
        architecture.verify(job, architecture.read(job, receipt["architecture_id"]))


def test_protected_room_volume_does_not_extend_the_full_room_width_back_over_the_approach():
    value = spec() | {"cut_depth": 2., "opening_width": 1.5}
    regions = architecture.protected_volume_bounds(value)
    def contained(point):
        return any(np.all(point > lower) and np.all(point < upper) for lower, upper in regions)
    assert not contained(np.array([1.7, 1., -.8]))
    assert contained(np.array([.8, 1., -.8]))
    assert contained(np.array([1.6, 1., 2.]))
    assert contained(np.array([0., 2.78, 2.]))


@pytest.mark.parametrize("depth", [.1, .6, 2., 4., 6.])
@pytest.mark.parametrize("method", [architecture.LEGACY_GEOMETRY, architecture.FINISHED_GEOMETRY, architecture.JOINED_GEOMETRY])
def test_protected_volume_encloses_actual_above_floor_room_and_tunnel_vertices(depth, method):
    value = spec() | {"cut_depth": depth}
    vertices, _ = architecture.room_geometry(value, method)
    local = architecture.local_coordinates(vertices, value)
    regions = architecture.protected_volume_bounds(value, method)
    for point in local[local[:, 1] > .005]:
        assert any(np.all(point >= lower - 1e-12) and np.all(point <= upper + 1e-12) for lower, upper in regions)


@pytest.mark.parametrize("key,value", [("origin", [True, False, True]), ("origin", [0, np.nan, 0]), ("origin", [101, 0, 0]),
    ("origin", [0, 0]), ("origin", [[0], [1, 2], 3]), ("origin", "invalid"), ("yaw_degrees", np.inf), ("yaw_degrees", 181), ("opening_width", True), ("opening_width", .7),
    ("opening_height", 1.8), ("cut_depth", 6.001), ("cut_depth", .099), ("room_width", 1.1), ("room_height", 2.15), ("wall_thickness", .01), ("extra", 1)])
def test_malformed_or_unbounded_architecture_refuses(key, value):
    changed = spec() | {key: value}
    with pytest.raises(evidence.EvidenceError):
        architecture.frame(changed)


def test_prepare_retains_authored_frame_and_never_mutates_active(architectural_scene):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), pointer["generation"], "Add a doorway and connected room, explicitly authored")
    assert receipt["base"] == pointer and receipt["method"] == architecture.METHOD
    assert set(receipt["artifacts"]) == {"authored-room.npz", "portal-frame.json", "replacement-membership.npz"}
    assert receipt["protected_elements"][0]["scale"] == 1
    assert scenes.active(job) == pointer
    architecture.verify(job, receipt)
    with pytest.raises(evidence.EvidenceError, match="does not belong"):
        architecture.artifact(job, receipt["architecture_id"], "../../meta.json")


@pytest.mark.parametrize("defect", ["generation", "scale", "frame", "partition", "native", "limit"])
def test_unsafe_architectural_baseline_refuses(architectural_scene, defect):
    job, pointer = architectural_scene
    def mutate(revision):
        if defect == "scale":
            revision["state"]["viewer"]["collision_shell"]["scale_to_world"] = 2
        elif defect == "frame":
            revision["artifacts"]["_world/collision_shell.json"] = scenes.store_json(job, {"verdict": "PASS", "geometry_frame": {"axis": "z-up"}})
        elif defect == "partition":
            revision["state"]["capture_partition"] = {"collision_id": "existing"}
        elif defect == "native":
            revision["state"]["viewer"]["elements"][0]["gaussian_appearance"] = {"gaussians_id": "existing"}
        elif defect == "limit":
            revision["state"]["architectural_portals"] = [None] * 4
    pointer = replace_active(job, pointer, mutate)
    with pytest.raises(evidence.EvidenceError):
        architecture.prepare(job, spec(), 1 if defect == "generation" else 0, "New room")
    assert scenes.active(job) == pointer


def test_architecture_receipt_and_stale_or_corrupt_artifacts_refuse(architectural_scene):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "New room")
    blob = architecture.artifact(job, receipt["architecture_id"], "portal-frame.json")
    original = blob.read_bytes()
    blob.chmod(0o600)
    blob.write_bytes(original + b" ")
    with pytest.raises(evidence.EvidenceError, match="corrupt"):
        architecture.verify(job, receipt)
    blob.write_bytes(original)
    changed = {**pointer, "generation": 1}
    manifests.atomic_write_json(scenes.root(job) / "active.json", changed)
    with pytest.raises(evidence.EvidenceError, match="older active"):
        architecture.verify(job, receipt)


def test_local_gate_failure_cannot_create_paired_proposal(architectural_scene):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "New room")
    stage_result(job, receipt, passed=False)
    with pytest.raises(scenes.SceneRevisionError, match="local geometry/navigation gates"):
        scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, "New room", 0)
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("legacy", [False, True])
def test_paired_visual_collision_semantics_apply_and_restore_together(architectural_scene, monkeypatch, legacy):
    if legacy:
        monkeypatch.setattr(navigation, "recipe", lambda: dict(navigation.LEGACY_RECIPE))
        monkeypatch.setattr(volumes, "default_method", lambda: volumes.LEGACY_METHOD)
    job, pointer = architectural_scene
    baseline = scenes.read_revision(job, pointer["revision_id"])
    receipt = architecture.prepare(job, spec(), 0, "New room")
    result = stage_result(job, receipt)
    proposal = scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, "New room", 0)
    assert scenes.active(job) == pointer
    preview = scenes.read_revision(job, proposal["preview_revision"])
    assert preview["artifacts"]["_world/collision_shell.glb"] == result["artifacts"]["collision_shell.glb"]
    assert preview["artifacts"]["_world/elements/extension.glb"] == result["artifacts"]["room.glb"]
    assert preview["state"]["architectural_portals"][0]["world_to_portal"] == receipt["clip"]["world_to_portal"]
    if legacy:
        assert "room_lower" not in preview["state"]["architectural_portals"][0]
        assert "material_lighting" not in preview["state"]["viewer"]["elements"][-1]
    else:
        assert preview["state"]["architectural_portals"][0]["room_lower"] == receipt["clip"]["room_lower"]
        assert preview["state"]["architectural_portals"][0]["room_upper"] == receipt["clip"]["room_upper"]
        assert preview["state"]["viewer"]["elements"][-1]["material_lighting"] == "authored-lit"
    assert preview["state"]["semantics"]["extension"]["architecture_id"] == receipt["architecture_id"]
    assert scenes.read_artifact_json(job, preview, "_studio/splat-visibility.json")["architectural_portals"] == preview["state"]["architectural_portals"]
    active = scenes.activate(job, proposal["proposal_id"], 0)
    with pytest.raises(scenes.SceneRevisionError, match="close its captured opening together"):
        scenes.propose(job, {"kind": "remove", "selected_slug": "extension"}, "Remove just the room", active["generation"])
    restored = scenes.restore(job, pointer["revision_id"], active["generation"])
    assert scenes.read_revision(job, restored["revision_id"])["state"] == baseline["state"]
    assert scenes.read_revision(job, restored["revision_id"])["artifacts"] == baseline["artifacts"]


@pytest.mark.parametrize("change", [{"navigation_method": navigation.LEGACY_METHOD}, {"capsule_radius_m": .1}, {"route": []}, {"center_floor_y": []}])
def test_passing_label_with_resealed_inconsistent_navigation_cannot_install(architectural_scene, change):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Synthetic inconsistent navigation fixture")
    stage_result(job, receipt, navigation_changes=change)
    with pytest.raises(scenes.SceneRevisionError, match="navigation"):
        scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, "New room", 0)
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("defect", ["missing-method", "legacy-method", "unknown-method", "missing-interior-gate", "failed-interior-gate", "numeric-interior-gate"])
def test_resealed_room_result_cannot_downgrade_appearance_or_hide_an_obstacle(architectural_scene, defect):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Synthetic inconsistent room appearance fixture")
    result = stage_result(job, receipt)
    if defect == "missing-method":
        result.pop("appearance_method")
    elif defect == "legacy-method":
        result["appearance_method"] = volumes.LEGACY_METHOD
    elif defect == "unknown-method":
        result["appearance_method"] = "cut-and-authored-room/v2"
    elif defect == "missing-interior-gate":
        result["gates"].pop("authored_room_interior_clear")
    else:
        result["gates"]["authored_room_interior_clear"] = 1 if defect == "numeric-interior-gate" else False
    result["sha256"] = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in result.items() if key != "sha256"})).hexdigest()
    manifests.atomic_write_json(architecture.directory(job, receipt["architecture_id"]) / "result.json", result)
    with pytest.raises(scenes.SceneRevisionError, match="appearance envelope|geometry/navigation gates"):
        scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, "New room", 0)
    assert scenes.active(job) == pointer


def test_explicit_included_prop_is_removed_only_with_the_paired_architecture(architectural_scene):
    job, pointer = architectural_scene
    baseline = scenes.read_revision(job, pointer["revision_id"])
    receipt = architecture.prepare(job, spec(), 0, "Explicitly include the box region in this private doorway", ["box"])
    assert receipt["protected_elements"] == [] and receipt["replaced_elements"][0]["rows"] == 4
    stage_result(job, receipt)
    proposal = scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, receipt["instruction"], 0)
    preview = scenes.read_revision(job, proposal["preview_revision"])
    assert preview["state"]["hidden_capture_slugs"] == ["box"]
    assert preview["state"]["semantics"]["box"]["active"] is False
    assert [entry["slug"] for entry in preview["state"]["viewer"]["elements"]] == ["extension"]
    assert scenes.active(job) == pointer
    current = scenes.activate(job, proposal["proposal_id"], 0)
    restored = scenes.restore(job, pointer["revision_id"], current["generation"])
    assert scenes.read_revision(job, restored["revision_id"])["state"] == baseline["state"]


@pytest.mark.parametrize("replaced", [["missing"], ["box", "box"], [False], "box"])
def test_invalid_included_prop_selection_refuses(architectural_scene, replaced):
    job, pointer = architectural_scene
    with pytest.raises(evidence.EvidenceError):
        architecture.prepare(job, spec(), 0, "New room", replaced)
    assert scenes.active(job) == pointer


def test_overlapping_protected_rows_cannot_be_included(architectural_scene):
    job, pointer = architectural_scene
    pointer = replace_active(job, pointer, lambda revision: revision["state"]["selections"]["elements"]["other"].update(rows=[0]))
    with pytest.raises(evidence.EvidenceError, match="disjoint verified"):
        architecture.prepare(job, spec(), 0, "New room", ["box"])
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("defect", ["clip", "room-lower", "room-upper", "missing-room-bound", "appearance-method", "missing-appearance-method",
    "room-lighting", "missing-room-lighting", "collider", "viewer-collider", "room", "room-file", "source", "membership", "semantic", "hidden"])
def test_architectural_revision_cannot_bypass_paired_clearance(architectural_scene, defect):
    job, _ = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Explicit box-region replacement", ["box"])
    stage_result(job, receipt)
    proposal = scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, receipt["instruction"], 0)
    preview = scenes.read_revision(job, proposal["preview_revision"])
    if defect == "clip":
        preview["state"]["architectural_portals"][0]["upper"][0] += 1
    elif defect in ("room-lower", "room-upper"):
        preview["state"]["architectural_portals"][0][defect.replace("-", "_")][0] += .1
    elif defect == "missing-room-bound":
        preview["state"]["architectural_portals"][0].pop("room_lower")
    elif defect == "appearance-method":
        preview["state"]["architectural_portals"][0]["appearance_method"] = volumes.LEGACY_METHOD
    elif defect == "missing-appearance-method":
        preview["state"]["architectural_portals"][0].pop("appearance_method")
    elif defect == "room-lighting":
        preview["state"]["viewer"]["elements"][0]["material_lighting"] = "unlit"
    elif defect == "missing-room-lighting":
        preview["state"]["viewer"]["elements"][0].pop("material_lighting")
    elif defect == "collider":
        preview["artifacts"]["_world/collision_shell.glb"] = {"sha256": "0" * 64, "bytes": 1}
    elif defect == "viewer-collider":
        preview["state"]["viewer"]["collision_shell"]["glb"] = "artifact:_world/shell.glb"
    elif defect == "room":
        preview["state"]["viewer"]["elements"].clear()
    elif defect == "room-file":
        preview["state"]["viewer"]["elements"][0]["files"]["glb"] = "artifact:_world/collision_shell.glb"
    elif defect == "source":
        preview["artifacts"]["_preview/langweb.ply"] = {"sha256": "0" * 64, "bytes": 1}
    elif defect == "membership":
        preview["state"]["selections"]["elements"]["box"]["rows"].append(6)
    elif defect == "semantic":
        preview["state"]["semantics"]["box"]["active"] = True
    else:
        preview["state"]["hidden_capture_slugs"].append("other")
    with pytest.raises(scenes.SceneRevisionError):
        scenes.verify_recovery_dependencies(job, preview)


@pytest.mark.parametrize("overlaps", [True, False])
def test_later_mesh_placement_must_leave_the_portal_clear(architectural_scene, monkeypatch, overlaps):
    job, _ = architectural_scene
    value = spec() | {"origin": [0., 0., 0.], "yaw_degrees": 0.}
    receipt = architecture.prepare(job, value, 0, "New room")
    stage_result(job, receipt)
    proposal = scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, "New room", 0)
    pointer = scenes.activate(job, proposal["proposal_id"], 0)
    mesh = job / "new-prop.glb"
    lower, upper = ((-.2, .4, -.1), (.2, 1.4, .1)) if overlaps else ((8, 0, 8), (9, 1, 9))
    mesh.write_bytes(_authored_glb(aabb_min=lower, aabb_max=upper))
    identity = manifests.file_identity(mesh)
    monkeypatch.setattr(scenes, "proposal_asset", lambda *args: (mesh, identity["sha256"], None))
    operation = {"kind": "place", "slug": "new-prop"}
    if overlaps:
        with pytest.raises(scenes.SceneRevisionError, match="overlaps a retained doorway"):
            scenes.propose(job, operation, "New prop", pointer["generation"])
    else:
        placed = scenes.propose(job, operation, "New prop", pointer["generation"])
        scenes.verify_recovery_dependencies(job, scenes.read_revision(job, placed["preview_revision"]))
    assert scenes.active(job) == pointer


def test_later_native_gaussian_placement_refuses_before_loading_an_asset(architectural_scene, monkeypatch):
    job, pointer = architectural_scene
    revision = scenes.read_revision(job, pointer["revision_id"])
    revision["state"]["architectural_portals"] = [{"architecture_id": "existing"}]
    monkeypatch.setattr(scenes, "proposal_asset", lambda *args: pytest.fail("Native asset must not load before portal clearance"))
    with pytest.raises(scenes.SceneRevisionError, match="footprint/portal clearance"):
        scenes.place_asset(job, revision, {"kind": "place", "slug": "native", "gaussians_id": "existing"}, pointer)


@pytest.mark.parametrize("method", [None, architecture.FINISHED_GEOMETRY, architecture.JOINED_GEOMETRY])
def test_architectural_api_prepares_but_does_not_launch_or_activate(architectural_scene, method):
    job, pointer = architectural_scene
    app = FastAPI()
    app.include_router(scene_studio_route.router, prefix="/api/splat")
    client = TestClient(app)
    url = f"/api/splat/jobs/{job.name}/studio/architectural-edits"
    body = {"expected_generation": 0, "instruction": "Authored room", "spec": spec()}
    if method:
        body["geometry_method"] = method
    response = client.post(url, json=body)
    assert response.status_code == 200, response.text
    identifier = response.json()["architecture_id"]
    listed = client.get(url).json()["edits"]
    assert len(listed) == 1 and listed[0]["result"] is None and listed[0]["stale"] is False
    assert listed[0]["geometry_method"] == (method or architecture.LEGACY_GEOMETRY)
    assert client.get(url + f"/{identifier}/artifact", params={"name": "portal-frame.json"}).status_code == 200
    assert client.get(url + f"/{identifier}/artifact", params={"name": "../../meta.json"}).status_code == 409
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("method", [None, False, "raised-floor-finish/v2"])
def test_architectural_api_refuses_unsupported_floor_before_preparation(architectural_scene, method):
    job, pointer = architectural_scene
    app = FastAPI()
    app.include_router(scene_studio_route.router, prefix="/api/splat")
    client = TestClient(app)
    response = client.post(f"/api/splat/jobs/{job.name}/studio/architectural-edits", json={
        "expected_generation": 0, "instruction": "Unsupported floor", "spec": spec(), "geometry_method": method})
    assert response.status_code == 422
    assert scenes.active(job) == pointer
    assert not list((scenes.root(job) / "architectural-edits").glob("architecture_*"))
