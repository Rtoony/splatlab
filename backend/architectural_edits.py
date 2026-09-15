"""Revision-bound authored portal frames with paired visual/collision room extensions."""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import math
from pathlib import Path
import re
import uuid

import numpy as np

import artifact_manifest as manifests
import architectural_navigation as navigation
import architectural_solids as solids
import architectural_volumes as volumes
import background_recovery as recovery
from dcc.architecture import box_mesh, room_boxes
from mesh.provenance import GENERATIVE_TAG, GLTF_EXTRAS_KEY
import glb_check
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as selections

IDENTIFIER = re.compile(r"architecture_[a-f0-9]{24}\Z")
METHOD = "authored-portal-and-connected-room/v1"
LEGACY_GEOMETRY = "nominal-floor/v1"
FINISHED_GEOMETRY = "raised-floor-finish/v1"
JOINED_GEOMETRY = "joined-floor-envelope/v1"
GATES = {"base_watertight", "cut_watertight", "positive_bounded_removal", "portal_volume_clear", "outside_surface_preserved",
         "approach_floor_continuous", "room_floor_continuous", "capsule_route_clear", "protected_elements_unchanged", "replacement_region_contained"}
FIELDS = {"origin", "yaw_degrees", "opening_width", "opening_height", "cut_depth", "room_width", "room_depth", "room_height", "wall_thickness"}


def required_gates(receipt):
    return (GATES | ({"authored_room_interior_clear"} if volumes.method(receipt["clip"]) == volumes.ROOM_METHOD else set())
            | ({"authored_room_watertight"} if geometry_method(receipt) == JOINED_GEOMETRY else set()))


def geometry_metadata(identifier, coordinate_frame, scope):
    return {GLTF_EXTRAS_KEY: GENERATIVE_TAG, "splatlab_architecture": {
        "method": METHOD, "architecture_id": identifier, "frame": coordinate_frame, "scope": scope}}


def frame(spec):
    if not isinstance(spec, dict) or set(spec) != FIELDS:
        raise evidence.EvidenceError("Architectural frame requires the exact dimensioned portal/room fields")
    raw_origin = spec["origin"]
    if not isinstance(raw_origin, list) or len(raw_origin) != 3 or any(type(value) not in (int, float) or not np.isfinite(value) or abs(value) > 100 for value in raw_origin):
        raise evidence.EvidenceError("Architectural threshold origin must be a finite world-metre vector within 100 m")
    origin = np.asarray(raw_origin, dtype=float)
    limits = {"yaw_degrees": (-180, 180), "opening_width": (.8, 3), "opening_height": (1.9, 3.5), "cut_depth": (.1, 6),
              "room_width": (1.5, 8), "room_depth": (1.5, 8), "room_height": (2.1, 5), "wall_thickness": (.08, .4)}
    for key, (lower, upper) in limits.items():
        value = spec[key]
        if type(value) not in (int, float) or not np.isfinite(value) or not lower <= value <= upper:
            raise evidence.EvidenceError(f"Architectural {key} must be between {lower} and {upper}")
    if spec["opening_width"] + .3 > spec["room_width"] or spec["opening_height"] + .15 > spec["room_height"]:
        raise evidence.EvidenceError("The extension must retain two jambs and a lintel around the opening")
    angle = math.radians(spec["yaw_degrees"])
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, 0., sine], [0., 1., 0.], [-sine, 0., cosine]])
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = [-(cosine * raw_origin[0] - sine * raw_origin[2]), -raw_origin[1],
                         -(sine * raw_origin[0] + cosine * raw_origin[2])]
    lower = np.array([-spec["opening_width"] / 2, .015, -spec["cut_depth"] / 2])
    upper = np.array([spec["opening_width"] / 2, spec["opening_height"], spec["cut_depth"] / 2])
    return {"origin": origin.astype(float), "rotation": rotation, "world_to_portal": transform, "lower": lower, "upper": upper}


def geometry_method(document):
    selected = document.get("geometry_method", LEGACY_GEOMETRY)
    if not isinstance(selected, str) or selected not in {LEGACY_GEOMETRY, FINISHED_GEOMETRY, JOINED_GEOMETRY}:
        raise evidence.EvidenceError("Unknown architectural geometry method")
    return selected


def floor_finish_m(spec, selected):
    geometry_method({"geometry_method": selected})
    coordinate = frame(spec)
    return float(coordinate["lower"][1]) + .001 if selected in {FINISHED_GEOMETRY, JOINED_GEOMETRY} else 0.


def authored_boxes(spec, selected=LEGACY_GEOMETRY):
    finish = floor_finish_m(spec, selected)
    boxes = room_boxes(spec["room_width"], spec["room_depth"], spec["room_height"], spec["wall_thickness"],
                       spec["opening_width"], spec["opening_height"], ceiling=True)
    offset = spec["cut_depth"] / 2
    boxes = [((center[0], center[1] + offset, center[2]), size) for center, size in boxes]
    boxes.append(((0, -.15, -spec["wall_thickness"] / 2),
                  (spec["opening_width"] + .1, spec["cut_depth"] + .3, spec["wall_thickness"])))
    for side in (-1, 1):
        boxes.append(((side * (spec["opening_width"] + spec["wall_thickness"]) / 2, 0, spec["opening_height"] / 2),
                      (spec["wall_thickness"], spec["cut_depth"] + spec["wall_thickness"], spec["opening_height"])))
    roof_width = spec["opening_width"] + (2 * spec["wall_thickness"] if selected == JOINED_GEOMETRY else 0)
    boxes.append(((0, 0, spec["opening_height"] + spec["wall_thickness"] / 2),
                  (roof_width, spec["cut_depth"] + spec["wall_thickness"], spec["wall_thickness"])))
    if finish:
        floors = [index for index, (center, size) in enumerate(boxes)
                  if center[2] == -spec["wall_thickness"] / 2 and size[2] == spec["wall_thickness"]]
        if len(floors) != 2:
            raise evidence.EvidenceError("Floor finish requires exactly the authored room and bridge slabs")
        for index in floors:
            center, size = boxes[index]
            boxes[index] = ((center[0], center[1], center[2] + finish), size)
    return boxes


def room_geometry(spec, selected=LEGACY_GEOMETRY):
    coordinate = frame(spec)
    boxes = authored_boxes(spec, selected)
    vertices, quads = solids.joined_box_mesh(boxes) if selected == JOINED_GEOMETRY else box_mesh(boxes)
    cosine, sine = map(float, coordinate["rotation"][0, [0, 2]])
    origin_x, origin_y, origin_z = spec["origin"]
    world = np.array([[cosine * vertex[0] + sine * vertex[1] + origin_x,
                       vertex[2] + origin_y,
                       -sine * vertex[0] + cosine * vertex[1] + origin_z] for vertex in vertices])
    triangles = np.array([[quad[0], quad[2], quad[1]] for quad in quads] + [[quad[0], quad[3], quad[2]] for quad in quads])
    return world, triangles


def local_coordinates(points, spec):
    coordinate = frame(spec)
    return (np.asarray(points) - coordinate["origin"]) @ coordinate["rotation"]


def protected_volume_bounds(spec, selected=LEGACY_GEOMETRY):
    frame(spec)
    finish = floor_finish_m(spec, selected)
    half_depth = spec["cut_depth"] / 2
    thickness = spec["wall_thickness"]
    regions = [
        (np.array([-spec["opening_width"] / 2 - thickness, .005, -half_depth - thickness / 2]),
         np.array([spec["opening_width"] / 2 + thickness, spec["opening_height"] + thickness, half_depth + thickness / 2])),
        (np.array([-spec["room_width"] / 2 - thickness, .005, half_depth - thickness / 2]),
         np.array([spec["room_width"] / 2 + thickness, spec["room_height"] + thickness, half_depth + spec["room_depth"] + thickness / 2])),
    ]
    if finish:
        regions.extend([
            (np.array([-(spec["opening_width"] + .1) / 2, finish - thickness, -half_depth - .3]),
             np.array([(spec["opening_width"] + .1) / 2, finish, half_depth])),
            (np.array([-spec["room_width"] / 2 - thickness, finish - thickness, half_depth - thickness / 2]),
             np.array([spec["room_width"] / 2 + thickness, finish, half_depth + spec["room_depth"] + thickness / 2])),
        ])
    return regions


def directory(job, identifier):
    if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid architectural edit identifier")
    output = scenes.root(job) / "architectural-edits" / identifier
    if output.resolve() != output.absolute():
        raise evidence.EvidenceError("Architectural edit directory cannot be symlinked")
    return output


def read(job, identifier, result=False):
    value = manifests.read_json(directory(job, identifier) / ("result.json" if result else "receipt.json"))
    digest = hashlib.sha256(scenes.canonical_bytes({key: item for key, item in value.items() if key != "sha256"})).hexdigest() if value else None
    if not value or value.get("sha256") != digest:
        raise evidence.EvidenceError("Architectural receipt is missing or corrupt")
    if result and value.get("prepared_sha256") != read(job, identifier)["sha256"]:
        raise evidence.EvidenceError("Architectural result belongs to different preparation")
    return value


def artifact(job, identifier, name, result=False):
    record = read(job, identifier, result)["artifacts"].get(name)
    if not record:
        raise evidence.EvidenceError("Artifact does not belong to this architectural edit")
    path = scenes.blob_path(job, record["sha256"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or manifests.sha256_file(path) != record["sha256"]:
        raise evidence.EvidenceError("Architectural artifact is missing or corrupt")
    return path


@contextmanager
def worker_slot(job, identifier):
    with (directory(job, identifier) / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise evidence.EvidenceError("Architectural worker already active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify(job, receipt):
    coordinate = frame(receipt["spec"])
    navigation.method(receipt.get("recipe"))
    expected_clip = volumes.contract(receipt["spec"], coordinate, volumes.method(receipt["clip"]))
    if receipt["clip"] != expected_clip:
        raise evidence.EvidenceError("Architectural clipping does not match its dimensioned collision frame")
    if scenes.active(job) != receipt["base"]:
        raise evidence.EvidenceError("Architectural edit belongs to an older active scene")
    revision = scenes.read_revision(job, receipt["base"]["revision_id"])
    selection_digest = hashlib.sha256(scenes.canonical_bytes(revision["state"].get("selections"))).hexdigest()
    if receipt.get("selections_sha256") != selection_digest:
        raise evidence.EvidenceError("Architectural preparation does not bind the current captured membership; prepare again")
    if revision["source_fingerprint"] != scenes.source_fingerprint(job):
        raise evidence.EvidenceError("Capture or calibration changed before architectural editing")
    for key, record in receipt["base_artifacts"].items():
        if revision["artifacts"].get(key) != record:
            raise evidence.EvidenceError("Architectural baseline artifact changed")
        scenes.artifact(job, revision["revision_id"], key)
    recovery.verify_receipt_sources(job, receipt, checksums=True)
    for name in receipt["artifacts"]:
        artifact(job, receipt["architecture_id"], name)
    expected_vertices, expected_faces = room_geometry(receipt["spec"], geometry_method(receipt))
    with np.load(artifact(job, receipt["architecture_id"], "authored-room.npz"), allow_pickle=False) as room:
        if set(room.files) != {"vertices", "faces"} or not np.array_equal(room["vertices"], expected_vertices) or not np.array_equal(room["faces"], expected_faces):
            raise evidence.EvidenceError("Authored room master differs from the dimensioned portal frame")


def prepare(job, spec, expected_generation, instruction, replaced_slugs=None, *, selected_geometry=LEGACY_GEOMETRY):
    coordinate = frame(spec)
    geometry_method({"geometry_method": selected_geometry})
    replaced_slugs = [] if replaced_slugs is None else replaced_slugs
    if not isinstance(replaced_slugs, list) or len(replaced_slugs) > 3 or any(not isinstance(slug, str) for slug in replaced_slugs) or len(set(replaced_slugs)) != len(replaced_slugs):
        raise evidence.EvidenceError("Choose at most three explicitly included captured props")
    if not isinstance(instruction, str) or not 1 <= len(instruction.strip()) <= 2000:
        raise evidence.EvidenceError("Describe the intended architectural change")
    with recovery.worker_slot(wait_seconds=3), scenes.write_lock(job):
        pointer = scenes.active(job)
        if not pointer or pointer["generation"] != expected_generation:
            raise evidence.EvidenceError("Active scene changed before architectural preparation")
        revision = scenes.read_revision(job, pointer["revision_id"])
        state = revision["state"]
        if state["viewer"].get("units") != "meters" or state["viewer"].get("calibration", {}).get("stale"):
            raise evidence.EvidenceError("Architecture requires a currently calibrated metre scene")
        if revision["source_fingerprint"] != scenes.source_fingerprint(job):
            raise evidence.EvidenceError("Refresh the changed captured baseline")
        if len(state.get("architectural_portals", [])) >= 4:
            raise evidence.EvidenceError("This bounded preview supports at most four active portal volumes")
        if state.get("capture_partition"):
            raise evidence.EvidenceError("A refined captured-instance partition needs explicit portal integration before structural cutting")
        if any(entry.get("gaussian_appearance") for entry in state["viewer"].get("elements", [])):
            raise evidence.EvidenceError("Existing native Gaussian layers need verified footprint/portal clearance before structural cutting")
        if list((job / "_scene/surfaces").glob("patch_*.ply")) or list((job / "_scene/ground").glob("*.npz")):
            raise evidence.EvidenceError("Measured or painted surface authority requires an explicit structural adapter")
        collider = state["viewer"].get("collision_shell", {})
        if collider.get("glb") != "artifact:_world/collision_shell.glb":
            raise evidence.EvidenceError("Architectural baseline must address its pinned captured collider")
        scale = collider.get("scale_to_world")
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
            raise evidence.EvidenceError("A frame-verified captured collider is required")
        report = scenes.read_artifact_json(job, revision, "_world/collision_shell.json")
        geometry_frame = report.get("geometry_frame", {})
        if report.get("verdict") not in {"PASS", "PASS_LOCAL_EDIT", "PASS_ARCHITECTURAL_EDIT"} or geometry_frame.get("axis") != "y-up" or geometry_frame.get("units") != "scene-units":
            raise evidence.EvidenceError("Architecture requires a graded Y-up captured collider")
        source = evidence.load(job)
        if not np.isclose(scale, source.calibration["meters_per_unit"], rtol=0, atol=1e-9):
            raise evidence.EvidenceError("Collider scale does not match the captured calibration")
        if "_preview/langweb.ply" not in revision["artifacts"]:
            raise evidence.EvidenceError("Architecture requires the captured Gaussian backdrop")
        protected = []
        replaced = []
        for entry in state["viewer"].get("elements", []):
            if entry["slug"] == "shell":
                continue
            key = "_world/elements/" + entry["slug"] + ".glb"
            if key not in revision["artifacts"]:
                raise evidence.EvidenceError("A protected scene element lacks its pinned geometry")
            record = {"slug": entry["slug"], "key": key, "role": entry.get("role"), "provenance": entry.get("provenance"), "scale": 1}
            if entry["slug"] in replaced_slugs:
                if entry.get("role") != "prop" or entry.get("provenance") in {"authored", "generated"}:
                    raise evidence.EvidenceError("Included architectural removals must be explicitly selected captured props")
                replaced.append(record)
            else:
                protected.append(record)
        if {item["slug"] for item in replaced} != set(replaced_slugs):
            raise evidence.EvidenceError("Included captured prop is not present in this scene")
        replacement_arrays = {}
        if replaced:
            document = state.get("selections") or {}
            positions = selections.read_ply_xyz(scenes.blob_path(job, revision["artifacts"]["_preview/langweb.ply"]["sha256"]))
            if len(positions) != document.get("n_rows") or not np.isfinite(positions).all():
                raise evidence.EvidenceError("Included props do not match the served captured row space")
            protected_rows = np.array([row for name, value in document.get("elements", {}).items() if name not in replaced_slugs for row in value.get("rows", [])], dtype=np.int64)
            for item in replaced:
                selected = document.get("elements", {}).get(item["slug"], {})
                rows = np.asarray(selected.get("rows", []))
                if (selected.get("coordinate_verified") is not True or rows.ndim != 1 or rows.dtype.kind not in "iu" or not 0 < len(rows) <= 200000
                        or np.any(rows < 0) or np.any(rows >= len(positions)) or len(np.unique(rows)) != len(rows) or np.isin(rows, protected_rows).any()):
                    raise evidence.EvidenceError("Included prop requires disjoint verified captured rows")
                replacement_arrays[item["slug"] + "-rows"] = rows
                replacement_arrays[item["slug"] + "-positions-world"] = positions[rows] @ evidence.Y_UP.T * scale
                item["rows"] = len(rows)
        identifier = "architecture_" + uuid.uuid4().hex[:24]
        output = directory(job, identifier)
        output.mkdir(parents=True)
        vertices, faces = room_geometry(spec, selected_geometry)
        np.savez_compressed(output / "authored-room.npz", vertices=vertices, faces=faces)
        clip = volumes.contract(spec, coordinate, volumes.default_method())
        manifests.atomic_write_json(output / "portal-frame.json", clip)
        np.savez_compressed(output / "replacement-membership.npz", **replacement_arrays)
        base_keys = {"_world/collision_shell.glb", "_world/collision_shell.json", "_preview/langweb.ply"} | {item["key"] for item in [*protected, *replaced]}
        receipt = selections.seal(job, output, "receipt.json", {"architecture_id": identifier, "method": METHOD,
            "created_at": manifests.utc_now(), "base": pointer, "spec": spec, "clip": clip, "instruction": instruction.strip(),
            "base_artifacts": {key: revision["artifacts"][key] for key in sorted(base_keys)}, "base_collision_report": report,
            "selections_sha256": hashlib.sha256(scenes.canonical_bytes(state.get("selections"))).hexdigest(),
            "protected_elements": protected, "replaced_elements": replaced, "sources": source.sources, "calibration": source.calibration,
            "recipe": navigation.recipe(), "geometry_method": selected_geometry,
            "scope": "Authored/inferred architectural frame, not measured structure; review required before paired appearance/collision activation",
            "code_sha256": manifests.sha256_file(Path(__file__))}, [output / "authored-room.npz", output / "portal-frame.json", output / "replacement-membership.npz"])
        verify(job, receipt)
        return receipt


def install(job, revision, identifier, pointer, slug, label):
    receipt, result = read(job, identifier), read(job, identifier, True)
    verify(job, receipt)
    if pointer != receipt["base"] or revision["artifacts"].get("_world/collision_shell.glb") != receipt["base_artifacts"]["_world/collision_shell.glb"]:
        raise evidence.EvidenceError("Architectural edit has a different collider baseline")
    if not isinstance(slug, str) or not scenes.SLUG_RE.fullmatch(slug) or slug == "shell" or slug in revision["state"]["semantics"]:
        raise evidence.EvidenceError("Choose a new lowercase architectural element name")
    if result.get("method") != METHOD or result.get("verdict") != "PASS_ARCHITECTURAL_EDIT" or set(result.get("gates", {})) != required_gates(receipt) or any(value is not True for value in result["gates"].values()):
        raise evidence.EvidenceError("Architectural candidate has not passed its local geometry/navigation gates")
    if result.get("appearance_method", volumes.LEGACY_METHOD) != volumes.method(receipt["clip"]):
        raise evidence.EvidenceError("Architectural result uses a different appearance envelope")
    if result.get("navigation_method", navigation.LEGACY_METHOD) != navigation.method(receipt["recipe"]):
        raise evidence.EvidenceError("Architectural result uses a different navigation method")
    if geometry_method(result) != geometry_method(receipt):
        raise evidence.EvidenceError("Architectural result uses a different geometry method")
    for name in result["artifacts"]:
        artifact(job, identifier, name, True)
    navigation.verify_document(receipt["spec"], receipt["recipe"], frame(receipt["spec"]),
        manifests.read_json(artifact(job, identifier, "navmesh.json", True)), identifier, result["gates"])
    room = artifact(job, identifier, "room.glb", True)
    glb_check.validate_glb(room)
    bounds = glb_check.position_bounds(room)
    if not bounds["identity_transforms"]:
        raise evidence.EvidenceError("Authored architecture must bake its world-metre frame")
    state = revision["state"]
    replaced_slugs = {item["slug"] for item in receipt["replaced_elements"]}
    state["viewer"]["elements"] = [entry for entry in state["viewer"]["elements"] if entry["slug"] not in replaced_slugs]
    state["hidden_capture_slugs"] = sorted(set(state["hidden_capture_slugs"]) | replaced_slugs)
    for replaced_slug in replaced_slugs:
        state["semantics"][replaced_slug].update(active=False, replaced_by_architecture=identifier)
    key = f"_world/elements/{slug}.glb"
    revision["artifacts"][key] = result["artifacts"]["room.glb"]
    for name in ("collision_shell.glb", "collision_shell.json", "navmesh.json"):
        revision["artifacts"]["_world/" + name] = result["artifacts"][name]
    for stage, value in (("receipt", receipt), ("result", result)):
        revision["artifacts"][f"_studio/architectural-edits/{identifier}/{stage}.json"] = scenes.store_json(job, value)
        for name, record in value["artifacts"].items():
            revision["artifacts"][f"_studio/architectural-edits/{identifier}/{name}"] = record
    state["viewer"]["collision_shell"] = {**state["viewer"]["collision_shell"], "gates": result["gates"],
        "glb": "artifact:_world/collision_shell.glb", "report": "artifact:_world/collision_shell.json"}
    state.setdefault("architectural_portals", []).append({"architecture_id": identifier, "slug": slug, **deepcopy(receipt["clip"]),
                                                         "replaced_capture_slugs": sorted(replaced_slugs), "provenance": "authored-inferred"})
    entry = {"slug": slug, "label": label or slug, "role": "environment", "provenance": "authored", "geometry_source": "dimensioned-connected-room",
             "architecture_id": identifier, "files": {"glb": "artifact:" + key}, "extent": bounds["extent"],
             "collision": {"ok": True, "strategy": "complex_as_simple", "hulls": 0},
             "classification": ["Authored/inferred portal frame; room and captured collider changed together; not measured structure"]}
    if volumes.method(receipt["clip"]) == volumes.ROOM_METHOD:
        entry["material_lighting"] = "authored-lit"
    if geometry_method(receipt) == FINISHED_GEOMETRY:
        entry["geometry_method"] = FINISHED_GEOMETRY
        entry["floor_finish_m"] = floor_finish_m(receipt["spec"], FINISHED_GEOMETRY)
        entry["classification"].append("Experimental 16 mm authored floor finish; clear opening and room heights are 16 mm below nominal dimensions")
    state["viewer"]["elements"].append(entry)
    state["semantics"][slug] = {"label": entry["label"], "provenance": "authored", "active": True, "architecture_id": identifier}
    state.setdefault("recovery_dependencies", {})[slug] = {key: receipt[key] for key in ("sources", "calibration")}
    return result["artifacts"]["room.glb"]


def verify_revision(job, revision):
    portals = revision["state"].get("architectural_portals", [])
    if not isinstance(portals, list) or len(portals) > 4:
        raise evidence.EvidenceError("Invalid revision architectural portal set")
    covered, identifiers = set(), set()
    current_selection = hashlib.sha256(scenes.canonical_bytes(revision["state"].get("selections"))).hexdigest()
    last_collision = None
    for portal in portals:
        if not isinstance(portal, dict):
            raise evidence.EvidenceError("Invalid revision architectural portal")
        identifier = portal.get("architecture_id")
        directory(job, identifier)
        if identifier in identifiers:
            raise evidence.EvidenceError("Duplicate architectural portal in revision")
        identifiers.add(identifier)
        root = f"_studio/architectural-edits/{identifier}/"
        receipt = scenes.read_artifact_json(job, revision, root + "receipt.json")
        result = scenes.read_artifact_json(job, revision, root + "result.json")
        for document in (receipt, result):
            expected = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in document.items() if key != "sha256"})).hexdigest()
            if document.get("sha256") != expected or document.get("method") != METHOD or document.get("architecture_id") != identifier:
                raise evidence.EvidenceError("Revision architectural evidence is corrupt")
        coordinate = frame(receipt["spec"])
        selected_navigation = navigation.method(receipt.get("recipe"))
        if result.get("navigation_method", navigation.LEGACY_METHOD) != selected_navigation:
            raise evidence.EvidenceError("Revision navigation method differs from its prepared evidence")
        selected_geometry = geometry_method(receipt)
        if geometry_method(result) != selected_geometry:
            raise evidence.EvidenceError("Revision geometry method differs from its prepared evidence")
        expected_clip = volumes.contract(receipt["spec"], coordinate, volumes.method(receipt["clip"]))
        if receipt["clip"] != expected_clip:
            raise evidence.EvidenceError("Architectural clipping differs from its dimensioned collision frame")
        if (result.get("prepared_sha256") != receipt["sha256"] or result.get("verdict") != "PASS_ARCHITECTURAL_EDIT"
                or set(result.get("gates", {})) != required_gates(receipt) or any(value is not True for value in result["gates"].values())
                or receipt.get("selections_sha256") != current_selection):
            raise evidence.EvidenceError("Revision architectural evidence is stale or failed its gates")
        if result.get("appearance_method", volumes.LEGACY_METHOD) != volumes.method(receipt["clip"]):
            raise evidence.EvidenceError("Revision appearance method differs from its prepared evidence")
        navigation.verify_document(receipt["spec"], receipt["recipe"], coordinate,
            scenes.read_artifact_json(job, revision, root + "navmesh.json"), identifier, result["gates"])
        if last_collision and receipt["base_artifacts"]["_world/collision_shell.glb"] != last_collision:
            raise evidence.EvidenceError("Architectural cuts do not form a consistent collider chain")
        last_collision = result["artifacts"]["collision_shell.glb"]
        if receipt["base_artifacts"]["_preview/langweb.ply"] != revision["artifacts"].get("_preview/langweb.ply"):
            raise evidence.EvidenceError("Architectural visibility addresses a different captured backdrop")
        for key in ("world_to_portal", "lower", "upper", "appearance_method", "room_lower", "room_upper"):
            if portal.get(key) != receipt["clip"].get(key):
                raise evidence.EvidenceError("Architectural appearance clipping differs from the retained collision frame")
        if portal.get("provenance") != "authored-inferred":
            raise evidence.EvidenceError("Authored architectural frames cannot be promoted to measured provenance")
        slug = portal.get("slug")
        entry = next((value for value in revision["state"]["viewer"]["elements"] if value["slug"] == slug), {})
        semantic = revision["state"]["semantics"].get(slug, {})
        if geometry_method(entry) != selected_geometry or entry.get("floor_finish_m", 0.) != floor_finish_m(receipt["spec"], selected_geometry):
            raise evidence.EvidenceError("Revision floor finish differs from its paired geometry recipe")
        if volumes.method(receipt["clip"]) == volumes.ROOM_METHOD and entry.get("material_lighting") != "authored-lit":
            raise evidence.EvidenceError("Authored-room lighting differs from its paired appearance recipe")
        if (entry.get("architecture_id") != identifier or entry.get("provenance") != "authored" or entry.get("role") != "environment"
                or entry.get("files", {}).get("glb") != f"artifact:_world/elements/{slug}.glb"
                or semantic.get("architecture_id") != identifier or semantic.get("active") is not True
                or revision["artifacts"].get(f"_world/elements/{slug}.glb") != result["artifacts"]["room.glb"]):
            raise evidence.EvidenceError("Architectural room, appearance and semantics must remain paired")
        replaced = {item["slug"] for item in receipt["replaced_elements"]}
        if set(portal.get("replaced_capture_slugs", [])) != replaced or covered & replaced:
            raise evidence.EvidenceError("Included captured removals do not match the architectural receipt")
        for removed in replaced:
            state = revision["state"]
            if (removed not in state["hidden_capture_slugs"] or state["semantics"].get(removed, {}).get("active") is not False
                    or state["semantics"][removed].get("replaced_by_architecture") != identifier
                    or any(value["slug"] == removed for value in state["viewer"]["elements"])):
                raise evidence.EvidenceError("Included captured prop removal is not atomic with its architecture")
        covered |= replaced
    if last_collision and revision["artifacts"].get("_world/collision_shell.glb") != last_collision:
        raise evidence.EvidenceError("Architectural appearance does not match the current captured collider")
    if portals and revision["state"]["viewer"].get("collision_shell", {}).get("glb") != "artifact:_world/collision_shell.glb":
        raise evidence.EvidenceError("Architectural viewer does not address the paired captured collider")
    return covered
