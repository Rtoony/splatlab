"""Explicit generated material on unknown support cells, retaining observed texture intact."""

from copy import deepcopy
from io import BytesIO
import hashlib
from pathlib import Path
import re
import struct
import uuid
from zipfile import ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

import artifact_manifest as manifests
import background_recovery as recovery
import glb_check
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as reviews
from mesh.provenance import GENERATIVE_TAG, GLTF_EXTRAS_KEY

COMPLETION_RE = re.compile(r"completion_[a-f0-9]{24}\Z")
MAX_IMAGE_BYTES = 16 * 1024 ** 2


def directory(job, identifier):
    if not COMPLETION_RE.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid background completion identifier")
    output = scenes.root(job) / "completions" / identifier
    if output.resolve() != output.absolute():
        raise evidence.EvidenceError("Completion paths cannot be symlinked")
    return output


def read(job, identifier, result=False):
    record = manifests.read_json(directory(job, identifier) / ("result.json" if result else "receipt.json"))
    if not record:
        raise evidence.EvidenceError("Background completion evidence is missing")
    digest = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in record.items() if key != "sha256"})).hexdigest()
    if record.get("sha256") != digest:
        raise evidence.EvidenceError("Background completion integrity check failed")
    if result and record.get("prepared_receipt_sha256") != read(job, identifier)["sha256"]:
        raise evidence.EvidenceError("Completion result belongs to different inputs")
    return record


def artifact(job, identifier, name, result=False):
    record = read(job, identifier, result)["artifacts"].get(name)
    if not record:
        raise evidence.EvidenceError("Artifact does not belong to this completion")
    path = scenes.blob_path(job, record["sha256"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or manifests.sha256_file(path) != record["sha256"]:
        raise evidence.EvidenceError("Completion artifact is missing or corrupt")
    return path


def recovery_input(job, receipt, name):
    path = recovery.artifact(job, receipt["recovery_id"], name)
    if manifests.sha256_file(path) != receipt["artifacts"][name]["sha256"]:
        raise evidence.EvidenceError("Observed recovery artifact is corrupt")
    return path


def observed_cells(job, receipt):
    size = receipt["report"]["texture_size"]
    if type(size) is not int or size not in {64, 128, 256}:
        raise evidence.EvidenceError("Completion requires a bounded recovery grid")
    path = recovery_input(job, receipt, "observations.npz")
    with ZipFile(path) as archive:
        if archive.getinfo("accepted.npy").file_size > 1024 ** 2:
            raise evidence.EvidenceError("Observation mask exceeds its decompression budget")
        with archive.open("accepted.npy") as handle:
            accepted = np.load(BytesIO(handle.read()), allow_pickle=False)
    if accepted.shape != (size, size) or accepted.dtype.kind != "b" or not 0 < accepted.sum() < accepted.size:
        raise evidence.EvidenceError("Completion needs a valid mix of observed and unknown cells")
    if abs(float(accepted.mean()) - receipt["report"]["supported_fraction"]) > 1e-10:
        raise evidence.EvidenceError("Recovery coverage and its observation mask disagree")
    with Image.open(recovery_input(job, receipt, "atlas.png")) as opened:
        if opened.size != (size, size) or opened.mode != "RGBA":
            raise evidence.EvidenceError("Observed atlas does not match its recovery grid")
        alpha = np.asarray(opened.getchannel("A"))
        if not np.array_equal(alpha, accepted.astype(np.uint8) * 255):
            raise evidence.EvidenceError("Observed texture alpha disagrees with its evidence mask")
    return accepted


def plane_arrays(plane):
    values = {name: np.asarray(plane[name], dtype=np.float64) for name in ("center", "normal", "right", "forward", "lower_uv", "upper_uv")}
    if any(value.shape != ((2,) if name.endswith("_uv") else (3,)) or not np.isfinite(value).all() for name, value in values.items()):
        raise evidence.EvidenceError("Invalid support-plane coordinates")
    axes = np.stack([values[name] for name in ("right", "normal", "forward")])
    span = values["upper_uv"] - values["lower_uv"]
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5) or values["normal"][1] < .94 or np.any(span <= 0) or np.any(span > 5):
        raise evidence.EvidenceError("Completion requires an orthonormal, bounded horizontal support plane")
    return values


def verify(job, receipt):
    if scenes.active(job) != receipt["base"]:
        raise evidence.EvidenceError("Completion is stale; the active scene changed")
    original = recovery.read(job, receipt["recovery_id"])
    if original["sha256"] != receipt["recovery_sha256"] or original["base"] != receipt["base"]:
        raise evidence.EvidenceError("Completion recovery identity changed")
    revision = scenes.read_revision(job, receipt["base"]["revision_id"])
    if revision["source_fingerprint"] != scenes.source_fingerprint(job):
        raise evidence.EvidenceError("Captured sources changed before completion")
    recovery.verify_receipt_sources(job, receipt, checksums=True)


def prepare(job, recovery_id, expected_generation, prompt):
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
        raise evidence.EvidenceError("Describe the intended generated material in 1–2000 characters")
    with recovery.worker_slot(wait_seconds=3), scenes.write_lock(job):
        original = recovery.read(job, recovery_id)
        pointer = scenes.active(job)
        if not pointer or pointer["generation"] != expected_generation or original["base"] != pointer:
            raise evidence.EvidenceError("Prepare completion against a current recovery and scene")
        plane = original["report"]["plane"]
        if original.get("render_vr_only") is not True or plane.get("selection", {}).get("method") != "photo-linked-sfm-anchors":
            raise evidence.EvidenceError("Choose photo-linked support anchors before proposing unseen geometry")
        plane_arrays(plane)
        accepted = observed_cells(job, original)
        identifier = "completion_" + uuid.uuid4().hex[:24]
        payload = {"schema": "dev.splatlab.background-completion/v1", "completion_id": identifier,
            "recovery_id": recovery_id, "recovery_sha256": original["sha256"], "base": pointer,
            "selected_slug": original["selected_slug"], "sources": original["sources"], "calibration": original["calibration"],
            "plane": plane, "grid_size": len(accepted), "observed_cells": int(accepted.sum()), "generated_cells": int((~accepted).sum()),
            "observed_fraction": float(accepted.mean()), "generated_fraction": float((~accepted).mean()),
            "prompt": prompt.strip(), "created_at": manifests.utc_now(), "render_vr_only": True,
            "geometry_source": "photo-anchored support plane extended into unknown cells",
            "generator_contract": "Externally supplied UV-aligned opaque square image; no automatic model launch or remote upload",
            "preservation_contract": "Observed atlas bytes and observed-cell geometry are retained; generated texture is used only on disjoint unknown cells",
            "scope": "creative candidate; generated appearance and unseen geometry are not captured evidence or navigation acceptance"}
        verify(job, payload)
        output = directory(job, identifier)
        output.mkdir(parents=True)
        for name in ("atlas.png", "support.png", "observations.npz"):
            (output / name).write_bytes(recovery_input(job, original, name).read_bytes())
        return reviews.seal(job, output, "receipt.json", payload, [output / name for name in ("atlas.png", "support.png", "observations.npz")])


def image_description(content):
    if not 0 < len(content) <= MAX_IMAGE_BYTES:
        raise evidence.EvidenceError("Generated material exceeds the 16 MiB upload budget")
    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.format not in {"PNG", "JPEG"} or opened.width != opened.height or not 64 <= opened.width <= 2048 or getattr(opened, "n_frames", 1) != 1:
                raise evidence.EvidenceError("Use one opaque square PNG/JPEG image between 64 and 2048 pixels")
            opened.load()
            if opened.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise evidence.EvidenceError("Generated material must be opaque; transparent pixels cannot certify a filled cell")
            return {"width": opened.width, "height": opened.height, "format": opened.format,
                    "mime_type": Image.MIME[opened.format], "extension": {"PNG": "png", "JPEG": "jpg"}[opened.format]}
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise evidence.EvidenceError("Generated material is not a decodable bounded image") from exc


def material_diagnostics(accepted, observed_png, generated_image):
    with Image.open(BytesIO(observed_png)) as opened:
        observed = np.asarray(opened.convert("RGB"), dtype=np.float64)
    with Image.open(BytesIO(generated_image)) as opened:
        generated = np.asarray(opened.convert("RGB"), dtype=np.float64)
    size = len(accepted)
    indices = np.floor((np.arange(size) + .5) * len(generated) / size).astype(int)
    sampled = generated[indices[:, None], indices[None, :]]
    anchored_error = np.abs(sampled - observed).mean(axis=2)
    mixed_jumps, generated_jumps, anchor_errors = [], [], []
    for first, second in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        boundary = accepted[first] != accepted[second]
        first_observed = accepted[first][boundary, None]
        known = np.where(first_observed, observed[first][boundary], observed[second][boundary])
        unknown = np.where(first_observed, sampled[second][boundary], sampled[first][boundary])
        predicted_known = np.where(first_observed, sampled[first][boundary], sampled[second][boundary])
        mixed_jumps.extend(np.abs(known - unknown).mean(axis=1).tolist())
        generated_jumps.extend(np.abs(predicted_known - unknown).mean(axis=1).tolist())
        anchor_errors.extend(np.abs(known - predicted_known).mean(axis=1).tolist())
    return {"schema": "dev.splatlab.material-compatibility/v1", "sample_space": "image-sRGB bytes at recovery-cell centers; nearest sampling",
            "observed_reference_mae_rgb8": float(anchored_error[accepted].mean()), "boundary_pairs": len(mixed_jumps),
            "boundary_anchor_mae_rgb8": float(np.mean(anchor_errors)), "boundary_jump_mae_rgb8": float(np.mean(mixed_jumps)),
            "generated_boundary_jump_mae_rgb8": float(np.mean(generated_jumps)), "automatic_acceptance": False,
            "scope": "Advisory UV/color disagreement only; legitimate material edges also create jumps. No hidden-region truth, photorealism or geometry score."}


def layered_glb(plane, accepted, observed_png, generated_image, image_mime):
    plane = plane_arrays(plane)
    size = len(accepted)
    coordinates = np.linspace(0, 1, size + 1)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    uv = np.stack([horizontal, vertical], axis=-1).reshape(-1, 2)
    planar = plane["lower_uv"] + uv * (plane["upper_uv"] - plane["lower_uv"])
    positions = (plane["center"] + planar[:, :1] * plane["right"] + planar[:, 1:] * plane["forward"]).astype("<f4")
    index_groups = []
    for cells in (accepted, ~accepted):
        rows, columns = np.nonzero(cells)
        starts = rows * (size + 1) + columns
        index_groups.append(np.stack([starts, starts + 1, starts + size + 2, starts, starts + size + 2, starts + size + 1], axis=1).astype("<u4").reshape(-1))
    payloads = [positions.tobytes(), uv.astype("<f4").tobytes(), *[indices.tobytes() for indices in index_groups], observed_png, generated_image]
    binary, views = bytearray(), []
    for payload in payloads:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
        binary.extend(payload)
        binary.extend(b"\0" * (-len(binary) % 4))
    tag = {GLTF_EXTRAS_KEY: GENERATIVE_TAG}
    layers = [{"appearance_source": "captured-photos", "geometry_source": "sfm-plane-fit", "cells": int(accepted.sum())},
              {"appearance_source": "generated-image", "geometry_source": "unobserved-plane-extrapolation", "cells": int((~accepted).sum())}]
    document = {"asset": {"version": "2.0", "generator": "SplatLab observed/generated support completion", "extras": tag},
        "extensionsUsed": ["KHR_materials_unlit"], "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "extras": tag}],
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
                      {"bufferView": 1, "componentType": 5126, "count": len(uv), "type": "VEC2"}]
                     + [{"bufferView": index + 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"} for index, indices in enumerate(index_groups)],
        "images": [{"bufferView": 4, "mimeType": "image/png"}, {"bufferView": 5, "mimeType": image_mime}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"source": index, "sampler": 0} for index in range(2)],
        "materials": [{"name": layer["appearance_source"], "extensions": {"KHR_materials_unlit": {}}, "doubleSided": True,
                       "alphaMode": "MASK" if index == 0 else "OPAQUE", "alphaCutoff": .5,
                       "pbrMetallicRoughness": {"baseColorTexture": {"index": index}, "metallicFactor": 0, "roughnessFactor": 1},
                       "extras": {**tag, **layer}} for index, layer in enumerate(layers)],
        "meshes": [{"extras": tag, "primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": index + 2,
                    "material": index, "extras": {**tag, **layer}} for index, layer in enumerate(layers)]}],
        "extras": {**tag, "geometry_source": "photo-anchored plane with unknown-cell extrapolation", "layers": layers,
                   "observed_texture_sha256": hashlib.sha256(observed_png).hexdigest(), "generated_texture_sha256": hashlib.sha256(generated_image).hexdigest(),
                   "coverage": "disjoint complete grid; generation never owns an observed cell"}}
    encoded = scenes.canonical_bytes(document)
    encoded += b" " * (-len(encoded) % 4)
    return struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary)) + struct.pack("<I4s", len(encoded), b"JSON") + encoded + struct.pack("<I4s", len(binary), b"BIN\0") + binary


def import_image(job, identifier, content, provider, model):
    if any(not isinstance(value, str) or not 1 <= len(value.strip()) <= 120 for value in (provider, model)):
        raise evidence.EvidenceError("Record a provider and model/source label, at most 120 characters each")
    description = image_description(content)
    with recovery.worker_slot(wait_seconds=3), scenes.write_lock(job):
        receipt = read(job, identifier)
        verify(job, receipt)
        output = directory(job, identifier)
        if (output / "result.json").exists() or (output / "candidate.glb").exists():
            raise evidence.EvidenceError("Completion output already exists; prepare a new candidate")
        original = recovery.read(job, receipt["recovery_id"])
        accepted = observed_cells(job, original)
        observed = artifact(job, identifier, "atlas.png").read_bytes()
        mesh = layered_glb(receipt["plane"], accepted, observed, content, description["mime_type"])
        (output / "candidate.glb").write_bytes(mesh)
        generated_name = "generated." + description["extension"]
        (output / generated_name).write_bytes(content)
        np.savez_compressed(output / "cell-provenance.npz", observed=accepted, generated=~accepted)
        glb_check.validate_glb(output / "candidate.glb")
        verify(job, receipt)
        result = {"completion_id": identifier, "prepared_receipt_sha256": receipt["sha256"], "status": "needs-review", "created_at": manifests.utc_now(),
            "generator": {"provider": provider.strip(), "model": model.strip(), "identity_source": "uploader-reported, not independently verified"},
            "image": description, "generated_image_name": generated_name, "observed_texture_sha256": hashlib.sha256(observed).hexdigest(),
            "generated_texture_sha256": hashlib.sha256(content).hexdigest(), "observed_cells": int(accepted.sum()), "generated_cells": int((~accepted).sum()),
            "triangles": int(accepted.size * 2), "observed_fraction": float(accepted.mean()), "generated_fraction": float((~accepted).mean()),
            "coverage_scope": "representation occupancy, not evidence coverage, photorealism, source fidelity or navigation acceptance",
            "material_diagnostics": material_diagnostics(accepted, observed, content),
            "implementation_sha256": manifests.sha256_file(Path(__file__)), "render_vr_only": True}
        return reviews.seal(job, output, "result.json", result, [output / name for name in ("candidate.glb", generated_name, "cell-provenance.npz")])


def proposal_asset(job, identifier, pointer, selected_slug=None):
    receipt = read(job, identifier)
    verify(job, receipt)
    result = read(job, identifier, result=True)
    if receipt["base"] != pointer or selected_slug and selected_slug != receipt["selected_slug"]:
        raise evidence.EvidenceError("Completion belongs to a different scene or captured object")
    if result.get("render_vr_only") is not True or result["observed_texture_sha256"] != receipt["artifacts"]["atlas.png"]["sha256"]:
        raise evidence.EvidenceError("Completion does not preserve the registered observed texture")
    return artifact(job, identifier, "candidate.glb", result=True), result["artifacts"]["candidate.glb"]["sha256"], receipt


def attach(job, revision, entry, slug, identifier):
    receipt = read(job, identifier)
    result = read(job, identifier, result=True)
    entry.update(geometry_source="photo-anchored-plane-with-unobserved-extension", appearance_source="captured-photos-and-generated-material",
                 classification=["observed texture preserved; unknown-cell geometry/appearance explicitly invented"],
                 completion={"completion_id": identifier, "recovery_id": receipt["recovery_id"], "observed_fraction": result["observed_fraction"],
                             "generated_fraction": result["generated_fraction"], "generator": result["generator"]})
    revision["state"].setdefault("recovery_dependencies", {})[slug] = {key: deepcopy(receipt[key]) for key in ("sources", "calibration")}
    original = recovery.read(job, receipt["recovery_id"])
    revision["artifacts"][f"_studio/{original['recovery_id']}.json"] = scenes.store_json(job, original)
    for name, identity in original["artifacts"].items():
        recovery_input(job, original, name)
        revision["artifacts"][f"_studio/recoveries/{original['recovery_id']}/{name}"] = identity
    for filename, record in (("receipt.json", receipt), ("result.json", result)):
        revision["artifacts"][f"_studio/completions/{identifier}/{filename}"] = scenes.store_json(job, record)
        for name, identity in record["artifacts"].items():
            artifact(job, identifier, name, result=filename == "result.json")
            revision["artifacts"][f"_studio/completions/{identifier}/{name}"] = identity
