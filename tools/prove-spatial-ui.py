#!/usr/bin/env python3
"""Check a staged frontend on an ephemeral loopback server without restarting services."""

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time

import uvicorn


def walking_proof_inputs(job, expected_generation):
    import artifact_manifest as manifests
    import scene_revisions as scenes

    pointer = scenes.active(job)
    revision = scenes.read_revision(job, pointer["revision_id"])
    sources = scenes.source_fingerprint(job)
    if (pointer["generation"] != expected_generation or revision["state"]["viewer"]["units"] != "meters"
            or revision["state"]["viewer"].get("calibration", {}).get("stale")
            or revision["source_fingerprint"] != sources):
        raise RuntimeError("Walking proof requires a current calibrated pinned revision")
    hashes = {key: manifests.sha256_file(scenes.local_file(job, key)) for key in sources["files"]}
    for key, record in revision["artifacts"].items():
        if manifests.sha256_file(scenes.artifact(job, pointer["revision_id"], key)) != record["sha256"]:
            raise RuntimeError("Walking input artifact checksum changed")
    return {"base": pointer, "state": revision["state"], "artifacts": revision["artifacts"],
            "source_fingerprint": sources, "source_hashes": hashes}


def audit_fixed_walking(job, snapshot, output):
    from PIL import Image
    import artifact_manifest as manifests

    if walking_proof_inputs(job, snapshot["base"]["generation"]) != snapshot:
        raise RuntimeError("Read-only walking proof changed its pinned scene or source content")
    report = manifests.read_json(output / "fixed-walking-browser-proof.json")
    if (report.get("status") != "passed" or report.get("base") != snapshot["base"]
            or report.get("final") != snapshot["base"] or report.get("exact_state_unchanged") is not True
            or report.get("exact_pixel_restore") is not True or report.get("page_errors") != []
            or report.get("body") != {"radiusM": .22, "totalHeightM": 1.7, "unitsPerMetre": 1}):
        raise RuntimeError("Walking browser report does not match its pinned input")
    labels = {"walking-baseline", "walking-admitted", "walking-after-short-movement", "walking-restored-view"}
    frames = report.get("frames", [])
    if len(frames) != len(labels) or {frame.get("label") for frame in frames} != labels:
        raise RuntimeError("Walking proof is missing required rendered frames")
    for frame in frames:
        image_path = output / (frame["label"] + ".png")
        if image_path.is_symlink() or manifests.sha256_file(image_path) != frame["sha256"]:
            raise RuntimeError("Walking frame checksum changed")
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.size != (960, 640) or image.size != (frame["width"], frame["height"]):
                raise RuntimeError("Walking frame dimensions changed")
            image.verify()
        with Image.open(image_path) as image:
            if image.convert("RGB").getcolors(maxcolors=15) is not None:
                raise RuntimeError("Walking frame is blank or lacks rendered color diversity")
    indexed = {frame["label"]: frame for frame in frames}
    if indexed["walking-baseline"]["sha256"] != indexed["walking-restored-view"]["sha256"]:
        raise RuntimeError("Walking saved frame does not restore exactly")
    trace_path = output / "walking-rendered-trace.json"
    if trace_path.is_symlink() or manifests.sha256_file(trace_path) != report.get("trace_sha256"):
        raise RuntimeError("Walking trace checksum changed")
    samples = json.loads(trace_path.read_text())
    if not isinstance(samples, list) or len(samples) < 5 or len(samples) != report.get("rendered_samples"):
        raise RuntimeError("Walking proof has insufficient rendered samples")
    previous_time = -math.inf
    for sample in samples:
        position = sample.get("position", [])
        if (sample.get("flying") is not False or sample.get("pointer_locked") is not True
                or sample.get("radius") != .22 or not math.isclose(sample.get("height", 0), 1.7, abs_tol=1e-9)
                or len(position) != 3 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in position)
                or not math.isfinite(sample.get("time", math.nan)) or sample["time"] <= previous_time):
            raise RuntimeError("Walking trace contains flight, altered dimensions or invalid rendered samples")
        previous_time = sample["time"]
    start = indexed["walking-admitted"]["position"]
    finish = samples[-1]["position"]
    travel = math.hypot(finish[0] - start[0], finish[2] - start[2])
    if not .05 < travel < .4 or not math.isclose(travel, report.get("horizontal_travel_m", -1), abs_tol=1e-9):
        raise RuntimeError("Walking trace does not support its stated short movement")
    audit = {"status": "passed", "base": snapshot["base"], "verified_frames": len(frames),
             "verified_samples": len(samples), "horizontal_travel_m": travel, "exact_sources_and_artifacts_unchanged": True,
             "scope": "Independent saved-image, trace and immutable-input checks; not doorway traversal or floor-repair acceptance"}
    manifests.atomic_write_json(output / "fixed-walking-audit.json", audit)
    return audit


def architectural_proof_inputs(job, identifier, expected_generation):
    import architectural_edits
    import architectural_navigation
    import architectural_volumes
    import artifact_manifest
    import scene_revisions

    receipt = architectural_edits.read(job, identifier)
    architectural_edits.verify(job, receipt)
    result = architectural_edits.read(job, identifier, True)
    if (receipt["base"]["generation"] != expected_generation or result.get("verdict") != "PASS_ARCHITECTURAL_EDIT"
            or result.get("architecture_id") != identifier or result.get("method") != architectural_edits.METHOD
            or set(result.get("gates", {})) != architectural_edits.required_gates(receipt)
            or any(value is not True for value in result["gates"].values())):
        raise RuntimeError("Architectural proof requires current passing geometry evidence")
    if result.get("appearance_method", architectural_volumes.LEGACY_METHOD) != architectural_volumes.method(receipt["clip"]):
        raise RuntimeError("Architectural proof appearance method differs from its prepared envelope")
    expected_triangles = len(architectural_edits.room_geometry(receipt["spec"], architectural_edits.geometry_method(receipt))[1])
    if result.get("metrics", {}).get("room_triangles") != expected_triangles:
        raise RuntimeError("Architectural proof triangle count differs from the verified room master")
    for name in result["artifacts"]:
        architectural_edits.artifact(job, identifier, name, True)
    if result.get("navigation_method", architectural_navigation.LEGACY_METHOD) != architectural_navigation.method(receipt["recipe"]):
        raise RuntimeError("Architectural proof navigation method differs from its preparation")
    architectural_navigation.verify_document(receipt["spec"], receipt["recipe"], architectural_edits.frame(receipt["spec"]),
        artifact_manifest.read_json(architectural_edits.artifact(job, identifier, "navmesh.json", True)), identifier, result["gates"])
    revision = scene_revisions.read_revision(job, receipt["base"]["revision_id"])
    return {"architecture_id": identifier, "base": receipt["base"], "result_sha256": result["sha256"],
            "authored_room_triangles": expected_triangles,
            "artifacts": revision["artifacts"], "source_fingerprint": revision["source_fingerprint"],
            "sources": receipt["sources"], "calibration": receipt["calibration"]}


def audit_architectural_walking_image(output, snapshot):
    from PIL import Image, ImageChops
    import artifact_manifest as manifests

    control = snapshot.get("render_control", {})
    if (control.get("method") != "scene-visibility-ab/v1" or control.get("same_camera_body_and_collision") is not True
            or control.get("scene_visible_after") is not True):
        raise RuntimeError("Rendered walking image lacks a same-pose scene visibility control")
    images = {}
    for label, record in (("original", snapshot), ("hidden", control.get("hidden", {})), ("restored", control.get("restored", {}))):
        expected = snapshot["filename"] if label == "original" else snapshot["filename"].replace(".png", f"-scene-{label}.png")
        if record.get("filename") != expected or Path(expected).name != expected:
            raise RuntimeError("Rendered walking control has an unexpected filename")
        image_path = output / expected
        if image_path.is_symlink() or not image_path.is_file() or manifests.sha256_file(image_path) != record.get("sha256"):
            raise RuntimeError("Rendered walking image or control checksum changed")
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.size != (960, 640):
                raise RuntimeError("Rendered walking image or control has invalid dimensions")
            image.verify()
        with Image.open(image_path) as image:
            images[label] = image.convert("RGB")
    if snapshot["sha256"] != control["restored"]["sha256"]:
        raise RuntimeError("Rendered walking control did not restore exact pixels")
    difference = ImageChops.difference(images["original"], images["hidden"])
    red, green, blue = difference.split()
    magnitude = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    changed_pixels = sum(magnitude.histogram()[8:])
    if changed_pixels < math.ceil(960 * 640 * .01):
        raise RuntimeError("Rendered walking image is blank or lacks a significant scene contribution")
    return {"filename": snapshot["filename"], "scene_contribution_pixels": changed_pixels,
            "method": control["method"], "exact_pixel_restore": True}


def audit_architectural_restoration(job, snapshot, output, preview_only=False):
    from PIL import Image
    import artifact_manifest as manifests
    import background_recovery
    import scene_revisions

    pointer = scene_revisions.active(job)
    baseline = scene_revisions.read_revision(job, snapshot["base"]["revision_id"])
    final = scene_revisions.read_revision(job, pointer["revision_id"])
    report = manifests.read_json(output / ("architectural-preview-proof.json" if preview_only else "architectural-browser-proof.json"))
    if (pointer["generation"] != snapshot["base"]["generation"] + (0 if preview_only else 2) or pointer.get("proposal_id") is not None
            or final["state"] != baseline["state"] or final["artifacts"] != snapshot["artifacts"]
            or scene_revisions.source_fingerprint(job) != snapshot["source_fingerprint"]
            or report.get("status") != ("passed_preview_only" if preview_only else "passed") or report.get("architecture_id") != snapshot["architecture_id"]
            or report.get("result_sha256") != snapshot["result_sha256"] or report.get("base") != snapshot["base"]
            or report.get("final") != pointer
            or (report.get("preview_dismissed") is not True if preview_only else
                report.get("exact_state_undo") is not True or report.get("exact_pixel_undo") is not True)):
        raise RuntimeError("Architectural browser evidence or exact restoration does not match the pinned input")
    required_frames = {"architecture-baseline", "architecture-draft", "architecture-guide-hidden", "architecture-preview",
        "architecture-preview-oblique", "architecture-splat-unclipped-diagnostic", "architecture-splat-reclipped",
        "architecture-mesh-cut", "architecture-mesh-uncut-diagnostic", "architecture-mesh-reclipped",
        "architecture-after-traversal", "architecture-applied-reloaded", "architecture-restored"}
    restored_label = "architecture-preview-dismissed" if preview_only else "architecture-restored"
    if preview_only:
        required_frames -= {"architecture-applied-reloaded", "architecture-restored"}
        required_frames.add(restored_label)
    frames = report.get("frames", [])
    if len(frames) != len(required_frames) or {frame.get("label") for frame in frames} != required_frames:
        raise RuntimeError("Architectural proof is missing required paired rendered frames")
    for frame in frames:
        image_path = output / (frame["label"] + ".png")
        if image_path.is_symlink() or manifests.sha256_file(image_path) != frame["png_sha256"]:
            raise RuntimeError("Architectural saved frame checksum changed")
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.size != (960, 640) or image.size != (frame["width"], frame["height"]):
                raise RuntimeError("Architectural saved frame dimensions differ from the renderer readback")
            image.verify()
        with Image.open(image_path) as image:
            if image.convert("RGB").getcolors(maxcolors=15) is not None:
                raise RuntimeError("Architectural saved frame is blank or lacks rendered color diversity")
    hashes = {frame["label"]: frame["png_sha256"] for frame in frames}
    for group in [("architecture-baseline", "architecture-guide-hidden", restored_label),
                  ("architecture-preview", "architecture-splat-reclipped", "architecture-after-traversal") + (() if preview_only else ("architecture-applied-reloaded",)),
                  ("architecture-mesh-cut", "architecture-mesh-reclipped")]:
        if len({hashes[name] for name in group}) != 1:
            raise RuntimeError("Saved architectural frames contradict exact paired appearance restoration")
    for first, second in [("architecture-baseline", "architecture-draft"), ("architecture-baseline", "architecture-preview"),
                          ("architecture-preview", "architecture-splat-unclipped-diagnostic"),
                          ("architecture-mesh-cut", "architecture-mesh-uncut-diagnostic")]:
        if hashes[first] == hashes[second]:
            raise RuntimeError("Saved architectural diagnostic frames show no visual change")
    expected_traces = {f"architecture-trace-{profile}-{direction}.json"
                       for profile in ("retained-worker", "viewer-default-dimensions") for direction in ("outward", "return")}
    traces = report.get("traces", [])
    if len(traces) != 4 or {trace.get("filename") for trace in traces} != expected_traces:
        raise RuntimeError("Architectural proof is missing a capsule profile or traversal direction")
    for trace in traces:
        trace_path = output / trace["filename"]
        if trace_path.is_symlink() or manifests.sha256_file(trace_path) != trace["sha256"]:
            raise RuntimeError("Saved architectural capsule trace checksum changed")
    expected_rendered = {name.replace("architecture-trace-", "architecture-rendered-") for name in expected_traces}
    rendered_traces = report.get("rendered_traces", [])
    if len(rendered_traces) != 4 or {trace.get("filename") for trace in rendered_traces} != expected_rendered:
        raise RuntimeError("Architectural proof is missing actual rendered traversal at a body profile or direction")
    walking_controls = []
    for trace in rendered_traces:
        trace_path = output / trace["filename"]
        if trace_path.is_symlink() or manifests.sha256_file(trace_path) != trace["sha256"]:
            raise RuntimeError("Saved rendered traversal checksum changed")
        document = manifests.read_json(trace_path)
        snapshots = document.get("snapshots", [])
        expected_images = {trace["filename"].replace(".json", f"-{index}.png") for index in range(3)}
        if (document.get("rendering_during_substeps") is not True or document.get("fixed_step_simulation") is not False
                or document.get("completed") is not True or document.get("error") is not None
                or len(snapshots) != 3 or {item.get("filename") for item in snapshots} != expected_images):
            raise RuntimeError("Rendered traversal is incomplete or lacks start/middle/end images")
        for walking_snapshot in snapshots:
            walking_controls.append(audit_architectural_walking_image(output, walking_snapshot))
    for key in final["artifacts"]:
        scene_revisions.artifact(job, final["revision_id"], key)
    background_recovery.verify_receipt_sources(job, snapshot, checksums=True)
    audit = {"status": "passed", "base": snapshot["base"], "final": pointer, "architecture_id": snapshot["architecture_id"],
        "result_sha256": snapshot["result_sha256"], "verified_frames": len(frames), "verified_traces": len(traces),
        "verified_rendered_traces": len(rendered_traces), "verified_walking_images": 3 * len(rendered_traces),
        "walking_render_controls": walking_controls, "verified_walking_control_images": 2 * len(walking_controls),
        "verified_artifacts": len(final["artifacts"]),
        "exact_state_and_artifact_restore": not preview_only, "preview_only": preview_only,
        "scene_never_activated": preview_only, "bound_original_sources_unchanged": True,
        "scope": "Independent local artifact/PNG/restoration verification, not visual quality or measured geometry acceptance"}
    manifests.atomic_write_json(output / ("architectural-preview-audit.json" if preview_only else "architectural-restoration-audit.json"), audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--playwright-module", type=Path, required=True)
    parser.add_argument("--route", default="")
    parser.add_argument("--studio-job", default="")
    parser.add_argument("--studio-generation", type=int, default=0)
    parser.add_argument("--studio-gaussian-object", default="")
    parser.add_argument("--studio-recovery", action="store_true")
    parser.add_argument("--studio-visibility", default="", help="Private recovery ID for unactivated depth-qualification comparison")
    parser.add_argument("--studio-structure", default="", help="Private retained structural study to review without scene mutation")
    parser.add_argument("--studio-architecture", default="", help="Private passing architectural study; tests actual shaders/navigation and applies/undoes one revision")
    parser.add_argument("--studio-architecture-preview", action="store_true", help="Keep the architectural GPU proof preview-only; never activate or claim an undo")
    parser.add_argument("--studio-walking", action="store_true", help="Read-only private fixed-body walking admission and actual rendered short movement")
    parser.add_argument("--studio-removal", action="store_true")
    parser.add_argument("--studio-inspect", action="store_true")
    parser.add_argument("--architectural-projection-comparison", action="store_true", help="Explicit private legacy/corrected projection and depth comparison")
    parser.add_argument("--studio-selection", action="store_true")
    parser.add_argument("--studio-refined", action="store_true")
    parser.add_argument("--studio-compound-receipt", type=Path)
    parser.add_argument("--studio-native-resume", type=Path)
    parser.add_argument("--browser-gpu", action="store_true")
    parser.add_argument("--studio-performance", action="store_true")
    parser.add_argument("--studio-support", action="store_true")
    parser.add_argument("--studio-completion", action="store_true")
    parser.add_argument("--studio-completion-layout", action="store_true")
    parser.add_argument("--studio-completion-comparison", type=Path)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.studio_architecture_preview and not args.studio_architecture:
        parser.error("Architectural preview-only proof requires an explicit architectural study")
    if args.architectural_projection_comparison and not os.environ.get("SPATIAL_INSPECT_ARCHITECTURE_PROOF"):
        parser.error("Architectural projection comparison requires a retained private architectural inspection")
    if os.environ.get("SPATIAL_INSPECT_ARCHITECTURE_PROOF"):
        private_root = Path(__file__).resolve().parents[1] / "data/spatial/scene-proof/outputs"
        if (not args.studio_inspect or not args.browser_gpu or args.studio_job != "splat_c0ffee"
                or not args.outputs_root or args.outputs_root.resolve() != private_root or args.route
                or any(value for name, value in vars(args).items() if name.startswith("studio_")
                       and name not in {"studio_job", "studio_generation", "studio_inspect"})):
            parser.error("Historical architectural appearance inspection requires the exact private GPU fixture")
        if args.output.exists() or args.output.is_symlink():
            parser.error("Historical appearance inspection needs a new output directory")
    if args.studio_compound_receipt:
        args.studio_refined = True
    if args.studio_completion_comparison:
        args.studio_completion = True
    if args.studio_native_resume and not args.studio_compound_receipt:
        parser.error("Native resumption requires the original compound receipt")
    if args.studio_performance and not args.browser_gpu:
        parser.error("Performance measurement requires --browser-gpu")
    if args.studio_walking:
        private_root = Path(__file__).resolve().parents[1] / "data/spatial/scene-proof/outputs"
        if not args.browser_gpu or args.studio_job != "splat_c0ffee" or not args.outputs_root or args.outputs_root.resolve() != private_root:
            parser.error("Walking proof requires --browser-gpu and the exact private fixture outputs root")
        if args.route or any(value for name, value in vars(args).items() if name.startswith("studio_")
                             and name not in {"studio_job", "studio_generation", "studio_walking"}):
            parser.error("Walking proof cannot be mixed with another proof mode")
        if args.output.exists() or args.output.is_symlink():
            parser.error("Walking proof needs a new output directory; preserve previous attempts")
    if args.studio_architecture:
        private_root = Path(__file__).resolve().parents[1] / "data/spatial/scene-proof/outputs"
        if not args.browser_gpu or args.studio_job != "splat_c0ffee" or not args.outputs_root or args.outputs_root.resolve() != private_root:
            parser.error("Architectural apply/undo proof requires --browser-gpu and the exact private fixture outputs root")
        if args.route or any(value for name, value in vars(args).items() if name.startswith("studio_")
                             and name not in {"studio_job", "studio_generation", "studio_architecture", "studio_architecture_preview"}):
            parser.error("Architectural proof cannot be mixed with another proof mode")
        if args.output.exists() or args.output.is_symlink():
            parser.error("Architectural proof needs a new output directory; preserve previous attempts")
    if not (args.dist / "index.html").is_file():
        parser.error("Build the staged frontend first")
    if not (args.playwright_module / "package.json").is_file():
        parser.error("The installed Playwright module path does not exist; no browser has started")
    if args.browser_gpu:
        gate = str(Path(__file__).resolve().parent / "splatlab-compute-gate.sh")
        if subprocess.run([gate, "--is-contained"], capture_output=True).returncode:
            parser.error("GPU browser proofs must run through splatlab-compute-gate.sh --run")
        subprocess.run([gate, "--check"], check=True)
    os.environ["SPLATLAB_FRONTEND_DIST"] = str(args.dist.resolve())
    os.environ["PORTAL_TOKEN"] = secrets.token_urlsafe(32)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    import main as application
    import capture_records
    if args.outputs_root:
        application.splat_route.DEFAULT_3D_ROOT = args.outputs_root.resolve()
    architecture_snapshot = None
    walking_snapshot = None
    if args.studio_walking:
        import artifact_manifest as manifests

        walking_snapshot = walking_proof_inputs(args.outputs_root.resolve() / args.studio_job, args.studio_generation)
        args.output.mkdir(parents=True)
        manifests.atomic_write_json(args.output / "walking-input-audit.json", walking_snapshot)
    if args.studio_architecture:
        import artifact_manifest as manifests

        architecture_snapshot = architectural_proof_inputs(args.outputs_root.resolve() / args.studio_job,
                                                          args.studio_architecture, args.studio_generation)
        args.output.mkdir(parents=True)
        manifests.atomic_write_json(args.output / "architectural-input-audit.json", architecture_snapshot)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(application.app, lifespan="off", access_log=False, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                raise RuntimeError("Temporary proof server failed to start")
            time.sleep(0.1)
        environment = capture_records.worker_env()
        environment.update(SPATIAL_PROOF_BASE=f"http://127.0.0.1:{port}",
                           SPATIAL_PROOF_TOKEN=os.environ["PORTAL_TOKEN"],
                           SPATIAL_PROOF_PLAYWRIGHT=str(args.playwright_module.resolve()),
                           SPATIAL_PROOF_ROUTE=args.route, SPATIAL_PROOF_OUTPUT=str(args.output.resolve()),
                           SPATIAL_STUDIO_JOB=args.studio_job,
                           SPATIAL_GAUSSIAN_OBJECT=args.studio_gaussian_object,
                           SPATIAL_STUDIO_RECOVERY="1" if args.studio_recovery else "",
                           SPATIAL_STUDIO_VISIBILITY=args.studio_visibility,
                           SPATIAL_STUDIO_STRUCTURE=args.studio_structure,
                           SPATIAL_STUDIO_ARCHITECTURE=args.studio_architecture,
                           SPATIAL_ARCHITECTURE_PREVIEW_ONLY="1" if args.studio_architecture_preview else "",
                           SPATIAL_INSPECT_ARCHITECTURE_PROOF=os.environ.get("SPATIAL_INSPECT_ARCHITECTURE_PROOF", ""),
                           SPATIAL_ARCHITECTURE_PROJECTION_COMPARISON="1" if args.architectural_projection_comparison else "",
                           SPATIAL_STUDIO_WALKING="1" if args.studio_walking else "",
                           SPATIAL_STUDIO_REMOVAL="1" if args.studio_removal else "",
                           SPATIAL_STUDIO_INSPECT="1" if args.studio_inspect else "",
                           SPATIAL_STUDIO_SELECTION="1" if args.studio_selection else "",
                           SPATIAL_STUDIO_REFINED="1" if args.studio_refined else "",
                           SPATIAL_COMPOUND_RECEIPT=str(args.studio_compound_receipt.resolve()) if args.studio_compound_receipt else "",
                           SPATIAL_NATIVE_RESUME=str(args.studio_native_resume.resolve()) if args.studio_native_resume else "",
                           SPATIAL_PROOF_GPU="1" if args.browser_gpu else "",
                           SPATIAL_STUDIO_PERFORMANCE="1" if args.studio_performance else "",
                           SPATIAL_STUDIO_SUPPORT="1" if args.studio_support else "",
                           SPATIAL_STUDIO_COMPLETION="1" if args.studio_completion or args.studio_completion_layout else "",
                           SPATIAL_COMPLETION_LAYOUT_ONLY="1" if args.studio_completion_layout else "",
                           SPATIAL_COMPLETION_COMPARISON=str(args.studio_completion_comparison.resolve()) if args.studio_completion_comparison else "",
                           SPATIAL_STUDIO_GENERATION=str(args.studio_generation))
        budget = 720 if args.studio_architecture or args.studio_native_resume or args.studio_completion_comparison else 360 if args.studio_walking or args.studio_performance or args.studio_visibility or args.studio_removal or args.studio_refined or args.studio_support or args.studio_completion or args.studio_gaussian_object else 180
        result = subprocess.run(["node", str(Path(__file__).with_suffix(".mjs"))], env=environment, timeout=budget)
        if result.returncode == 0 and args.architectural_projection_comparison:
            import artifact_manifest as manifests
            report = manifests.read_json(args.output / "architectural-appearance-diagnostic.json")
            if (report.get("projection_comparison") is not True or report.get("active_scene_unchanged") is not True
                    or report.get("page_errors") != [] or report.get("shader_errors") != []
                    or len(report.get("frames", [])) != 18):
                raise RuntimeError("Requested architectural projection/depth comparison did not run completely")
        if result.returncode == 0 and architecture_snapshot:
            audit_architectural_restoration(args.outputs_root.resolve() / args.studio_job, architecture_snapshot, args.output, args.studio_architecture_preview)
        if result.returncode == 0 and walking_snapshot:
            audit_fixed_walking(args.outputs_root.resolve() / args.studio_job, walking_snapshot, args.output)
        return result.returncode
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
        if args.studio_architecture or args.studio_removal or args.studio_refined or args.studio_support or args.studio_completion:
            import artifact_manifest as manifests
            import scene_revisions

            private_root = Path(__file__).resolve().parents[1] / "data/spatial/scene-proof/outputs"
            if args.studio_job == "splat_c0ffee" and args.outputs_root and args.outputs_root.resolve() == private_root:
                checkpoint = manifests.read_json(args.output / "private-rollback-checkpoint.json")
                job = private_root / args.studio_job
                pointer = scene_revisions.active(job)
                owned = checkpoint.get("rollback_from", [{"revision_id": checkpoint["preview_revision"],
                    "generation": checkpoint["base"]["generation"] + 1}]) if checkpoint else []
                owned_pointer = bool(pointer and any(pointer["revision_id"] == candidate["revision_id"]
                    and pointer["generation"] == candidate["generation"] for candidate in owned))
                if checkpoint and pointer and not owned_pointer:
                    current = scene_revisions.read_revision(job, pointer["revision_id"])
                    owned_pointer = any(pointer["generation"] == candidate["generation"] and current["parent"] == candidate["parent"]
                        and current["operation"] == {"kind": "restore", "revision_id": candidate["target"]}
                        for candidate in checkpoint.get("restore_targets", []) if candidate["target"] != checkpoint["base"]["revision_id"])
                if checkpoint and pointer and owned_pointer and scene_revisions.read_revision(job, pointer["revision_id"])["state"] != scene_revisions.read_revision(job, checkpoint["base"]["revision_id"])["state"]:
                    restored = scene_revisions.restore(job, checkpoint["base"]["revision_id"], pointer["generation"])
                    manifests.atomic_write_json(args.output / "failed-proof-restore.json", {
                        "applied": pointer, "restored": restored, "scope": "private proof cleanup, not a browser-undo acceptance claim",
                    })


if __name__ == "__main__":
    raise SystemExit(main())
