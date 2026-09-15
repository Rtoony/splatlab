import io
import os
import subprocess
import sys

from PIL import Image
import pytest

import artifact_manifest as manifests
from test_architectural_proof import TOOL, runner
from test_scene_revisions import scene


@pytest.fixture
def walking_proof(scene, tmp_path):
    job, pointer = scene
    snapshot = runner.walking_proof_inputs(job, 0)
    output = tmp_path / "walking-proof"
    output.mkdir()
    frames = []
    for label in ("walking-baseline", "walking-admitted", "walking-after-short-movement", "walking-restored-view"):
        image = Image.new("RGB", (960, 640), (10, 20, 30))
        for index in range(16):
            image.putpixel((index, 0), (index, 15, 80))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_path = output / (label + ".png")
        image_path.write_bytes(buffer.getvalue())
        frames.append({"label": label, "width": 960, "height": 640,
                       "sha256": manifests.sha256_file(image_path), "position": [0, 1.48, 0]})
    samples = [{"time": 100 + index * 16, "position": [index * .02, 1.48, 0],
                "radius": .22, "height": 1.7, "flying": False, "pointer_locked": True} for index in range(8)]
    trace = output / "walking-rendered-trace.json"
    manifests.atomic_write_json(trace, samples)
    report = {"status": "passed", "base": pointer, "final": pointer, "page_errors": [],
              "body": {"radiusM": .22, "totalHeightM": 1.7, "unitsPerMetre": 1},
              "exact_state_unchanged": True, "exact_pixel_restore": True, "frames": frames,
              "trace_sha256": manifests.sha256_file(trace), "rendered_samples": len(samples), "horizontal_travel_m": .14,
              "scope": "Synthetic audit unit fixture, not a browser or real navigation result"}
    manifests.atomic_write_json(output / "fixed-walking-browser-proof.json", report)
    return job, snapshot, output, report, samples


def test_walking_audit_verifies_images_originals_and_short_fixed_body_trace(walking_proof):
    job, snapshot, output, _, _ = walking_proof
    audit = runner.audit_fixed_walking(job, snapshot, output)
    assert audit["verified_frames"] == 4 and audit["verified_samples"] == 8
    assert audit["exact_sources_and_artifacts_unchanged"]


@pytest.mark.parametrize("defect", ["source", "body", "flight", "pointer", "time", "short-trace", "travel", "image", "blank", "missing-frame"])
def test_walking_audit_refuses_changed_sources_and_incomplete_or_misleading_evidence(walking_proof, defect):
    job, snapshot, output, report, samples = walking_proof
    if defect == "source":
        source = job / "_world/shell.glb"
        before = source.stat()
        content = source.read_bytes()
        source.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif defect == "body":
        samples[2]["height"] = 1.2
    elif defect == "flight":
        samples[2]["flying"] = True
    elif defect == "pointer":
        samples[2]["pointer_locked"] = False
    elif defect == "time":
        samples[2]["time"] = samples[1]["time"]
    elif defect == "short-trace":
        samples[:] = samples[:2]
    elif defect == "travel":
        report["horizontal_travel_m"] = .2
    elif defect == "image":
        report["frames"][0]["sha256"] = "0" * 64
    elif defect == "blank":
        image_path = output / "walking-baseline.png"
        Image.new("RGB", (960, 640)).save(image_path)
        report["frames"][0]["sha256"] = manifests.sha256_file(image_path)
    else:
        report["frames"].pop()
    trace = output / "walking-rendered-trace.json"
    manifests.atomic_write_json(trace, samples)
    report["trace_sha256"] = manifests.sha256_file(trace)
    manifests.atomic_write_json(output / "fixed-walking-browser-proof.json", report)
    with pytest.raises(RuntimeError):
        runner.audit_fixed_walking(job, snapshot, output)
    assert not (output / "fixed-walking-audit.json").exists()


@pytest.mark.parametrize("defect", ["gpu", "root", "job", "mixed", "existing"])
def test_walking_cli_refuses_unsafe_inputs_before_server_or_gpu(tmp_path, defect):
    output = tmp_path / "new-proof"
    root = TOOL.parents[1] / "data/spatial/scene-proof/outputs"
    command = [sys.executable, str(TOOL), "--dist", str(tmp_path / "absent-dist"), "--playwright-module", "absent",
               "--output", str(output), "--studio-job", "splat_badbad" if defect == "job" else "splat_c0ffee",
               "--studio-walking", "--outputs-root", str(tmp_path if defect == "root" else root)]
    if defect != "gpu":
        command.append("--browser-gpu")
    if defect == "mixed":
        command.append("--studio-selection")
    if defect == "existing":
        output.mkdir()
        (output / "retained.txt").write_text("Keep the previous failed attempt")
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "Walking" in result.stderr
    if defect == "existing":
        assert (output / "retained.txt").read_text() == "Keep the previous failed attempt"
    else:
        assert not output.exists()
