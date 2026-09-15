import copy
import importlib.util
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

import artifact_manifest as manifests


TOOL = Path(__file__).resolve().parents[2] / "tools/prepare-dual-fisheye-pilot.py"
MODULE_SPEC = importlib.util.spec_from_file_location("dual_fisheye_pilot", TOOL)
pilot = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(pilot)


@pytest.fixture
def capture(tmp_path):
    source = tmp_path / "raw" / "source.insv"
    source.parent.mkdir()
    source.write_bytes(b"synthetic test media, not a real capture")
    return {"capture_id": "capture_" + "a" * 24,
            "source": {"path": str(source), **manifests.file_identity(source)},
            "streams": [{"index": 0, "width": 3840, "height": 3840},
                        {"index": 2, "width": 3840, "height": 3840}]}


def log_text(offset=0, count=5):
    return "[Parsed_showinfo_1 @ address] config in time_base: 1/30000, frame_rate: 30000/1001\n" + "".join(
        f"[Parsed_showinfo_1 @ address] n: {ordinal} pts: {ordinal * 30030 + offset} pts_time: 0\n"
        for ordinal in range(count))


def fake_decode(command, stdout, stderr, timeout, env):
    directory = Path(command[-1]).parent
    assert timeout > 0
    assert "CUDA_VISIBLE_DEVICES" not in env
    for ordinal in range(5):
        Image.new("RGB", (512, 512), (ordinal * 20, 0, 0)).save(directory / f"decoded-{ordinal:06d}.jpg")
    stderr.write(log_text())
    return subprocess.CompletedProcess(command, 0)


def test_plan_preserves_raw_lens_identity_and_group_splits(capture):
    result = pilot.make_plan(capture, 0, 30, 5, 512)
    assert [group["source_decoded_frame_index"] for group in result["groups"]] == [0, 30, 60, 90, 120]
    assert [group["split"] for group in result["groups"]] == ["train", "train", "test", "val", "train"]
    assert result["calibration"]["rig_extrinsics"] == "unknown"
    assert result["calibration"]["physical_optical_centers_assumed_equal"] is False
    assert result["clock"]["exposure_sync_verified"] is False


@pytest.mark.parametrize("args", [(-1, 30, 5, 512), (0, 0, 5, 512), (0, 30, 4, 512),
                                 (0, 30, 97, 512), (0, 30, 5, 513), (3600, 30, 5, 512),
                                 (0, 30, 5, 4096), (True, 30, 5, 512)])
def test_plan_refuses_unbounded_or_ambiguous_selection(capture, args):
    with pytest.raises(ValueError):
        pilot.make_plan(capture, *args)


@pytest.mark.parametrize("change", ["one_stream", "not_square", "duplicate_stream", "missing_hash"])
def test_plan_refuses_unsupported_inputs(capture, change):
    if change == "one_stream":
        capture["streams"].pop()
    elif change == "not_square":
        capture["streams"][0]["height"] = 1920
    elif change == "duplicate_stream":
        capture["streams"][1]["index"] = 0
    else:
        capture["source"].pop("sha256")
    with pytest.raises(ValueError):
        pilot.make_plan(capture, 0, 30, 5, 512)


def test_decode_command_never_stitches_resamples_or_rotates(tmp_path):
    command = pilot.decode_command(tmp_path / "source.insv", 2, [0, 30], 512, tmp_path)
    filters = command[command.index("-vf") + 1]
    assert filters.startswith("select=eq(n\\,0)+eq(n\\,30),showinfo")
    assert "v360" not in filters and "fps=" not in filters and "setpts" not in filters
    assert "-noautorotate" in command and "-copyts" in command and "-ss" not in command
    assert command[command.index("-map") + 1] == "0:2"
    assert command[command.index("-hwaccel") + 1] == "none"


def test_pts_use_integer_time_base_not_rounded_display_or_nominal_fps():
    values = pilot.parse_timestamps(log_text(11), 5)
    assert values[1]["pts"] == 30041
    assert values[1]["time_s"] == 30041 / 30000
    assert pilot.paired_timestamps(values, copy.deepcopy(values))
    changed = copy.deepcopy(values)
    changed[0]["pts"] += 1
    assert not pilot.paired_timestamps(values, changed)
    assert not pilot.paired_timestamps(values, changed[:-1])


@pytest.mark.parametrize("log", ["", log_text(count=4), log_text() + log_text(),
                                 log_text().replace("pts: 30030", "pts: 0"),
                                 log_text().replace("n: 3", "n: 9"),
                                 log_text().replace("1/30000", "0/1"),
                                 log_text().replace("1/30000", "1/0")])
def test_timestamp_failures_refuse_calibrated_pairing(log):
    with pytest.raises(ValueError):
        pilot.parse_timestamps(log, 5)


def test_success_keeps_both_lenses_in_each_split_and_sources_unchanged(capture, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot.subprocess, "run", fake_decode)
    output = tmp_path / "pilot"
    result = pilot.extract(pilot.make_plan(capture, 0, 30, 5, 512), output, 30)
    assert result["status"] == "decoded-needs-camera-and-mask-review"
    assert len(result["views"]) == 10
    assert result["clock"]["paired_container_pts_match"] is True
    for group in result["groups"]:
        views = [view for view in result["views"] if view["group_id"] == group["group_id"]]
        assert {view["lens_id"] for view in views} == {"lens-0", "lens-1"}
        assert {view["split"] for view in views} == {group["split"]}
    assert len((output / "train-images.txt").read_text().splitlines()) == 6
    assert manifests.same_file_identity(Path(capture["source"]["path"]), capture["source"])


@pytest.mark.parametrize("failure", ["pts_mismatch", "decode_error", "timeout", "truncated_image"])
def test_failures_leave_diagnostic_receipt_but_no_training_list(capture, tmp_path, monkeypatch, failure):
    def decode(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        result = fake_decode(command, **kwargs)
        if failure == "pts_mismatch" and "lens-1" in command[-1]:
            kwargs["stderr"].seek(0)
            kwargs["stderr"].truncate()
            kwargs["stderr"].write(log_text(1))
        if failure == "decode_error":
            return subprocess.CompletedProcess(command, 1)
        if failure == "truncated_image":
            (Path(command[-1]).parent / "decoded-000000.jpg").write_bytes(b"bad")
        return result

    monkeypatch.setattr(pilot.subprocess, "run", decode)
    output = tmp_path / "pilot"
    with pytest.raises((OSError, ValueError, subprocess.SubprocessError)):
        pilot.extract(pilot.make_plan(capture, 0, 30, 5, 512), output, 30)
    receipt = json.loads((output / "pilot.json").read_text())
    assert receipt["status"] == "failed-not-for-reconstruction"
    assert not (output / "train-images.txt").exists()


def test_changed_source_and_existing_output_refuse_before_decode(capture, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not decode"))
    plan = pilot.make_plan(capture, 0, 30, 5, 512)
    with pytest.raises(ValueError, match="new output"):
        pilot.extract(plan, tmp_path, 30)
    with pytest.raises(ValueError, match="outside"):
        pilot.extract(plan, Path(capture["source"]["path"]).parent / "derived", 30)
    Path(capture["source"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        pilot.extract(plan, tmp_path / "pilot", 30)
