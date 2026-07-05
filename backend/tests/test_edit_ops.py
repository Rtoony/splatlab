from __future__ import annotations

import asyncio
import struct
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_ops  # noqa: E402
import splat_route  # noqa: E402


# =============================================================================
# fixtures
# =============================================================================


@pytest.fixture()
def outputs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "outputs" / "3d"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", root)
    monkeypatch.setattr(edit_ops, "MAX_VERSIONS", 3)
    monkeypatch.setattr(edit_ops, "_EDIT_LOCKS", {})  # per-test isolation (and per-event-loop safety)
    return root


@pytest.fixture()
def stub_transform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny shell-script stand-in for splat-transform, wired in via the REAL
    SPLAT_TRANSFORM_BIN override that splat_route._splat_transform_path() honors.
    Behavior is driven by marker files in the returned control dir:
      fail_all   -> every invocation exits non-zero
      fail_regen -> only regen invocations (output splat.spz.*/web.ply.*/langweb.ply.*
                    tmp names) exit non-zero
    Otherwise it copies input -> output (argv shape: [bin, -w?, src, flags..., dst]).
    STUB_SLEEP (seconds) makes invocations slow so tests can overlap requests."""
    ctl = tmp_path / "stub-ctl"
    ctl.mkdir()
    script = tmp_path / "stub-splat-transform.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -u\n"
        'if [ "${1:-}" = "-w" ]; then in="$2"; else in="$1"; fi\n'
        'for a in "$@"; do out="$a"; done\n'
        'if [ -n "${STUB_SLEEP:-}" ]; then sleep "$STUB_SLEEP"; fi\n'
        'if [ -f "$STUB_CTL/fail_all" ]; then echo "stub: forced failure" >&2; exit 7; fi\n'
        'case "$(basename "$out")" in\n'
        "  splat.spz.*|web.ply.*|langweb.ply.*)\n"
        '    if [ -f "$STUB_CTL/fail_regen" ]; then echo "stub: forced regen failure" >&2; exit 7; fi\n'
        "    ;;\n"
        "esac\n"
        'cp "$in" "$out"\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv("SPLAT_TRANSFORM_BIN", str(script))
    monkeypatch.setenv("STUB_CTL", str(ctl))
    monkeypatch.delenv("STUB_SLEEP", raising=False)
    return ctl


def _make_job(root: Path, job_id: str, *, status: str = "completed", with_preview: bool = True) -> Path:
    job_dir = root / job_id
    if with_preview:
        preview = job_dir / "_preview"
        preview.mkdir(parents=True)
        (preview / "splat.ply").write_bytes(b"PLYDATA-v0")
    else:
        job_dir.mkdir(parents=True)
    splat_route._write_meta(
        job_id,
        {
            "job_id": job_id,
            "status": status,
            "output_dir": str(job_dir),
            "mode": "3d",
            "input_path": "test-input",
            "created_at": splat_route._utc_now(),
        },
    )
    return job_dir


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(edit_ops.router, prefix="/api/splat")
    return TestClient(app)


def _write_ply(path: Path, props: list[str], rows: list[list[float]]) -> None:
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {len(rows)}\n"
    header += "".join(f"property float {p}\n" for p in props)
    header += "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("latin1"))
        for row in rows:
            f.write(struct.pack("<" + "f" * len(props), *row))


def _read_ply_rows(path: Path) -> tuple[int, list[list[float]]]:
    data = path.read_bytes()
    hdr_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:hdr_end].decode("latin1")
    import re

    props = re.findall(r"property float (\w+)", header)
    n = int(re.search(r"element vertex (\d+)", header).group(1))
    rows = []
    off = hdr_end
    row_size = 4 * len(props)
    for _ in range(n):
        rows.append(list(struct.unpack("<" + "f" * len(props), data[off : off + row_size])))
        off += row_size
    return n, rows


# =============================================================================
# 1. id validation / editable-job gating
# =============================================================================


def test_require_editable_job_rejects_bad_id(outputs_root: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        edit_ops._require_editable_job("not-a-splat-id")
    assert exc.value.status_code == 404


def test_require_editable_job_rejects_missing_job(outputs_root: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        edit_ops._require_editable_job("splat_deadbeef")
    assert exc.value.status_code == 404


def test_require_editable_job_rejects_running_job(outputs_root: Path) -> None:
    _make_job(outputs_root, "splat_dead0001", status="running")
    with pytest.raises(HTTPException) as exc:
        edit_ops._require_editable_job("splat_dead0001")
    assert exc.value.status_code == 409


def test_require_editable_job_rejects_no_preview(outputs_root: Path) -> None:
    _make_job(outputs_root, "splat_dead0002", with_preview=False)
    with pytest.raises(HTTPException) as exc:
        edit_ops._require_editable_job("splat_dead0002")
    assert exc.value.status_code == 409


def test_require_editable_job_accepts_healthy_job(outputs_root: Path) -> None:
    job_dir = _make_job(outputs_root, "splat_dead0003")
    meta, resolved_dir = edit_ops._require_editable_job("splat_dead0003")
    assert meta["job_id"] == "splat_dead0003"
    assert resolved_dir == job_dir


# =============================================================================
# 2. op -> CLI argv builder correctness (no subprocess execution)
# =============================================================================


def test_op_to_argv_crop_box() -> None:
    op = edit_ops.CropBoxOp(type="crop_box", min=(-1, -2, -3), max=(1, 2, 3))
    assert edit_ops._op_to_argv(op) == ["-B", "-1,-2,-3,1,2,3"]


def test_op_to_argv_crop_sphere() -> None:
    op = edit_ops.CropSphereOp(type="crop_sphere", center=(0, 0, 0), radius=2.5)
    assert edit_ops._op_to_argv(op) == ["-S", "0,0,0,2.5"]


def test_op_to_argv_filter_value_both_bounds() -> None:
    op = edit_ops.FilterValueOp(type="filter_value", name="opacity", min=0.1, max=0.9)
    assert edit_ops._op_to_argv(op) == ["-V", "opacity,gte,0.1", "-V", "opacity,lte,0.9"]


def test_op_to_argv_filter_value_one_bound() -> None:
    op = edit_ops.FilterValueOp(type="filter_value", name="scale_0", min=None, max=2.0)
    assert edit_ops._op_to_argv(op) == ["-V", "scale_0,lte,2"]


def test_filter_value_rejects_unknown_name() -> None:
    with pytest.raises(Exception):
        edit_ops.FilterValueOp(type="filter_value", name="not_a_real_prop", min=0.1)


def test_filter_value_rejects_no_bounds() -> None:
    with pytest.raises(Exception):
        edit_ops.FilterValueOp(type="filter_value", name="opacity")


def test_op_to_argv_filter_floaters_default() -> None:
    op = edit_ops.FilterFloatersOp(type="filter_floaters")
    assert edit_ops._op_to_argv(op) == ["-G"]


def test_op_to_argv_filter_floaters_explicit() -> None:
    op = edit_ops.FilterFloatersOp(type="filter_floaters", size=0.05, op=0.1, min=0.004)
    assert edit_ops._op_to_argv(op) == ["-G", "0.05,0.1,0.004"]


def test_filter_floaters_rejects_partial_params() -> None:
    with pytest.raises(Exception):
        edit_ops.FilterFloatersOp(type="filter_floaters", size=0.05)


def test_op_to_argv_filter_cluster_with_seed() -> None:
    op = edit_ops.FilterClusterOp(type="filter_cluster", res=1.0, op=0.999, min=0.1, seed_pos=(1, 2, 3))
    assert edit_ops._op_to_argv(op) == ["-D", "1,0.999,0.1", "--seed-pos", "1,2,3"]


def test_op_to_argv_decimate_n() -> None:
    op = edit_ops.DecimateOp(type="decimate", n=500000)
    assert edit_ops._op_to_argv(op) == ["-F", "500000"]


def test_op_to_argv_decimate_pct() -> None:
    op = edit_ops.DecimateOp(type="decimate", pct=25)
    assert edit_ops._op_to_argv(op) == ["-F", "25%"]


def test_decimate_rejects_both_n_and_pct() -> None:
    with pytest.raises(Exception):
        edit_ops.DecimateOp(type="decimate", n=100, pct=10)


def test_decimate_rejects_neither() -> None:
    with pytest.raises(Exception):
        edit_ops.DecimateOp(type="decimate")


def test_op_to_argv_translate_rotate_scale() -> None:
    assert edit_ops._op_to_argv(edit_ops.TranslateOp(type="translate", x=1, y=-2, z=3)) == ["-t", "1,-2,3"]
    assert edit_ops._op_to_argv(edit_ops.RotateOp(type="rotate", x=90, y=0, z=0)) == ["-r", "90,0,0"]
    assert edit_ops._op_to_argv(edit_ops.ScaleOp(type="scale", factor=2)) == ["-s", "2"]


def test_build_apply_argv_chains_in_given_order() -> None:
    ops = [
        edit_ops.CropBoxOp(type="crop_box", min=(-1, -1, -1), max=(1, 1, 1)),
        edit_ops.DecimateOp(type="decimate", pct=50),
        edit_ops.TranslateOp(type="translate", x=1, y=0, z=0),
    ]
    argv = edit_ops._build_apply_argv("splat-transform", Path("/tmp/in.ply"), ops, Path("/tmp/out.ply"))
    assert argv == [
        "splat-transform",
        "-w",
        "/tmp/in.ply",
        "-B",
        "-1,-1,-1,1,1,1",
        "-F",
        "50%",
        "-t",
        "1,0,0",
        "/tmp/out.ply",
    ]


def test_ops_need_gpu_true_for_voxelization_actions() -> None:
    assert edit_ops._ops_need_gpu([edit_ops.FilterFloatersOp(type="filter_floaters")])
    assert edit_ops._ops_need_gpu([edit_ops.FilterClusterOp(type="filter_cluster")])
    assert not edit_ops._ops_need_gpu([edit_ops.TranslateOp(type="translate", x=1, y=0, z=0)])


def test_ops_change_topology() -> None:
    assert edit_ops._ops_change_topology([edit_ops.DecimateOp(type="decimate", pct=10)])
    assert edit_ops._ops_change_topology([edit_ops.CropSphereOp(type="crop_sphere", center=(0, 0, 0), radius=1)])
    assert not edit_ops._ops_change_topology(
        [edit_ops.TranslateOp(type="translate", x=1, y=0, z=0), edit_ops.ScaleOp(type="scale", factor=2)]
    )


def test_apply_ops_request_requires_at_least_one_op() -> None:
    with pytest.raises(Exception):
        edit_ops.ApplyOpsRequest(ops=[])


# =============================================================================
# 3. matrix decomposition (merge transforms) — pure math, no subprocess
# =============================================================================


def test_decompose_identity_returns_none() -> None:
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert edit_ops._decompose_similarity_matrix(identity) is None


def test_decompose_pure_translation() -> None:
    m = [[1, 0, 0, 5], [0, 1, 0, -2], [0, 0, 1, 0.5], [0, 0, 0, 1]]
    translate, rotate, scale = edit_ops._decompose_similarity_matrix(m)
    assert translate == pytest.approx((5, -2, 0.5))
    assert rotate == (0.0, 0.0, 0.0)
    assert scale is None


def test_decompose_pure_uniform_scale() -> None:
    m = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]
    translate, rotate, scale = edit_ops._decompose_similarity_matrix(m)
    assert translate == (0.0, 0.0, 0.0)
    assert rotate == (0.0, 0.0, 0.0)
    assert scale == pytest.approx(2.0)


@pytest.mark.parametrize(
    "axis_idx,angle_deg,expected_flag_axis",
    [(0, 90.0, "x"), (1, 90.0, "y"), (2, 90.0, "z")],
)
def test_decompose_single_axis_rotation_matches_verified_sign_flip(axis_idx, angle_deg, expected_flag_axis) -> None:
    """Cross-checks _decompose_similarity_matrix against the EMPIRICALLY measured
    splat-transform sign convention (module docstring / _axis_rotation_flag):
    -r must carry -angle for x/y but +angle for z to reproduce a standard
    Rx/Ry/Rz(angle_deg) rotation."""
    import math

    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    if axis_idx == 0:
        rot3 = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis_idx == 1:
        rot3 = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        rot3 = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m = [[*rot3[0], 0.0], [*rot3[1], 0.0], [*rot3[2], 0.0], [0, 0, 0, 1]]
    translate, rotate, scale = edit_ops._decompose_similarity_matrix(m)
    assert translate == (0.0, 0.0, 0.0)
    assert scale is None
    expected = edit_ops._axis_rotation_flag(expected_flag_axis, angle_deg)
    assert rotate == pytest.approx(expected, abs=1e-3)


def test_decompose_rejects_nonuniform_scale() -> None:
    m = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]]
    with pytest.raises(HTTPException) as exc:
        edit_ops._decompose_similarity_matrix(m)
    assert exc.value.status_code == 422
    assert "non-uniform" in exc.value.detail


def test_decompose_rejects_shear() -> None:
    # equal-norm, non-orthogonal columns: col0=(1,0,0), col1=(0.70711,0.70711,0)
    m = [[1, 0.70711, 0, 0], [0, 0.70711, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    with pytest.raises(HTTPException) as exc:
        edit_ops._decompose_similarity_matrix(m)
    assert exc.value.status_code == 422
    assert "shear" in exc.value.detail


def test_decompose_rejects_reflection() -> None:
    m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    with pytest.raises(HTTPException) as exc:
        edit_ops._decompose_similarity_matrix(m)
    assert exc.value.status_code == 422
    assert "reflection" in exc.value.detail


def test_decompose_rejects_multiaxis_rotation() -> None:
    # cyclic axis permutation (rotation by 120 deg about (1,1,1)) — a genuine
    # rotation matrix, but not aligned to any single principal axis.
    m = [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    with pytest.raises(HTTPException) as exc:
        edit_ops._decompose_similarity_matrix(m)
    assert exc.value.status_code == 422
    assert "multi-axis" in exc.value.detail


def test_decompose_rejects_degenerate_axis() -> None:
    m = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    with pytest.raises(HTTPException) as exc:
        edit_ops._decompose_similarity_matrix(m)
    assert exc.value.status_code == 422
    assert "degenerate" in exc.value.detail


def test_decompose_rejects_non_4x4() -> None:
    with pytest.raises(HTTPException) as exc:
        edit_ops._decompose_similarity_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert exc.value.status_code == 422


# =============================================================================
# 4. merge transform argv (TRS ordering + matrix path) — no subprocess
# =============================================================================


def test_merge_transform_to_argv_trs_orders_scale_rotate_translate() -> None:
    spec = edit_ops.MergeTransformSpec(translate=(1, 2, 3), rotate=(10, 20, 30), scale=2.0)
    assert edit_ops._merge_transform_to_argv(spec) == ["-s", "2", "-r", "10,20,30", "-t", "1,2,3"]


def test_merge_transform_to_argv_trs_partial() -> None:
    spec = edit_ops.MergeTransformSpec(translate=(1, 0, 0))
    assert edit_ops._merge_transform_to_argv(spec) == ["-t", "1,0,0"]


def test_merge_transform_to_argv_matrix_translation_only() -> None:
    m = [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    spec = edit_ops.MergeTransformSpec(matrix=m)
    assert edit_ops._merge_transform_to_argv(spec) == ["-t", "5,0,0"]


def test_merge_transform_to_argv_matrix_identity_is_noop() -> None:
    m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    spec = edit_ops.MergeTransformSpec(matrix=m)
    assert edit_ops._merge_transform_to_argv(spec) == []


def test_merge_transform_spec_rejects_matrix_and_trs_together() -> None:
    with pytest.raises(Exception):
        edit_ops.MergeTransformSpec(matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], translate=(1, 0, 0))


def test_build_merge_argv_applies_per_job_transforms() -> None:
    argv = edit_ops._build_merge_argv(
        "splat-transform",
        [("splat_a", Path("/tmp/a.ply")), ("splat_b", Path("/tmp/b.ply"))],
        {"splat_b": edit_ops.MergeTransformSpec(translate=(1, 0, 0))},
        Path("/tmp/out.ply"),
    )
    assert argv == [
        "splat-transform",
        "-w",
        "/tmp/a.ply",
        "/tmp/b.ply",
        "-t",
        "1,0,0",
        "/tmp/out.ply",
    ]


# =============================================================================
# 5. relevancy decode — the worker's REAL wire contract (quantize_relevancy):
#    q over the vector's OWN [min,max]; rel = X-Min + q/255*(X-Max - X-Min)
# =============================================================================


def test_decode_relevancy_bytes_dequantizes_with_xmin_xmax() -> None:
    """The BLOCKER case: X-Min=0.4/X-Max=0.6 — bytes must map into [0.4, 0.6],
    NOT to the absolute q/255 range-normalized values."""
    data = bytes([0, 64, 128, 191, 255])
    scores = edit_ops._decode_relevancy_bytes(data, 5, 0.4, 0.6)
    assert scores == pytest.approx(
        [0.4, 0.4 + 64 / 255 * 0.2, 0.4 + 128 / 255 * 0.2, 0.4 + 191 / 255 * 0.2, 0.6]
    )
    # exact mask at threshold 0.45: only byte 0 falls below (old q/255 decode
    # would ALSO have put byte 64 = 0.251 below — a different, wrong mask)
    assert [s >= 0.45 for s in scores] == [False, True, True, True, True]


def test_decode_relevancy_bytes_constant_vector_min_eq_max() -> None:
    # worker sends all-zero bytes with rmin == rmax for a constant vector
    assert edit_ops._decode_relevancy_bytes(bytes(4), 4, 0.7, 0.7) == pytest.approx([0.7] * 4)


def test_decode_relevancy_bytes_rejects_fp16_length() -> None:
    # the worker never sends fp16 — 2N bytes is a contract violation, not a dtype hint
    data = struct.pack("<4e", 0.1, 0.2, 0.3, 0.4)
    with pytest.raises(HTTPException) as exc:
        edit_ops._decode_relevancy_bytes(data, 4, 0.0, 1.0)
    assert exc.value.status_code == 502


def test_decode_relevancy_bytes_length_mismatch_raises() -> None:
    with pytest.raises(HTTPException) as exc:
        edit_ops._decode_relevancy_bytes(b"\x00\x01\x02", 4, 0.0, 1.0)
    assert exc.value.status_code == 502


def test_worker_relevancy_scores_missing_dequant_headers_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(config_path: str, lfdir: str, clean_text: str) -> httpx.Response:
        return httpx.Response(200, content=bytes(4))  # no X-Min/X-Max

    monkeypatch.setattr(edit_ops, "_post_worker_relevancy", fake_post)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(edit_ops._worker_relevancy_scores("cfg", "lf", "chair", 4))
    assert exc.value.status_code == 502
    assert "X-Min" in exc.value.detail


def test_worker_relevancy_scores_count_mismatch_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(config_path: str, lfdir: str, clean_text: str) -> httpx.Response:
        return httpx.Response(
            200, content=bytes(5), headers={"X-Count": "5", "X-Min": "0.0", "X-Max": "1.0"}
        )

    monkeypatch.setattr(edit_ops, "_post_worker_relevancy", fake_post)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(edit_ops._worker_relevancy_scores("cfg", "lf", "chair", 4))
    assert exc.value.status_code == 502
    assert "X-Count" in exc.value.detail


# =============================================================================
# 6. PLY row-mask rewrite round-trip (pure stdlib, header-driven)
# =============================================================================


def test_rewrite_ply_masked_round_trip(tmp_path: Path) -> None:
    props = ["x", "y", "z", "f_dc_0", "opacity"]
    n = 20
    rows = [[float(i), float(i) * 2, float(i) * 3, 0.5, 1.0] for i in range(n)]
    src = tmp_path / "splat.ply"
    _write_ply(src, props, rows)

    mask = [i % 2 == 0 for i in range(n)]  # keep evens -> 10 rows
    dst = tmp_path / "masked.ply"
    kept = edit_ops._rewrite_ply_masked(src, dst, mask)

    assert kept == 10
    n_out, rows_out = _read_ply_rows(dst)
    assert n_out == 10
    assert [r[0] for r in rows_out] == [float(i) for i in range(0, 20, 2)]


def test_rewrite_ply_masked_rejects_length_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "splat.ply"
    _write_ply(src, ["x"], [[1.0], [2.0]])
    with pytest.raises(ValueError):
        edit_ops._rewrite_ply_masked(src, tmp_path / "out.ply", [True])


def test_parse_ply_header_rejects_ascii_format(tmp_path: Path) -> None:
    p = tmp_path / "ascii.ply"
    p.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1.0\n")
    with pytest.raises(ValueError, match="binary_little_endian"):
        edit_ops._parse_ply_header(p)


def test_parse_ply_header_rejects_list_property(tmp_path: Path) -> None:
    p = tmp_path / "withface.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\n"
        "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    )
    p.write_bytes(header.encode("latin1") + struct.pack("<f", 1.0))
    with pytest.raises(ValueError):
        edit_ops._parse_ply_header(p)


def test_parse_ply_header_rejects_unknown_type(tmp_path: Path) -> None:
    p = tmp_path / "weird.ply"
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty weirdtype x\nend_header\n"
    p.write_bytes(header.encode("latin1") + b"\x00")
    with pytest.raises(ValueError, match="unknown"):
        edit_ops._parse_ply_header(p)


# =============================================================================
# 7. snapshot / prune / revert cycle
# =============================================================================


def test_snapshot_creates_version_and_manifest(outputs_root: Path) -> None:
    job_dir = _make_job(outputs_root, "splat_dead0004")
    vdir, manifest, created = edit_ops._snapshot_preview(job_dir, op="apply", params={"ops": ["noop"]})
    assert created is True
    assert vdir.is_dir()
    assert manifest["seq"] == 1
    assert manifest["op"] == "apply"
    assert manifest["files"] == ["splat.ply"]
    assert manifest["splat_ply_stat"] is not None
    versions = edit_ops._list_version_dirs(job_dir)
    assert len(versions) == 1
    assert versions[0][1]["seq"] == 1


def test_snapshot_dedupes_unchanged_preview_content(outputs_root: Path) -> None:
    """Two snapshots with NO mutation in between must not burn a version slot."""
    job_dir = _make_job(outputs_root, "splat_dead0009")
    d1, m1, c1 = edit_ops._snapshot_preview(job_dir, op="apply", params={})
    assert c1 is True and m1["seq"] == 1
    d2, m2, c2 = edit_ops._snapshot_preview(job_dir, op="apply", params={})
    assert c2 is False
    assert d2 == d1 and m2["seq"] == 1
    assert len(edit_ops._list_version_dirs(job_dir)) == 1
    # content change -> a real new snapshot again
    (job_dir / "_preview" / "splat.ply").write_bytes(b"CHANGED-CONTENT")
    d3, m3, c3 = edit_ops._snapshot_preview(job_dir, op="apply", params={})
    assert c3 is True and m3["seq"] == 2


def test_prune_versions_keeps_only_newest_cap(outputs_root: Path) -> None:
    job_dir = _make_job(outputs_root, "splat_dead0005")
    for i in range(6):
        (job_dir / "_preview" / "splat.ply").write_bytes(f"v{i}".encode())
        edit_ops._snapshot_preview(job_dir, op="apply", params={"i": i})
        # pruning is the caller's post-success step now, never snapshot's own
        edit_ops._prune_versions(job_dir)
    versions = edit_ops._list_version_dirs(job_dir)
    seqs = sorted(m["seq"] for _d, m in versions)
    assert len(versions) == edit_ops.MAX_VERSIONS
    assert seqs == [4, 5, 6]  # newest 3 of 6 snapshots kept (cap=3 from fixture)


def test_revert_restores_prior_content(outputs_root: Path, client: TestClient) -> None:
    job_id = "splat_dead0006"
    job_dir = _make_job(outputs_root, job_id)
    edit_ops._snapshot_preview(job_dir, op="apply", params={})  # v1, content = "PLYDATA-v0"

    (job_dir / "_preview" / "splat.ply").write_bytes(b"PLYDATA-v1-EDITED")

    resp = client.post(f"/api/splat/jobs/{job_id}/edit/revert", json={"version": 1})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert "splat.ply" in body["restored_files"]
    assert (job_dir / "_preview" / "splat.ply").read_bytes() == b"PLYDATA-v0"

    # revert itself snapshotted the pre-revert state -> 2 versions now exist
    versions = client.get(f"/api/splat/jobs/{job_id}/edit/versions").json()["versions"]
    assert len(versions) == 2


def test_revert_missing_version_404(outputs_root: Path, client: TestClient) -> None:
    job_id = "splat_dead0007"
    _make_job(outputs_root, job_id)
    resp = client.post(f"/api/splat/jobs/{job_id}/edit/revert", json={"version": 99})
    assert resp.status_code == 404


def test_revert_bad_job_id_404(outputs_root: Path, client: TestClient) -> None:
    resp = client.post("/api/splat/jobs/not-a-real-id/edit/revert", json={"version": 1})
    assert resp.status_code == 404


def test_list_versions_empty_for_fresh_job(outputs_root: Path, client: TestClient) -> None:
    job_id = "splat_dead0008"
    _make_job(outputs_root, job_id)
    resp = client.get(f"/api/splat/jobs/{job_id}/edit/versions")
    assert resp.status_code == 200
    assert resp.json()["versions"] == []


# =============================================================================
# 8. endpoint-level flows against a STUB splat-transform (SPLAT_TRANSFORM_BIN)
# =============================================================================

_TRANSLATE_OPS = {"ops": [{"type": "translate", "x": 1, "y": 0, "z": 0}]}


def test_apply_round_trip_success(outputs_root: Path, client: TestClient, stub_transform: Path) -> None:
    job_id = "splat_aaaa0001"
    job_dir = _make_job(outputs_root, job_id)
    src = job_dir / "_preview" / "splat.ply"
    src.write_bytes(b"PLY-CONTENT-ORIGINAL")

    resp = client.post(f"/api/splat/jobs/{job_id}/edit/apply", json=_TRANSLATE_OPS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["warnings"] == []
    assert body["version_before"] == 1
    # stub copies input -> output, so content round-trips through tmp+replace
    assert src.read_bytes() == b"PLY-CONTENT-ORIGINAL"
    # snapshot v1 preserves the pre-edit scene
    versions = edit_ops._list_version_dirs(job_dir)
    assert len(versions) == 1
    assert (versions[0][0] / "splat.ply").read_bytes() == b"PLY-CONTENT-ORIGINAL"
    # derived artifacts regenerated from the edited splat.ply
    assert (job_dir / "_preview" / "splat.spz").is_file()
    assert (job_dir / "_preview" / "web.ply").is_file()
    # no tmp litter left behind
    assert list((job_dir / "_preview").glob("*.edit-tmp")) == []


def test_apply_failure_leaves_source_and_removes_snapshot(
    outputs_root: Path, client: TestClient, stub_transform: Path
) -> None:
    (stub_transform / "fail_all").write_text("1")
    job_id = "splat_aaaa0002"
    job_dir = _make_job(outputs_root, job_id)
    src = job_dir / "_preview" / "splat.ply"
    src.write_bytes(b"PLY-CONTENT-ORIGINAL")

    resp = client.post(f"/api/splat/jobs/{job_id}/edit/apply", json=_TRANSLATE_OPS)
    assert resp.status_code == 500
    assert "splat-transform failed" in resp.json()["detail"]
    # source untouched
    assert src.read_bytes() == b"PLY-CONTENT-ORIGINAL"
    # the failed op's snapshot was discarded — repeated failures can't churn the cap
    assert edit_ops._list_version_dirs(job_dir) == []
    assert list((job_dir / "_preview").glob("*.edit-tmp")) == []


def test_apply_regen_failure_unlinks_stale_artifacts(
    outputs_root: Path, client: TestClient, stub_transform: Path
) -> None:
    """If .spz/web.ply/langweb.ply regeneration fails, the STALE pre-edit copies must
    be removed (splat_route's fmt fallback only triggers on a MISSING file) and the
    warnings must say so."""
    (stub_transform / "fail_regen").write_text("1")
    job_id = "splat_aaaa0003"
    job_dir = _make_job(outputs_root, job_id)
    preview = job_dir / "_preview"
    (preview / "splat.spz").write_bytes(b"STALE-SPZ")
    (preview / "web.ply").write_bytes(b"STALE-WEB")
    (preview / "langweb.ply").write_bytes(b"STALE-LANGWEB")
    lf = job_dir / "_langfield"
    lf.mkdir()
    (lf / "gauss_emb.npz").write_bytes(b"stub")  # language field present -> langweb applies

    resp = client.post(f"/api/splat/jobs/{job_id}/edit/apply", json=_TRANSLATE_OPS)
    assert resp.status_code == 200, resp.text
    warnings = resp.json()["warnings"]
    assert any("splat.spz removed" in w for w in warnings)
    assert any("web.ply removed" in w for w in warnings)
    assert any("langweb.ply removed" in w for w in warnings)
    # the stale artifacts are really gone, so the missing-file fallbacks are real
    assert not (preview / "splat.spz").exists()
    assert not (preview / "web.ply").exists()
    assert not (preview / "langweb.ply").exists()


def test_semantic_delete_dequantizes_with_worker_headers(
    outputs_root: Path, client: TestClient, stub_transform: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end semantic delete with a mocked worker response INCLUDING the
    X-Min=0.4/X-Max=0.6 dequantization headers. Bytes [0,64,128,191,255] at
    threshold 0.45 must match rows 1-4 (dequantized 0.4502..0.6) and keep ONLY
    row 0 — the old absolute q/255 decode would have kept rows 0 AND 1."""
    job_id = "splat_bbbb0001"
    job_dir = _make_job(outputs_root, job_id)
    ply = job_dir / "_preview" / "splat.ply"
    _write_ply(ply, ["x", "y", "z"], [[float(i), 0.0, 0.0] for i in range(5)])
    lf = job_dir / "_langfield"
    lf.mkdir()
    (lf / "gauss_emb.npz").write_bytes(b"stub")
    (job_dir / "config.yml").write_text("stub: true\n")

    async def fake_has() -> bool:
        return True

    async def fake_post(config_path: str, lfdir: str, clean_text: str) -> httpx.Response:
        return httpx.Response(
            200,
            content=bytes([0, 64, 128, 191, 255]),
            headers={"X-Count": "5", "X-Min": "0.4", "X-Max": "0.6"},
        )

    monkeypatch.setattr(edit_ops, "_worker_has_relevancy_endpoint", fake_has)
    monkeypatch.setattr(edit_ops, "_post_worker_relevancy", fake_post)

    payload = {"text": "chair", "threshold": 0.45, "mode": "delete", "cleanup": False}
    resp = client.post(f"/api/splat/jobs/{job_id}/edit/semantic", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["matched"] == 4
    assert body["kept"] == 1
    assert body["cleanup"] is False
    assert body["rows_before_cleanup"] == 1
    assert body["rows_after_cleanup"] == 1
    assert body["language_field_stale"] is True
    n_out, rows_out = _read_ply_rows(ply)
    assert n_out == 1
    assert rows_out[0][0] == 0.0  # exactly the sub-threshold gaussian survived

    # the edit marked the field STALE -> a second semantic edit is refused loudly
    resp2 = client.post(f"/api/splat/jobs/{job_id}/edit/semantic", json=payload)
    assert resp2.status_code == 409
    assert "stale" in resp2.json()["detail"]


def test_semantic_rejects_stale_language_field(
    outputs_root: Path, client: TestClient, stub_transform: Path
) -> None:
    job_id = "splat_bbbb0002"
    job_dir = _make_job(outputs_root, job_id)
    lf = job_dir / "_langfield"
    lf.mkdir()
    (lf / "gauss_emb.npz").write_bytes(b"stub")
    (lf / "STALE").write_text("2026-07-04\n")
    resp = client.post(
        f"/api/splat/jobs/{job_id}/edit/semantic",
        json={"text": "chair", "threshold": 0.5, "mode": "delete"},
    )
    assert resp.status_code == 409
    assert "stale" in resp.json()["detail"]


def test_concurrent_applies_second_gets_409(
    outputs_root: Path, stub_transform: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two overlapping mutating POSTs on ONE job: exactly one proceeds, the other is
    rejected immediately with 409 (non-blocking per-job lock, no queueing)."""
    monkeypatch.setenv("STUB_SLEEP", "0.3")  # make the winning transform slow enough to overlap
    job_id = "splat_cccc0001"
    _make_job(outputs_root, job_id)
    app = FastAPI()
    app.include_router(edit_ops.router, prefix="/api/splat")

    async def main() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            url = f"/api/splat/jobs/{job_id}/edit/apply"
            r1, r2 = await asyncio.gather(c.post(url, json=_TRANSLATE_OPS), c.post(url, json=_TRANSLATE_OPS))
            return r1, r2

    r1, r2 = asyncio.run(main())
    statuses = sorted([r1.status_code, r2.status_code])
    assert statuses == [200, 409], (r1.text, r2.text)
    loser = r1 if r1.status_code == 409 else r2
    assert "edit is already in progress" in loser.json()["detail"]
