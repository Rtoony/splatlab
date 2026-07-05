"""Client-side language heatmap backend: quantization wire format, the langweb
artifact (row-order-preserving SH strip), the app-side /langfield/relevancy proxy,
and a skippable integration probe that re-verifies splat-transform's order
preservation on a synthetic 3DGS .ply.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import langfield_worker  # noqa: E402
import splat_route  # noqa: E402


# ---------------------------------------------------------------------------
# quantize_relevancy — the wire format the client dequantizes
# ---------------------------------------------------------------------------


def _dequantize(payload: bytes, rmin: float, rmax: float) -> np.ndarray:
    q = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
    return rmin + q / 255.0 * (rmax - rmin)


def test_quantize_roundtrip_within_half_step():
    rng = np.random.default_rng(42)
    rel = rng.uniform(0.0, 1.0, size=5000).astype(np.float32)
    payload, rmin, rmax = langfield_worker.quantize_relevancy(rel)
    assert len(payload) == rel.size
    assert rmin == pytest.approx(float(rel.min()))
    assert rmax == pytest.approx(float(rel.max()))
    back = _dequantize(payload, rmin, rmax)
    half_step = (rmax - rmin) / 255.0 / 2.0
    assert np.max(np.abs(back - rel)) <= half_step + 1e-6


def test_quantize_extremes_map_to_0_and_255():
    rel = np.array([0.2, 0.9, 0.55], dtype=np.float32)
    payload, rmin, rmax = langfield_worker.quantize_relevancy(rel)
    q = np.frombuffer(payload, dtype=np.uint8)
    assert q[0] == 0 and q[1] == 255
    assert (rmin, rmax) == (pytest.approx(0.2), pytest.approx(0.9))


def test_quantize_constant_vector_roundtrips_to_constant():
    rel = np.full(17, 0.5, dtype=np.float32)
    payload, rmin, rmax = langfield_worker.quantize_relevancy(rel)
    assert len(payload) == 17
    assert rmin == rmax == pytest.approx(0.5)
    assert np.array_equal(_dequantize(payload, rmin, rmax), rel)


def test_quantize_empty_vector():
    payload, rmin, rmax = langfield_worker.quantize_relevancy(np.array([], dtype=np.float32))
    assert payload == b"" and rmin == 0.0 and rmax == 0.0


def test_worker_relevancy_503_when_heavy_deps_missing():
    # Under the test interpreter torch/nerfstudio are absent, so the worker app
    # must refuse /relevancy with a 503 (never a crash) — same as /query.
    client = TestClient(langfield_worker.app)
    resp = client.post(
        "/relevancy", json={"config": "/nope/config.yml", "lfdir": "/nope", "text": "keys"}
    )
    assert resp.status_code == 503
    assert "not ready" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# langweb artifact — command builder
# ---------------------------------------------------------------------------


def test_langweb_command_strips_harmonics_without_decimate(tmp_path: Path):
    cmd = splat_route._langweb_command("/opt/bin/splat-transform", tmp_path)
    assert cmd == [
        "/opt/bin/splat-transform",
        str(tmp_path / "_preview" / "splat.ply"),
        "--filter-harmonics",
        "0",
        str(tmp_path / "_preview" / "langweb.ply"),
    ]
    # Decimation reorders/merges rows and breaks the gauss_emb.npz row match —
    # guard that it can never sneak into this specific artifact's argv.
    assert "--decimate" not in cmd


# ---------------------------------------------------------------------------
# app-side routes: fmt=langweb + /langfield/relevancy proxy
# ---------------------------------------------------------------------------


JOB_ID = "splat_c0ffee"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _make_langfield_job(outputs: Path, job_id: str = JOB_ID) -> Path:
    """A finished language-field job on disk: preview splat.ply + gauss_emb.npz
    sidecar + a nerfstudio config.yml (what _find_latest_config looks for)."""
    job_dir = outputs / job_id
    preview = job_dir / "_preview"
    preview.mkdir(parents=True)
    (preview / "splat.ply").write_bytes(b"raw-ply-bytes")
    lfdir = job_dir / "_langfield"
    lfdir.mkdir()
    (lfdir / "gauss_emb.npz").write_bytes(b"npz")
    run_dir = job_dir / "processed" / "splatfacto" / "2026-07-04_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yml").write_text("method_name: splatfacto\n")
    return job_dir


def test_preview_fmt_langweb_serves_artifact_when_present(client: tuple[TestClient, Path]):
    http, outputs = client
    job_dir = _make_langfield_job(outputs)
    (job_dir / "_preview" / "langweb.ply").write_bytes(b"langweb-ply-bytes")
    resp = http.get(f"/api/splat/jobs/{JOB_ID}/preview/file", params={"fmt": "langweb"})
    assert resp.status_code == 200
    assert resp.content == b"langweb-ply-bytes"
    assert resp.headers["content-disposition"].endswith(f'{JOB_ID}.ply"')


def test_preview_fmt_langweb_falls_back_to_raw_ply(client: tuple[TestClient, Path]):
    http, outputs = client
    _make_langfield_job(outputs)  # no langweb.ply written
    resp = http.get(f"/api/splat/jobs/{JOB_ID}/preview/file", params={"fmt": "langweb"})
    assert resp.status_code == 200
    assert resp.content == b"raw-ply-bytes"


def test_relevancy_proxy_503_when_worker_down(client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch):
    http, outputs = client
    _make_langfield_job(outputs)

    async def _down(config_path: str, lfdir: str, clean_text: str):
        return None  # what the helper returns on connect error / non-200

    monkeypatch.setattr(splat_route, "_langfield_worker_relevancy", _down)
    resp = http.post(f"/api/splat/jobs/{JOB_ID}/langfield/relevancy", json={"text": "keys"})
    assert resp.status_code == 503
    assert "warm worker" in resp.json()["detail"]


def test_relevancy_proxy_forwards_body_and_headers(client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch):
    http, outputs = client
    _make_langfield_job(outputs)
    payload = bytes([0, 128, 255, 42])
    matches = {"focus": [0.0, 1.0, 2.0], "radius": 0.5, "matches": []}
    seen: dict[str, str] = {}

    async def _warm(config_path: str, lfdir: str, clean_text: str):
        seen["config"] = config_path
        seen["lfdir"] = lfdir
        seen["text"] = clean_text
        return httpx.Response(
            200,
            content=payload,
            headers={
                "X-Count": "4",
                "X-Min": "0.1",
                "X-Max": "0.9",
                "X-Matches": json.dumps(matches, separators=(",", ":")),
                "X-Internal-Junk": "must-not-leak",
            },
        )

    monkeypatch.setattr(splat_route, "_langfield_worker_relevancy", _warm)
    resp = http.post(f"/api/splat/jobs/{JOB_ID}/langfield/relevancy", json={"text": "my keys!"})
    assert resp.status_code == 200
    assert resp.content == payload
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["x-count"] == "4"
    assert resp.headers["x-min"] == "0.1"
    assert resp.headers["x-max"] == "0.9"
    assert json.loads(resp.headers["x-matches"]) == matches
    assert "x-internal-junk" not in resp.headers
    # the proxy sanitized the query text before forwarding (same as /query)
    assert seen["text"] == "my keys"
    assert seen["lfdir"].endswith("_langfield")


def test_relevancy_proxy_404_without_language_field(client: tuple[TestClient, Path]):
    http, outputs = client
    job_dir = outputs / JOB_ID
    (job_dir / "_preview").mkdir(parents=True)  # a job with no _langfield sidecar
    resp = http.post(f"/api/splat/jobs/{JOB_ID}/langfield/relevancy", json={"text": "keys"})
    assert resp.status_code == 404


def test_relevancy_proxy_400_on_empty_text(client: tuple[TestClient, Path]):
    http, outputs = client
    _make_langfield_job(outputs)
    resp = http.post(f"/api/splat/jobs/{JOB_ID}/langfield/relevancy", json={"text": "   "})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# integration probe: splat-transform --filter-harmonics 0 preserves row order
# ---------------------------------------------------------------------------


N_ROWS = 256
GS_FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)  # 62 float32 props — the exact splatfacto ns-export column layout


def _write_gs_ply(path: Path, data: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {data.shape[0]}\n"
        + "".join(f"property float {name}\n" for name in GS_FIELDS)
        + "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())


def _read_binary_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        fields: list[tuple[str, str]] = []
        count = 0
        while True:
            line = f.readline().decode("ascii").strip()
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
            elif parts and parts[0] == "property":
                fields.append((parts[2], "<f4"))
            elif line == "end_header":
                break
        dtype = np.dtype(fields)
        return np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)


@pytest.mark.skipif(
    splat_route._splat_transform_path() is None,
    reason="splat-transform binary not available",
)
def test_splat_transform_harmonics_strip_preserves_row_order(tmp_path: Path):
    """The design invariant behind langweb.ply: --filter-harmonics 0 (no decimate)
    must keep row i == row i so client relevancy indexes line up with the splat."""
    rng = np.random.default_rng(7)
    data = rng.uniform(-1.0, 1.0, size=(N_ROWS, len(GS_FIELDS))).astype(np.float32)
    # distinct, monotonically tagged positions so any reorder is unmissable
    data[:, 0] = np.arange(N_ROWS, dtype=np.float32) / 16.0
    src = tmp_path / "in.ply"
    dst = tmp_path / "langweb.ply"
    _write_gs_ply(src, data)

    cmd = splat_route._langweb_command(splat_route._splat_transform_path(), tmp_path)
    # point the builder's fixed _preview paths at our tmp files
    cmd[1], cmd[-1] = str(src), str(dst)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    out = _read_binary_ply(dst)
    assert len(out) == N_ROWS                      # full count — nothing decimated
    assert not any(n.startswith("f_rest_") for n in out.dtype.names)  # SH stripped
    for col_idx, col in enumerate(("x", "y", "z")):
        assert np.array_equal(out[col], data[:, col_idx]), f"row order changed in {col}"
