"""Lossless Gaussian exports and descriptive, non-survey reconstruction review."""

from html import escape
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Review distributions need nonempty finite values")
    return dict(zip(("minimum", "p01", "median", "p99", "maximum"), map(float, np.quantile(values, [0, .01, .5, .99, 1]))))


def export_splats(destination, parameters):
    destination = Path(destination)
    required = {"means": 3, "scales": 3, "quats": 4, "features_dc": 3, "opacities": 1}
    arrays = {name: np.asarray(parameters[name]) for name in (*required, "features_rest")}
    if any(values.dtype != np.dtype("float32") for values in arrays.values()):
        raise ValueError("Lossless review requires float32 checkpoint parameters")
    count = len(arrays["means"])
    if not 1 <= count <= 3_000_000:
        raise ValueError("Review supports one to three million Gaussian rows")
    for name, width in required.items():
        if arrays[name].shape != (count, width):
            raise ValueError(f"Invalid Gaussian {name} dimensions")
    rest = arrays["features_rest"]
    if rest.ndim != 3 or rest.shape[0] != count or rest.shape[1] not in {0, 3, 8, 15} or rest.shape[2] != 3:
        raise ValueError("Unsupported spherical-harmonic dimensions")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Nonfinite Gaussian parameters cannot be silently filtered")
    norms = np.linalg.norm(arrays["quats"].astype(np.float64), axis=1)
    if np.any(norms <= 1e-8):
        raise ValueError("Degenerate Gaussian rotation")
    logs = arrays["scales"].astype(np.float64)
    if np.any(logs < np.log(np.finfo(np.float32).tiny)) or np.any(logs > np.log(np.finfo(np.float32).max)):
        raise ValueError("Gaussian scale cannot be represented safely")
    scales = np.exp(logs)
    aspect = np.exp(logs.max(axis=1) - logs.min(axis=1))
    opacity_logits = arrays["opacities"][:, 0].astype(np.float64)
    opacity = np.exp(-np.logaddexp(0, -opacity_logits))
    fields = [(name, "<f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
    fields += [(f"f_dc_{index}", "<f4") for index in range(3)]
    fields += [(f"f_rest_{index}", "<f4") for index in range(rest.shape[1] * 3)]
    fields += [("opacity", "<f4")]
    fields += [(f"scale_{index}", "<f4") for index in range(3)]
    fields += [(f"rot_{index}", "<f4") for index in range(4)]
    vertices = np.zeros(count, dtype=fields)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = arrays["means"][:, index]
    for index in range(3):
        vertices[f"f_dc_{index}"] = arrays["features_dc"][:, index]
        vertices[f"scale_{index}"] = arrays["scales"][:, index]
    packed_rest = rest.transpose(0, 2, 1).reshape(count, -1)
    for index in range(packed_rest.shape[1]):
        vertices[f"f_rest_{index}"] = packed_rest[:, index]
    for index in range(4):
        vertices[f"rot_{index}"] = arrays["quats"][:, index]
    vertices["opacity"] = arrays["opacities"][:, 0]
    statistics = {"rows": count, "sh_degree": int(np.sqrt(rest.shape[1] + 1)) - 1,
                  "row_policy": "all checkpoint rows retained in original order; no opacity culling or quantization",
                  "coordinate_scope": "normalized training frame, not established metres or survey coordinates",
                  "normal_scope": "zero PLY placeholders, not measured surface normals",
                  "minimum": arrays["means"].min(axis=0).tolist(), "maximum": arrays["means"].max(axis=0).tolist(),
                  "opacity": distribution(opacity), "scale_training_units": distribution(scales),
                  "axis_ratio": distribution(aspect), "quaternion_norm": distribution(norms),
                  "opacity_below_1_over_255_rows": int((opacity < 1 / 255).sum()),
                  "geometry_error": None, "geometry_error_reason": "Parameter distributions are not surface accuracy or floater measurements"}
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {count}\n"
    header += "".join(f"property float {name}\n" for name, _dtype in fields) + "end_header\n"
    with destination.open("xb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)
    return statistics


def depth_statistics(path):
    depth = np.load(path, allow_pickle=False, mmap_mode="r")
    if depth.ndim not in {2, 3} or depth.ndim == 3 and depth.shape[-1] != 1 or not depth.size or depth.size > 16_777_216 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("Invalid retained depth dimensions or dtype")
    finite = np.isfinite(depth)
    valid = finite & (depth > 0)
    return {"pixels": depth.size, "finite_fraction": float(finite.mean()), "positive_fraction": float(valid.mean()),
            "positive_depth_training_units": distribution(depth[valid]) if valid.any() else None,
            "scope": "rendered expected depth, not an independent geometric reference or visibility mask"}


def review_html(result, runs):
    names = list(result["metrics"])
    rows = []
    for name in names:
        metrics = result["metrics"][name]
        ssim = "n/a" if metrics["ssim"] is None else f"{metrics['ssim']:.4f}"
        rows.append(f"<tr><th>{escape(name)}</th><td>{metrics['psnr_db']:.3f}</td><td>{ssim}</td>"
                    f"<td>{metrics['gaussians']:,}</td><td>{metrics['training_seconds']:.1f}</td>"
                    f"<td><a href='{escape(name)}/model.ply'>Full SH splat</a></td></tr>")
    sections = []
    for view in runs[names[0]]["evaluation"]["views"]:
        filename = Path(view["image"]).stem
        figures = [("Reference", f"../{names[0]}/renders/{filename}-reference.png")]
        figures += [(name, f"../{name}/renders/{filename}.png") for name in names]
        content = "".join(f"<figure><figcaption>{escape(label)}</figcaption><a href='{escape(path)}'><img loading='lazy' src='{escape(path)}' alt='{escape(label)} at {escape(filename)}'></a></figure>" for label, path in figures)
        sections.append(f"<section><h2>Held-out view {escape(filename)}</h2><div class='views'>{content}</div></section>")
    return "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" + """
<title>SplatLab reconstruction review</title>
<style>
body{font:16px system-ui;background:#10141b;color:#e8edf5;margin:24px}a{color:#8bc8ff}
h1{font-size:1.7rem}.notice{padding:16px;border:1px solid #ba924e;border-radius:8px;max-width:1000px}
.table{overflow:auto}table{border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #46515f}
.views{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto}figcaption{padding:8px}
section{margin-top:32px}@media(max-width:650px){body{margin:12px}.views{grid-template-columns:1fr}}
</style><h1>Paired reconstruction review</h1>
<p class='notice'>Private research outputs — needs human review; no method promoted. All images use the same held-out cameras.
Opacity and parameter distributions do not measure geometric accuracy. Exports retain all SH coefficients and checkpoint rows,
in normalized training coordinates, not survey metres. Rendering differences between external viewers remain unvalidated.</p>
""" + f"<p>{result['held_out_views']} held-out views; {result['budget']['iterations_per_arm']:,} iterations per arm. <a href='review.json'>Checksummed review and descriptive geometry statistics</a></p>" + "<div class='table'><table><tr><th>Method</th><th>PSNR dB</th><th>SSIM</th><th>Gaussians</th><th>Training seconds</th><th>Export</th></tr>" + "".join(rows) + "</table></div>" + "".join(sections) + "</html>"
