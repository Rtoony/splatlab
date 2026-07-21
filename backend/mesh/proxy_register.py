#!/usr/bin/env python3
"""P5c: register a generated proxy splat onto an isolated scene object.

Bbox-similarity initialization (robust p1-p99), then Open3D point-to-point ICP
WITH scaling to refine — the naive init alone left a ~40 cm offset on the
garden table (debris-pulled centroids). The full similarity transform is
applied to the RAW 3DGS fields: xyz affine, per-axis log-scales += ln(s),
quaternions composed with the ICP rotation (wxyz convention).

Runs in the dn-splatter-probe env.

Usage: proxy_register.py <proxy_splat.ply> <target_object.ply> <out.ply>
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement


def _xyz(v):
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both [N,4] or [4] wxyz."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:  # rare near-180° case; adequate for ICP-scale corrections
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4.0
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
        return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def main() -> int:
    proxy_path, target_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    proxy = PlyData.read(proxy_path)
    target = PlyData.read(target_path)
    pv, tv = proxy["vertex"], target["vertex"]
    pxyz, txyz = _xyz(pv), _xyz(tv)

    # Init: robust bbox similarity.
    plo, phi = np.percentile(pxyz, 1, axis=0), np.percentile(pxyz, 99, axis=0)
    tlo, thi = np.percentile(txyz, 1, axis=0), np.percentile(txyz, 99, axis=0)
    s1 = float(np.linalg.norm(thi - tlo) / max(np.linalg.norm(phi - plo), 1e-9))
    t1 = (tlo + thi) / 2 - s1 * (plo + phi) / 2
    init_xyz = pxyz * s1 + t1

    # Refine: ICP with scaling on downsampled clouds.
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(init_xyz)).voxel_down_sample(0.02)
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(txyz)).voxel_down_sample(0.02)
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=0.12,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    M = np.asarray(icp.transformation)          # [s2*R2 | t2]
    sR, t2 = M[:3, :3], M[:3, 3]
    s2 = float(np.cbrt(max(np.linalg.det(sR), 1e-12)))
    R2 = sR / s2

    # Compose the FULL similarity on original proxy coords:
    # x' = s2 R2 (s1 x + t1) + t2  =  (s1 s2) R2 x + (s2 R2 t1 + t2)
    s_total = s1 * s2
    t_total = s2 * (R2 @ t1) + t2
    new_xyz = (s_total * (pxyz @ R2.T) + t_total).astype(np.float32)

    data = np.array(pv.data)
    data["x"], data["y"], data["z"] = new_xyz[:, 0], new_xyz[:, 1], new_xyz[:, 2]
    lns = np.float32(np.log(s_total))
    for sc in ("scale_0", "scale_1", "scale_2"):
        if sc in data.dtype.names:
            data[sc] = (np.asarray(pv[sc]) + lns).astype(np.float32)
    if all(f"rot_{i}" in data.dtype.names for i in range(4)):
        q = np.stack([pv[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
        qR = _rot_to_quat_wxyz(R2)
        q_new = _quat_mul_wxyz(qR[None, :], q)
        for i in range(4):
            data[f"rot_{i}"] = q_new[:, i].astype(np.float32)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(out_path)

    report = {
        "gaussians": int(len(data)),
        "init_scale": round(s1, 4),
        "icp_fitness": round(float(icp.fitness), 3),
        "icp_rmse": round(float(icp.inlier_rmse), 4),
        "icp_scale": round(s2, 4),
        "icp_rotation_deg": round(float(np.degrees(np.arccos(
            np.clip((np.trace(R2) - 1) / 2, -1, 1)))), 2),
        "total_scale": round(s_total, 4),
    }
    Path(out_path).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
