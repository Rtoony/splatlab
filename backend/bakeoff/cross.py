"""Cross-pipeline scoring: a model reconstructed by ANOTHER SfM (its own frame,
scale and gauge) rendered at SplatLab's own held-out cameras.

The two reconstructions of one capture share instants, so their camera
centres pair up one-to-one; a least-squares similarity (Umeyama) between the
paired centres maps SplatLab's frame into the other model's frame. The CAMERAS
are moved, never the splats — a camera needs only a rotation and a centre, so
the other model's SH bands stay untouched. Projection is scale-invariant, so
the intrinsics carry over unchanged.
"""
from __future__ import annotations

import numpy as np

GL_TO_CV = np.diag([1.0, -1.0, -1.0])   # nerfstudio/OpenGL camera axes → OpenCV (x right, y down, z forward)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) minimising ||dst - (s R src + t)||² over paired rows."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need >= 3 paired points of equal shape")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    e = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[2, 2] = -1
    r = u @ e @ vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ e) / var)
    t = mu_d - s * r @ mu_s
    return s, r, t


def align_report(src: np.ndarray, dst: np.ndarray, s: float, r: np.ndarray, t: np.ndarray) -> dict:
    res = np.linalg.norm(dst - (s * src @ r.T + t), axis=1)
    extent = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    return {"pairs": int(len(src)), "scale": s, "rmse": float(np.sqrt((res ** 2).mean())),
            "median": float(np.median(res)), "p95": float(np.percentile(res, 95)),
            "extent": extent, "rmse_over_extent": float(np.sqrt((res ** 2).mean()) / extent)}


def c2w_gl_to_viewmat(c2w_gl: np.ndarray, s: float = 1.0, r: np.ndarray | None = None,
                      t: np.ndarray | None = None) -> np.ndarray:
    """A nerfstudio (OpenGL) camera-to-world in frame F → an OpenCV world-to-camera
    4x4 in frame G, where points map F → G as x_G = s R x_F + t."""
    r = np.eye(3) if r is None else np.asarray(r, float)
    t = np.zeros(3) if t is None else np.asarray(t, float)
    c2w = np.asarray(c2w_gl, float)
    rot_cv = c2w[:3, :3] @ GL_TO_CV
    centre = c2w[:3, 3]
    rot_g = r @ rot_cv
    centre_g = s * r @ centre + t
    view = np.eye(4)
    view[:3, :3] = rot_g.T
    view[:3, 3] = -rot_g.T @ centre_g
    return view


def colmap_centre(qvec_wxyz: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP world-to-camera (q, t) → camera centre in world."""
    w, x, y, z = np.asarray(qvec_wxyz, float) / np.linalg.norm(qvec_wxyz)
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return -rot.T @ np.asarray(tvec, float)
