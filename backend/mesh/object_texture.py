#!/usr/bin/env python3
"""Simplified, TEXTURED Blender asset from a splat capture.

twin_finish.py already turns an object mesh + its gaussians into a colored,
decimated GLB — but the colour lives in VERTEX COLOURS, so fidelity is bound to
the vertex budget. Decimating to a genuinely simple asset (a few thousand faces)
therefore throws the capture's colour away with the geometry: the two are
coupled. This stage decouples them. Geometry is simplified as far as you like;
colour is baked into a UV texture map sampled from the gaussians, so a 5k-face
model can still carry 1k-2k of colour detail.

Pipeline (order is load-bearing, see the notes at each step):
  crop out the ground -> repair -> drop floater components
  -> screened-Poisson refit -> Taubin smooth -> quadric decimate
  -> xatlas UV unwrap -> rasterize the atlas
  -> per-texel gaussian colour (adaptive kernel, outward-side filtered)
  -> seam dilate -> GLB with baseColorTexture -> MANDATORY readback

On xatlas: twin_finish.py's docstring records that chart generation does NOT
converge on raw fragmented TSDF meshes (lab finding, >1 h runs killed). That
finding stands and is exactly why unwrapping happens LAST here — after
reconstruction and decimation have reduced the input to a small, connected
surface. Unwrapping an 8k-face Poisson component is a categorically different
problem from unwrapping a 200k-face 50-cluster raw fusion, and measures 0.2-0.8 s.

Degrades honestly: if unwrap or bake fails, the GLB is still written with
vertex colours and the report says texture.baked=false plus the reason. It
never claims a texture it did not produce, and the readback re-opens the
written GLB to prove the UVs and image actually survived export.

Measured end to end on three objects across two captures (2026-07-25):
fire-hydrant 0.85 m, round-wooden-table 1.6 m, garden flower-vase — 5-9 s each.

Dense visual shells (200k-600k faces) are a different regime and were measured
separately on 2026-07-26. The bake is flat there — ~5 s at any face count, see
bake_texture — so the remaining cost is the UV unwrap, which is superlinear and
dominates everything else past ~200k faces. See the note at the unwrap call.

Usage: object_texture.py <mesh.ply> <splat.ply> <out.glb>
       [--meters-per-unit MPU] [--target-faces 8000] [--texture-size 1024]
       [--smooth] [--smooth-iterations 2] [--smooth-feature-deg 40.0]
       [--min-component-frac 0.02] [--crop-bbox JSON | --no-crop]
       [--crop-margin 0.15] [--no-reconstruct] [--poisson-depth 8]
       [--report PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

C0 = 0.28209479177387814  # SH DC -> RGB


def _stage(msg: str, t0: float) -> float:
    """Progress to stderr with elapsed time. A silent multi-minute filter is
    indistinguishable from a hang; this stage ran 10 min unattended once."""
    now = time.time()
    print(f"  [{now - t0:6.1f}s] {msg}", file=sys.stderr, flush=True)
    return now


def load_solid_gaussians(splat_ply: Path):
    """Same convention as twin_finish.py / batch_isolate.py: SH DC term to RGB,
    keep only gaussians the model is actually confident about."""
    v = PlyData.read(str(splat_ply))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    rgb = np.clip(0.5 + C0 * np.stack(
        [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32), 0, 1)
    solid = opac > 0.5
    return xyz[solid], rgb[solid]


def crop_to_box(mesh, lo, hi, margin: float):
    """Keep only faces inside an axis-aligned box, expanded by `margin` of its
    own size on every side.

    This is what removes the ground. A TSDF object mesh is fused from a
    generous crop, so it welds in the sidewalk/floor the object stands on —
    which is not the object and which wrecks both the silhouette and the face
    budget. object.json's bbox_tight is the SEMANTIC bbox (the langfield-
    selected gaussians), so it already excludes ground by construction; feed it
    in and the fused ground falls away. Same frame as mesh.ply (scene units).
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    pad = (hi - lo) * margin
    lo, hi = lo - pad, hi + pad
    v = np.asarray(mesh.vertices)
    inside = np.all((v >= lo) & (v <= hi), axis=1)
    keep = inside[mesh.faces].all(axis=1)
    return keep, lo, hi


def mesh_from_points(xyz, depth: int = 8, trim_pct: float = 8.0, knn: int = 32):
    """Screened-Poisson surface straight from a gaussian point cloud.

    Used when there is no TSDF mesh to start from — which is the normal case
    for scene-lane instances, since batch_isolate.py claims gaussians into
    object.ply and never meshes them.

    orient_normals_consistent_tangent_plane is not optional: estimate_normals
    returns normals with arbitrary sign, and Poisson interprets that sign as
    which side is 'inside'. Left unoriented it reconstructs inside-out shards
    instead of a surface. Density trimming afterwards removes the closure
    Poisson invents wherever the capture saw nothing.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(knn)
    rec, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=False)
    dens = np.asarray(dens)
    if len(dens) and float(np.ptp(dens)) > 0:
        rec.remove_vertices_by_mask(dens < np.quantile(dens, trim_pct / 100.0))
        rec.remove_unreferenced_vertices()
    return trimesh.Trimesh(vertices=np.asarray(rec.vertices),
                           faces=np.asarray(rec.triangles), process=False)


def clean_and_simplify(mesh, target_faces: int, min_component_frac: float,
                       smooth: bool, smooth_iters: int, smooth_feature_deg: float,
                       remesh: bool = True, reconstruct: bool = True,
                       poisson_depth: int = 8, poisson_trim_pct: float = 8.0,
                       sample_points: int = 120_000):
    """repair -> drop floaters -> Poisson refit -> smooth -> decimate.

    Reconstruction is THE quality step. A TSDF object mesh is an open shell full
    of holes, slivers and detached boundary shards; simplifying it only yields a
    smaller torn shell (live-checked on the hydrant: crop + remesh + decimate
    still rendered as flapping fragments). Screened Poisson refits ONE smooth
    manifold surface through points sampled off that shell, which is what makes
    the output read as a solid object rather than confetti.

    Open3D, not pymeshlab, does the reconstruction: pymeshlab's
    generate_surface_reconstruction_screened_poisson did not return within 240 s
    on this 49k-face input even at depth 6 with preclean off (probed
    2026-07-25), while Open3D produced 197k faces in 1.5 s at depth 8. Normals
    come from the triangle mesh itself rather than being estimated from a bare
    point cloud, so orientation is exact and Poisson does not invert patches.

    Density trimming is mandatory: Poisson always closes a surface, so regions
    the capture never saw get invented. Dropping the lowest-density vertices
    removes that fabrication instead of shipping it as geometry.
    """
    import open3d as o3d

    tm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)))
    stats = {"faces_in": int(len(tm.triangles))}
    t_s = time.time()

    tm.remove_duplicated_vertices()
    tm.remove_duplicated_triangles()
    tm.remove_degenerate_triangles()
    tm.remove_unreferenced_vertices()
    stats["faces_after_repair"] = int(len(tm.triangles))

    def drop_small_components(m, frac):
        if frac <= 0 or len(m.triangles) == 0:
            return m
        labels, counts, _ = m.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        if len(counts) <= 1:
            return m
        keep_min = max(4, int(counts.max() * frac))
        drop = np.isin(labels, np.flatnonzero(counts < keep_min))
        if drop.any() and not drop.all():
            m.remove_triangles_by_mask(drop)
            m.remove_unreferenced_vertices()
        return m

    tm = drop_small_components(tm, min_component_frac)
    stats["faces_after_cleanup"] = int(len(tm.triangles))
    t_s = _stage(f"repair+cleanup -> {len(tm.triangles)} faces", t_s)

    reconstructed = False
    if reconstruct and len(tm.triangles) >= 100:
        try:
            tm.compute_vertex_normals()
            pcd = tm.sample_points_uniformly(number_of_points=sample_points)
            rec, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth, linear_fit=False)
            dens = np.asarray(dens)
            faces_raw = int(len(rec.triangles))
            if len(dens) and float(np.ptp(dens)) > 0:
                rec.remove_vertices_by_mask(dens < np.quantile(dens, poisson_trim_pct / 100.0))
                rec.remove_unreferenced_vertices()
            rec = drop_small_components(rec, 0.05)
            if len(rec.triangles) > 100:
                tm = rec
                reconstructed = True
                stats["poisson"] = {
                    "depth": poisson_depth, "sample_points": sample_points,
                    "trim_pct": poisson_trim_pct,
                    "faces_raw": faces_raw, "faces_after_trim": int(len(tm.triangles)),
                }
            else:
                stats["poisson_error"] = "reconstruction collapsed after density trim"
        except Exception as exc:  # noqa: BLE001 — reconstruction is best-effort
            stats["poisson_error"] = f"{type(exc).__name__}: {exc}"[:160]
        t_s = _stage(f"poisson -> {len(tm.triangles)} faces (ok={reconstructed})", t_s)
    stats["faces_after_reconstruct"] = int(len(tm.triangles))
    stats["reconstructed"] = reconstructed

    if smooth and len(tm.triangles):
        # Taubin, not Laplacian: it alternates shrink/inflate passes so the
        # object does not shrivel (the same shrink-resistance twin_finish.py
        # chose its pymeshlab filter for).
        tm = tm.filter_smooth_taubin(number_of_iterations=max(1, smooth_iters) * 5)
        tm.remove_unreferenced_vertices()

    if len(tm.triangles) > target_faces:
        tm = tm.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
        tm.remove_unreferenced_vertices()
        tm.remove_degenerate_triangles()
    stats["faces_out"] = int(len(tm.triangles))
    _stage(f"smooth+decimate -> {len(tm.triangles)} faces", t_s)

    return (np.asarray(tm.vertices, dtype=np.float64),
            np.asarray(tm.triangles, dtype=np.int64), stats)


# Working-set caps for the rasterizer and the neighbour query. Both stages run
# on ONE flat array covering many faces at once, so without a cap a 600k-face
# shell would allocate tens of GB where an 8k-face prop allocates megabytes.
# Slabbing keeps peak RAM flat and face count only buys more slabs; the numbers
# are chosen so a slab's transient arrays stay under ~1 GB.
_RASTER_TEXEL_BATCH = 4_000_000   # candidate texels per rasterizer slab
_QUERY_ELEMENT_BATCH = 8_000_000  # texels x k elements per KD query slab


def _face_slabs(ends, budget: int):
    """Yield face ranges [lo, hi) whose candidate texels fit in `budget`.

    Splits BETWEEN faces, never inside one. A face's texels have to stay
    contiguous and in face order because the final scatter is last-writer-wins
    (see _rasterize_atlas), and one face whose UV bbox alone exceeds the budget
    still gets its own slab rather than stalling.
    """
    total = len(ends)
    lo = 0
    while lo < total:
        base = int(ends[lo - 1]) if lo else 0
        hi = max(int(np.searchsorted(ends, base + budget, side="right")), lo + 1)
        yield lo, hi
        lo = hi


def _rasterize_atlas(verts_scene, faces, uv_px, size: int, vnormals):
    """All atlas triangles rasterized at once — the vectorized form of what used
    to be a Python for-loop over every face.

    The loop was O(faces) interpreter round trips, each allocating a meshgrid
    and a handful of small arrays. Numpy does the same work on one flat array of
    candidate texels — every pixel of every face's UV bounding box laid end to
    end — with the per-face scalars broadcast through an index map.

    Measured 2026-07-26, whole-bake wall time on a 2048 atlas (the world shell,
    480k gaussians) — looped vs vectorized: 8k faces 5.4 -> 5.2 s, 50k 7.1 ->
    4.9 s, 100k 8.8 -> 5.8 s, 200k 11.8 -> 5.0 s, 400k 18.5 -> 4.9 s, 600k
    27.2 -> 6.3 s; peak RSS at 400k 2.70 -> 0.83 GB. Vectorized the bake is
    FLAT in face count because the work is then bounded by covered TEXELS, which
    the atlas size caps — a 600k-face shell costs the same as an 8k-face prop.

    Returns (px, py, pos, nrm) in EXACTLY the order the loop emitted them:
    faces in index order, and within a face the bbox pixel centres in row-major
    order. That order is load-bearing, not cosmetic — tex[py, px] = cols is a
    last-writer-wins scatter, so two faces sharing a texel have to resolve the
    same way they always did. Returns Nones when nothing was covered.
    """
    if len(faces) == 0:
        return None, None, None, None

    tri_uv = uv_px[faces]                       # (F, 3, 2), float32 as before
    tx, ty = tri_uv[:, :, 0], tri_uv[:, :, 1]
    x0 = np.maximum(np.floor(tx.min(axis=1)).astype(np.int64), 0)
    x1 = np.minimum(np.ceil(tx.max(axis=1)).astype(np.int64), size - 1)
    y0 = np.maximum(np.floor(ty.min(axis=1)).astype(np.int64), 0)
    y1 = np.minimum(np.ceil(ty.max(axis=1)).astype(np.int64), size - 1)
    on_atlas = (x1 >= x0) & (y1 >= y0)          # else the loop skipped the face
    bw = np.where(on_atlas, x1 - x0 + 1, 0)
    npix = bw * np.where(on_atlas, y1 - y0 + 1, 0)
    ends = np.cumsum(npix)

    ax, ay = tx[:, 0], ty[:, 0]
    bx, by = tx[:, 1], ty[:, 1]
    cx, cy = tx[:, 2], ty[:, 2]
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    ok_den = np.abs(den) >= 1e-12
    # Degenerate faces are masked out below; substituting 1 only keeps the
    # division from raising, it never reaches the output.
    safe_den = np.where(ok_den, den, np.float32(1.0))

    px_all, py_all, pos_all, nrm_all = [], [], [], []
    for lo, hi in _face_slabs(ends, _RASTER_TEXEL_BATCH):
        sub = slice(lo, hi)
        base = int(ends[lo - 1]) if lo else 0
        n = int(ends[hi - 1]) - base
        if n:
            fi = np.repeat(np.arange(lo, hi, dtype=np.int64), npix[sub])
            # position of each candidate inside its own face's bbox
            local = np.arange(n, dtype=np.int64) - (ends[fi] - npix[fi] - base)
            wf = bw[fi]
            xs = (x0[fi] + local % wf) + 0.5
            ys = (y0[fi] + local // wf) + 0.5

            # barycentric coords of every pixel centre in the bbox
            dxc, dyc = xs - cx[fi], ys - cy[fi]
            inv = safe_den[fi]
            w0 = ((by[fi] - cy[fi]) * dxc + (cx[fi] - bx[fi]) * dyc) / inv
            w1 = ((cy[fi] - ay[fi]) * dxc + (ax[fi] - cx[fi]) * dyc) / inv
            w2 = 1.0 - w0 - w1
            hit = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4) & ok_den[fi]
            sel = np.flatnonzero(hit)
        else:
            fi = sel = np.empty(0, dtype=np.int64)

        # Sub-pixel or degenerate UV triangles: no pixel centre falls inside
        # them, so the plain rasterizer writes nothing and the face samples
        # black at render time. At a few thousand faces in a 1024 atlas this is
        # common, and it is exactly the black speckling seen on the first bake.
        # Guarantee every face owns at least one texel, at its UV centroid.
        f_hit = fi[sel] - lo
        cnt = np.bincount(f_hit, minlength=hi - lo)
        fb = np.flatnonzero((cnt == 0) & on_atlas[sub])

        # Interleave centroid texels back into face order (see docstring).
        out_cnt = np.where(cnt > 0, cnt, on_atlas[sub].astype(np.int64))
        out_end = np.cumsum(out_cnt)
        m = int(out_end[-1]) if len(out_end) else 0
        if m == 0:
            continue
        out_start = out_end - out_cnt
        dest_hit = out_start[f_hit] + (np.arange(len(sel), dtype=np.int64)
                                       - (np.cumsum(cnt) - cnt)[f_hit])
        dest_fb = out_start[fb]

        b_px = np.empty(m, dtype=np.int32)
        b_py = np.empty(m, dtype=np.int32)
        b_pos = np.empty((m, 3), dtype=np.float32)
        b_nrm = np.empty((m, 3), dtype=np.float32) if vnormals is not None else None

        if len(sel):
            tri = faces[fi[sel]]
            u0, u1, u2 = w0[sel, None], w1[sel, None], w2[sel, None]
            b_px[dest_hit] = (xs[sel] - 0.5).astype(np.int32)
            b_py[dest_hit] = (ys[sel] - 0.5).astype(np.int32)
            b_pos[dest_hit] = (u0 * verts_scene[tri[:, 0]]
                               + u1 * verts_scene[tri[:, 1]]
                               + u2 * verts_scene[tri[:, 2]]).astype(np.float32)
            if vnormals is not None:
                b_nrm[dest_hit] = (u0 * vnormals[tri[:, 0]]
                                   + u1 * vnormals[tri[:, 1]]
                                   + u2 * vnormals[tri[:, 2]]).astype(np.float32)
        if len(fb):
            gfb = fb + lo
            tri = faces[gfb]
            b_px[dest_fb] = np.clip(np.round(tx[gfb].mean(axis=1)),
                                    0, size - 1).astype(np.int32)
            b_py[dest_fb] = np.clip(np.round(ty[gfb].mean(axis=1)),
                                    0, size - 1).astype(np.int32)
            b_pos[dest_fb] = verts_scene[tri].mean(axis=1).astype(np.float32)
            if vnormals is not None:
                b_nrm[dest_fb] = vnormals[tri].mean(axis=1)

        px_all.append(b_px)
        py_all.append(b_py)
        pos_all.append(b_pos)
        if b_nrm is not None:
            nrm_all.append(b_nrm)

    if not pos_all:
        return None, None, None, None
    return (np.concatenate(px_all), np.concatenate(py_all),
            np.concatenate(pos_all),
            np.concatenate(nrm_all) if nrm_all else None)


def bake_texture(verts_scene, faces, uvs, gx, gc, size: int, k: int = 24,
                 vnormals=None):
    """Rasterize every atlas triangle and colour each covered texel from the
    gaussians nearest that texel's 3D position.

    Texel -> 3D goes through barycentric interpolation on the SCENE-space
    vertices, because that is the frame the gaussians live in. Sampling in the
    exported (scaled, Y-up) frame would silently query the wrong neighbourhood.

    Returns (rgb_uint8 HxWx3, coverage_mask HxW bool).
    """
    tex = np.zeros((size, size, 3), dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)

    # Collect every covered texel first, then do batched KD queries — a
    # per-triangle query would be ~10k round trips into the tree.
    px, py, pos, nrm = _rasterize_atlas(verts_scene, faces, uvs * (size - 1),
                                        size, vnormals)
    if pos is None:
        return None, None
    if nrm is not None:
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)

    # DROP SHADOWED SAMPLES. The final write is a last-writer-wins scatter, so
    # when several faces cover one texel only the last sample can ever be seen —
    # colouring the rest is pure waste, and at 400k faces in a 2048 atlas the
    # duplicates are a large minority of all samples. Keeping the last index per
    # texel is exactly what the scatter would have resolved to, so the image is
    # unchanged; the neighbour query just gets a smaller input. A full-atlas
    # index buffer beats sorting: O(size^2 + n) with trivial constants.
    winner = np.full(size * size, -1, dtype=np.int32)
    winner[py.astype(np.int64) * size + px] = np.arange(len(px), dtype=np.int32)
    keep = winner[winner >= 0]
    px, py, pos = px[keep], py[keep], pos[keep]
    if nrm is not None:
        nrm = nrm[keep]

    kq = min(k, len(gx))
    tree = cKDTree(gx)
    cols = np.empty((len(pos), 3), dtype=np.float32)
    # Every texel's colour depends only on its own neighbours, so slabbing the
    # query is arithmetically identical to one giant call — it just bounds the
    # (n, k) intermediates, which are what actually blow up at scale.
    step = max(1, _QUERY_ELEMENT_BATCH // max(kq, 1))
    for s in range(0, len(pos), step):
        e = min(s + step, len(pos))
        dist, idx = tree.query(pos[s:e], k=kq, workers=-1)
        if kq == 1:
            cols[s:e] = gc[idx]
            continue
        # ADAPTIVE GAUSSIAN kernel, not 1/d. The gaussians are far sparser than
        # the atlas: ~20k solid gaussians over a ~1 m object is ~7 mm spacing,
        # while a 1024 atlas texel is ~1 mm, so every gaussian owns ~50 texels.
        # With 1/d weights the nearest gaussian dominates its whole
        # neighbourhood and the bake comes out as hard-edged Voronoi patches
        # (visible on the first hydrant bake). Weighting by exp(-(d/h)^2) with h
        # set per texel from its own neighbour distances blends across several
        # gaussians and follows local density instead of imposing a fixed
        # radius.
        h = dist[:, min(3, kq - 1)][:, None]
        h = np.maximum(h, 1e-6)
        w = np.exp(-(dist / h) ** 2)
        if nrm is not None:
            # SIDE FILTER. Poisson refits the surface, so a texel's nearest
            # gaussian in raw 3D distance can sit on the far wall of a thin
            # feature or inside the body — and interior gaussians are dark,
            # which is where the black blotches on the first bakes came from.
            # Keep only gaussians on the outward side of the local surface.
            side = np.einsum("nkc,nc->nk", gx[idx] - pos[s:e, None, :], nrm[s:e])
            ok = side > -h * 0.5
            # a texel with nothing on its outward side falls back to unfiltered
            # rather than going black
            w = np.where(ok.any(axis=1, keepdims=True), w * ok, w)
        wsum = w.sum(axis=1, keepdims=True)
        w = np.where(wsum > 0, w / np.maximum(wsum, 1e-12), 1.0 / kq)
        cols[s:e] = np.einsum("nk,nkc->nc", w, gc[idx])

    tex[py, px] = cols
    mask[py, px] = True
    return (np.clip(tex, 0, 1) * 255).astype(np.uint8), mask


def dilate_atlas(tex, mask, passes: int = 24):
    """Bleed colour outward past chart borders.

    A GPU samples with bilinear filtering and mipmaps, so texels just OUTSIDE a
    chart still get read at the seam. Leaving them black draws dark rims around
    every chart — the classic unpadded-atlas artifact. Nearest-neighbour bleed
    is enough and needs no extra dependency beyond what is already imported.
    """
    out = tex.copy()
    filled = mask.copy()
    for _ in range(passes):
        holes = ~filled
        if not holes.any():
            break
        acc = np.zeros_like(out, dtype=np.float32)
        cnt = np.zeros(filled.shape, dtype=np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = np.roll(np.roll(out, dy, axis=0), dx, axis=1).astype(np.float32)
            svalid = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
            acc += src * svalid[..., None]
            cnt += svalid
        grow = holes & (cnt > 0)
        out[grow] = (acc[grow] / cnt[grow][:, None]).astype(np.uint8)
        filled = filled | grow
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("splat")
    ap.add_argument("out_glb")
    ap.add_argument("--meters-per-unit", type=float, default=None)
    ap.add_argument("--target-faces", type=int, default=8_000)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--smooth-iterations", type=int, default=2)
    ap.add_argument("--smooth-feature-deg", type=float, default=40.0)
    ap.add_argument("--min-component-frac", type=float, default=0.02)
    # Ground removal. Pass object.json's bbox_tight VERBATIM (scene units) — the
    # semantic bbox excludes the ground the TSDF fused in. Omit to keep the mesh
    # as-is. Taken as one JSON value, not two coordinate triples: scene coords
    # are routinely negative and argparse reads a leading '-' as an option flag,
    # so "--crop-min -0.16,..." fails even from a subprocess arg list.
    ap.add_argument("--crop-bbox", default=None,
                    help='{"min":[x,y,z],"max":[x,y,z]} in scene units')
    ap.add_argument("--crop-margin", type=float, default=0.15,
                    help="expand the crop box by this fraction of its own size")
    ap.add_argument("--no-crop", action="store_true",
                    help="keep the mesh as-is (no ground removal)")
    ap.add_argument("--auto-crop-pct", type=float, default=1.0,
                    help="percentile trim when deriving the crop from the gaussians")
    ap.add_argument("--no-remesh", action="store_true")
    ap.add_argument("--no-reconstruct", action="store_true",
                    help="skip screened-Poisson refit (keeps the raw open shell)")
    ap.add_argument("--poisson-depth", type=int, default=8)
    ap.add_argument("--poisson-trim-pct", type=float, default=8.0)
    # The floor guards against reconstructing noise. 1000 suited the single-
    # object lane (tens of thousands of gaussians per object); scene-lane props
    # are legitimately sparse — Bonsai's bike bottle is 288 gaussians and was
    # rejected outright. Low but non-zero, and overridable.
    ap.add_argument("--min-gaussians", type=int, default=250)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    t0 = time.time()
    out_glb = Path(args.out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    gx, gc = load_solid_gaussians(Path(args.splat))
    if len(gx) < args.min_gaussians:
        print(f"FATAL: only {len(gx)} solid gaussians "
              f"(floor {args.min_gaussians}; lower with --min-gaussians)", file=sys.stderr)
        return 1

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    points_source = False
    if len(mesh.faces) == 0:
        # POINT-CLOUD SOURCE. The scene lane (batch_isolate.py) claims each
        # instance's gaussians into object.ply and never builds a per-object
        # mesh — so for whole-scene work there is no TSDF to start from, and
        # running one per instance would be the most expensive step in the
        # pipeline. Poisson takes an oriented point cloud natively, so the
        # gaussians ARE a valid surface source. Normals must be made globally
        # consistent first: with per-point normals left arbitrarily flipped,
        # Poisson produces inside-out shards rather than a surface.
        _log_pts = len(gx)
        depth = args.poisson_depth
        if _log_pts < 1_000:
            depth = min(depth, 6)
        elif _log_pts < 5_000:
            depth = min(depth, 7)
        knn = max(8, min(32, _log_pts // 20))
        mesh = mesh_from_points(gx, depth=depth, knn=knn,
                                trim_pct=args.poisson_trim_pct)
        points_source = True
        _stage(f"surface from {_log_pts} gaussians (depth {depth}, knn {knn}) -> {len(mesh.faces)} faces", t0)
        if len(mesh.faces) < 4:
            print("FATAL: point-cloud reconstruction produced no surface", file=sys.stderr)
            return 1

    crop_stats = None
    lo = hi = None
    crop_source = None
    if args.crop_bbox:
        try:
            box = json.loads(args.crop_bbox)
            lo = [float(x) for x in box["min"]]
            hi = [float(x) for x in box["max"]]
            crop_source = "explicit"
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"FATAL: --crop-bbox must be {{'min':[x,y,z],'max':[x,y,z]}}: {exc}",
                  file=sys.stderr)
            return 1
        if len(lo) != 3 or len(hi) != 3:
            print("FATAL: --crop-bbox min/max must each have 3 components", file=sys.stderr)
            return 1
    elif not args.no_crop:
        # AUTO CROP, preferring the tightest trustworthy source.
        #
        # 1. object.json's bbox_tight — the UN-EXPANDED semantic members. This
        #    is the only source that reliably excludes ground.
        # 2. percentiles of object.ply — a fallback for older object lanes
        #    (the garden capture's flower-vase predates bbox_tight entirely, and
        #    a feature that only works on captures built after some date is not
        #    a feature).
        #
        # object.ply itself is deliberately NOT the first choice: it holds the
        # EXPANDED member set (the hydrant's 3,999-gaussian pool expands to
        # 28,627 at expand=1.6), and that expansion reaches into the ground —
        # cropping to it keeps 100% of the mesh and removes nothing.
        sidecar = Path(args.splat).parent / "object.json"
        tight = None
        if sidecar.is_file():
            try:
                tight = json.loads(sidecar.read_text()).get("bbox_tight")
            except (OSError, json.JSONDecodeError):
                tight = None
        if tight and "min" in tight and "max" in tight:
            lo = [float(x) for x in tight["min"]]
            hi = [float(x) for x in tight["max"]]
            crop_source = f"auto ({sidecar.name} bbox_tight)"
        else:
            lo = np.percentile(gx, args.auto_crop_pct, axis=0).tolist()
            hi = np.percentile(gx, 100.0 - args.auto_crop_pct, axis=0).tolist()
            crop_source = (f"auto (object gaussians p{args.auto_crop_pct:g}"
                           f"-p{100 - args.auto_crop_pct:g}; no bbox_tight sidecar)")

    if lo is not None:
        keep, lo_p, hi_p = crop_to_box(mesh, lo, hi, args.crop_margin)
        kept = int(keep.sum())
        if kept < 4:
            print(f"FATAL: crop box kept {kept} faces — box is wrong or empty",
                  file=sys.stderr)
            return 1
        crop_stats = {
            "source": crop_source,
            "faces_before": int(len(mesh.faces)), "faces_kept": kept,
            "kept_frac": round(kept / max(len(mesh.faces), 1), 4),
            "box_min": [round(float(x), 4) for x in lo_p],
            "box_max": [round(float(x), 4) for x in hi_p],
            "margin": args.crop_margin,
        }
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()

    verts_scene, faces, stats = clean_and_simplify(
        mesh, args.target_faces, args.min_component_frac,
        args.smooth, args.smooth_iterations, args.smooth_feature_deg,
        remesh=not args.no_remesh,
        reconstruct=(not args.no_reconstruct) and not points_source,
        poisson_depth=args.poisson_depth, poisson_trim_pct=args.poisson_trim_pct)
    if len(faces) < 4:
        print(f"FATAL: cleanup left {len(faces)} faces", file=sys.stderr)
        return 1
    stats["crop"] = crop_stats
    stats["geometry_source"] = "gaussians (poisson)" if points_source else "mesh"

    # ---- UV unwrap (best effort; vertex colours remain the honest fallback) ----
    #
    # ⚠ THIS is the wall on dense shells, not the bake. xatlas.parametrize is
    # steeply superlinear in face count — measured 2026-07-26 on the world shell
    # at 0.2 s / 8k, 2.1 s / 50k, 10.3 s / 100k, 76.2 s / 200k and 1564.2 s
    # (26 min) / 400k, which is what actually blew the 10-minute budget on the
    # first 400k run. The bake below is ~5 s flat at any of those sizes.
    # Unwrapping the SAME 400k mesh in 4 spatial chunks and packing each chart
    # set into its own UV tile took 118 s total, so the cost is chart generation
    # over one huge mesh rather than the work itself — but chunking changes the
    # atlas layout, i.e. the output, so it is a deliberate design call for the
    # dense-shell lane and not something to slip in as an optimization.
    texture_report = None
    uvs = None
    t_uv = time.time()
    try:
        import xatlas
        vmapping, indices, uvs = xatlas.parametrize(verts_scene, faces)
        verts_scene = verts_scene[vmapping]
        faces = indices.astype(np.int64)
        uv_seconds = round(time.time() - t_uv, 1)
    except Exception as exc:  # noqa: BLE001 — unwrap is optional, never fatal
        uvs = None
        texture_report = {"baked": False, "reason": f"unwrap failed: {type(exc).__name__}: {exc}"[:200]}

    # ---- vertex colours: always produced, they are the fallback AND useful in DCC ----
    kq = min(16, len(gx))
    dist, idx = cKDTree(gx).query(verts_scene.astype(np.float32), k=kq, workers=-1)
    if kq == 1:
        vcols = gc[idx]
    else:
        h = np.maximum(dist[:, min(3, kq - 1)][:, None], 1e-6)
        w = np.exp(-(dist / h) ** 2)
        wsum = w.sum(axis=1, keepdims=True)
        w = np.where(wsum > 0, w / np.maximum(wsum, 1e-12), 1.0 / kq)
        vcols = np.einsum("nk,nkc->nc", w, gc[idx])
    vcols_u8 = (np.clip(vcols, 0, 1) * 255).astype(np.uint8)

    # ---- bake ----
    tex_img = None
    if uvs is not None:
        vn = trimesh.Trimesh(vertices=verts_scene, faces=faces,
                             process=False).vertex_normals
        tex, mask = bake_texture(verts_scene, faces, uvs, gx, gc, args.texture_size,
                                 vnormals=np.asarray(vn, dtype=np.float32))
        if tex is None:
            texture_report = {"baked": False, "reason": "atlas rasterized zero texels"}
        else:
            coverage = float(mask.mean())
            tex = dilate_atlas(tex, mask, passes=max(16, args.texture_size // 48))
            tex_img = Image.fromarray(tex, mode="RGB")
            # Also write the atlas beside the GLB. The GLB embeds it, but a
            # standalone PNG is what lets the UI show a thumbnail, lets a human
            # eyeball chart packing, and lets an artist retouch the map and
            # re-pack it without unpacking the binary.
            atlas_path = out_glb.with_name(out_glb.stem + "_atlas.png")
            tex_img.save(atlas_path)
            texture_report = {
                "baked": True,
                "size": args.texture_size,
                "coverage": round(coverage, 4),
                "unwrap_seconds": uv_seconds,
                "charts_uv_range": [round(float(uvs.min()), 4), round(float(uvs.max()), 4)],
                "atlas_png": atlas_path.name,
                "atlas_bytes": atlas_path.stat().st_size,
            }

    # ---- export: scale to metres, Y-up (same convention as twin_finish.py) ----
    scale = args.meters_per_unit or 1.0
    v = verts_scene * scale
    yup = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1).astype(np.float32)
    out = trimesh.Trimesh(vertices=yup, faces=faces, process=False)
    if tex_img is not None:
        out.visual = trimesh.visual.TextureVisuals(
            uv=uvs, image=tex_img,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=tex_img, metallicFactor=0.0, roughnessFactor=0.75),
        )
    else:
        out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=vcols_u8)
    out.export(str(out_glb))

    # ---- MANDATORY readback (house rule: never trust an exporter's exit code) ----
    back = trimesh.load(str(out_glb), force="mesh")
    if len(back.faces) != len(faces):
        out_glb.unlink(missing_ok=True)
        print(f"FATAL: GLB readback mismatch ({len(back.faces)} != {len(faces)})", file=sys.stderr)
        return 1
    if tex_img is not None:
        has_uv = getattr(getattr(back.visual, "uv", None), "shape", None) is not None
        img_back = getattr(getattr(back.visual, "material", None), "baseColorTexture", None)
        if not has_uv or img_back is None:
            out_glb.unlink(missing_ok=True)
            print("FATAL: GLB readback lost UVs or texture", file=sys.stderr)
            return 1
        texture_report["readback_uv_count"] = int(back.visual.uv.shape[0])
        texture_report["readback_texture_size"] = list(img_back.size)

    report = {
        "verts": int(len(yup)), "faces": int(len(faces)),
        "solid_gaussians": int(len(gx)),
        "units": "meters" if args.meters_per_unit else "scene-units (uncalibrated)",
        "meters_per_unit": float(args.meters_per_unit) if args.meters_per_unit else None,
        "up_axis": "Y",
        "extent": [round(float(x), 3) for x in back.extents],
        "simplify": stats,
        "texture": texture_report,
        "glb_bytes": out_glb.stat().st_size,
        "seconds": round(time.time() - t0, 1),
        "smoothing": (
            {"applied": True, "iterations": args.smooth_iterations,
             "feature_deg": args.smooth_feature_deg}
            if args.smooth else {"applied": False}
        ),
    }
    report_path = Path(args.report) if args.report else out_glb.parent / "object_texture.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
