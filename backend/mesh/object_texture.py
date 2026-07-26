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

    # Collect every covered texel first, then do ONE batched KD query — a
    # per-triangle query would be ~10k round trips into the tree.
    px_all, py_all, pos_all, nrm_all = [], [], [], []
    uv_px = uvs * (size - 1)

    for f in faces:
        t = uv_px[f]
        x0 = max(int(np.floor(t[:, 0].min())), 0)
        x1 = min(int(np.ceil(t[:, 0].max())), size - 1)
        y0 = max(int(np.floor(t[:, 1].min())), 0)
        y1 = min(int(np.ceil(t[:, 1].max())), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        xs = xs.ravel() + 0.5
        ys = ys.ravel() + 0.5

        # barycentric coords of every pixel centre in the bbox
        tri3 = verts_scene[f]
        ax, ay = t[0]; bx, by = t[1]; cx, cy = t[2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        inside = None
        if abs(den) >= 1e-12:
            w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
            w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)

        if inside is None or not inside.any():
            # Sub-pixel or degenerate UV triangle: no pixel centre falls inside
            # it, so the plain rasterizer writes nothing and the face samples
            # black at render time. At a few thousand faces in a 1024 atlas
            # this is common, and it is exactly the black speckling seen on the
            # first bake. Guarantee every face owns at least one texel.
            cu = int(np.clip(round(t[:, 0].mean()), 0, size - 1))
            cv = int(np.clip(round(t[:, 1].mean()), 0, size - 1))
            px_all.append(np.array([cu], dtype=np.int32))
            py_all.append(np.array([cv], dtype=np.int32))
            pos_all.append(tri3.mean(axis=0)[None, :])
            if vnormals is not None:
                nrm_all.append(vnormals[f].mean(axis=0)[None, :])
            continue

        w0, w1, w2 = w0[inside], w1[inside], w2[inside]
        pos = (w0[:, None] * tri3[0] + w1[:, None] * tri3[1] + w2[:, None] * tri3[2])
        px_all.append((xs[inside] - 0.5).astype(np.int32))
        py_all.append((ys[inside] - 0.5).astype(np.int32))
        pos_all.append(pos)
        if vnormals is not None:
            tn = vnormals[f]
            nrm_all.append(w0[:, None] * tn[0] + w1[:, None] * tn[1] + w2[:, None] * tn[2])

    if not pos_all:
        return None, None

    px = np.concatenate(px_all)
    py = np.concatenate(py_all)
    pos = np.concatenate(pos_all).astype(np.float32)
    nrm = np.concatenate(nrm_all).astype(np.float32) if nrm_all else None
    if nrm is not None:
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)

    kq = min(k, len(gx))
    dist, idx = cKDTree(gx).query(pos, k=kq, workers=-1)
    if kq == 1:
        cols = gc[idx]
    else:
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
            side = np.einsum("nkc,nc->nk", gx[idx] - pos[:, None, :], nrm)
            ok = side > -h * 0.5
            # a texel with nothing on its outward side falls back to unfiltered
            # rather than going black
            w = np.where(ok.any(axis=1, keepdims=True), w * ok, w)
        wsum = w.sum(axis=1, keepdims=True)
        w = np.where(wsum > 0, w / np.maximum(wsum, 1e-12), 1.0 / kq)
        cols = np.einsum("nk,nkc->nc", w, gc[idx])

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
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    t0 = time.time()
    out_glb = Path(args.out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if len(mesh.faces) == 0:
        print("FATAL: empty mesh", file=sys.stderr)
        return 1
    gx, gc = load_solid_gaussians(Path(args.splat))
    if len(gx) < 1000:
        print(f"FATAL: only {len(gx)} solid gaussians", file=sys.stderr)
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
        remesh=not args.no_remesh, reconstruct=not args.no_reconstruct,
        poisson_depth=args.poisson_depth, poisson_trim_pct=args.poisson_trim_pct)
    if len(faces) < 4:
        print(f"FATAL: cleanup left {len(faces)} faces", file=sys.stderr)
        return 1
    stats["crop"] = crop_stats

    # ---- UV unwrap (best effort; vertex colours remain the honest fallback) ----
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
