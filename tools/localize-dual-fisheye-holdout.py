#!/usr/bin/env python3
"""Localize held-out raw images against frozen geometry inside the existing compute gate."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import artifact_manifest as manifests
from fisheye_reference import project_fisheye, verify_sources
from heldout_localization import read_keypoints, read_tracks, select_correspondences, validate_training_database


GATE = Path(__file__).resolve().parent / "splatlab-compute-gate.sh"


def run(pilot_root: Path, sfm_root: Path, output: Path) -> dict:
    if subprocess.run([str(GATE), "--is-contained"], capture_output=True, timeout=5).returncode:
        raise ValueError("Launch this GPU-capable dependency through splatlab-compute-gate.sh --run")
    pilot_root, sfm_root, output = pilot_root.resolve(), sfm_root.resolve(), output.resolve()
    if output.exists() or any(output.is_relative_to(root) for root in (pilot_root, sfm_root)):
        raise ValueError("Choose a new output outside frozen inputs")
    pilot, cameras, poses, lens_cameras, sources, snapshot = verify_sources(pilot_root, sfm_root)
    for filename in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
        path = sfm_root / "sparse/0" / filename
        sources["model/" + filename] = path
        snapshot["model/" + filename] = manifests.sha256_file(path)
    import pycolmap

    if not pycolmap.__version__.startswith("4.1."):
        raise ValueError("Review compatibility before changing installed pycolmap 4.1")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"schema": "dev.splatlab.frozen-holdout-localization/v1", "status": "running-not-accepted",
               "started_at": manifests.utc_now(), "tool_sha256": manifests.sha256_file(Path(__file__)),
               "source_hashes": snapshot, "pycolmap_version": pycolmap.__version__, "poses": [],
               "geometry_refined": False, "intrinsics_refined": False, "appearance_evaluated": False,
               "metric_scale": "unknown", "uses_gpu": False,
               "limitations": ["Pose residuals fit held-out feature matches; they are not held-out rendered-image scores.",
                               "No held-out point is triangulated, no joint bundle adjustment or rig acceptance occurs.",
                               "Independent held-out camera estimates remain in arbitrary training-model coordinates."]}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    try:
        database_path = output / "localization.db"
        source_database = sfm_root / "database.db"
        database_hashes = {str(path): manifests.sha256_file(path) for path in
                           (source_database, Path(str(source_database) + "-wal")) if path.exists()}
        with closing(sqlite3.connect(source_database.as_uri() + "?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(database_path)) as database:
                original.backup(database)
        tracks = read_tracks(sfm_root / "sparse/0/images.txt")
        with closing(sqlite3.connect(database_path)) as database, database:
            training = validate_training_database(database, tracks)
            for identifier, camera in cameras.items():
                result = database.execute("UPDATE cameras SET params=?,prior_focal_length=1 WHERE camera_id=?",
                                          (np.array(camera["params"], dtype=np.float64).tobytes(), identifier))
                if result.rowcount != 1:
                    raise ValueError("Estimated lens camera is missing from training database")
        heldout = [view for view in pilot["views"] if view["split"] != "train"]
        extraction = pycolmap.FeatureExtractionOptions(num_threads=4, use_gpu=False, max_image_size=pilot["output_width"])
        extraction.sift.max_num_features = 4096
        extraction.sift.first_octave = 0
        for lens, identifier in lens_cameras.items():
            names = [view["image"] for view in heldout if view["lens_id"] == lens]
            pycolmap.extract_features(database_path, pilot_root / "images", image_names=names,
                reader_options=pycolmap.ImageReaderOptions(existing_camera_id=identifier),
                extraction_options=extraction, device=pycolmap.Device.cpu)
        pycolmap.match_exhaustive(database_path,
            matching_options=pycolmap.FeatureMatchingOptions(num_threads=4, use_gpu=False), device=pycolmap.Device.cpu)
        reconstruction = pycolmap.Reconstruction(sfm_root / "sparse/0")
        if {image.name for image in reconstruction.images.values()} != set(poses):
            raise ValueError("Binary model and frozen training text disagree")
        for identifier, camera in cameras.items():
            if not np.allclose(reconstruction.cameras[identifier].params, camera["params"], atol=1e-12, rtol=0):
                raise ValueError("Binary and text lens intrinsics disagree")
        for image in reconstruction.images.values():
            pose = poses[image.name]
            expected = np.column_stack((pose["rotation"], pose["translation"]))
            if not np.allclose(image.cam_from_world().matrix(), expected, atol=1e-10, rtol=0):
                raise ValueError("Binary and text training poses disagree")
        estimation = pycolmap.AbsolutePoseEstimationOptions(estimate_focal_length=False)
        estimation.ransac.max_error = 4
        estimation.ransac.random_seed = 0
        estimation.ransac.num_threads = 1
        refinement = pycolmap.AbsolutePoseRefinementOptions(refine_focal_length=False, refine_extra_params=False)
        with closing(sqlite3.connect(database_path)) as database:
            image_ids = dict(database.execute("SELECT name,image_id FROM images"))
            pairs = list(database.execute("SELECT pair_id,rows,cols,data FROM two_view_geometries WHERE rows>0"))
            for view in heldout:
                query_id = image_ids[view["image"]]
                keypoints = read_keypoints(database, query_id)
                selected = select_correspondences(query_id, len(keypoints), pairs, training)
                record = {"image": view["image"], "split": view["split"], "group_id": view["group_id"],
                          "physical_lens": view["lens_id"], "matches": len(selected),
                          "status": "not-localized", "camera_to_world_opencv": None}
                if len(selected) >= 20:
                    pixels = keypoints[[item[0] for item in selected]]
                    positions = np.array([reconstruction.points3D[item[1]].xyz for item in selected])
                    identifier = lens_cameras[view["lens_id"]]
                    camera = reconstruction.cameras[identifier]
                    before = camera.params.copy()
                    result = pycolmap.estimate_and_refine_absolute_pose(pixels, positions, camera, estimation, refinement)
                    if not np.array_equal(before, camera.params):
                        raise ValueError("Pose-only estimator changed frozen lens intrinsics")
                    if result is not None:
                        inliers = np.asarray(result["inlier_mask"], dtype=bool)
                        world_to_camera = np.asarray(result["cam_from_world"].matrix())
                        if (inliers.shape != (len(selected),) or world_to_camera.shape != (3, 4)
                                or not np.isfinite(world_to_camera).all()):
                            raise ValueError("Pose estimator returned invalid evidence")
                        projected, in_cone = project_fisheye(
                            positions @ world_to_camera[:, :3].T + world_to_camera[:, 3], cameras[identifier]["params"])
                        retained = inliers & in_cone
                        residual = np.linalg.norm(projected[retained] - pixels[retained], axis=1)
                        usable = len(residual) >= 20 and retained.mean() >= .25 and float(np.percentile(residual, 95)) <= 4
                        record.update(inliers=int(inliers.sum()), supported_cone_inliers=int(retained.sum()),
                                      supported_inlier_fraction=float(retained.mean()),
                                      mean_inlier_reprojection_px=float(residual.mean()) if len(residual) else None,
                                      p95_inlier_reprojection_px=float(np.percentile(residual, 95)) if len(residual) else None)
                        if usable:
                            transform = np.eye(4)
                            transform[:3] = world_to_camera
                            record.update(status="localized-needs-review", camera_to_world_opencv=np.linalg.inv(transform).tolist())
                receipt["poses"].append(record)
                manifests.atomic_write_json(output / "receipt.json", receipt)
        if snapshot != {name: manifests.sha256_file(path) for name, path in sources.items()}:
            raise ValueError("Frozen image/model inputs changed during localization")
        if database_hashes != {name: manifests.sha256_file(Path(name)) for name in database_hashes}:
            raise ValueError("Original training database changed during localization")
        receipt["source_database_hashes"] = database_hashes
        receipt["localized"] = sum(record["status"] == "localized-needs-review" for record in receipt["poses"])
        receipt["total_heldout"] = len(heldout)
        receipt["status"] = "localized-needs-review" if receipt["localized"] == len(heldout) else "incomplete-localization"
        receipt["database_sha256"] = manifests.sha256_file(database_path)
    except Exception as error:
        receipt.update(status="failed-not-accepted", error=str(error))
        raise
    finally:
        receipt.update(finished_at=manifests.utc_now(), seconds=round(time.monotonic() - started, 3))
        manifests.atomic_write_json(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--sfm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.pilot, args.sfm, args.output)
    print(json.dumps({key: result[key] for key in ("status", "localized", "total_heldout", "seconds")}, indent=2))
    return 0 if result["status"] == "localized-needs-review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
