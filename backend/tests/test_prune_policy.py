"""Owner decision 2026-09-30: finished splats are never auto-deleted. The old 10-newest-unpinned cap rm -rf'd
every older scene when a 16-job re-run finished (incl. the storage room's walkable world). Only stale failed or
stopped jobs that never produced a splat may be cleaned up."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


def test_completed_scenes_are_never_deleted_and_failed_splats_are_kept(tmp_path, monkeypatch):
    metas = []
    for i in range(15):                                                   # well past the old cap of 10
        metas.append({"job_id": f"splat_c{i:02d}", "status": "completed", "pinned": False,
                      "created_at": f"2026-09-{i + 1:02d}T00:00:00+00:00", "output_dir": str(tmp_path / f"c{i}")})
    old = "2026-01-01T00:00:00+00:00"
    with_splat = tmp_path / "f1"
    (with_splat / splat_route.PREVIEW_DIRNAME).mkdir(parents=True)
    (with_splat / splat_route.PREVIEW_DIRNAME / "splat.ply").write_bytes(b"ply")
    metas += [{"job_id": "splat_f0", "status": "failed", "pinned": False, "created_at": old,
               "output_dir": str(tmp_path / "f0")},                                       # no splat -> cleaned
              {"job_id": "splat_f1", "status": "failed", "pinned": False, "created_at": old,
               "output_dir": str(with_splat)},                                            # has a splat -> kept
              {"job_id": "splat_s2", "status": "stopped", "pinned": True, "created_at": old,
               "output_dir": str(tmp_path / "s2")}]                                       # pinned -> kept
    deleted: list[str] = []
    monkeypatch.setattr(splat_route, "_all_metas", lambda: metas)
    monkeypatch.setattr(splat_route, "_delete_job_files", lambda job_id: deleted.append(job_id))
    monkeypatch.setattr(splat_route, "JOBS", {})
    assert splat_route._prune_old_jobs() == 1
    assert deleted == ["splat_f0"]
    assert not hasattr(splat_route, "KEEP_UNPINNED_COMPLETED")
