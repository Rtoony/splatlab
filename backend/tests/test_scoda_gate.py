import csv
import json
from pathlib import Path

from PIL import Image

from health import scoda_gate as sg


def _img(p: Path, size=(32, 24)):
    Image.new("RGB", size, (120, 90, 60)).save(p)


def test_pick_reference_views_is_evenly_spaced_and_bounded(tmp_path: Path):
    for i in range(20):
        _img(tmp_path / f"f{i:02d}.jpg")
    (tmp_path / "notes.txt").write_text("x")
    picks = sg.pick_reference_views(tmp_path, 5)
    assert [p.name for p in picks] == ["f00.jpg", "f05.jpg", "f10.jpg", "f14.jpg", "f19.jpg"]
    assert sg.pick_reference_views(tmp_path, 0) == []
    assert len(sg.pick_reference_views(tmp_path, 100)) == 20


def test_stage_reference_views_downscales_and_is_idempotent(tmp_path: Path):
    src = tmp_path / "images"; src.mkdir()
    for i in range(3):
        _img(src / f"{i}.png", size=(1600, 1200))
    out = sg.stage_reference_views(src, tmp_path / "Ref_views", "job1", n=3, max_side=400)
    assert [p.name for p in out] == ["000.png", "001.png", "002.png"]
    assert Image.open(out[0]).size == (400, 300)
    mtime = out[0].stat().st_mtime
    sg.stage_reference_views(src, tmp_path / "Ref_views", "job1", n=3, max_side=400)
    assert out[0].stat().st_mtime == mtime


def test_input_csv_and_command_shape(tmp_path: Path):
    rd = tmp_path / "renders"; rd.mkdir()
    _img(rd / "cam0001_eval.png"); _img(rd / "cam0001_yaw+12.png"); (rd / "cams.json").write_text("{}")
    n = sg.write_input_csv(rd, "job1", tmp_path / "in.csv")
    rows = list(csv.DictReader((tmp_path / "in.csv").open()))
    assert n == 2 and rows[0]["scene"] == "job1" and rows[0]["image"].endswith("cam0001_eval.png")
    cmd = sg.scoda_command(tmp_path / "in.csv", tmp_path / "ref", tmp_path / "stats", tmp_path / "out.csv")
    assert cmd[1].endswith("run_scoda.py") and "--no-use-pca" in cmd and cmd[cmd.index("--device") + 1] == "cpu"


def test_aggregate_splits_eval_and_novel_and_compare_ranks(tmp_path: Path):
    cams = {"rendered": [{"file": "a_eval.png", "novel": False}, {"file": "a_yaw+12.png", "novel": True},
                         {"file": "a_yaw-12.png", "novel": True}]}
    rows = [{"image": "/x/a_eval.png", "Q_F": "0.9", "Q_R": "0.8", "Q_final": "0.85"},
            {"image": "/x/a_yaw+12.png", "Q_F": "0.5", "Q_R": "0.6", "Q_final": "0.55"},
            {"image": "/x/a_yaw-12.png", "Q_F": "", "Q_R": "", "Q_final": "bad"}]
    agg = sg.aggregate(rows, cams)
    assert agg["overall"]["n"] == 2 and agg["eval_pose"]["mean"] == 0.85 and agg["novel_pose"]["mean"] == 0.55
    good = {"overall": {"mean": 0.8}}; bad = {"overall": {"mean": 0.4}}
    assert sg.compare({"good": good, "bad": bad}, ["good", "bad"])["ranking_correct"] is True
    assert sg.compare({"good": good, "bad": bad}, ["bad", "good"])["ranking_correct"] is False
    assert sg.compare({"good": good, "bad": {}}, ["good", "bad"])["ranking_correct"] is False


def test_run_gate_without_renders_reports_and_dry_run_never_calls_scoda(tmp_path: Path, monkeypatch):
    job = tmp_path / "splat_abc123"; (job / "processed" / "images").mkdir(parents=True)
    _img(job / "processed" / "images" / "0.jpg")
    res = sg.run_gate(job, dry_run=True)
    assert res["renders_present"] is False and "error" in res
    rd = job / "_health" / "scoda" / "renders"; rd.mkdir(parents=True)
    _img(rd / "cam0000_eval.png"); (rd / "cams.json").write_text(json.dumps({"rendered": [{"file": "cam0000_eval.png", "novel": False}]}))
    called = []
    monkeypatch.setattr(sg.subprocess, "run", lambda *a, **k: called.append(a) or (_ for _ in ()).throw(AssertionError("must not run")))
    res = sg.run_gate(job, dry_run=True)
    assert res["renders"] == 1 and res["reference_views"] == 1 and not called and res["report_only"] is True
