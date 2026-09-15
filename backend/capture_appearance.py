"""Source-bound, all-view appearance comparisons for captured scene refinements."""

from html import escape
import json
from pathlib import Path
import shutil

from PIL import Image

import artifact_manifest as manifests
from capture_surfels import verify_snapshot
from fisheye_evaluation import summarize_scores
from reference_delivery import artifact_path


CAMERA_KEYS = ("width", "height", "fx", "fy", "cx", "cy", "camera_to_world_opengl")


def read_evaluation(directory: Path):
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "dev.splatlab.fisheye-baseline-evaluation/v1"
            or receipt.get("status") != "evaluated-needs-review"
            or receipt.get("test_split_evaluated") is not False):
        raise ValueError("Require a completed validation-only capture evaluation")
    snapshot = {receipt_path: manifests.sha256_file(receipt_path)}
    snapshot.update({artifact_path(directory, name): digest for name, digest in receipt["files"].items()})
    verify_snapshot(snapshot)
    records = receipt.get("validation", [])
    if not records or len({record["image"] for record in records}) != len(records):
        raise ValueError("Validation image identities must be nonempty and unique")
    for ordinal, record in enumerate(records):
        for kind in ("reference", "render"):
            name = f"renders/{ordinal:03d}-{kind}.png"
            if name not in receipt["files"]:
                raise ValueError("Every compared image must be sealed by its evaluation")
            with Image.open(directory / name) as picture:
                if picture.size != (record["width"], record["height"]):
                    raise ValueError("Compared image dimensions differ from their camera")
                picture.verify()
    return receipt, snapshot


def comparison_html(receipt):
    sections = []
    for ordinal, record in enumerate(receipt["views"]):
        figures = "".join(
            f'<figure><figcaption>{label}</figcaption><a href="images/{ordinal:03d}-{kind}.png">'
            f'<img loading="lazy" src="images/{ordinal:03d}-{kind}.png" alt="{label}"></a></figure>'
            for kind, label in (("reference", "Captured source"), ("before", "Before refinement"),
                                ("after", "After refinement")))
        sections.append(f'<section><h2>{escape(record["image"])}</h2><div class="views">{figures}</div></section>')
    before, after = receipt["before"], receipt["after"]

    def score(value):
        return "exact match" if value is None else f"{value:.2f} dB"

    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chanate captured reality — before and after</title><style>
body{margin:0;padding:24px;background:#111b24;color:#edf4f9;font:16px system-ui}main{max-width:1700px;margin:auto}
h1{font-size:clamp(1.8rem,4vw,3rem)}h2{font-size:1rem;overflow-wrap:anywhere}p{max-width:1100px;line-height:1.6}
a{color:#9fe1f2}.notice{padding:16px;border-left:3px solid #edc07c;background:#22323e}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
figure{margin:0}figcaption{padding:8px 0;color:#bed0df}img{width:100%;display:block;border-radius:8px}section{border-top:1px solid #344856;padding-top:8px;margin-top:24px}
@media(max-width:700px){body{padding:14px}.views{grid-template-columns:1fr}}</style><main>
<p>REALITY RE-GENERATOR / OBSERVED CAPTURE MODE</p><h1>Does more training improve this scene?</h1>''' + (
        f'<p><strong>{"Aggregate appearance improves; inspect every view." if receipt["aggregate_appearance_improved"] else "No overall appearance gain: retain the earlier model as the default."}</strong></p>'
        f'<p>{len(receipt["views"])} matched validation views, all included. '
        f'{before["iterations"]:,} → {after["iterations"]:,} training iterations; '
        f'{before["splats"]:,} → {after["splats"]:,} splats. '
        f'Pixel-weighted masked appearance score: {score(before["summary"]["masked_psnr_db"])} → '
        f'{score(after["summary"]["masked_psnr_db"])}.</p>'
        '<p class="notice">These are actual 3D model renders, not AI-generated illustrations. '
        'Cameras, source photographs, masks and split identity are held fixed. Longer training also changes '
        'the learning-rate schedule; this is a bounded practical refinement, not an equal-compute benchmark. '
        'Overlapping crops are not independent captures. New detail may remain blurry, distorted or missing; '
        'appearance scores are not geometric accuracy. Nothing is accepted as measured architecture.</p>'
        '<p><a href="receipt.json">Complete comparison receipt</a> · '
        '<a href="before-receipt.json">Original run</a> · <a href="after-receipt.json">Refined run</a></p>'
        + "".join(sections) + '</main></html>')


def build_comparison(before_root: Path, after_root: Path, output: Path):
    before_root, after_root, output = before_root.resolve(), after_root.resolve(), output.resolve()
    if output.exists() or any(output.is_relative_to(root) for root in (before_root, after_root)):
        raise ValueError("Choose a new comparison directory outside its frozen inputs")
    before, before_snapshot = read_evaluation(before_root)
    after, after_snapshot = read_evaluation(after_root)
    frozen_keys = [name for name in before["source_hashes"] if name.endswith("/transforms.json")]
    if len(frozen_keys) != 1 or after["source_hashes"].get(frozen_keys[0]) != before["source_hashes"][frozen_keys[0]]:
        raise ValueError("Comparisons require the same exact frozen training camera document")
    dataset_root = str(Path(frozen_keys[0]).parent) + "/"
    frozen_before = {name: digest for name, digest in before["source_hashes"].items() if name.startswith(dataset_root)}
    frozen_after = {name: digest for name, digest in after["source_hashes"].items() if name.startswith(dataset_root)}
    if not frozen_before or frozen_before != frozen_after:
        raise ValueError("Training source photographs, masks or split evidence changed")
    after_indices = {record["image"]: ordinal for ordinal, record in enumerate(after["validation"])}
    if set(after_indices) != {record["image"] for record in before["validation"]}:
        raise ValueError("Comparisons must include the same complete validation view set")
    records, copies = [], []
    for ordinal, prior in enumerate(before["validation"]):
        after_index = after_indices[prior["image"]]
        current = after["validation"][after_index]
        if any(prior.get(key) != current.get(key) for key in CAMERA_KEYS):
            raise ValueError("Compared camera matrices and intrinsics must match exactly")
        reference = f"renders/{ordinal:03d}-reference.png"
        after_reference = f"renders/{after_index:03d}-reference.png"
        if before["files"][reference] != after["files"][after_reference]:
            raise ValueError("Compared source photographs must be byte-identical")
        records.append({"image": prior["image"], "before": prior, "after": current,
                        "reference_sha256": before["files"][reference]})
        copies.extend(((before_root / reference, f"images/{ordinal:03d}-reference.png"),
                       (before_root / f"renders/{ordinal:03d}-render.png", f"images/{ordinal:03d}-before.png"),
                       (after_root / f"renders/{after_index:03d}-render.png", f"images/{ordinal:03d}-after.png")))
    receipt = {"schema": "dev.splatlab.capture-appearance-comparison/v1", "status": "compared-needs-owner-review",
               "created_at": manifests.utc_now(), "generated_completion": False, "accepted_geometry_changed": False,
               "test_appearance_evaluated": False, "all_validation_views_included": True,
               "source_hashes": {str(path): digest for path, digest in {**before_snapshot, **after_snapshot}.items()},
               "dataset_hashes": frozen_before, "views": records}
    for label, result in (("before", before), ("after", after)):
        receipt[label] = {"method": result.get("method", "splatfacto"), "iterations": result["training_iterations"],
                          "splats": result["gaussians"]["rows"], "summary": summarize_scores(result["validation"])}
    receipt["aggregate_appearance_improved"] = all(
        receipt["after"]["summary"][label + "_mse"] < receipt["before"]["summary"][label + "_mse"]
        for label in ("full", "masked"))
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for source, relative in copies:
        shutil.copyfile(source, output / relative)
    for label, root in (("before", before_root), ("after", after_root)):
        shutil.copyfile(root / "receipt.json", output / f"{label}-receipt.json")
    (output / "index.html").write_text(comparison_html(receipt))
    verify_snapshot({**before_snapshot, **after_snapshot})
    receipt["files"] = {path.relative_to(output).as_posix(): manifests.sha256_file(path)
                        for path in output.rglob("*") if path.is_file()}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    return receipt
