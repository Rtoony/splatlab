"""CPU-side metrics and review pages for fixed-camera fisheye reconstructions."""

from html import escape
import math
import re

import numpy as np


def image_scores(reference, rendered, mask):
    reference, rendered = np.asarray(reference), np.asarray(rendered)
    mask = np.asarray(mask)
    if (reference.ndim != 3 or reference.shape[-1] != 3 or reference.shape != rendered.shape
            or mask.shape != reference.shape[:2] or mask.dtype != np.bool_ or not mask.any()
            or not np.isfinite(reference).all() or not np.isfinite(rendered).all()
            or np.any(reference < 0) or np.any(reference > 1) or np.any(rendered < 0) or np.any(rendered > 1)):
        raise ValueError("Need finite matching RGB images in [0,1] and a nonempty boolean mask")
    difference = reference.astype(np.float64) - rendered.astype(np.float64)
    result = {"pixels": int(mask.size), "masked_pixels": int(mask.sum())}
    for label, values in (("full", difference), ("masked", difference[mask])):
        mse = float(np.mean(values ** 2))
        result[label + "_mse"] = mse
        result[label + "_mae"] = float(np.mean(np.abs(values)))
        result[label + "_psnr_db"] = -10 * math.log10(mse) if mse > 0 else None
    result["perfect_match"] = result["full_mse"] == 0
    return result


def summarize_scores(records):
    if not records:
        raise ValueError("No evaluated views")
    summary = {"views": len(records)}
    for label, weight in (("full", "pixels"), ("masked", "masked_pixels")):
        count = sum(record[weight] for record in records)
        mse = sum(record[label + "_mse"] * record[weight] for record in records) / count
        summary[label + "_mse"] = mse
        summary[label + "_psnr_db"] = -10 * math.log10(mse) if mse > 0 else None
    return summary


def validation_scope(records):
    matches = [re.fullmatch(r"(?:[^/]+/)?lens-[01]-(frame-\d{6})-(?:left|centre|right)\.png", record["image"])
               for record in records]
    if records and all(matches):
        count = len({match.group(1) for match in matches})
        return f"{len(records)} validation crops share {count} held-out timestamps and overlapping directions."
    return f"{len(records)} validation views; independent timestamp grouping is not available in these image identities."


def review_html(receipt):
    rows = []
    for index, record in enumerate(receipt["validation"]):
        figures = "".join(f'<figure><figcaption>{label}</figcaption><a href="renders/{index:03d}-{kind}.png">'
                          f'<img loading="lazy" src="renders/{index:03d}-{kind}.png" alt="{label}"></a></figure>'
                          for label, kind in (("Real held-out photo", "reference"), ("Reconstructed view", "render"), ("Absolute error × 3", "error")))
        score = record["masked_psnr_db"]
        label = f"{score:.2f} dB masked PSNR" if score is not None else "Exact RGB match"
        rows.append(f'<section><h2>{escape(record["image"])} · {label}</h2><div class="views">{figures}</div></section>')
    summary = receipt["summary"]
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Condo exterior — real reconstruction</title><style>
body{font:16px system-ui;background:#101923;color:#e6eff8;margin:0;padding:28px}main{max-width:1500px;margin:auto}
h1{font-size:clamp(1.8rem,4vw,3rem);margin:8px 0}h2{font-size:1rem;overflow-wrap:anywhere}p{line-height:1.6;max-width:1100px}
a{color:#87d6fa}.tag{color:#82dac1;text-transform:uppercase;letter-spacing:.14em}.notice{border-left:3px solid #e4b868;padding:8px 18px;background:#202b35}
.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{padding:8px 0;color:#adbed1}img{width:100%;display:block;border-radius:8px}
section{margin:28px 0;border-top:1px solid #344452;padding-top:12px}.links{display:flex;gap:20px;flex-wrap:wrap} @media(max-width:650px){body{padding:16px}.views{grid-template-columns:1fr}}
</style><main><div class="tag">SplatLab / captured reality / private review</div><h1>Your condo, reconstructed from raw video.</h1>''' + (
        f'<p>{receipt["gaussians"]["rows"]:,} trained Gaussians · {receipt["training_iterations"]:,} iterations · '
        f'{summary["views"]} unseen validation crops. Masked PSNR: {summary["masked_psnr_db"]:.2f} dB. '
        'These are actual model renders, not generated illustrations or replayed source photos.</p>'
        '<p class="notice">Exterior pilot, not an accepted building model. Arbitrary scale; no condo registration. '
        f'{validation_scope(receipt["validation"])} Crops are not independent captures. '
        'Masks may exclude predicted operator/sky regions as well as lens rim/nadir, but cannot recover unseen surfaces. '
        'No independent surface-accuracy claim.</p>'
        '<div class="links"><a href="splat.ply" download>Download full Gaussian splat</a>'
        '<a href="receipt.json">Metrics and provenance</a></div>' + "".join(rows) + '</main></html>')
