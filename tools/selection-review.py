#!/usr/bin/env python3
"""Prepare or seal a review-only selection study; GPU workers are separate gated commands."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import selection_reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "refine", "finalize"])
    parser.add_argument("job", type=Path)
    parser.add_argument("--selected-slug")
    parser.add_argument("--expected-generation", type=int)
    parser.add_argument("--review-id")
    parser.add_argument("--contribution-weighted", action="store_true", help="Opt-in footprint votes with mixed-footprint abstention; requires a sealed contribution worker run")
    parser.add_argument("--spatial-margin-m", type=float, help="Refinement margin from zero to 0.2 metres; default retains the conservative 0.05 metre bound")
    args = parser.parse_args()
    if args.contribution_weighted and args.action != "refine":
        parser.error("--contribution-weighted is only valid for refinement")
    if args.spatial_margin_m is not None and args.action != "refine":
        parser.error("--spatial-margin-m is only valid for refinement")
    if args.action == "prepare":
        if args.selected_slug is None or args.expected_generation is None:
            parser.error("Preparation requires --selected-slug and --expected-generation")
        result = selection_reviews.prepare(args.job.resolve(), args.selected_slug, args.expected_generation)
        result = {key: result[key] for key in ("review_id", "base", "core_count", "scope")}
    else:
        if args.review_id is None:
            parser.error("Refinement/finalization requires --review-id")
        if args.action == "refine":
            result = selection_reviews.refine(args.job.resolve(), args.review_id, contribution_weighted=args.contribution_weighted,
                spatial_margin_m=.05 if args.spatial_margin_m is None else args.spatial_margin_m)
        else:
            result = selection_reviews.finalize(args.job.resolve(), args.review_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
