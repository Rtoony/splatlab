# Architecture scaffold: from a capture to reviewable building evidence

A building-agnostic pipeline (2026-09-15) that turns a captured exterior into explicit planes, openings and
measurements against a `building.json` model, and files them as review-queue evidence the owner accepts or
rejects. Nothing in it promotes a reconstruction to accepted geometry.

## Stages

| Step | Tool (env) | In | Out |
|---|---|---|---|
| 1 Structure study | `tools/capture-structure.py prepare/masks/analyze` (app venv → SAM3 env via the compute gate, needs a lease JSON with `no_new_starts_after`) — or, for an ORDINARY job (phone video, DSLR stills), `tools/structure-study-from-job.py --job splat_xxx --output DIR` then the same `masks` / `analyze` | X5: fisheye reference package + frozen SfM (`sparse/0`; the reference pins the SfM package by hash). Job: `processed/transforms.json` + `processed/sparse/0/*.bin` (poses, one OPENCV camera, `applied_transform`); frames are undistorted + downsized to one pinhole intrinsics, COLMAP features rescaled to the frame resolution (COLMAP may have run at full size), support = error ≤ 2 px, track ≥ 3, reprojection ≤ 3 px | N views (24 X5 / `--views` job) with SAM3 masks per prompt (`--prompts` overrides the default list: building exterior wall, pavement, sky, vegetation, vehicle, window, garage door, door), tracks + reserved check points |
| 2 Metric depth | `tools/moge-depth-views.py` (sam3d-body env, gated, ~10 s) | structure study | MoGe-2 depth per view scaled by that view's own SfM tracks; `global_meters_per_unit` = an independent metric scale for the SfM frame |
| 3 Scaffold | `tools/architecture-scaffold.py --source moge` (splatops env, CPU, ~20 s) | 1 + 2 (+ registration + model, optional) | `scaffold.json` (planes, plumb wall bases anchored to sparse tracks, openings cast by rays onto walls with exclusive wall assignment, gaps, reserved-track checks, up + Manhattan frame), `scaffold.ply`, per-view overlays, `contact.png`, receipt |
| 4 Registration | `tools/register-by-openings.py` (when no manual alignment exists) | scaffold + MoGe receipt + model + a wall/opening to anchor on | similarity file in the alignment-receipt shape (`cases[].fit.matrix`): planes → rotation, MoGe → scale, a door or the window pair → translation |
| 5 Evidence | `tools/building-evidence.py` | scaffold + MoGe + registration + model + `--building-target kind:id` | `evidence.json` payloads; `--apply` posts to the review API (`--api`), `--resume` finishes a partial post |

Core math lives in `backend/architecture/` (`scaffold_core.py`, `model_match.py`), numpy/scipy only, tested.
Without a pavement plane the scaffold keeps SVD wall bases (not plumb) and skips the up vector; the step-3 receipt hashes
whatever of `surfels.npz` / evaluation receipt exists, so a study directory can stand in for `--evaluation`.
One-shot for a new exterior job: `bash ~/scripts/condo-exterior-job-2026-09-16.sh --job=splat_xxx --apply [--post]`.

## What the evidence contains

Per matched opening a Measurement (width, height, sill rows in metres; basis = visible mask edge, not framing);
a High Question when the model has the same kind of opening elsewhere on that wall (a mirrored door); a Question
per unmodelled opening inside the footprint; one heights Question when window sills disagree with level elevations
(expressed against `--floor-datums`, e.g. `"garage slab=0,studio FF=0.3048"`); a scale-check Reference (MoGe-2 vs
the registration's scale, when the registration is not MoGe itself); a wall-line Reference (offset + angle of every
registered wall plane against the nearest model line); a Measurement per captured projection/recess between two
parallel walls (same level, or cross-level when a model `dimensions` entry documents it).

## Measured behaviour worth knowing

- 2DGS surfel quaternion normals after 3,000 steps are ~random; the model's rendered depth splits walls into layers
  20–60 cm apart. MoGe-2 depth anchored per view to the tracks is view-consistent (street wall: one plane, RMS 9.7 cm).
- Openings must be assigned to one wall by depth agreement on the ring of wall around the mask (glass fools monocular
  depth); attaching an instance to every wall whose footprint contains it produced ghost windows on the plane behind.
- Registration by a single door can be mirrored if the model has the door on the wrong side; anchor on a window pair
  and let the door show up as a position question instead.
- MoGe-2's scale agreed with a garage-door-based alignment to 1 % on the condo but disagreed with a tape reference by
  20 % on another job: treat it as an independent estimate, not a survey.

## Worked example (2286 Chanate)

`~/scripts/splatlab-architecture-scaffold-2026-09-15.sh`, `~/scripts/condo-second-clip-scaffold-2026-09-15.sh`,
`~/scripts/condo-evidence-2026-09-15.sh`; `tools/condo-*.py` are thin wrappers that fill in the condo's target ids and
datums. Report: `~/reports/2026-09-15-splatlab-condo-architecture-scaffold.md`.
