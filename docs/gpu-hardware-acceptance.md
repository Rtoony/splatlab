# GPU Hardware Acceptance

This is an attended post-maintenance evidence gate. It never removes the
hardware-maintenance marker, starts Flight A, installs a unit, or authorizes GPU
work. A `PASS_PRE_FLIGHT_A` receipt is only an input to a separate reviewed
operator decision.

## Preconditions

- Complete the powered-down GPU, 12V-2x6, EPS, support, and connector inspection.
- Load BIOS 3202 defaults, leave memory at JEDEC with XMP disabled, and disable
  ASUS AI/MultiCore overclocking.
- Complete at least four full MemTest86 passes with zero errors.
- Boot normally and keep the canonical maintenance marker present and unchanged.
- Stop every supervisor-denied workload. Run acceptance from a plain local
  terminal, not from Codex, Claude, or another `aipc-safe-run-*.scope`.
- Do not place secrets in evidence. Evidence records and artifacts are local,
  non-secret maintenance data.

## Prepare Private Records

After the final boot, create deliberately incomplete templates in a new private
directory:

```bash
python3 tools/gpu-hardware-acceptance.py \
  --write-evidence-templates \
  /home/rtoony/reports/splatlab-safe-evaluation-2026-07-11/operator-evidence
```

The generator creates a `0700` directory and three exclusive `0600` JSON files.
It refuses to overwrite existing files. The templates have false/placeholding
values and no artifacts, so they cannot pass acceptance without attended edits.

For each record:

1. Set `recorded_at` to the actual completion time in `YYYY-MM-DDTHH:MM:SSZ` UTC.
2. Replace only assertions personally verified during the attended work.
3. Put at least one corresponding non-secret artifact in the same private
   directory, with mode `0600` and a basename containing only letters, digits,
   `.`, `_`, or `-`. Acceptable artifacts are a photo, an exported firmware or
   MemTest report, or a plain-text operator-attestation note recording the
   assertions the operator personally verified. Photographs are optional.
4. Add each artifact basename and its lowercase SHA-256 to `artifacts`.
5. Keep the generated host, boot ID, operator, and UID binding unchanged.

The gate trusts explicit operator assertions for facts it cannot measure. It
rejects symlinks, hard links, relative paths, duplicate JSON keys, public
files/directories, files older than 24 hours, future timestamps, mismatched
file/record timestamps, wrong host or boot IDs, missing artifacts, and artifact
hash mismatches. MemTest evidence is recorded after the final boot and binds the
saved MemTest artifact from the attended run to that boot.

## Run Acceptance

Pass the completed structured JSON records, the exact staged ASUS package, the
explicit counts, and every attended confirmation flag:

```bash
python3 tools/gpu-hardware-acceptance.py \
  --firmware-package /home/rtoony/Downloads/firmware/rog-maximus-z890-hero/ROG-MAXIMUS-Z890-HERO-ASUS-3202.ZIP \
  --firmware-evidence /absolute/private/path/firmware-evidence.template.json \
  --physical-evidence /absolute/private/path/physical-evidence.template.json \
  --memtest-result /absolute/private/path/memtest86-evidence.template.json \
  --memtest-passes 4 \
  --memtest-errors 0 \
  --confirm-gpu-reseated \
  --confirm-gpu-support \
  --confirm-12v-2x6 \
  --confirm-eps \
  --confirm-connectors-undamaged \
  --confirm-bios-defaults \
  --confirm-jedec-memory \
  --confirm-ai-overclocking-disabled
```

The process holds `/run/user/1000/nexus-heavy-work.lock` from its first hardware
check through receipt fsync. It also rejects all Flight A supervisor competing
units and active interactive AI scopes, measures Intel ME from sysfs, verifies
the enabled boot-persistent CPU guard, requires RAPL PL1/PL2 of 125/177 W at the
start and end, and samples those limits throughout the fixed 15-minute idle
window.

Any failed or unreadable condition produces `BLOCKED` or exits before acceptance.
The maintenance marker must still be present afterward.
