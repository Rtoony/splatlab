#!/usr/bin/env bash
# P6f acceptance gate — scene assembly with the fidelity dial.
# Read-only. No GPU/Blender of its own. Exit 0 = P6f holds.
#
# Checks: (1) toolchain files present; (2) IF a live scene.report.json exists
# for the pinned garden scene, the contamination gate passed, every proxy
# element carries the generative tag and every captured/ground-derived
# element does NOT (independently re-verified here, not just trusting the
# report), and the manifest state is exactly "built" (never auto-approved).
set -uo pipefail

GARDEN=/home/rtoony/projects/splatcli/outputs/3d/splat_32d926d9
MESH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../backend/mesh" && pwd)"
APP_PY=/home/rtoony/projects/splatlab/.venv/bin/python
FAILS=0

say() { printf '%s %s\n' "$1" "$2"; }
check() {  # check <name> <cmd...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then say "OK  " "$name"; else say "FAIL" "$name"; FAILS=$((FAILS+1)); fi
}

for f in scene_assemble.py blender_assemble.py; do
  check "$f present" test -f "$MESH_DIR/$f"
done
check "blender 4.5.11 binary present" test -f /home/rtoony/tools/blender-4.5.11-linux-x64/blender

REPORT="$GARDEN/_regen/scene.report.json"
if [ -f "$REPORT" ]; then
  check "garden scene.report.json contamination gate passed" python3 -c "
import json
r = json.load(open('$REPORT'))
assert r['assemble']['contamination_gate']['ok'] is True, r['assemble']['contamination_gate']
assert r['assemble']['n_built'] >= 1, 'zero elements built'
"
  check "manifest state is 'built', never auto-approved" python3 -c "
import json
r = json.load(open('$REPORT'))
assert r['manifest']['state'] == 'built', r['manifest']['state']
"
  check "independent glTF-extras re-verification matches manifest provenance" "$APP_PY" -c "
import json, struct, sys
sys.path.insert(0, '$MESH_DIR')
import provenance
r = json.load(open('$REPORT'))
with open('$GARDEN/_regen/scene.glb', 'rb') as f:
    f.read(12)
    chunk_len, chunk_type = struct.unpack('<II', f.read(8))
    gltf = json.loads(f.read(chunk_len))
by_slug = {e['slug']: e['provenance'] for e in r['manifest']['elements']}
seen = set()
for node in gltf.get('nodes', []):
    name = node.get('name', '')
    slug = next((s for s in by_slug if name == s or name.startswith(s + '_')), None)
    if slug is None:
        continue
    seen.add(slug)
    extras = node.get('extras') or {}
    has_tag = provenance.GENERATIVE_TAG in (extras.get('splatlab_provenance') or '')
    should = by_slug[slug] == 'proxy'
    assert has_tag == should, f'{slug}: provenance={by_slug[slug]!r} tag_present={has_tag}'
missing = set(by_slug) - seen
assert not missing, f'elements missing from GLB nodes: {missing}'
"
  check "scene.blend present" test -f "$GARDEN/_regen/scene.blend"
else
  say "SKIP" "scene.report.json not built yet (run POST /jobs/{id}/scene/assemble first)"
fi

if [ "$FAILS" -eq 0 ]; then echo "GATE_P6F: PASS"; exit 0; fi
echo "GATE_P6F: FAIL ($FAILS check(s))"; exit 1
