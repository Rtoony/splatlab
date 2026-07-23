#!/usr/bin/env bash
# P6a acceptance gate — scene-regeneration foundations & provenance rails.
# Read-only on job data (scratch in mktemp). No GPU. Exit 0 = P6a holds.
#
# Checks: (1) scene-lane source jobs pinned; (2) hydrant proxy artifact carries
# the 4x4 transform + crop camera + in-file generative tag; (3) the survey lane
# REFUSES tagged geometry (executable contamination test, both entry scripts);
# (4) scene_manifest schema round-trips and fails loud when broken;
# (5) sam3_doctor shallow preflight healthy.
set -uo pipefail

HYDRANT=/home/rtoony/projects/splatcli/outputs/3d/splat_513e89171d
GARDEN=/home/rtoony/projects/splatcli/outputs/3d/splat_32d926d9
MESH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../backend/mesh" && pwd)"
PROBE_PY=/home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python
FAILS=0

say() { printf '%s %s\n' "$1" "$2"; }
check() {  # check <name> <cmd...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then say "OK  " "$name"; else say "FAIL" "$name"; FAILS=$((FAILS+1)); fi
}

# (1) pins
check "hydrant pinned" python3 -c "
import json,sys; sys.exit(0 if json.load(open('$HYDRANT/meta.json')).get('pinned') else 1)"
check "garden pinned" python3 -c "
import json,sys; sys.exit(0 if json.load(open('$GARDEN/meta.json')).get('pinned') else 1)"

# (2) hydrant proxy artifact carries transform + crop cam + in-file tag
check "proxy.json has transform_4x4 + crop_camera_id" python3 -c "
import json,sys
p=json.load(open('$HYDRANT/_objects/fire-hydrant/proxy.json'))
t=p.get('transform_4x4'); ok=(t is not None and len(t)==4 and all(len(r)==4 for r in t)
    and p.get('crop_camera_id') is not None and p.get('provenance')=='proxy')
sys.exit(0 if ok else 1)"
check "proxy.ply carries in-file generative tag" python3 -c "
import sys; sys.path.insert(0,'$MESH_DIR'); import provenance
sys.exit(0 if provenance.ply_is_generative('$HYDRANT/_objects/fire-hydrant/proxy.ply') else 1)"

# (3) contamination test: survey scripts must refuse tagged geometry LOUDLY
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
python3 - "$SCRATCH/tagged.ply" <<'EOF'
import sys
sys.path.insert(0, "/home/rtoony/projects/splatlab/backend/mesh")
from provenance import GENERATIVE_TAG
lines = ["ply", "format ascii 1.0", f"comment {GENERATIVE_TAG}",
         "element vertex 3", "property float x", "property float y", "property float z",
         "element face 1", "property list uchar int vertex_indices", "end_header",
         "0 0 0", "1 0 0", "0 1 0", "3 0 1 2", ""]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
PARAMS='{"job_id":"gate","meters_per_unit":1.0,"epsg":2226,"geo":{"lat":38.44,"lon":-122.71,"heading_deg":0.0,"anchor_scene":[0,0]}}'
refusal() {  # refusal <script> — non-zero exit AND REFUSED on stderr
  local out
  out="$("$PROBE_PY" "$MESH_DIR/$1" "$SCRATCH/tagged.ply" "$SCRATCH/out" --params-json "$PARAMS" 2>&1)"
  local rc=$?
  [ "$rc" -ne 0 ] && grep -q "REFUSED" <<<"$out"
}
check "ground_extract refuses tagged ply" refusal ground_extract.py
check "geo_export refuses tagged ply" refusal geo_export.py

# (4) manifest schema round-trip + fail-loud
check "scene_manifest round-trip + fail-loud" python3 -c "
import sys; sys.path.insert(0,'$MESH_DIR')
import scene_manifest as sm, tempfile
d = tempfile.mkdtemp()
m = sm.new_manifest('splat_gate','scene-units',None)
sm.add_element(m, slug='t', provenance='proxy', files={'ply':'p.ply'},
               transform_4x4=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
sm.write_manifest(d, m); sm.read_manifest(d)
m['doctrine']='trust me'
try:
    sm.validate_manifest(m); sys.exit(1)
except sm.ManifestError:
    sys.exit(0)"

# (5) SAM 3 preflight (shallow)
check "sam3_doctor shallow" python3 "$MESH_DIR/sam3_doctor.py"

if [ "$FAILS" -eq 0 ]; then echo "GATE_P6A: PASS"; exit 0; fi
echo "GATE_P6A: FAIL ($FAILS check(s))"; exit 1
