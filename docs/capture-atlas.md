# GPS / 360 capture atlas

The atlas is SplatLab's private **observed site-context layer**: original recordings,
timestamped GPS tracks and source-linked review panoramas. It is not an architectural
model, terrain surface, registered camera solution or Gaussian reconstruction.

## Current condo packet

- Eight original INSV recordings, 617.867 seconds total; their LRV companions are not
  additional independent camera captures.
- 20 existing provisional review panoramas; four declared reference files are absent.
- 5,998 GPS fixes across six recordings; two short clips have no GPS/review views.
- Eleven review views have tentative locations using unverified container clocks.
- Clip `VID_20260908_123612_00_002.insv` has its first GPS fix **361.09 seconds after
  container start**, outside its 277.377-second video interval. Its nine review images
  remain unplaced. No automatic offset, guessed index-to-track match or model transform
  is applied.

The local originals remain under
`/mnt/storage/media-ingest/insta360-x5/2026-09-08/raw`. The removable SSD is not needed.
Source manifests and photographs are not edited by this workflow.

## Build and open

From the SplatLab root:

```sh
.venv/bin/python tools/capture-atlas.py \
  --handoff /mnt/storage/media-ingest/insta360-x5/2026-09-08/handoff \
  --raw-root /mnt/storage/media-ingest/insta360-x5/2026-09-08/raw \
  --title 'Condo surroundings · GPS / 360 capture atlas'
```

The command reuses unchanged, previously inspected source identities or inspects
and hashes newly selected originals. Its raw hashes must match the handoff. It reads
embedded metadata without video decoding, GPU inference, cloud APIs or model downloads.
Every successful build creates a new `data/spatial/capture-atlases/atlas_*/` snapshot.
Incomplete `.building-*` directories are not listed as completed atlases.

`index.html` embeds its images and scripts and opens offline. Keep the whole directory
for the separate downloadable `atlas.json` and `tracks.geojson` files. `receipt.json`
records hashes for all delivered files. Originals are referenced by basename/hash;
raw machine paths are not exported in the atlas manifest.

An optional bounded local preview is available with:

```sh
.venv/bin/python tools/serve-capture-atlas.py ATLAS_ID --seconds 28800
```

It prints its chosen `http://127.0.0.1:PORT/` address, permits only loopback Host values
and sealed artifact filenames, disables caching, and closes after its time budget.
This is a private preview, not a new registered/public app. Do not tunnel it or expose
the condo location automatically.

## Product integration

The existing authenticated capture router now provides:

- `GET /api/splat/capture-atlases`: completed snapshots.
- `GET /api/splat/capture-atlases/{atlas_id}/{filename}`: hash-checked, allowlisted
  `index.html`, `atlas.json`, `tracks.geojson` and review JPEGs.

The Routes page links available atlases. These source changes and their private build
are staged; no production SplatLab restart or public routing change is part of this
packet. The separate condo publishing/correction session keeps ownership of its app.

## Data and rendering rules

- Each recording is a separate track. Inter-fix gaps over two seconds, invalid fixes,
  non-increasing times and jumps over the declared 25 m/s display threshold split
  tracks. Thresholds are display diagnostics, not a measurement certification.
- Interpolation is limited to accepted temporal segments. Missing/unknown clocks do
  not acquire image locations. Operator-verified clocks still do not certify GPS accuracy.
- WGS84 coordinates are projected to a local east/north tangent display plane at
  zero geographic height. Metres apply to that display frame only. GPS altitude remains
  telemetry with unknown datum; it is not used as terrain or model elevation.
- JSON retains source hashes, exact GPS UTC values, approximate review seek times,
  clock status, missing files, segment breaks and an explicitly absent model transform.
- GeoJSON exports separate two-dimensional segments with timestamp arrays and source
  hashes, not fabricated connected paths between recordings or model alignment.
- The perspective panorama uses CPU Canvas2D and redraws only on load/interaction.
  Drag/arrow keys change viewing direction; scroll or plus/minus changes FOV. The
  180-degree inspection rotation changes the view only, not source pixels or calibration.
- Source images/provenance remain inspectable. No external tiles, CDN scripts, APIs,
  generated imagery or continuous GPU render loop are required.

## Verification

```sh
env PYTHONPATH=backend pytest -q \
  backend/tests/test_capture_atlas.py \
  backend/tests/test_capture_atlas_server.py \
  backend/tests/test_capture_routes.py
node --test --test-isolation=none tools/test-capture-atlas.mjs
node tools/prove-capture-atlas.mjs ATLAS_DIRECTORY NEW_PROOF_DIRECTORY PLAYWRIGHT_MODULE
```

The HTTP tests require local socket access; a sandbox without that access may hang
at TestClient rather than establish a code failure. The browser proof uses existing
Playwright with GPU/3D software rasterization disabled. It verifies all source views,
actual look-around pixels, exact reset, map keyboard selection, empty clips, mobile
width, zero external requests/JS errors and unchanged snapshot hashes.

## Next reconstruction step

Use clip003's existing unstitched pilot as a separate experiment, not the atlas's sparse
generic stitches. Read-only rig diagnostics are available through:

```sh
env OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/review-dual-fisheye-rig.py \
  --images data/spatial/condo-raw-sfm-2026-09-08-01/sparse/0/images.txt \
  --training-list data/spatial/condo-raw-pilot-2026-09-08-01/train-images.txt \
  --output NEW_DIAGNOSTIC_JSON
```

The current 12-pair solution has median/max relative-rotation departures of
0.1210/0.1925 degrees, but estimated lens separation ranges from 0.01424 to 0.14458
arbitrary units. This is a diagnostic observation, not a rig acceptance result.
Next: localize held-out images against frozen training geometry; compare a constrained
rig where justified; verify fisheye projection support before a small Gaussian run.
Use the existing compute gate for any GPU-capable command. Preserve held-out isolation,
private model registration gates and separation of reconstructed versus generated content.
