# SplatLab systemd units

Captured from the live system on 2026-07-27. These files previously existed
**only** under `~/.config/systemd/user/` and nowhere in the repo, so the rails
that actually confine SplatLab — the compute gate, the slice, the secrets
ordering barrier, the worker port — were unreviewable and unrecoverable from a
clone. `deploy/` shipped only the retired Flight A units.

They are copied verbatim, so this directory is a record of what runs, not an
idealised version of it. Nothing here contains a secret value: the units
reference `nexus-svc-inject` and the RAM-only `/dev/shm/nexus-env-*` drop, per
the zero-disk policy.

## Layout

| File | What it does |
|---|---|
| `user/splatlab.service` | The app: FastAPI on `127.0.0.1:3416`. |
| `user/splatlab-langfield.service` | The warm language-field query worker. |
| `user/splatlab-blender-mcp.service` | The restricted Blender MCP, loopback `:9877`. |
| `user/splatlab.slice` | The resource boundary: E-cores 8-15, 400% CPU, 32/48 GiB. |
| `user/*.service.d/50-nexus-secrets-ready.conf` | Boot-ordering barrier — without it the service starts ~6s before `/dev/shm/nexus_session` exists and comes up secretless. |
| `user/*.service.d/60-safety-guard.conf` | Safety guard. |
| `user/*.service.d/70-shared-slice.conf` | Puts the unit in `splatlab.slice`. |
| `user/*.service.d/80-compute-gate.conf` | `ExecCondition` on `tools/splatlab-compute-gate.sh`. |
| `user/splatlab.service.d/90-langfield-worker-url.conf` | Points the app at the worker. |
| `user/splatlab-langfield.service.d/90-worker-port-3443.conf` | Sets the worker's listen port (3443). |

## Worker port (moved 2026-09-30)

The worker listens on **3443**. It used to listen on 3425 — but Panel Studio has owned 3425 since 2026-08-27
(registered in the Nexus manifest), so SplatLab's language queries were answered by Panel Studio with a non-200:
the app never woke its own worker and every search 503'd. The drop-in is renamed to what it does
(`90-worker-port-3443.conf`, was the misleading `90-supervised-port-3418.conf`), the app's
`90-langfield-worker-url.conf` points at 3443, and both code defaults (`splat_route.LANGFIELD_WORKER_URL`,
`langfield_worker.PORT`) match, so a missing drop-in still lands on the right port.

Installing the drop-ins is a live systemd change, so it is a script the operator runs:
`bash ~/scripts/splatlab-langfield-port-3443-2026-09-30.sh --apply` (dry-run by default; backs up, verifies,
prints the rollback).

**2. Flight A units are historical.** `splatlab-flight-a@.service`,
`splatlab-flight-a-boot-recovery.service` and
`splatlab.service.d/90-flight-a-recovery.conf` belong to the retired
hardware-acceptance ladder. They are kept for reference and must not be
installed, enabled or started — see
`~/reports/2026-07-14-system-intent-and-indefinite-splatlab-pause.md`.

## Re-capturing after a change

```bash
cd ~/projects/splatlab
for f in splatlab.service splatlab-langfield.service splatlab-blender-mcp.service splatlab.slice; do
  cp ~/.config/systemd/user/$f deploy/systemd/user/
done
for d in splatlab.service.d splatlab-langfield.service.d; do
  cp ~/.config/systemd/user/$d/*.conf deploy/systemd/user/$d/
done
git -C ~/projects/splatlab diff --stat deploy/systemd
```

Always re-read the diff before committing: a unit that gained an inline secret
must never be committed. The current set references the vault, never values.
