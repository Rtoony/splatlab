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
| `user/splatlab-langfield.service.d/90-supervised-port-3418.conf` | Sets the worker's listen port. |

## Two known discrepancies

**1. The `90-supervised-port-3418.conf` filename is wrong.** Its contents set
`--port 3425`, which is where the worker actually listens and what
`90-langfield-worker-url.conf` points the app at. There is no 3418. The
investigation that surfaced this read it as a three-way port conflict
(3417/3418/3425); it is really one live port plus a misleading filename plus a
stale code default.

The code defaults were the actionable half and are now fixed —
`splat_route.LANGFIELD_WORKER_URL` and `langfield_worker.PORT` both default to
3425, so if these drop-ins ever go missing the app fails over to the right port
instead of a dead one.

Renaming the file is a live systemd mutation and is deliberately left to the
operator. To do it:

```bash
cd ~/.config/systemd/user/splatlab-langfield.service.d
git -C ~/projects/splatlab show HEAD:deploy/systemd/user/splatlab-langfield.service.d/90-supervised-port-3418.conf > 90-worker-port-3425.conf
rm 90-supervised-port-3418.conf
systemctl --user daemon-reload
systemctl --user restart splatlab-langfield.service
systemctl --user show splatlab-langfield.service -p ExecStart | grep -o 'port 3425'   # receipt
```

Then rename the copy in this directory to match.

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
