// Export Center — the /view workspace Export tab (professionalization wave 6).
// Four lanes against the real backend surfaces (shapes mirrored in
// contracts.ts, read from source 2026-07-25):
//   1. Portable formats — GET/POST /jobs/{id}/exports        (export_route.py)
//   2. Game engines     — POST /unreal-bundle + POST /collision (export_route.py)
//   3. Mesh / DCC       — POST /jobs/{id}/mesh                (splat_route.py)
//   4. Survey / CAD     — POST /jobs/{id}/geo/contours        (geo_route.py)
// Builds are long BLOCKING POSTs. Busy state layers the useActivity() poll
// (server-truth in-process locks: `exporting` covers formats/collision/UE
// bundle, `meshing` covers the mesh build) over local isPending, so a
// reloaded tab still sees a running build. Contours holds a lock the
// activity poll does NOT expose — a concurrent run surfaces as the backend's
// own verbatim 409.
// Real-data receipts baked in (STATUS.md "PROFESSIONALIZATION WAVES 1-3"):
// CPU SOG at 10 iterations blew the 60-min timeout on a 1.32M-gaussian scene
// (default LOW + big-scene warning), streamed-SOG hits an upstream
// splat-transform bug ("Missing lod assignment"), and failed artifacts carry
// the tool-log tail under `error` (no `reason` on failures) — rendered as a
// collapsed monospace tail.
import { useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  buildCollision,
  buildGroundContours,
  buildPortableExports,
  buildSplatMesh,
  buildUnrealBundle,
  fetchPortableExports,
} from "@/lib/api";
import { useActivity } from "@/lib/use-activity";
import { relTime } from "@/lib/format";
import type {
  SplatCollisionRequest,
  SplatExportArtifact,
  SplatExportManifest,
  SplatJob,
} from "@/lib/contracts";
import { DownloadMenu } from "@/components/gallery/download-menu";
import { Button, Input, SectionLabel, Skeleton, useToast } from "@/components/ui";
import {
  AlertTriangle,
  Box,
  Check,
  Download,
  Gamepad2,
  Layers,
  Loader2,
  Map,
  PackageOpen,
  Shapes,
  X,
} from "lucide-react";

// "Big scene" for the SOG-iterations timeout warning: the live failure was at
// 1.32M gaussians; 1M also matches the backend's own streamed-SOG threshold.
const BIG_SCENE_GAUSSIANS = 1_000_000;

// ── tiny shared pieces ───────────────────────────────────────────────────────

function BusyBanner({ children }: { children: ReactNode }) {
  return (
    <p className="flex items-start gap-1.5 rounded-lg border border-cyan-300/25 bg-cyan-400/10 px-2.5 py-2 text-[11px] leading-snug text-cyan-100">
      <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin" /> {children}
    </p>
  );
}

function ErrorBanner({ message }: { message: string | null | undefined }) {
  if (!message) return null;
  return (
    <p className="flex items-start gap-1.5 rounded-lg border border-red-400/25 bg-red-400/10 px-2.5 py-2 text-xs leading-snug text-red-200">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {message}
    </p>
  );
}

function AmberNote({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2.5 py-1.5 text-[11px] leading-snug text-amber-100/85">
      {children}
    </p>
  );
}

// Collapsed tool-log tail for FAILED artifacts. Failures carry the last lines
// of the converter log under `error` (there is no short `reason` on failures
// — `reason` only exists on skips), so the honest rendering is the log tail.
function ErrorTail({ error }: { error: string }) {
  const [open, setOpen] = useState(false);
  const tail = error.trim().split("\n").slice(-6).join("\n");
  return (
    <div className="mt-1.5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-[10px] font-semibold uppercase tracking-wide text-red-300/80 hover:text-red-200"
      >
        {open ? "hide error log" : "show error log"}
      </button>
      {open && (
        <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg border border-red-400/20 bg-black/40 p-2 font-mono text-[10px] leading-relaxed text-red-200/90">
          {tail}
        </pre>
      )}
    </div>
  );
}

function Disclosure({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500 hover:text-zinc-300"
      >
        {open ? "▾ advanced" : "▸ advanced"} {label && `· ${label}`}
      </button>
      {open && <div className="mt-2 space-y-2">{children}</div>}
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
  disabled,
  hint,
  widthClass = "w-24",
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
  hint?: string;
  widthClass?: string;
}) {
  return (
    <label className="flex items-center justify-between gap-2">
      <span className="min-w-0 text-[11px] text-zinc-400">
        {label}
        {hint && <span className="block text-[10px] text-zinc-600">{hint}</span>}
      </span>
      <Input
        size="xs"
        inputMode="decimal"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={`${widthClass} shrink-0`}
      />
    </label>
  );
}

function CheckRow({
  checked,
  onChange,
  disabled,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  label: string;
  hint?: string;
}) {
  return (
    <label className={`flex items-start gap-2 ${disabled ? "opacity-50" : "cursor-pointer"}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="mt-0.5 h-3.5 w-3.5 accent-cyan-400"
      />
      <span className="min-w-0">
        <span className="block text-xs font-medium text-zinc-200">{label}</span>
        {hint && <span className="block text-[10px] leading-snug text-zinc-500">{hint}</span>}
      </span>
    </label>
  );
}

// Segmented control (radio semantics) — mode/version pickers.
function Segmented<T extends string>({
  options,
  value,
  onChange,
  disabled,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded-lg border border-white/12">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          disabled={disabled}
          onClick={() => onChange(o.value)}
          className={`px-2.5 py-1 text-[11px] font-semibold transition ${
            o.value === value ? "bg-cyan-400/20 text-cyan-100" : "bg-white/[0.03] text-zinc-400 hover:text-zinc-200"
          } disabled:cursor-not-allowed disabled:opacity-50`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function DownloadRow({ href, label, hint, filename }: { href: string; label: string; hint?: string; filename?: string }) {
  return (
    <a
      href={href}
      download={filename}
      className="flex items-center justify-between gap-2 rounded-lg border border-white/10 bg-white/[0.02] px-2.5 py-1.5 transition hover:border-cyan-400/30 hover:bg-white/5"
    >
      <span className="min-w-0">
        <span className="block truncate text-xs font-medium text-zinc-100">{label}</span>
        {hint && <span className="block text-[10px] text-zinc-500">{hint}</span>}
      </span>
      <Download className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
    </a>
  );
}

function LaneCard({ icon: Icon, title, children }: { icon: typeof Box; title: string; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
      <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
        <Icon className="h-3.5 w-3.5 text-cyan-200" /> {title}
      </p>
      {children}
    </div>
  );
}

// Positive-finite / integer parsers returning null on bad input — every form
// pre-validates so users get a friendly message instead of a FastAPI 422.
function parseNum(raw: string): number | null {
  const v = Number(raw.trim());
  return Number.isFinite(v) ? v : null;
}
function parseInt_(raw: string): number | null {
  const v = parseNum(raw);
  return v !== null && Number.isInteger(v) ? v : null;
}

// ── 1. Portable formats ──────────────────────────────────────────────────────

const PORTABLE_FORMATS: { key: string; label: string; hint: string; ext: string }[] = [
  { key: "spz", label: "SPZ", hint: "cross-tool interchange · smallest file", ext: "spz" },
  { key: "sog", label: "SOG (bundled)", hint: "web delivery · CPU-compressed", ext: "sog" },
  { key: "streamed-sog", label: "Streamed SOG", hint: "multi-LOD web runtime · manifest + chunks", ext: "json" },
  { key: "gltf", label: "Gaussian GLB", hint: "KHR_gaussian_splatting · DCC import", ext: "glb" },
];

const CHIP_STYLE: Record<string, string> = {
  ready: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
  failed: "border-red-400/30 bg-red-400/10 text-red-300",
  skipped: "border-white/15 bg-white/5 text-zinc-400",
  stale: "border-amber-400/30 bg-amber-400/10 text-amber-300",
  "not-built": "border-white/10 bg-white/[0.03] text-zinc-500",
};

function StatusChip({ status }: { status: SplatExportArtifact["status"] | "not-built" }) {
  const text =
    status === "ready" ? (
      <>
        <Check className="h-3 w-3" /> ready
      </>
    ) : status === "failed" ? (
      <>
        <X className="h-3 w-3" /> failed
      </>
    ) : status === "not-built" ? (
      "not built"
    ) : (
      status
    );
  return (
    <span
      className={`inline-flex items-center gap-0.5 rounded-full border px-1.5 py-0.5 text-[10px] font-semibold ${CHIP_STYLE[status]}`}
    >
      {text}
    </span>
  );
}

function PortableFormatsCard({
  job,
  manifest,
  manifestLoading,
  manifestError,
  exporting,
}: {
  job: SplatJob;
  manifest: SplatExportManifest | undefined;
  manifestLoading: boolean;
  manifestError: string | null;
  exporting: boolean;
}) {
  const qc = useQueryClient();
  const toast = useToast();

  const [spzVersion, setSpzVersion] = useState<3 | 4>(4);
  // Backend default is 10 — the UI defaults LOW: 10 iterations exceeded the
  // 60-min conversion timeout live on the 1.32M-gaussian proof scene.
  const [sogIterations, setSogIterations] = useState("2");
  const [lodChunkCount, setLodChunkCount] = useState("512");
  const [lodChunkExtent, setLodChunkExtent] = useState("16");
  const [forceStreamed, setForceStreamed] = useState(false);
  const [overwrite, setOverwrite] = useState(false);

  const iterations = parseInt_(sogIterations);
  const chunkCount = parseInt_(lodChunkCount);
  const chunkExtent = parseNum(lodChunkExtent);
  const formError =
    iterations === null || iterations < 1 || iterations > 100
      ? "SOG iterations must be a whole number from 1 to 100."
      : chunkCount === null || chunkCount < 32 || chunkCount > 4096
        ? "LOD chunk count must be a whole number from 32 to 4096 (thousands of splats per chunk)."
        : chunkExtent === null || chunkExtent <= 0 || chunkExtent > 10000
          ? "LOD chunk extent must be a positive number up to 10,000 scene units."
          : null;

  const gaussians = manifest?.source?.gaussian_count ?? job.stats?.gaussians ?? null;
  const bigScene = gaussians !== null && gaussians >= BIG_SCENE_GAUSSIANS;
  const showTimeoutWarning = iterations !== null && iterations > 5 && bigScene;

  const build = useMutation({
    mutationFn: () =>
      buildPortableExports(job.job_id, {
        spz_version: spzVersion,
        sog_iterations: iterations ?? 2,
        lod_chunk_count_k: chunkCount ?? 512,
        lod_chunk_extent: chunkExtent ?? 16,
        force_streamed_sog: forceStreamed,
        overwrite,
      }),
    onSuccess: (data) => {
      qc.setQueryData(["portable-exports", job.job_id], data);
      const r = data.result;
      if (!r) {
        toast("Export build finished", "success");
        return;
      }
      const parts = [
        r.built.length && `built ${r.built.join(", ")}`,
        r.cached.length && `current ${r.cached.join(", ")}`,
        r.skipped.length && `skipped ${r.skipped.join(", ")}`,
        r.failures.length && `${r.failures.length} failed`,
      ].filter(Boolean);
      toast(`Formats: ${parts.join(" · ")}`, r.failures.length ? "error" : "success");
    },
  });

  const busy = exporting || build.isPending;
  const stale = manifest?.status === "stale";

  return (
    <LaneCard icon={Box} title="Portable formats">
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">
        SPZ, SOG, and GLB derived from the canonical full-fidelity PLY, with SHA-256 checksums in a
        signed-off manifest.
      </p>

      <div className="mt-2 space-y-2">
        {busy && (
          <BusyBanner>
            An export build is running for this scene. Large scenes can take many minutes — leave this
            open or check back.
          </BusyBanner>
        )}
        <ErrorBanner message={(build.error as Error | null)?.message} />
        {manifestError && <ErrorBanner message={manifestError} />}
        {stale && (
          <div className="flex items-center justify-between gap-2 rounded-lg border border-amber-400/25 bg-amber-400/10 px-2.5 py-2">
            <span className="text-[11px] leading-snug text-amber-100/90">
              The scene changed since these formats were built — they no longer match the current splat.
            </span>
            <Button type="button" size="sm" variant="outline" disabled={busy || formError !== null} onClick={() => build.mutate()}>
              Rebuild
            </Button>
          </div>
        )}

        {manifestLoading ? (
          <div className="space-y-1.5">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : (
          <div className="space-y-1.5">
            {PORTABLE_FORMATS.map((f) => {
              const artifact = manifest?.artifacts?.[f.key];
              const status = artifact?.status ?? "not-built";
              return (
                <div key={f.key} className="rounded-lg border border-white/10 bg-white/[0.02] px-2.5 py-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-zinc-100">{f.label}</p>
                      <p className="text-[10px] text-zinc-500">{f.hint}</p>
                    </div>
                    <div className="flex shrink-0 items-center gap-1.5">
                      <StatusChip status={status} />
                      {status === "ready" && artifact?.url && (
                        <a
                          href={artifact.url}
                          download={`${job.job_id}.${f.ext}`}
                          className="rounded-md border border-white/15 bg-white/5 px-2 py-1 text-[11px] font-semibold text-zinc-200 hover:bg-white/10"
                        >
                          Download
                        </a>
                      )}
                    </div>
                  </div>
                  {status === "skipped" && artifact?.reason && (
                    <p className="mt-1 text-[10px] leading-snug text-zinc-500">{artifact.reason}</p>
                  )}
                  {status === "failed" && artifact?.error && <ErrorTail error={artifact.error} />}
                </div>
              );
            })}
            {manifest?.status === "ready" && manifest.manifest_url && (
              <DownloadRow
                href={manifest.manifest_url}
                label="Export manifest"
                hint="bounds · scale · provenance · SHA-256"
              />
            )}
          </div>
        )}
      </div>

      <div className="mt-3 space-y-2 border-t border-white/10 pt-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] text-zinc-400">SPZ version</span>
          <Segmented
            options={[
              { value: "3", label: "v3" },
              { value: "4", label: "v4 (default)" },
            ]}
            value={String(spzVersion) as "3" | "4"}
            onChange={(v) => setSpzVersion(Number(v) as 3 | 4)}
            disabled={busy}
          />
        </div>
        <NumField
          label="SOG iterations"
          hint="1–100 · higher compresses better but is much slower"
          value={sogIterations}
          onChange={setSogIterations}
          disabled={busy}
          widthClass="w-20"
        />
        {showTimeoutWarning && (
          <AmberNote>
            High iterations can exceed the 60-min build timeout on large scenes
            {gaussians !== null && <> (this one has {gaussians.toLocaleString()} gaussians)</>}.
          </AmberNote>
        )}
        <Disclosure label="streamed SOG · overwrite">
          <AmberNote>Streamed SOG is currently unreliable on large scenes (upstream tool bug).</AmberNote>
          <NumField
            label="LOD chunk count (k)"
            hint="32–4096 · thousands of splats per chunk"
            value={lodChunkCount}
            onChange={setLodChunkCount}
            disabled={busy}
            widthClass="w-20"
          />
          <NumField
            label="LOD chunk extent"
            hint="scene units per chunk"
            value={lodChunkExtent}
            onChange={setLodChunkExtent}
            disabled={busy}
            widthClass="w-20"
          />
          <CheckRow
            checked={forceStreamed}
            onChange={setForceStreamed}
            disabled={busy}
            label="Force streamed SOG"
            hint="Scenes under 1M gaussians auto-skip it (bundled SOG is more efficient); this overrides."
          />
          <CheckRow
            checked={overwrite}
            onChange={setOverwrite}
            disabled={busy}
            label="Overwrite current artifacts"
            hint="Rebuild formats that are already up to date instead of reusing them."
          />
        </Disclosure>
        {formError && <p className="text-[11px] leading-snug text-rose-300/90">{formError}</p>}
        <Button
          type="button"
          size="sm"
          className="w-full"
          disabled={busy || formError !== null}
          onClick={() => build.mutate()}
        >
          {build.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Box className="h-3.5 w-3.5" />}
          {manifest?.status === "ready" ? "Rebuild formats" : "Build formats"}
        </Button>
        <p className="text-[10px] leading-snug text-zinc-600">
          Builds run on the server and block until finished — already-current formats are reused unless
          Overwrite is on.
        </p>
      </div>
    </LaneCard>
  );
}

// ── 2. Game engines ──────────────────────────────────────────────────────────

function GameEnginesCard({
  job,
  manifest,
  exporting,
}: {
  job: SplatJob;
  manifest: SplatExportManifest | undefined;
  exporting: boolean;
}) {
  const qc = useQueryClient();
  const toast = useToast();

  const unreal = useMutation({
    mutationFn: () => buildUnrealBundle(job.job_id),
    onSuccess: (bundle) => {
      toast(`UE 5.6 bundle packaged — ${bundle.files} files`, "success");
      void qc.invalidateQueries({ queryKey: ["portable-exports", job.job_id] });
    },
  });

  // Collision form. Visible: mode + voxel size. Everything else defaults to
  // the backend's CollisionRequest values under Advanced.
  const [mode, setMode] = useState<"exterior" | "interior" | "raw">("exterior");
  const [voxelSize, setVoxelSize] = useState("0.05");
  const [opacityThreshold, setOpacityThreshold] = useState("0.1");
  const [fillSize, setFillSize] = useState("1.6");
  const [clusterResolution, setClusterResolution] = useState("1.0");
  const [clusterOpacity, setClusterOpacity] = useState("0.999");
  const [clusterMinContribution, setClusterMinContribution] = useState("0.1");
  const [carve, setCarve] = useState(false);
  const [carveHeight, setCarveHeight] = useState("1.6");
  const [carveRadius, setCarveRadius] = useState("0.2");
  const [meshStyle, setMeshStyle] = useState<"smooth" | "faces">("smooth");
  const [seedX, setSeedX] = useState("0");
  const [seedY, setSeedY] = useState("0");
  const [seedZ, setSeedZ] = useState("0");

  function collisionRequest(): SplatCollisionRequest | string {
    const voxel = parseNum(voxelSize);
    if (voxel === null || voxel <= 0 || voxel > 10) return "Voxel size must be > 0 and ≤ 10 scene units.";
    const opacity = parseNum(opacityThreshold);
    if (opacity === null || opacity < 0 || opacity > 1) return "Opacity threshold must be within 0–1.";
    const fill = parseNum(fillSize);
    if (fill === null || fill <= 0 || fill > 100) return "Fill size must be > 0 and ≤ 100.";
    const cRes = parseNum(clusterResolution);
    if (cRes === null || cRes <= 0 || cRes > 100) return "Cluster resolution must be > 0 and ≤ 100.";
    const cOp = parseNum(clusterOpacity);
    if (cOp === null || cOp < 0 || cOp > 1) return "Cluster opacity must be within 0–1.";
    const cMin = parseNum(clusterMinContribution);
    if (cMin === null || cMin < 0 || cMin > 1) return "Cluster min contribution must be within 0–1.";
    const cH = parseNum(carveHeight);
    if (cH === null || cH <= 0 || cH > 20) return "Carve height must be > 0 and ≤ 20.";
    const cR = parseNum(carveRadius);
    if (cR === null || cR <= 0 || cR > 10) return "Carve radius must be > 0 and ≤ 10.";
    const sx = parseNum(seedX);
    const sy = parseNum(seedY);
    const sz = parseNum(seedZ);
    if (sx === null || sy === null || sz === null) return "Seed position fields must be numbers.";
    return {
      mode,
      voxel_size: voxel,
      opacity_threshold: opacity,
      fill_size: fill,
      cluster_resolution: cRes,
      cluster_opacity: cOp,
      cluster_min_contribution: cMin,
      carve,
      carve_height: cH,
      carve_radius: cR,
      mesh_style: meshStyle,
      seed_position: [sx, sy, sz],
    };
  }

  const collisionReq = collisionRequest();
  const collisionInvalid = typeof collisionReq === "string";

  const collision = useMutation({
    mutationFn: () => buildCollision(job.job_id, collisionReq as SplatCollisionRequest),
    onSuccess: () => {
      toast("Collision artifacts built", "success");
      void qc.invalidateQueries({ queryKey: ["portable-exports", job.job_id] });
    },
  });

  const busy = exporting || unreal.isPending || collision.isPending;
  const bundle = manifest?.unreal_bundle;
  const collisionArtifact = manifest?.collision;
  const exportsReady = manifest?.status === "ready";

  return (
    <LaneCard icon={Gamepad2} title="Game engines">
      {/* UE 5.6 bundle */}
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">
        One Unreal 5.6 handoff zip: gaussians in every built format, collision, geometry twins, survey
        artifacts, and an import contract with checksums.
      </p>
      <div className="mt-2 space-y-2">
        {busy && exporting && !unreal.isPending && !collision.isPending && (
          <BusyBanner>An export operation is already running for this scene — builds share one lock.</BusyBanner>
        )}
        <ErrorBanner message={(unreal.error as Error | null)?.message} />
        {bundle && (
          <div className="space-y-1.5">
            {bundle.zip_url && (
              <DownloadRow
                href={bundle.zip_url}
                label="Unreal 5.6 bundle .zip"
                hint={`${bundle.files} files · built ${relTime(bundle.built_at)}`}
                filename={`${job.job_id}-unreal-5.6.zip`}
              />
            )}
            {bundle.manifest_url && (
              <DownloadRow href={bundle.manifest_url} label="Bundle manifest" hint="import contract · per-file checksums" />
            )}
          </div>
        )}
        <Button type="button" size="sm" className="w-full" disabled={busy || !exportsReady} onClick={() => unreal.mutate()}>
          {unreal.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Layers className="h-3.5 w-3.5" />}
          {bundle ? "Repackage UE 5.6 bundle" : "Package UE 5.6 bundle"}
        </Button>
        {!exportsReady && (
          <p className="text-[10px] leading-snug text-zinc-600">
            Build (or rebuild) the portable formats above first — the bundle packages them.
          </p>
        )}
      </div>

      {/* Collision */}
      <div className="mt-3 space-y-2 border-t border-white/10 pt-3">
        <p className="text-xs font-semibold text-zinc-100">Collision (GPU voxelization)</p>
        <p className="text-[11px] leading-snug text-zinc-500">
          Voxel occupancy grid + a walkable collision mesh for engine physics. A render-lane artifact —
          never survey truth.
        </p>
        <ErrorBanner message={(collision.error as Error | null)?.message} />
        {collisionArtifact?.status === "ready" && (
          <div className="space-y-1.5">
            {collisionArtifact.mesh_url && (
              <DownloadRow
                href={collisionArtifact.mesh_url}
                label="Collision mesh .glb"
                hint={`${collisionArtifact.mode} mode · built ${relTime(collisionArtifact.built_at)}`}
                filename={`${job.job_id}-collision.glb`}
              />
            )}
            {collisionArtifact.voxel_url && (
              <DownloadRow
                href={collisionArtifact.voxel_url}
                label="Voxel grid .json"
                hint="the .bin payload it references ships inside the UE bundle"
              />
            )}
          </div>
        )}
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] text-zinc-400">Mode</span>
          <Segmented
            options={[
              { value: "exterior", label: "exterior" },
              { value: "interior", label: "interior" },
              { value: "raw", label: "raw" },
            ]}
            value={mode}
            onChange={setMode}
            disabled={busy}
          />
        </div>
        <p className="text-[10px] leading-snug text-zinc-600">
          {mode === "exterior"
            ? "Outdoor/open scene — fills downward from the floor."
            : mode === "interior"
              ? "Room scan — flood-fills outward from the seed point (set it under Advanced)."
              : "Occupied voxels only, no fill."}
        </p>
        <NumField
          label="Voxel size"
          hint="scene units · smaller = finer + slower"
          value={voxelSize}
          onChange={setVoxelSize}
          disabled={busy}
          widthClass="w-20"
        />
        <Disclosure label="carve · cluster · seed">
          <NumField label="Opacity threshold" hint="0–1 · voxel occupancy cutoff" value={opacityThreshold} onChange={setOpacityThreshold} disabled={busy} widthClass="w-20" />
          <NumField label="Fill size" hint="interior/exterior fill kernel" value={fillSize} onChange={setFillSize} disabled={busy} widthClass="w-20" />
          <NumField label="Cluster resolution" value={clusterResolution} onChange={setClusterResolution} disabled={busy} widthClass="w-20" />
          <NumField label="Cluster opacity" hint="0–1" value={clusterOpacity} onChange={setClusterOpacity} disabled={busy} widthClass="w-20" />
          <NumField label="Cluster min contribution" hint="0–1" value={clusterMinContribution} onChange={setClusterMinContribution} disabled={busy} widthClass="w-20" />
          <CheckRow
            checked={carve}
            onChange={setCarve}
            disabled={busy}
            label="Carve a capsule at the seed"
            hint="Clears the scanner's own standing spot out of the collision volume."
          />
          {carve && (
            <>
              <NumField label="Carve height" value={carveHeight} onChange={setCarveHeight} disabled={busy} widthClass="w-20" />
              <NumField label="Carve radius" value={carveRadius} onChange={setCarveRadius} disabled={busy} widthClass="w-20" />
            </>
          )}
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] text-zinc-400">Mesh style</span>
            <Segmented
              options={[
                { value: "smooth", label: "smooth" },
                { value: "faces", label: "faces" },
              ]}
              value={meshStyle}
              onChange={setMeshStyle}
              disabled={busy}
            />
          </div>
          <div>
            <p className="text-[11px] text-zinc-400">
              Seed position <span className="text-[10px] text-zinc-600">(scene units · interior flood-fill start)</span>
            </p>
            <div className="mt-1 grid grid-cols-3 gap-1.5">
              <Input size="xs" inputMode="decimal" placeholder="x" value={seedX} onChange={(e) => setSeedX(e.target.value)} disabled={busy} />
              <Input size="xs" inputMode="decimal" placeholder="y" value={seedY} onChange={(e) => setSeedY(e.target.value)} disabled={busy} />
              <Input size="xs" inputMode="decimal" placeholder="z" value={seedZ} onChange={(e) => setSeedZ(e.target.value)} disabled={busy} />
            </div>
          </div>
        </Disclosure>
        {collisionInvalid && <p className="text-[11px] leading-snug text-rose-300/90">{collisionReq as string}</p>}
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="w-full"
          disabled={busy || collisionInvalid}
          onClick={() => collision.mutate()}
        >
          {collision.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Shapes className="h-3.5 w-3.5" />}
          {collisionArtifact ? "Rebuild collision" : "Build collision"}
        </Button>
      </div>
    </LaneCard>
  );
}

// ── 3. Mesh / DCC ────────────────────────────────────────────────────────────

function MeshDccCard({ job, meshing }: { job: SplatJob; meshing: boolean }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [finetune, setFinetune] = useState(false);
  const [finish, setFinish] = useState(false);
  const [gate, setGate] = useState(false);

  const build = useMutation({
    mutationFn: () => buildSplatMesh(job.job_id, { finetune, finish, gate }),
    onSuccess: (resp) => {
      toast(
        resp.cached
          ? "Mesh already built — links refreshed (turn on Fine-tune to force a rebuild)"
          : "Mesh built",
        "success",
      );
      void qc.invalidateQueries({ queryKey: ["status"] });
    },
  });

  const busy = meshing || build.isPending;
  const links = [
    { url: job.mesh_file_url, label: "Mesh .ply", hint: "raw triangle mesh", filename: `${job.job_id}.ply` },
    { url: job.mesh_glb_url, label: "Mesh .glb", hint: "drag straight into Blender", filename: `${job.job_id}.glb` },
    {
      url: job.twin_glb_url,
      label: "Twin .glb",
      hint: "splat-colored · decimated · real meters · Y-up",
      filename: `${job.job_id}-twin.glb`,
    },
  ].filter((l) => l.url);

  return (
    <LaneCard icon={Shapes} title="Mesh / DCC">
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">
        A conventional triangle mesh from the trained checkpoint, plus the colored, decimated digital
        twin for Blender, CAD, and printing.
      </p>
      <div className="mt-2 space-y-2">
        {busy && (
          <BusyBanner>
            A mesh/object build is running for this scene — mesh ~2 min, fine-tune ~10–15 min. Safe to
            leave this tab.
          </BusyBanner>
        )}
        <ErrorBanner message={(build.error as Error | null)?.message} />
        {links.length > 0 ? (
          <div className="space-y-1.5">
            {links.map((l) => (
              <DownloadRow key={l.label} href={l.url!} label={l.label} hint={l.hint} filename={l.filename} />
            ))}
          </div>
        ) : (
          <p className="text-[11px] text-zinc-500">No mesh built for this scene yet.</p>
        )}
        <div className="space-y-1.5 border-t border-white/10 pt-2">
          <CheckRow
            checked={finetune}
            onChange={setFinetune}
            disabled={busy}
            label="Fine-tune first (DN escalation)"
            hint="~10–15 min of real GPU training · rebuilds the mesh · roughly doubles connectivity on fragmentary scenes"
          />
          <CheckRow
            checked={finish}
            onChange={setFinish}
            disabled={busy || !job.preview_available}
            label="Twin finish"
            hint={
              job.preview_available
                ? "Splat→mesh color transfer + decimate → meters/Y-up twin.glb (~6 s)"
                : "Needs the exported splat preview — generate the preview first."
            }
          />
          <CheckRow
            checked={gate}
            onChange={setGate}
            disabled={busy}
            label="Fidelity gate"
            hint="Renders the mesh through 6 capture cameras — PSNR/SSIM/coverage receipt vs the real photos"
          />
          <Button type="button" size="sm" className="w-full" disabled={busy} onClick={() => build.mutate()}>
            {build.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Shapes className="h-3.5 w-3.5" />}
            {links.length > 0 ? "Rebuild mesh" : "Build mesh"}
          </Button>
          <p className="text-[10px] leading-snug text-zinc-600">
            mesh ~2 min · finetune ~10–15 min. Without Fine-tune, an existing mesh is reused instantly.
          </p>
        </div>
      </div>
    </LaneCard>
  );
}

// ── 4. Survey / CAD ──────────────────────────────────────────────────────────

function PrereqRow({ ok, okText, missingText, fix }: { ok: boolean; okText: string; missingText: string; fix: string }) {
  return (
    <div
      className={`flex items-start gap-1.5 rounded-lg border px-2.5 py-1.5 text-[11px] leading-snug ${
        ok ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-200/90" : "border-amber-300/20 bg-amber-300/10 text-amber-100/85"
      }`}
    >
      {ok ? <Check className="mt-0.5 h-3.5 w-3.5 shrink-0" /> : <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />}
      <span>
        <span className="font-medium">{ok ? okText : missingText}</span>
        {!ok && <span className="block text-amber-100/70">{fix}</span>}
      </span>
    </div>
  );
}

function SurveyCadCard({ job }: { job: SplatJob }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [minorFt, setMinorFt] = useState("0.5");
  const [majorFt, setMajorFt] = useState("2.5");
  const [epsg, setEpsg] = useState("2226");
  const [cellM, setCellM] = useState("0.25");
  const [tinFaces, setTinFaces] = useState(false);

  const hasScale = typeof job.meters_per_unit === "number" && job.meters_per_unit > 0;
  const hasGeo = Boolean(job.geo);
  const prereqsOk = hasScale && hasGeo;
  // Ground sampling needs a mesh OR a language field (semantic AUTO). Soft
  // hint only — the backend's 409 spells out the fix if it comes to that.
  const groundSourceMissing = !job.mesh_file_url && !job.langfield_available;

  const minor = parseNum(minorFt);
  const major = parseNum(majorFt);
  const epsgN = parseInt_(epsg);
  const cell = parseNum(cellM);
  const formError =
    minor === null || major === null || minor <= 0 || minor > major || major > 100
      ? "Contour intervals need 0 < minor ≤ major ≤ 100 (feet)."
      : epsgN === null || epsgN < 1000 || epsgN > 999999
        ? "EPSG must be a plausible code (1000–999999)."
        : cell === null || cell < 0.05 || cell > 5
          ? "Grid cell must be within 0.05–5.0 meters."
          : null;

  const build = useMutation({
    mutationFn: () =>
      buildGroundContours(job.job_id, {
        minor_ft: minor ?? 0.5,
        major_ft: major ?? 2.5,
        epsg: epsgN ?? 2226,
        cell_m: cell ?? 0.25,
        tin_faces: tinFaces,
      }),
    onSuccess: () => {
      toast("Ground contours built — DXF and PNEZD points are ready below", "success");
      void qc.invalidateQueries({ queryKey: ["status"] });
    },
  });

  const busy = build.isPending;
  const links = [
    { url: job.survey_dxf_url, label: "Site DXF", hint: "georeferenced TIN · grid coordinates" },
    { url: job.survey_landxml_url, label: "LandXML surface", hint: "imports as a Civil 3D surface" },
    { url: job.contours_dxf_url, label: "Contours DXF", hint: "ground contours · office layers" },
    { url: job.ground_points_url, label: "Ground points", hint: "PNEZD · survey point file" },
    { url: job.sections_url, label: "Sections view", hint: "cross-sections vs the scene · PNG" },
    { url: job.surface_iso_url, label: "3D surface view", hint: "isometric ground TIN · PNG" },
  ].filter((l) => l.url);

  return (
    <LaneCard icon={Map} title="Survey / CAD">
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">
        Real deliverables on real grid coordinates: a georeferenced TIN surface (DXF + LandXML), ground
        contours on office layers, and PNEZD points — straight from the splat to Civil 3D.
      </p>
      <div className="mt-2 space-y-2">
        {links.length > 0 && (
          <div className="space-y-1.5">
            {links.map((l) => (
              <DownloadRow key={l.label} href={l.url!} label={l.label} hint={l.hint} />
            ))}
          </div>
        )}

        <div className="space-y-1.5 border-t border-white/10 pt-2">
          <p className="text-xs font-semibold text-zinc-100">Build ground contours</p>
          <PrereqRow
            ok={hasScale}
            okText={`Scale calibrated — ${Number(job.meters_per_unit).toPrecision(3)} m per scene unit`}
            missingText="Scale isn't calibrated yet"
            fix="Open the Measure tab, measure a distance you know, and set its real length — that pins meters-per-unit."
          />
          <PrereqRow
            ok={hasGeo}
            okText={hasGeo ? `Geo anchor set — ${job.geo!.lat.toFixed(5)}, ${job.geo!.lon.toFixed(5)}` : ""}
            missingText="No geo anchor yet"
            fix="Open the ⋯ menu → Locate in the world and pin the scene on the map — that anchors grid coordinates."
          />
          {groundSourceMissing && (
            <AmberNote>
              Ground extraction needs a mesh (build one in Mesh / DCC above) or a language field for
              semantic ground.
            </AmberNote>
          )}
          {busy && (
            <BusyBanner>
              Running the ground → TIN → contour pipeline — typically a few minutes on big sites. Leave
              this open or check back.
            </BusyBanner>
          )}
          <ErrorBanner message={(build.error as Error | null)?.message} />
          <div className="grid grid-cols-2 gap-1.5">
            <NumField label="Minor interval (ft)" value={minorFt} onChange={setMinorFt} disabled={busy} widthClass="w-16" />
            <NumField label="Major interval (ft)" value={majorFt} onChange={setMajorFt} disabled={busy} widthClass="w-16" />
          </div>
          <Disclosure label="projection · sampling">
            <NumField label="EPSG" hint="default 2226 — NAD83 / CA zone 2 (US ft)" value={epsg} onChange={setEpsg} disabled={busy} widthClass="w-24" />
            <NumField label="Grid cell (m)" hint="0.05–5.0 · use 0.5–1.0 for big sites" value={cellM} onChange={setCellM} disabled={busy} widthClass="w-20" />
            <CheckRow
              checked={tinFaces}
              onChange={setTinFaces}
              disabled={busy}
              label="Draw TIN faces as review linework"
              hint="Adds the triangulation itself to the DXF for QC."
            />
          </Disclosure>
          {formError && <p className="text-[11px] leading-snug text-rose-300/90">{formError}</p>}
          <Button
            type="button"
            size="sm"
            className="w-full"
            disabled={busy || !prereqsOk || formError !== null}
            onClick={() => build.mutate()}
          >
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Map className="h-3.5 w-3.5" />}
            {job.contours_dxf_url ? "Rebuild ground contours" : "Build ground contours"}
          </Button>
          <p className="text-[10px] leading-snug text-zinc-600">
            Mesh/semantic ground → PNEZD points → CDT TIN → contour DXF on office layers, with receipt
            renders. Semantic ground engages automatically when the scene has a language field.
          </p>
        </div>
      </div>
    </LaneCard>
  );
}

// ── the lane ─────────────────────────────────────────────────────────────────

export function ExportLane({ job }: { job: SplatJob }) {
  const activity = useActivity();
  const flags = activity.data?.jobs[job.job_id];
  const exporting = Boolean(flags?.exporting);
  const meshing = Boolean(flags?.meshing);
  const completed = job.status === "completed";

  // One shared manifest fetch for the Portable + Game-engine cards (same
  // query key the DownloadMenu uses, so the cache is shared with it).
  const manifestQuery = useQuery({
    queryKey: ["portable-exports", job.job_id],
    queryFn: () => fetchPortableExports(job.job_id),
    enabled: completed && Boolean(job.preview_available),
    retry: false,
  });

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <PackageOpen className="h-4 w-4 text-cyan-300" />
          <SectionLabel>Export</SectionLabel>
        </div>
        <DownloadMenu job={job} />
      </div>

      {!completed ? (
        <p className="rounded-2xl border border-white/10 bg-white/[0.02] p-3 text-xs leading-relaxed text-zinc-400">
          Exports need a completed scene — this one is <span className="font-semibold text-zinc-200">{job.status}</span>.
          Every artifact here derives from the final trained splat, so there's nothing to build yet.
        </p>
      ) : (
        <>
          <PortableFormatsCard
            job={job}
            manifest={manifestQuery.data}
            manifestLoading={manifestQuery.isLoading}
            manifestError={manifestQuery.error ? (manifestQuery.error as Error).message : null}
            exporting={exporting}
          />
          <GameEnginesCard job={job} manifest={manifestQuery.data} exporting={exporting} />
          <MeshDccCard job={job} meshing={meshing} />
          <SurveyCadCard job={job} />
        </>
      )}
    </div>
  );
}
