// Edit lane — the /view workspace Edit tab. Native Edit-mode v1: wires four
// of edit_ops.py's proven server-side operations (floater cleanup, decimate,
// rigid transforms here; crop-box lives with crop-sphere in the Measure tab's
// Spark tool panel where click-to-place picking exists). Every apply is
// snapshot-versioned server-side (max 5 kept) — the shared Undo row reverts
// to the version_before the last lane edit returned. The SuperSplat escape
// hatch stays for full manual editing, and the Import card closes the
// roundtrip: an externally edited .ply comes back as a versioned edit.
import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { applyEditOps, fetchEditVersions, revertEdit, uploadEditedPly } from "@/lib/api";
import { relTime } from "@/lib/format";
import type { SplatEditApplyResponse, SplatEditOp, SplatJob } from "@/lib/contracts";
import { SemanticEditPanel } from "@/components/edit/semantic-edit";
import { Button, Input, SectionLabel, useToast } from "@/components/ui";
import { Crosshair, ExternalLink, History, Loader2, Move3d, Shrink, Sparkles, Undo2, UploadCloud, Wand2, Wrench } from "lucide-react";

// Backend rails (edit_ops.py pydantic models) — pre-validate so the user gets
// a friendly message instead of a 422 blob.
const ROTATE_LIMIT = 360;
const SCALE_MAX = 1000;
const COORD_BOUND = 1e5;

type PendingOp = "floaters" | "decimate" | "transform" | "import" | "undo" | null;

// "" and "-" mean "not entered" → identity. Anything else must parse.
function parseField(raw: string, identity: number): number {
  const t = raw.trim();
  if (t === "" || t === "-") return identity;
  return Number(t);
}

// Canonical splat PLYs run from a few MB to 1GB+ — MB is the honest unit.
function fmtFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function LangfieldTopologyWarning({ stale }: { stale: boolean }) {
  return (
    <p className="mt-2 rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[10px] leading-snug text-amber-100/85">
      {stale
        ? "This scene's language field is already stale (the scene was edited), so this operation can't make search any worse."
        : "This scene has a language field. This operation changes the gaussian count, so search and paint will stop working until the language field is rebuilt."}
    </p>
  );
}

export function EditLane({ job, onEdited }: { job: SplatJob; onEdited?: () => void }) {
  const toast = useToast();
  const qc = useQueryClient();
  // Restore-point timeline (server keeps at most 5 snapshots).
  const versionsQuery = useQuery({
    queryKey: ["edit-versions", job.job_id],
    queryFn: () => fetchEditVersions(job.job_id),
    staleTime: 5_000,
  });
  const [restoreArmed, setRestoreArmed] = useState<number | null>(null);
  const [restoring, setRestoring] = useState<number | null>(null);
  const [pending, setPending] = useState<PendingOp>(null);
  // Backend {detail} verbatim — the edit routes write human-readable reasons
  // (langfield-stale 409, edit-lock 423, arbiter 503, splat-transform log tail).
  const [error, setError] = useState<string | null>(null);
  // Non-fatal apply warnings (derived-artifact regen failures) — worth showing.
  const [warnings, setWarnings] = useState<string[]>([]);
  // Snapshot seq of the last successful lane apply → the shared Undo target.
  const [lastVersion, setLastVersion] = useState<number | null>(null);
  // Langfield staleness is NOT tracked locally: `job.langfield_stale` arrives
  // via the status poll (STALE-marker truth, same source as the 409 guard),
  // so it survives reloads and sees edits made from other tabs/tools.

  // Decimate: pct is the percentage of gaussians KEPT (splat-transform -F n%
  // — "Use n% to keep a percentage of Gaussians"), not removed.
  const [keepPct, setKeepPct] = useState(50);
  const [decimateArmed, setDecimateArmed] = useState(false);

  // Transform disclosure: string-held numeric fields (blank = identity).
  const [transformOpen, setTransformOpen] = useState(false);
  const [tX, setTX] = useState("");
  const [tY, setTY] = useState("");
  const [tZ, setTZ] = useState("");
  const [rX, setRX] = useState("");
  const [rY, setRY] = useState("");
  const [rZ, setRZ] = useState("");
  const [scale, setScale] = useState("");
  const [transformArmed, setTransformArmed] = useState(false);

  // Import edited .ply (SuperSplat roundtrip return leg).
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importArmed, setImportArmed] = useState(false);
  const [importPct, setImportPct] = useState<number | null>(null);

  useEffect(() => {
    if (!decimateArmed) return;
    const t = window.setTimeout(() => setDecimateArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [decimateArmed]);

  useEffect(() => {
    if (!transformArmed) return;
    const t = window.setTimeout(() => setTransformArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [transformArmed]);

  useEffect(() => {
    if (!importArmed) return;
    const t = window.setTimeout(() => setImportArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [importArmed]);

  const gaussians = job.stats?.gaussians ?? null;
  const estRemaining = gaussians !== null ? Math.round((gaussians * keepPct) / 100) : null;

  const superSplatHref = job.preview_file_url
    ? `/supersplat/?load=${encodeURIComponent(job.preview_file_url)}&filename=${encodeURIComponent(`${job.job_id}.ply`)}`
    : null;

  async function runApply(kind: Exclude<PendingOp, "undo" | null>, ops: SplatEditOp[], describe: (resp: SplatEditApplyResponse) => string) {
    setPending(kind);
    setError(null);
    setWarnings([]);
    try {
      const resp = await applyEditOps(job.job_id, ops);
      setLastVersion(resp.version_before);
      setWarnings(resp.warnings ?? []);
      toast(describe(resp), "success");
      void qc.invalidateQueries({ queryKey: ["edit-versions", job.job_id] });
      onEdited?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Edit failed.");
    } finally {
      setPending(null);
    }
  }

  // Removed count = pre-edit count (last status poll) − post-edit count (the
  // response's freshly recomputed job.stats.gaussians). Both server-derived;
  // if either is missing we say so instead of inventing a number.
  function removalToast(resp: SplatEditApplyResponse, verb: string): string {
    const after = resp.job.stats?.gaussians ?? null;
    if (gaussians !== null && after !== null) {
      return `${verb} ${Math.max(0, gaussians - after).toLocaleString()} splats · restore point v${resp.version_before} saved`;
    }
    return `Done · restore point v${resp.version_before} saved`;
  }

  function runFloaters() {
    // {} → the splat-transform CLI's own -G defaults (backend passes through).
    void runApply("floaters", [{ type: "filter_floaters" }], (resp) => removalToast(resp, "Removed"));
  }

  function runDecimate() {
    void runApply("decimate", [{ type: "decimate", pct: keepPct }], (resp) => removalToast(resp, "Merged away"));
  }

  // Batch only the non-identity ops into ONE apply call. Backend executes ops
  // in list order (splat-transform "ACTIONS executed in order") — scale and
  // rotate act about the origin, translate goes LAST so the typed offset is
  // in final world units, not re-scaled by a later op.
  function buildTransformOps(): SplatEditOp[] | string {
    const sc = parseField(scale, 1);
    const rx = parseField(rX, 0);
    const ry = parseField(rY, 0);
    const rz = parseField(rZ, 0);
    const tx = parseField(tX, 0);
    const ty = parseField(tY, 0);
    const tz = parseField(tZ, 0);
    const all = [sc, rx, ry, rz, tx, ty, tz];
    if (all.some((v) => !Number.isFinite(v))) return "Transform fields must be numbers (blank = no change).";
    if (sc <= 0 || sc > SCALE_MAX) return `Scale must be > 0 and ≤ ${SCALE_MAX}.`;
    if ([rx, ry, rz].some((v) => Math.abs(v) > ROTATE_LIMIT)) return "Rotation is limited to ±360° per axis.";
    if ([tx, ty, tz].some((v) => Math.abs(v) > COORD_BOUND)) return `Translation is limited to ±${COORD_BOUND.toLocaleString()} per axis.`;
    const ops: SplatEditOp[] = [];
    if (sc !== 1) ops.push({ type: "scale", factor: sc });
    if (rx !== 0 || ry !== 0 || rz !== 0) ops.push({ type: "rotate", x: rx, y: ry, z: rz });
    if (tx !== 0 || ty !== 0 || tz !== 0) ops.push({ type: "translate", x: tx, y: ty, z: tz });
    return ops;
  }

  const transformOps = buildTransformOps();
  const transformInvalid = typeof transformOps === "string";
  const transformCount = transformInvalid ? 0 : transformOps.length;

  function runTransform() {
    const ops = buildTransformOps();
    if (typeof ops === "string" || ops.length === 0) return;
    // Rigid translate/rotate/scale do NOT invalidate the language field —
    // the backend won't write a STALE marker, so the poll won't flag one.
    void runApply(
      "transform",
      ops,
      (resp) => `Transform applied (${ops.map((o) => o.type).join(" → ")}) · restore point v${resp.version_before} saved`,
    );
  }

  async function runUndo() {
    if (lastVersion === null) return;
    setPending("undo");
    setError(null);
    try {
      const resp = await revertEdit(job.job_id, lastVersion);
      setLastVersion(null);
      setWarnings([]);
      toast(`Restored v${resp.reverted_to} (${resp.restored_files.length} files)`, "success");
      void qc.invalidateQueries({ queryKey: ["edit-versions", job.job_id] });
      onEdited?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Undo failed.");
    } finally {
      setPending(null);
    }
  }

  function chooseImportFile(f: File | null) {
    setImportArmed(false);
    setError(null);
    if (!f) {
      setImportFile(null);
      return;
    }
    // Pre-flight the extension so a wrong pick fails before a 300MB+ upload;
    // the backend re-validates the actual PLY header regardless.
    if (!f.name.toLowerCase().endsWith(".ply")) {
      setImportFile(null);
      setError(`'${f.name}' is not a .ply file — in SuperSplat use File → Export → PLY.`);
      return;
    }
    setImportFile(f);
  }

  async function runImport() {
    const file = importFile;
    if (!file) return;
    setPending("import");
    setError(null);
    setWarnings([]);
    setImportPct(0);
    try {
      const resp = await uploadEditedPly(job.job_id, file, setImportPct);
      setLastVersion(resp.version_before);
      setWarnings(resp.warnings ?? []);
      setImportFile(null);
      setImportArmed(false);
      toast(
        `Imported ${resp.gaussians.toLocaleString()} splats from ${file.name} · restore point v${resp.version_before} saved`,
        "success",
      );
      void qc.invalidateQueries({ queryKey: ["status"] });
      void qc.invalidateQueries({ queryKey: ["edit-versions", job.job_id] });
      onEdited?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Import failed.");
    } finally {
      setPending(null);
      setImportPct(null);
    }
  }

  const busy = pending !== null;

  async function restoreVersion(seq: number) {
    setRestoring(seq);
    setError(null);
    try {
      const resp = await revertEdit(job.job_id, seq);
      setRestoreArmed(null);
      setLastVersion(null);
      toast(`Restored v${resp.reverted_to} (${resp.restored_files.length} files)`, "success");
      void qc.invalidateQueries({ queryKey: ["edit-versions", job.job_id] });
      onEdited?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Restore failed.");
    } finally {
      setRestoring(null);
    }
  }

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center gap-2">
        <Wrench className="h-4 w-4 text-cyan-300" />
        <SectionLabel>Edit</SectionLabel>
      </div>

      {error && (
        <div className="rounded-xl border border-red-400/30 bg-red-500/10 px-3 py-2 text-xs leading-relaxed text-red-200">
          {error}
        </div>
      )}
      {warnings.length > 0 && (
        <div className="rounded-xl border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-[11px] leading-relaxed text-amber-100/85">
          {warnings.map((w, i) => (
            <p key={i}>{w}</p>
          ))}
        </div>
      )}
      {lastVersion !== null && (
        <div className="flex items-center justify-between gap-2 rounded-xl border border-white/10 bg-white/[0.02] px-3 py-2">
          <span className="text-xs text-zinc-400">
            Restore point <span className="font-semibold text-zinc-200">v{lastVersion}</span> saved before the last edit.
          </span>
          <Button type="button" size="sm" variant="outline" onClick={() => void runUndo()} disabled={busy}>
            {pending === "undo" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Undo2 className="h-3.5 w-3.5" />} Undo
          </Button>
        </div>
      )}
      {job.langfield_stale && (
        <p className="text-[11px] leading-snug text-amber-300/90">
          Language field is stale — the scene was edited after the field was built. Re-run this scene with
          Language search on to rebuild it (retrains the scene).
        </p>
      )}

      {/* Clean up floaters */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
        <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
          <Wand2 className="h-3.5 w-3.5 text-cyan-200" /> Clean up floaters
        </p>
        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
          Removes gaussians that don't contribute to any solid surface (server-side voxel pass, engine
          defaults). There's no live preview for this one — you'll get the exact removed count afterward,
          and Undo brings it all back.
        </p>
        {job.langfield_available && <LangfieldTopologyWarning stale={Boolean(job.langfield_stale)} />}
        <Button type="button" size="sm" className="mt-2 w-full" onClick={runFloaters} disabled={busy}>
          {pending === "floaters" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
          Clean up floaters
        </Button>
      </div>

      {/* Decimate */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
        <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
          <Shrink className="h-3.5 w-3.5 text-cyan-200" /> Decimate
        </p>
        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
          Simplify the scene by progressive pairwise merging — lighter files, faster viewing, softer detail.
        </p>
        <div className="mt-2 flex items-center gap-2">
          <span className="shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">keep</span>
          <input
            type="range"
            min={10}
            max={90}
            step={5}
            value={keepPct}
            onChange={(e) => {
              setKeepPct(Number(e.target.value));
              setDecimateArmed(false);
            }}
            className="w-full"
            disabled={busy}
          />
          <span className="w-10 shrink-0 text-right text-xs text-zinc-400">{keepPct}%</span>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">
          {estRemaining !== null
            ? `~${estRemaining.toLocaleString()} of ${gaussians!.toLocaleString()} splats would remain.`
            : "Splat count unknown for this scene — the result count shows after applying."}
        </p>
        {job.langfield_available && <LangfieldTopologyWarning stale={Boolean(job.langfield_stale)} />}
        <Button
          type="button"
          size="sm"
          variant={decimateArmed ? "primary" : "outline"}
          className={`mt-2 w-full ${decimateArmed ? "border-red-400/50 bg-red-400/20 text-red-100" : ""}`}
          onClick={() => (decimateArmed ? runDecimate() : setDecimateArmed(true))}
          disabled={busy}
        >
          {pending === "decimate" ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : decimateArmed ? (
            `Sure? Permanently merges down to ${keepPct}%`
          ) : (
            `Decimate to ${keepPct}%`
          )}
        </Button>
      </div>

      {/* Transform */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
        <button
          type="button"
          onClick={() => setTransformOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-2 text-left"
        >
          <span className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
            <Move3d className="h-3.5 w-3.5 text-cyan-200" /> Transform
          </span>
          <span className="text-[10px] uppercase tracking-wide text-zinc-500">{transformOpen ? "hide" : "show"}</span>
        </button>
        {transformOpen && (
          <div className="mt-2 space-y-2">
            <p className="text-xs leading-relaxed text-zinc-400">
              Move, rotate (degrees), and uniformly scale the whole scene. Blank fields mean no change; all
              entered values apply as one edit. Rigid transforms don't affect the language field.
            </p>
            <div>
              <p className="text-[10px] uppercase tracking-wide text-zinc-500">Translate (scene units)</p>
              <div className="mt-1 grid grid-cols-3 gap-1.5">
                <Input size="xs" inputMode="decimal" placeholder="x" value={tX} onChange={(e) => setTX(e.target.value)} disabled={busy} />
                <Input size="xs" inputMode="decimal" placeholder="y" value={tY} onChange={(e) => setTY(e.target.value)} disabled={busy} />
                <Input size="xs" inputMode="decimal" placeholder="z" value={tZ} onChange={(e) => setTZ(e.target.value)} disabled={busy} />
              </div>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wide text-zinc-500">Rotate (degrees, ±360)</p>
              <div className="mt-1 grid grid-cols-3 gap-1.5">
                <Input size="xs" inputMode="decimal" placeholder="x°" value={rX} onChange={(e) => setRX(e.target.value)} disabled={busy} />
                <Input size="xs" inputMode="decimal" placeholder="y°" value={rY} onChange={(e) => setRY(e.target.value)} disabled={busy} />
                <Input size="xs" inputMode="decimal" placeholder="z°" value={rZ} onChange={(e) => setRZ(e.target.value)} disabled={busy} />
              </div>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wide text-zinc-500">Uniform scale</p>
              <Input
                size="xs"
                inputMode="decimal"
                placeholder="1.0 (no change)"
                value={scale}
                onChange={(e) => setScale(e.target.value)}
                disabled={busy}
                className="mt-1 w-28"
              />
            </div>
            {typeof transformOps === "string" && (
              <p className="text-[11px] leading-snug text-rose-300/90">{transformOps}</p>
            )}
            <Button
              type="button"
              size="sm"
              variant={transformArmed ? "primary" : "outline"}
              className={`w-full ${transformArmed ? "border-red-400/50 bg-red-400/20 text-red-100" : ""}`}
              onClick={() => (transformArmed ? runTransform() : setTransformArmed(true))}
              disabled={busy || transformInvalid || transformCount === 0}
            >
              {pending === "transform" ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : transformArmed ? (
                `Sure? Permanently applies ${transformCount} op${transformCount === 1 ? "" : "s"}`
              ) : transformCount === 0 ? (
                "Apply transform (enter a change first)"
              ) : (
                `Apply transform (${transformCount} op${transformCount === 1 ? "" : "s"})`
              )}
            </Button>
          </div>
        )}
      </div>

      {/* Crop tools pointer — box + sphere need click-to-place picking, which
          lives in the Spark viewer's tool panel (Measure tab). */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
        <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
          <Crosshair className="h-3.5 w-3.5 text-cyan-200" /> Crop to sphere or box
        </p>
        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
          In the <span className="text-zinc-200">Measure</span> tab's tool panel — pick a center in the
          scene, preview exactly what gets removed in red, apply with undo.
        </p>
      </div>

      {/* Restore points */}
      {(versionsQuery.data?.versions.length ?? 0) > 0 && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
          <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
            <History className="h-3.5 w-3.5 text-cyan-200" /> Restore points
          </p>
          <p className="mt-1 text-xs leading-relaxed text-zinc-400">
            Every edit saves a snapshot first. Restoring saves your current state as a new restore point, so a
            restore is itself undoable. {versionsQuery.data?.max_versions ?? 5} max — oldest are dropped.
          </p>
          <div className="mt-2 space-y-1.5">
            {[...(versionsQuery.data?.versions ?? [])].sort((a, b) => b.seq - a.seq).map((v) => (
              <div key={v.seq} className="flex items-center justify-between gap-2 rounded-lg border border-white/10 bg-white/[0.02] px-2.5 py-1.5">
                <span className="min-w-0 truncate text-xs text-zinc-300" title={v.params ? JSON.stringify(v.params) : undefined}>
                  <span className="font-mono font-semibold text-zinc-100">v{v.seq}</span>
                  <span className="text-zinc-500"> · {v.op} · {relTime(v.ts)}</span>
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant={restoreArmed === v.seq ? "primary" : "outline"}
                  disabled={busy || restoring !== null}
                  onClick={() => (restoreArmed === v.seq ? void restoreVersion(v.seq) : setRestoreArmed(v.seq))}
                >
                  {restoring === v.seq ? <Loader2 className="h-3 w-3 animate-spin" /> : restoreArmed === v.seq ? "Sure?" : "Restore"}
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Semantic edit — text-driven delete/isolate/extract via the language
          field. Renders its own honest disabled card when the field is
          missing or stale (polled job flags), so it mounts unconditionally. */}
      <SemanticEditPanel job={job} onEdited={onEdited} />

      {/* Import edited .ply — the return leg of the SuperSplat roundtrip. */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
        <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
          <UploadCloud className="h-3.5 w-3.5 text-cyan-200" /> Import edited .ply
        </p>
        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
          Bring an externally edited copy of this scene back as the new canonical splat. Any editor that
          exports gaussian-splat .ply works — previews and downloads regenerate automatically.
        </p>
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (!busy && e.dataTransfer.files[0]) chooseImportFile(e.dataTransfer.files[0]);
          }}
          onClick={() => !busy && importInputRef.current?.click()}
          className={`mt-2 rounded-xl border border-dashed border-white/15 bg-white/[0.02] px-3 py-3 text-center transition-colors ${
            busy ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:border-cyan-400/40"
          }`}
        >
          <input
            ref={importInputRef}
            type="file"
            accept=".ply"
            className="hidden"
            disabled={busy}
            onChange={(e) => {
              chooseImportFile(e.target.files?.[0] ?? null);
              e.target.value = ""; // re-picking the same file must re-fire onChange
            }}
          />
          {pending === "import" ? (
            <div className="space-y-2">
              <Loader2 className="mx-auto h-5 w-5 animate-spin text-cyan-300" />
              {importPct !== null && (
                <>
                  <div className="mx-auto h-1.5 w-2/3 overflow-hidden rounded-full bg-white/10">
                    <div className="h-full bg-cyan-400 transition-all" style={{ width: `${importPct}%` }} />
                  </div>
                  <p className="text-[11px] text-zinc-400">Uploading… {importPct}%</p>
                </>
              )}
            </div>
          ) : importFile ? (
            <p className="text-xs text-zinc-200">
              <span className="font-semibold">{importFile.name}</span>
              <span className="text-zinc-500"> · {fmtFileSize(importFile.size)}</span>
            </p>
          ) : (
            <p className="text-xs text-zinc-500">Drop a .ply here or click to browse</p>
          )}
        </div>
        {importFile && pending !== "import" && (
          <Button
            type="button"
            size="sm"
            variant={importArmed ? "primary" : "outline"}
            className={`mt-2 w-full ${importArmed ? "border-red-400/50 bg-red-400/20 text-red-100" : ""}`}
            onClick={() => (importArmed ? void runImport() : setImportArmed(true))}
            disabled={busy}
          >
            {importArmed ? "Sure? Replaces this scene's splat — a restore point is saved first" : `Import ${importFile.name}`}
          </Button>
        )}
      </div>

      {superSplatHref && (
        <a
          href={superSplatHref}
          target="_blank"
          rel="noreferrer"
          className="flex items-start gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-3 transition hover:border-cyan-400/30"
        >
          <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-cyan-200" />
          <span className="min-w-0">
            <span className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
              Edit in SuperSplat <ExternalLink className="h-3 w-3 text-zinc-500" />
            </span>
            <span className="mt-0.5 block text-xs leading-relaxed text-zinc-400">
              The full manual editor — select with rect/brush/picker, delete gaussians, transform, export. Opens in
              a new tab with this scene loaded. When you're done in SuperSplat: File → Export → PLY, then drop the
              file back here.
            </span>
          </span>
        </a>
      )}
    </div>
  );
}
