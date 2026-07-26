// Objects panel — the /view workspace Objects tab (single-object lane, P5b/c).
// Lists every built object from GET /jobs/{id}/objects (the object.json
// isolation receipts + an on-disk files{} map) and drives the multi-minute
// POST /jobs/{id}/objects extraction: name an object → its gaussians as a
// Blender-ready splat, a tight fine-voxel mesh, and opt-in twin finish /
// TripoSplat proxy. Request shape mirrors ObjectIsolateBody (splat_route.py
// ~4501, read from source 2026-07-25). Busy state layers the useActivity()
// poll's `meshing` flag (mesh, object-isolate, and scene-lane builds share
// one server-side lock) over local isPending, so a reloaded tab still sees a
// running build. Scene-WIDE decomposition is a different lane — the P6 Scene
// panel card at the bottom keeps that door.
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchSplatObjects, isolateSplatObject } from "@/lib/api";
import { useActivity } from "@/lib/use-activity";
import type { SplatJob, SplatObjectEntry, SplatObjectFileFormat } from "@/lib/contracts";
import { Button, Input, SectionLabel, Skeleton, useToast } from "@/components/ui";
import { AlertTriangle, Boxes, Download, Layers, Loader2, Scissors } from "lucide-react";

// Download labels per _OBJECT_FILES key (splat_route.py ~4776), in render order.
const OBJECT_FILES: { fmt: SplatObjectFileFormat; label: string; hint: string }[] = [
  { fmt: "splat", label: "Splat .ply", hint: "the object's gaussians · SuperSplat/Blender" },
  { fmt: "ply", label: "Mesh .ply", hint: "raw triangle mesh" },
  { fmt: "glb", label: "Mesh .glb", hint: "raw mesh · drag into Blender" },
  { fmt: "twin", label: "Twin .glb", hint: "colored + decimated · Blender-ready" },
  { fmt: "receipt", label: "Mesh receipt", hint: "render check · PNG" },
  { fmt: "twin-top", label: "Twin top view", hint: "receipt · PNG" },
  { fmt: "twin-oblique", label: "Twin oblique view", hint: "receipt · PNG" },
  { fmt: "textured", label: "Textured .glb", hint: "simplified + UV colour map · Blender-ready" },
  { fmt: "textured-atlas", label: "Colour atlas", hint: "the baked texture map · PNG" },
  { fmt: "proxy", label: "Proxy .ply", hint: "AI-generated stand-in · render/VR only, never survey" },
  { fmt: "proxy-preview", label: "Proxy preview", hint: "WEBP thumbnail" },
];

function parseNum(raw: string): number | null {
  const v = Number(raw.trim());
  return Number.isFinite(v) ? v : null;
}
function parseIntStrict(raw: string): number | null {
  const v = parseNum(raw);
  return v !== null && Number.isInteger(v) ? v : null;
}

function ObjectCard({ job, entry }: { job: SplatJob; entry: SplatObjectEntry }) {
  const files = OBJECT_FILES.filter((f) => entry.files[f.fmt]);
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-zinc-100">{entry.query}</p>
          <p className="truncate font-mono text-[10px] text-zinc-500">{entry.slug}</p>
        </div>
        <span className="shrink-0 rounded-full border border-white/15 bg-white/5 px-2 py-0.5 text-[10px] font-semibold text-zinc-300">
          {entry.expanded_members.toLocaleString()} gaussians
        </span>
      </div>
      <p className="mt-1 text-[10px] leading-snug text-zinc-500">
        {entry.pool_members.toLocaleString()} in the core cluster · expanded ×{entry.expand} at floor{" "}
        {entry.rel_floor}
        {entry.clusters_found > 1 && <> · {entry.clusters_found} candidate clusters found</>}
      </p>
      {files.length > 0 && (
        <div className="mt-2 space-y-1">
          {files.map((f) => (
            <a
              key={f.fmt}
              href={entry.files[f.fmt]!}
              className="flex items-center justify-between gap-2 rounded-lg border border-white/10 bg-white/[0.02] px-2.5 py-1.5 transition hover:border-cyan-400/30 hover:bg-white/5"
            >
              <span className="min-w-0">
                <span className="block truncate text-xs font-medium text-zinc-100">{f.label}</span>
                <span className="block text-[10px] text-zinc-500">{f.hint}</span>
              </span>
              <Download className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
            </a>
          ))}
        </div>
      )}
      <p className="sr-only">job {job.job_id}</p>
    </div>
  );
}

export function ObjectsPanel({ job, onOpenScenePanel }: { job: SplatJob; onOpenScenePanel: () => void }) {
  const toast = useToast();
  const qc = useQueryClient();
  const activity = useActivity();
  const meshing = Boolean(activity.data?.jobs[job.job_id]?.meshing);

  const listing = useQuery({
    queryKey: ["objects", job.job_id],
    queryFn: () => fetchSplatObjects(job.job_id),
    retry: false,
  });

  // Extraction form — defaults mirror ObjectIsolateBody's exactly.
  const [query, setQuery] = useState("");
  const [expand, setExpand] = useState("1.6");
  const [relFloor, setRelFloor] = useState("0.30");
  const [mesh, setMesh] = useState(true);
  const [proxy, setProxy] = useState(false);
  const [finish, setFinish] = useState(false);
  const [targetFaces, setTargetFaces] = useState("10000");
  const [cluster, setCluster] = useState("0");
  const [texture, setTexture] = useState(false);
  const [textureFaces, setTextureFaces] = useState("8000");
  const [textureSize, setTextureSize] = useState("1024");
  const [textureCrop, setTextureCrop] = useState(true);
  const [smooth, setSmooth] = useState(false);
  const [smoothIterations, setSmoothIterations] = useState("2");
  const [smoothFeatureDeg, setSmoothFeatureDeg] = useState("40");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const t = window.setTimeout(() => setArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [armed]);

  const extract = useMutation({
    mutationFn: () =>
      isolateSplatObject(job.job_id, {
        query: query.trim(),
        cluster: parseIntStrict(cluster) ?? 0,
        expand: parseNum(expand) ?? 1.6,
        rel_floor: parseNum(relFloor) ?? 0.3,
        mesh,
        proxy,
        finish: mesh && finish,
        finish_target_faces: parseIntStrict(targetFaces) ?? 10000,
        texture: mesh && texture,
        texture_target_faces: parseIntStrict(textureFaces) ?? 8000,
        texture_size: parseIntStrict(textureSize) ?? 1024,
        texture_crop: textureCrop,
        smooth,
        smooth_iterations: parseIntStrict(smoothIterations) ?? 2,
        smooth_feature_deg: parseNum(smoothFeatureDeg) ?? 40,
      }),
    onSuccess: (resp) => {
      toast(
        `"${resp.object.query}" isolated — ${resp.object.expanded_members.toLocaleString()} gaussians` +
          (resp.object.clusters_found > 1 ? ` (${resp.object.clusters_found} clusters found)` : ""),
        "success",
      );
      setQuery("");
      void qc.invalidateQueries({ queryKey: ["objects", job.job_id] });
      void qc.invalidateQueries({ queryKey: ["status"] });
    },
    onSettled: () => setArmed(false),
  });

  // Gates, most specific first. The two langfield reasons are DISTINCT: stale
  // means a field exists but no longer matches the edited gaussians; missing
  // means the scene never built one. Both fixes retrain the scene.
  const gateReason = !["completed"].includes(job.status)
    ? `Object extraction needs a completed scene — this one is ${job.status}.`
    : job.langfield_stale
      ? "Language field is stale (the scene was edited after it was built) — object extraction needs a matching field. Re-run this scene with Language search on to rebuild it (retrains the scene)."
      : !job.langfield_available
        ? "No language field built for this scene — object extraction needs one. Re-run this scene with Language search on to build it (retrains the scene)."
        : null;

  const q = query.trim();
  const expandN = parseNum(expand);
  const relFloorN = parseNum(relFloor);
  const clusterN = parseIntStrict(cluster);
  const facesN = parseIntStrict(targetFaces);
  const smoothItersN = parseIntStrict(smoothIterations);
  const featureDegN = parseNum(smoothFeatureDeg);
  const formError =
    q.length > 0 && q.length < 2
      ? "Describe the object in at least 2 characters."
      : q.length > 80
        ? "Keep the description under 80 characters."
        : q.startsWith("-")
          ? "The description can't start with a dash."
          : expandN === null || expandN < 1 || expandN > 3
            ? "Expand must be within 1.0–3.0."
            : relFloorN === null || relFloorN <= 0 || relFloorN >= 1
              ? "Relevancy floor must be between 0 and 1 (exclusive)."
              : clusterN === null || clusterN < 0 || clusterN > 5
                ? "Cluster must be a whole number from 0 to 5."
                : finish && (facesN === null || facesN < 1000 || facesN > 100000)
                  ? "Finish target faces must be a whole number from 1,000 to 100,000."
                  : smooth && (smoothItersN === null || smoothItersN < 1 || smoothItersN > 10)
                    ? "Smoothing iterations must be a whole number from 1 to 10."
                    : smooth && (featureDegN === null || featureDegN < 5 || featureDegN > 90)
                      ? "Feature angle must be within 5–90 degrees."
                      : null;

  const busy = meshing || extract.isPending;
  const canSubmit = !gateReason && !busy && q.length >= 2 && formError === null;

  const objects = listing.data?.objects ?? [];

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center gap-2">
        <Boxes className="h-4 w-4 text-cyan-300" />
        <SectionLabel>Objects</SectionLabel>
      </div>

      {/* Built objects */}
      {listing.isLoading ? (
        <div className="space-y-2">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
        </div>
      ) : listing.error ? (
        <p className="rounded-xl border border-red-400/25 bg-red-400/10 px-3 py-2 text-xs leading-snug text-red-200">
          Couldn't list this scene's objects: {(listing.error as Error).message}
        </p>
      ) : objects.length > 0 ? (
        <div className="space-y-2">
          {objects.map((entry) => (
            <ObjectCard key={entry.slug} job={job} entry={entry} />
          ))}
        </div>
      ) : (
        <p className="text-[11px] leading-relaxed text-zinc-500">
          No objects extracted from this scene yet — name one below to get its isolated splat, mesh, and a
          Blender-ready twin.
        </p>
      )}

      {/* Extract form */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
        <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
          <Scissors className="h-3.5 w-3.5 text-cyan-200" /> Extract an object
        </p>
        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
          Name anything in the scene ("the fire hydrant") — the language field finds its gaussians and
          builds a tight fine-voxel mesh of just that object.
        </p>
        {gateReason ? (
          <p className="mt-2 rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[11px] leading-snug text-amber-100/85">
            {gateReason}
          </p>
        ) : (
          <div className="mt-2 space-y-2">
            {busy && (
              <p className="flex items-start gap-1.5 rounded-lg border border-cyan-300/25 bg-cyan-400/10 px-2.5 py-2 text-[11px] leading-snug text-cyan-100">
                <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin" /> An object/mesh build is
                running for this scene — multi-minute; safe to leave this tab.
              </p>
            )}
            {extract.error && (
              <p className="flex items-start gap-1.5 rounded-lg border border-red-400/25 bg-red-400/10 px-2.5 py-2 text-xs leading-snug text-red-200">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {(extract.error as Error).message}
              </p>
            )}
            <Input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setArmed(false);
              }}
              placeholder='What should be extracted? e.g. "the fire hydrant"'
              disabled={busy}
              // The viewer listens for WASD on window — keep typing out of it.
              onKeyDown={(e) => e.stopPropagation()}
              onKeyUp={(e) => e.stopPropagation()}
            />
            <div className="grid grid-cols-2 gap-1.5">
              <label className="flex items-center justify-between gap-2">
                <span className="text-[11px] text-zinc-400">
                  Expand
                  <span className="block text-[10px] text-zinc-600">1.0–3.0 · reach past the core</span>
                </span>
                <Input size="xs" inputMode="decimal" value={expand} onChange={(e) => setExpand(e.target.value)} disabled={busy} className="w-16 shrink-0" />
              </label>
              <label className="flex items-center justify-between gap-2">
                <span className="text-[11px] text-zinc-400">
                  Relevancy floor
                  <span className="block text-[10px] text-zinc-600">0–1 · lower keeps more</span>
                </span>
                <Input size="xs" inputMode="decimal" value={relFloor} onChange={(e) => setRelFloor(e.target.value)} disabled={busy} className="w-16 shrink-0" />
              </label>
            </div>
            <p className="text-[10px] leading-snug text-zinc-600">
              The defaults (1.6 / 0.30) captured a full table with legs and vase in grading; tighter values
              return just the named surface.
            </p>
            <label className={`flex items-start gap-2 ${busy ? "opacity-50" : "cursor-pointer"}`}>
              <input type="checkbox" checked={mesh} onChange={(e) => { setMesh(e.target.checked); if (!e.target.checked) { setFinish(false); setTexture(false); } }} disabled={busy} className="mt-0.5 h-3.5 w-3.5 accent-cyan-400" />
              <span className="min-w-0">
                <span className="block text-xs font-medium text-zinc-200">Build a mesh</span>
                <span className="block text-[10px] leading-snug text-zinc-500">Tight fine-voxel TSDF mesh of just this object (adds most of the build time).</span>
              </span>
            </label>
            <label className={`flex items-start gap-2 ${busy || !mesh ? "opacity-50" : "cursor-pointer"}`}>
              <input type="checkbox" checked={mesh && finish} onChange={(e) => setFinish(e.target.checked)} disabled={busy || !mesh} className="mt-0.5 h-3.5 w-3.5 accent-cyan-400" />
              <span className="min-w-0">
                <span className="block text-xs font-medium text-zinc-200">Twin finish</span>
                <span className="block text-[10px] leading-snug text-zinc-500">
                  {mesh
                    ? "Splat-color transfer + decimate → a simple, colored Blender asset."
                    : "Needs the mesh — turn \"Build a mesh\" back on."}
                </span>
              </span>
            </label>
            {mesh && finish && (
              <label className="flex items-center justify-between gap-2 pl-5">
                <span className="text-[11px] text-zinc-400">
                  Finish target faces
                  <span className="block text-[10px] text-zinc-600">1,000–100,000 · 10k ≈ hero-prop range</span>
                </span>
                <Input size="xs" inputMode="numeric" value={targetFaces} onChange={(e) => setTargetFaces(e.target.value)} disabled={busy} className="w-20 shrink-0" />
              </label>
            )}
            <label className={`flex items-start gap-2 ${busy || !mesh ? "opacity-50" : "cursor-pointer"}`}>
              <input type="checkbox" checked={mesh && texture} onChange={(e) => setTexture(e.target.checked)} disabled={busy || !mesh} className="mt-0.5 h-3.5 w-3.5 accent-cyan-400" />
              <span className="min-w-0">
                <span className="block text-xs font-medium text-zinc-200">Textured asset</span>
                <span className="block text-[10px] leading-snug text-zinc-500">
                  {mesh
                    ? "Poisson refit + decimate + UV unwrap, with the splat's colour baked into a texture map — colour no longer capped by the face budget."
                    : "Needs the mesh — turn \"Build a mesh\" back on."}
                </span>
              </span>
            </label>
            {mesh && texture && (
              <div className="space-y-2 pl-5">
                <label className="flex items-center justify-between gap-2">
                  <span className="text-[11px] text-zinc-400">
                    Texture target faces
                    <span className="block text-[10px] text-zinc-600">500–200,000 · 8k is a simple, solid asset</span>
                  </span>
                  <Input size="xs" inputMode="numeric" value={textureFaces} onChange={(e) => setTextureFaces(e.target.value)} disabled={busy} className="w-20 shrink-0" />
                </label>
                <label className="flex items-center justify-between gap-2">
                  <span className="text-[11px] text-zinc-400">
                    Texture size
                    <span className="block text-[10px] text-zinc-600">256–4096 px · square atlas</span>
                  </span>
                  <Input size="xs" inputMode="numeric" value={textureSize} onChange={(e) => setTextureSize(e.target.value)} disabled={busy} className="w-20 shrink-0" />
                </label>
                <label className="flex items-start gap-2 cursor-pointer">
                  <input type="checkbox" checked={textureCrop} onChange={(e) => setTextureCrop(e.target.checked)} disabled={busy} className="mt-0.5 h-3.5 w-3.5 accent-cyan-400" />
                  <span className="min-w-0">
                    <span className="block text-[11px] text-zinc-400">Cut the ground away</span>
                    <span className="block text-[10px] leading-snug text-zinc-600">TSDF fuses the floor into the object; this crops to the object's own extent.</span>
                  </span>
                </label>
              </div>
            )}
            <label className={`flex items-start gap-2 ${busy ? "opacity-50" : "cursor-pointer"}`}>
              <input type="checkbox" checked={proxy} onChange={(e) => setProxy(e.target.checked)} disabled={busy} className="mt-0.5 h-3.5 w-3.5 accent-cyan-400" />
              <span className="min-w-0">
                <span className="block text-xs font-medium text-zinc-200">Generate a proxy</span>
                <span className="block text-[10px] leading-snug text-zinc-500">
                  AI-generated clean replacement, registered at the object's pose. Plausible, not faithful —
                  render/VR only, never a survey artifact.
                </span>
              </span>
            </label>
            <div>
              <button
                type="button"
                onClick={() => setAdvancedOpen((v) => !v)}
                className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500 hover:text-zinc-300"
              >
                {advancedOpen ? "▾ advanced" : "▸ advanced"} · cluster · smoothing
              </button>
              {advancedOpen && (
                <div className="mt-2 space-y-2">
                  <label className="flex items-center justify-between gap-2">
                    <span className="text-[11px] text-zinc-400">
                      Cluster index
                      <span className="block text-[10px] text-zinc-600">0–5 · 0 = best match; try 1+ if it grabbed the wrong instance</span>
                    </span>
                    <Input size="xs" inputMode="numeric" value={cluster} onChange={(e) => setCluster(e.target.value)} disabled={busy} className="w-14 shrink-0" />
                  </label>
                  <label className={`flex items-start gap-2 ${busy ? "opacity-50" : "cursor-pointer"}`}>
                    <input type="checkbox" checked={smooth} onChange={(e) => setSmooth(e.target.checked)} disabled={busy} className="mt-0.5 h-3.5 w-3.5 accent-cyan-400" />
                    <span className="min-w-0">
                      <span className="block text-xs font-medium text-zinc-200">Curvature-adaptive smoothing</span>
                      <span className="block text-[10px] leading-snug text-zinc-500">
                        Smooths TSDF reconstruction noise on flat/cylindrical surfaces while protecting sharp
                        detail (bolt heads, cap rims). Applies to the twin finish.
                      </span>
                    </span>
                  </label>
                  {smooth && (
                    <div className="grid grid-cols-2 gap-1.5 pl-5">
                      <label className="flex items-center justify-between gap-2">
                        <span className="text-[11px] text-zinc-400">Iterations<span className="block text-[10px] text-zinc-600">1–10</span></span>
                        <Input size="xs" inputMode="numeric" value={smoothIterations} onChange={(e) => setSmoothIterations(e.target.value)} disabled={busy} className="w-12 shrink-0" />
                      </label>
                      <label className="flex items-center justify-between gap-2">
                        <span className="text-[11px] text-zinc-400">Feature angle<span className="block text-[10px] text-zinc-600">5–90°</span></span>
                        <Input size="xs" inputMode="numeric" value={smoothFeatureDeg} onChange={(e) => setSmoothFeatureDeg(e.target.value)} disabled={busy} className="w-12 shrink-0" />
                      </label>
                    </div>
                  )}
                </div>
              )}
            </div>
            {formError && <p className="text-[11px] leading-snug text-rose-300/90">{formError}</p>}
            <Button
              type="button"
              size="sm"
              variant={armed ? "primary" : "outline"}
              className={`w-full ${armed ? "border-red-400/50 bg-red-400/20 text-red-100" : ""}`}
              disabled={!canSubmit}
              onClick={() => (armed ? extract.mutate() : setArmed(true))}
            >
              {extract.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : armed ? (
                "Sure? Multi-minute GPU build"
              ) : (
                <>
                  <Scissors className="h-3.5 w-3.5" /> Extract object
                </>
              )}
            </Button>
            <p className="text-[10px] leading-snug text-zinc-600">
              Multi-minute GPU build (isolation + per-object mesh); safe to leave this tab — the busy banner
              tracks the server, not this page.
            </p>
          </div>
        )}
      </div>

      {/* Scene-wide decomposition is a different lane (P6). */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
        <p className="text-sm font-semibold text-zinc-100">Scene decomposition (P6)</p>
        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
          Enumerate every object in the scene, isolate them from the background, build proxies, extract the
          ground, and reassemble — with an approval gate before anything ships.
        </p>
        <Button size="sm" className="mt-2 w-full" onClick={onOpenScenePanel}>
          <Layers className="h-3.5 w-3.5" /> Open Scene panel
        </Button>
      </div>
    </div>
  );
}
