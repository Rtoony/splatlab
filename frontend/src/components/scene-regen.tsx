// P6 Control Panel: the GUI front-end for the six scene-regeneration REST
// endpoints (P6a-P6f, backend-only until now) — enumerate -> isolate ->
// proxy -> ground -> assemble -> approve. A lazy-loaded full-screen modal,
// mirroring geo-locate.tsx's mount pattern exactly (job/onClose props,
// `const X = lazy(() => import(...))` in the host page).
//
// Six independently-triggered, freely re-runnable stages with no unifying
// `stage` field — deliberately NOT StageRail (splat.tsx), which is
// contractually tied to one job's auto-advancing single-process stage list.
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchSceneAssemble,
  fetchSceneGround,
  fetchSceneInventory,
  fetchSceneIsolate,
  fetchSceneProxy,
} from "@/lib/api";
import type {
  SceneAssembleReport,
  SceneGroundReport,
  SceneInventoryReport,
  SceneIsolateReport,
  SceneProxyReport,
  SplatJob,
  SplatSceneSummary,
} from "@/lib/contracts";
import { Button, Card, Input, SectionLabel } from "@/components/ui";
import ReceiptLightbox from "@/components/receipt-lightbox";
import {
  AlertTriangle,
  Boxes,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Download,
  Layers,
  Loader2,
  Mountain,
  PackageSearch,
  RefreshCw,
  ShieldCheck,
  Wand2,
  X,
} from "lucide-react";

type StageKey = "inventory" | "isolate" | "proxy" | "ground" | "assemble";

const STAGES: { key: StageKey; label: string; icon: typeof Boxes }[] = [
  { key: "inventory", label: "Inventory", icon: PackageSearch },
  { key: "isolate", label: "Isolate", icon: Boxes },
  { key: "proxy", label: "Proxy", icon: Wand2 },
  { key: "ground", label: "Ground", icon: Mountain },
  { key: "assemble", label: "Assemble", icon: Layers },
];

// Bypasses api.ts's apiRequest() (which throws the raw response body, often
// FastAPI's `{"detail": "..."}` JSON) so the three known gate causes --
// require_heavy_work_admitted()'s maintenance/backup interlock, the
// _mesh_export_lock "already running" 409, and each route's own prerequisite
// check -- surface as the exact human-readable `detail` string the backend
// wrote, not a JSON blob. Matches spark-scene-viewer.tsx's existing pattern.
async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const resp = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    const detail = String(
      ((await resp.json().catch(() => null)) as { detail?: string } | null)?.detail ?? `HTTP ${resp.status}`,
    );
    throw new Error(detail);
  }
  return (await resp.json()) as T;
}

// No server-side cascade invalidation exists (re-running inventory doesn't
// mark isolate/proxy/ground/assemble stale) -- this is the one place that
// catches it, by comparing built_at across meta["scene"].
function staleBanner(scene: SplatSceneSummary | undefined, stage: StageKey, upstream: StageKey): string | null {
  const cur = scene?.[stage]?.built_at;
  const up = scene?.[upstream]?.built_at;
  if (!cur || !up) return null;
  if (up > cur) return `Built from an earlier ${upstream} — re-run ${stage} to pick up the newer ${upstream}.`;
  return null;
}

interface StageDotProps {
  active: boolean;
  running: boolean;
  done: boolean;
}
function StageDot({ active, running, done }: StageDotProps) {
  const color = running ? "bg-cyan-300 animate-pulse" : done ? "bg-emerald-400" : "bg-zinc-600";
  return <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${active ? "ring-2 ring-cyan-300/40" : ""} ${color}`} />;
}

function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <p className="flex items-start gap-1.5 rounded-lg border border-red-400/25 bg-red-400/10 px-2.5 py-2 text-xs leading-snug text-red-200">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {message}
    </p>
  );
}

function StaleNote({ text }: { text: string | null }) {
  if (!text) return null;
  return (
    <p className="flex items-start gap-1.5 rounded-lg border border-amber-400/25 bg-amber-400/10 px-2.5 py-2 text-xs leading-snug text-amber-200">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {text}
    </p>
  );
}

// ── modal shell ────────────────────────────────────────────────────────────
export default function SceneRegenModal({ job, onClose }: { job: SplatJob; onClose: () => void }) {
  const jobId = job.job_id;
  const queryClient = useQueryClient();
  const [activeStage, setActiveStage] = useState<StageKey>("inventory");
  const [lightbox, setLightbox] = useState<string | null>(null);
  // Assembly's fidelity dial is lifted here (not into AssemblePanel) so it
  // survives switching to another tab and back.
  const [fidelityMode, setFidelityMode] = useState<"faithful" | "styled">("styled");
  const [overrides, setOverrides] = useState<Record<string, "captured" | "proxy">>({});

  const invalidateStatus = () => queryClient.invalidateQueries({ queryKey: ["status"] });

  const inventoryMutation = useMutation({
    mutationFn: (body: Record<string, unknown>) => postJSON<SceneInventoryReport>(`/api/splat/jobs/${jobId}/scene/inventory`, body),
    onSuccess: () => {
      invalidateStatus();
      queryClient.invalidateQueries({ queryKey: ["scene-inventory", jobId] });
    },
  });
  const isolateMutation = useMutation({
    mutationFn: (body: Record<string, unknown>) => postJSON<SceneIsolateReport>(`/api/splat/jobs/${jobId}/scene/isolate`, body),
    onSuccess: () => {
      invalidateStatus();
      queryClient.invalidateQueries({ queryKey: ["scene-isolate", jobId] });
    },
  });
  const proxyMutation = useMutation({
    mutationFn: () => postJSON<SceneProxyReport>(`/api/splat/jobs/${jobId}/scene/proxy`, {}),
    onSuccess: () => {
      invalidateStatus();
      queryClient.invalidateQueries({ queryKey: ["scene-proxy", jobId] });
    },
  });
  const groundMutation = useMutation({
    mutationFn: (body: Record<string, unknown>) => postJSON<SceneGroundReport>(`/api/splat/jobs/${jobId}/scene/ground`, body),
    onSuccess: () => {
      invalidateStatus();
      queryClient.invalidateQueries({ queryKey: ["scene-ground", jobId] });
    },
  });
  const assembleMutation = useMutation({
    mutationFn: (body: Record<string, unknown>) => postJSON<SceneAssembleReport>(`/api/splat/jobs/${jobId}/scene/assemble`, body),
    onSuccess: () => {
      invalidateStatus();
      queryClient.invalidateQueries({ queryKey: ["scene-assemble", jobId] });
    },
  });
  const approveMutation = useMutation({
    mutationFn: () => postJSON<{ job_id: string; state: string }>(`/api/splat/jobs/${jobId}/scene/assemble/approve`),
    onSuccess: () => {
      invalidateStatus();
      queryClient.invalidateQueries({ queryKey: ["scene-assemble", jobId] });
    },
  });

  const busy =
    inventoryMutation.isPending ||
    isolateMutation.isPending ||
    proxyMutation.isPending ||
    groundMutation.isPending ||
    assembleMutation.isPending ||
    approveMutation.isPending;

  const scene = job.scene;
  const title = job.input_path?.split("/").pop() || job.job_id;

  return (
    <div className="fixed inset-0 z-50 flex bg-black/70 backdrop-blur-sm">
      <div className="relative m-2 flex min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#05070d] sm:m-4">
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-white/10 px-4 py-3">
          <div className="flex min-w-0 items-center gap-2">
            <Layers className="h-4 w-4 shrink-0 text-cyan-300" />
            <SectionLabel>Scene regeneration</SectionLabel>
            <span className="truncate text-xs text-zinc-500">— {title}</span>
          </div>
          <button type="button" onClick={onClose} className="rounded-full p-1 text-zinc-400 transition hover:bg-white/10 hover:text-zinc-100" title="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        {busy && (
          <div className="flex shrink-0 items-center gap-2 border-b border-cyan-400/20 bg-cyan-400/10 px-4 py-1.5 text-xs text-cyan-100">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> A build is running for this scene — this can take several minutes. Leave this tab open.
          </div>
        )}

        <div className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-white/10 px-3 py-2">
          {STAGES.map(({ key, label, icon: Icon }) => {
            const running =
              (key === "inventory" && inventoryMutation.isPending) ||
              (key === "isolate" && isolateMutation.isPending) ||
              (key === "proxy" && proxyMutation.isPending) ||
              (key === "ground" && groundMutation.isPending) ||
              (key === "assemble" && (assembleMutation.isPending || approveMutation.isPending));
            const done = Boolean(scene?.[key]);
            return (
              <button
                key={key}
                type="button"
                onClick={() => setActiveStage(key)}
                className={`flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                  activeStage === key ? "bg-white/10 text-zinc-100" : "text-zinc-500 hover:text-zinc-200"
                }`}
              >
                <StageDot active={activeStage === key} running={running} done={done} />
                <Icon className="h-3.5 w-3.5" /> {label}
              </button>
            );
          })}
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="mx-auto max-w-4xl">
            {activeStage === "inventory" && (
              <InventoryPanel jobId={jobId} scene={scene} busy={busy} mutation={inventoryMutation} openLightbox={setLightbox} />
            )}
            {activeStage === "isolate" && (
              <IsolatePanel jobId={jobId} scene={scene} busy={busy} mutation={isolateMutation} openLightbox={setLightbox} />
            )}
            {activeStage === "proxy" && (
              <ProxyPanel jobId={jobId} scene={scene} busy={busy} mutation={proxyMutation} openLightbox={setLightbox} />
            )}
            {activeStage === "ground" && (
              <GroundPanel jobId={jobId} scene={scene} busy={busy} mutation={groundMutation} openLightbox={setLightbox} />
            )}
            {activeStage === "assemble" && (
              <AssemblePanel
                jobId={jobId}
                scene={scene}
                busy={busy}
                assembleMutation={assembleMutation}
                approveMutation={approveMutation}
                fidelityMode={fidelityMode}
                setFidelityMode={setFidelityMode}
                overrides={overrides}
                setOverrides={setOverrides}
                openLightbox={setLightbox}
              />
            )}
          </div>
        </div>
      </div>
      {lightbox && <ReceiptLightbox src={lightbox} onClose={() => setLightbox(null)} />}
    </div>
  );
}

// ── shared bits ─────────────────────────────────────────────────────────────
function NumField({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <label className="flex items-center justify-between gap-2 text-xs">
      <span className="text-zinc-400">{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
        className="w-24 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-right text-xs text-zinc-200"
      />
    </label>
  );
}

function Thumb({ src, alt, onOpen }: { src: string; alt: string; onOpen: () => void }) {
  return (
    <button type="button" onClick={onOpen} className="block aspect-square w-full overflow-hidden rounded-lg border border-white/10 bg-black/30">
      <img src={src} alt={alt} loading="lazy" className="h-full w-full object-cover" onError={(e) => ((e.currentTarget as HTMLImageElement).style.opacity = "0.15")} />
    </button>
  );
}

// ── Inventory (P6b) ─────────────────────────────────────────────────────────
function InventoryPanel({
  jobId,
  scene,
  busy,
  mutation,
  openLightbox,
}: {
  jobId: string;
  scene: SplatSceneSummary | undefined;
  busy: boolean;
  mutation: ReturnType<typeof useMutation<SceneInventoryReport, Error, Record<string, unknown>>>;
  openLightbox: (src: string) => void;
}) {
  const [maxNouns, setMaxNouns] = useState(12);
  const [views, setViews] = useState(8);
  const [minViews, setMinViews] = useState(2);
  const [voteFrac, setVoteFrac] = useState(0.3);
  const [overrideNouns, setOverrideNouns] = useState(false);
  const [nounsText, setNounsText] = useState("");
  const [vetoedOpen, setVetoedOpen] = useState(false);
  const [camIdx, setCamIdx] = useState(0);

  const reportQuery = useQuery({
    queryKey: ["scene-inventory", jobId],
    queryFn: () => fetchSceneInventory(jobId),
    enabled: Boolean(scene?.inventory),
    retry: false,
  });
  // Prefer the mutation's own just-returned report over the polled fetch --
  // shows the real result immediately instead of waiting on job.scene to
  // update via the ["status"] poll before the GET query even enables.
  const data = mutation.data ?? reportQuery.data;

  function run() {
    const body: Record<string, unknown> = {
      max_nouns: maxNouns,
      views,
      min_views: minViews,
      vote_frac: voteFrac,
    };
    if (overrideNouns) {
      body.nouns = nounsText.split(",").map((s) => s.trim()).filter(Boolean);
    }
    mutation.mutate(body);
  }

  const instances = data?.instances ?? [];
  const vetoed = data?.vetoed ?? [];
  const camName = String(camIdx).padStart(3, "0");
  const camUrl = `/api/splat/jobs/${jobId}/scene/inventory/file?fmt=overlay&name=${camName}`;

  return (
    <div className="space-y-4">
      <p className="text-xs leading-snug text-zinc-500">
        Names every object in the scene at once — Qwen3-VL + language-field vocab source the nouns, SAM 3 finds every
        instance. Needs a built language field. Re-POST with an explicit noun list to correct what it found (the HITL
        edit path).
      </p>
      {!scene?.inventory && !mutation.data && (
        <p className="text-[11px] text-zinc-500">Not run yet for this scene.</p>
      )}

      <Card className="space-y-2.5 border-white/10 bg-white/5 p-3">
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <NumField label="Max nouns" value={maxNouns} onChange={setMaxNouns} min={1} max={24} />
          <NumField label="Views" value={views} onChange={setViews} min={4} max={16} />
          <NumField label="Min views" value={minViews} onChange={setMinViews} min={1} max={8} />
          <NumField label="Vote frac" value={voteFrac} onChange={setVoteFrac} step={0.05} min={0.05} max={0.95} />
        </div>
        <label className="flex items-center gap-1.5 text-xs text-zinc-400">
          <input type="checkbox" checked={overrideNouns} onChange={(e) => setOverrideNouns(e.target.checked)} className="accent-cyan-300" />
          Override auto-detected nouns
        </label>
        {overrideNouns && (
          <Input
            value={nounsText}
            onChange={(e) => setNounsText(e.target.value)}
            placeholder="round wooden table, flower vase, …  (comma-separated; empty = none)"
          />
        )}
        <Button type="button" size="sm" onClick={run} disabled={busy} className="w-full">
          {mutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <PackageSearch className="h-3.5 w-3.5" />}
          Run inventory
        </Button>
        <ErrorBanner message={mutation.isError ? mutation.error.message : null} />
      </Card>

      {instances.length > 0 && (
        <div className="space-y-2">
          <SectionLabel>{instances.length} instance(s) found</SectionLabel>
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
            {instances.map((inst) => (
              <div key={inst.slug} className="space-y-1">
                <Thumb
                  src={`/api/splat/jobs/${jobId}/scene/inventory/file?fmt=crop&name=${inst.slug}`}
                  alt={inst.label}
                  onOpen={() => openLightbox(`/api/splat/jobs/${jobId}/scene/inventory/file?fmt=crop&name=${inst.slug}`)}
                />
                <p className="truncate text-[11px] font-medium text-zinc-200" title={inst.label}>
                  {inst.label}
                </p>
                <p className="text-[10px] tabular-nums text-zinc-500">
                  {inst.n_members.toLocaleString()} pts · {inst.n_views} views
                  {inst.regression && Object.keys(inst.regression).length > 0 && (
                    <span className="ml-1 rounded bg-cyan-400/15 px-1 text-cyan-200">
                      IoU {Object.values(inst.regression)[0].iou.toFixed(2)}
                    </span>
                  )}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {vetoed.length > 0 && (
        <div>
          <button type="button" onClick={() => setVetoedOpen((v) => !v)} className="flex items-center gap-1 text-[11px] font-semibold text-zinc-500 hover:text-zinc-300">
            <ChevronDown className={`h-3 w-3 transition-transform ${vetoedOpen ? "" : "-rotate-90"}`} /> {vetoed.length} vetoed
          </button>
          {vetoedOpen && (
            <ul className="mt-1 space-y-0.5 pl-4 text-[11px] text-zinc-500">
              {vetoed.map((v, i) => (
                <li key={i}>
                  {v.noun ?? "(none)"} — {v.reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {data && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <SectionLabel>Camera-overlay receipt</SectionLabel>
            <div className="flex items-center gap-1">
              <button type="button" onClick={() => setCamIdx((i) => Math.max(0, i - 1))} className="rounded p-1 text-zinc-400 hover:bg-white/10">
                <ChevronLeft className="h-3.5 w-3.5" />
              </button>
              <span className="text-[10px] tabular-nums text-zinc-500">cam {camName}</span>
              <button type="button" onClick={() => setCamIdx((i) => (i + 1) % Math.max(1, views))} className="rounded p-1 text-zinc-400 hover:bg-white/10">
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
          <button type="button" onClick={() => openLightbox(camUrl)} className="block w-full overflow-hidden rounded-lg border border-white/10">
            <img src={camUrl} alt="camera overlay receipt" className="w-full object-contain" onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")} />
          </button>
        </div>
      )}
    </div>
  );
}

// ── Isolate (P6c) ────────────────────────────────────────────────────────────
function IsolatePanel({
  jobId,
  scene,
  busy,
  mutation,
  openLightbox,
}: {
  jobId: string;
  scene: SplatSceneSummary | undefined;
  busy: boolean;
  mutation: ReturnType<typeof useMutation<SceneIsolateReport, Error, Record<string, unknown>>>;
  openLightbox: (src: string) => void;
}) {
  const [minMembers, setMinMembers] = useState(200);
  const [views, setViews] = useState(2);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [recallExpand, setRecallExpand] = useState(false);
  const [dilationMult, setDilationMult] = useState(10);
  const [relFloor, setRelFloor] = useState(0.3);

  const reportQuery = useQuery({
    queryKey: ["scene-isolate", jobId],
    queryFn: () => fetchSceneIsolate(jobId),
    enabled: Boolean(scene?.isolate),
    retry: false,
  });
  const data = mutation.data ?? reportQuery.data;

  const prereqMissing = !scene?.inventory;
  const stale = staleBanner(scene, "isolate", "inventory");

  function run() {
    mutation.mutate({
      min_members: minMembers,
      views,
      recall_expand: recallExpand,
      dilation_mult: dilationMult,
      rel_floor: relFloor,
    });
  }

  const receiptUrl = `/api/splat/jobs/${jobId}/scene/isolate/file?fmt=receipt`;

  return (
    <div className="space-y-4">
      <p className="text-xs leading-snug text-zinc-500">
        Claims each inventory instance's gaussians into a Blender-ready object, and writes the true background
        complement. A thin instance is skipped, never aborts the batch.
      </p>
      <StaleNote text={stale} />
      {prereqMissing && <StaleNote text="Run Inventory first — batch isolation needs it." />}

      <Card className="space-y-2.5 border-white/10 bg-white/5 p-3">
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <NumField label="Min members" value={minMembers} onChange={setMinMembers} step={10} min={1} />
          <NumField label="Views" value={views} onChange={setViews} min={1} max={8} />
        </div>
        <button type="button" onClick={() => setAdvancedOpen((v) => !v)} className="flex items-center gap-1 text-[11px] font-semibold text-zinc-500 hover:text-zinc-300">
          <ChevronDown className={`h-3 w-3 transition-transform ${advancedOpen ? "" : "-rotate-90"}`} /> Advanced
        </button>
        {advancedOpen && (
          <div className="space-y-2 border-t border-white/10 pt-2">
            <label className="flex items-center gap-1.5 text-xs text-zinc-400">
              <input type="checkbox" checked={recallExpand} onChange={(e) => setRecallExpand(e.target.checked)} className="accent-cyan-300" />
              Recall-expand (hybrid SAM3 + relevancy-gated dilation)
            </label>
            <p className="text-[10px] leading-snug text-zinc-500">
              Reduces ghost-disc residue on multi-object, well-covered scenes (garden-class). Proven insufficient on
              tight single-object close-ups (hydrant-class) — opt-in, not a default.
            </p>
            {recallExpand && (
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <NumField label="Dilation mult" value={dilationMult} onChange={setDilationMult} step={1} min={0.1} max={100} />
                <NumField label="Relevancy floor" value={relFloor} onChange={setRelFloor} step={0.05} min={0.05} max={0.95} />
              </div>
            )}
          </div>
        )}
        <Button type="button" size="sm" onClick={run} disabled={busy || prereqMissing} className="w-full">
          {mutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Boxes className="h-3.5 w-3.5" />}
          Run isolation
        </Button>
        <ErrorBanner message={mutation.isError ? mutation.error.message : null} />
      </Card>

      {data && (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <Card className="border-white/10 bg-white/5 p-2">
              <p className="tabular-nums text-lg font-semibold text-zinc-100">{data.sanity.n_claimed.toLocaleString()}</p>
              <p className="text-[10px] text-zinc-500">claimed</p>
            </Card>
            <Card className="border-white/10 bg-white/5 p-2">
              <p className="tabular-nums text-lg font-semibold text-zinc-100">{data.sanity.n_background.toLocaleString()}</p>
              <p className="text-[10px] text-zinc-500">background</p>
            </Card>
            <Card className={`border-white/10 bg-white/5 p-2 ${data.sanity.sanity_sum_ok ? "" : "border-red-400/40"}`}>
              <p className={`flex items-center justify-center gap-1 text-lg font-semibold ${data.sanity.sanity_sum_ok ? "text-emerald-300" : "text-red-300"}`}>
                {data.sanity.sanity_sum_ok ? <Check className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
              </p>
              <p className="text-[10px] text-zinc-500">conserved</p>
            </Card>
          </div>

          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-[10px] uppercase tracking-wide text-zinc-500">
                <th className="pb-1 font-semibold">Instance</th>
                <th className="pb-1 font-semibold">Status</th>
                <th className="pb-1 text-right font-semibold">Points</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {data.instances.map((inst) => (
                <tr key={inst.slug}>
                  <td className="py-1 text-zinc-200">{inst.label}</td>
                  <td className={`py-1 ${inst.status === "built" ? "text-emerald-300" : "text-zinc-500"}`}>{inst.status}</td>
                  <td className="py-1 text-right tabular-nums text-zinc-400">{inst.n_members_final?.toLocaleString() ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div>
            <SectionLabel>Background-removed receipt</SectionLabel>
            <button type="button" onClick={() => openLightbox(receiptUrl)} className="mt-1 block w-full overflow-hidden rounded-lg border border-white/10">
              <img src={receiptUrl} alt="background-removed receipt" className="w-full object-contain" onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Proxy (P6d) ──────────────────────────────────────────────────────────────
function ProxyPanel({
  jobId,
  scene,
  busy,
  mutation,
  openLightbox,
}: {
  jobId: string;
  scene: SplatSceneSummary | undefined;
  busy: boolean;
  mutation: ReturnType<typeof useMutation<SceneProxyReport, Error, void>>;
  openLightbox: (src: string) => void;
}) {
  const reportQuery = useQuery({
    queryKey: ["scene-proxy", jobId],
    queryFn: () => fetchSceneProxy(jobId),
    enabled: Boolean(scene?.proxy),
    retry: false,
  });
  const data = mutation.data ?? reportQuery.data;

  const prereqMissing = !scene?.isolate || scene.isolate.n_built === 0;
  const stale = staleBanner(scene, "proxy", "isolate");

  return (
    <div className="space-y-4">
      <p className="text-xs leading-snug text-zinc-500">
        TripoSplat-generates + ICP-registers a clean proxy for every isolated instance, under one shared GPU lease.
        Runs serially per object inside one request — the slowest stage here. A per-instance failure ships without a
        proxy; the captured object stays valid on its own.
      </p>
      <StaleNote text={stale} />
      {prereqMissing && <StaleNote text="Run Isolate first — batch proxy needs at least one built instance." />}

      <Button type="button" size="sm" onClick={() => mutation.mutate()} disabled={busy || prereqMissing} className="w-full">
        {mutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
        Run batch proxy
      </Button>
      <ErrorBanner message={mutation.isError ? mutation.error.message : null} />

      {data && (
        <div className="space-y-3">
          <p className="text-[11px] tabular-nums text-zinc-500">
            {data.n_built} built · {data.n_skipped} skipped
          </p>
          {data.instances.map((inst) => (
            <Card key={inst.slug} className="space-y-1.5 border-white/10 bg-white/5 p-2.5">
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium text-zinc-200">{inst.label}</p>
                <span className={`text-[10px] font-semibold ${inst.status === "built" ? "text-emerald-300" : "text-amber-300"}`}>{inst.status}</span>
              </div>
              {inst.status === "built" ? (
                <>
                  <button
                    type="button"
                    onClick={() => openLightbox(`/api/splat/jobs/${jobId}/scene/proxy/file?fmt=triptych&name=${inst.slug}`)}
                    className="block w-full overflow-hidden rounded-lg border border-white/10"
                  >
                    <img
                      src={`/api/splat/jobs/${jobId}/scene/proxy/file?fmt=triptych&name=${inst.slug}`}
                      alt={`${inst.label} triptych`}
                      className="w-full object-contain"
                    />
                  </button>
                  <p className="text-[10px] tabular-nums text-zinc-500">
                    fitness {inst.icp_fitness?.toFixed(3) ?? "—"} · rmse {inst.icp_rmse?.toFixed(4) ?? "—"} · scale {inst.total_scale?.toFixed(3) ?? "—"}
                  </p>
                </>
              ) : null}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Ground (P6e) ─────────────────────────────────────────────────────────────
function GroundPanel({
  jobId,
  scene,
  busy,
  mutation,
  openLightbox,
}: {
  jobId: string;
  scene: SplatSceneSummary | undefined;
  busy: boolean;
  mutation: ReturnType<typeof useMutation<SceneGroundReport, Error, Record<string, unknown>>>;
  openLightbox: (src: string) => void;
}) {
  const [semanticThresh, setSemanticThresh] = useState(0.5);
  const [cellUnits, setCellUnits] = useState(0.03);

  const reportQuery = useQuery({
    queryKey: ["scene-ground", jobId],
    queryFn: () => fetchSceneGround(jobId),
    enabled: Boolean(scene?.ground),
    retry: false,
  });
  const data = mutation.data ?? reportQuery.data;

  return (
    <div className="space-y-4">
      <p className="text-xs leading-snug text-zinc-500">
        Real captured ground gaussians → a scene-unit TIN → a splat-colored GLB. Independent of Isolate/Proxy — only
        needs the completed job and a built language field. Fails loud below the ground-coverage floor rather than
        emitting a silent empty mesh.
      </p>

      <Card className="space-y-2.5 border-white/10 bg-white/5 p-3">
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <NumField label="Semantic thresh" value={semanticThresh} onChange={setSemanticThresh} step={0.05} min={0.05} max={0.95} />
          <NumField label="Cell units" value={cellUnits} onChange={setCellUnits} step={0.005} min={0.001} />
        </div>
        <Button
          type="button"
          size="sm"
          onClick={() => mutation.mutate({ semantic_thresh: semanticThresh, cell_units: cellUnits })}
          disabled={busy}
          className="w-full"
        >
          {mutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Mountain className="h-3.5 w-3.5" />}
          Run ground build
        </Button>
        <ErrorBanner message={mutation.isError ? mutation.error.message : null} />
      </Card>

      {data && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            <Card className="border-white/10 bg-white/5 p-2">
              <p className="tabular-nums text-lg font-semibold text-zinc-100">{data.ground_points.toLocaleString()}</p>
              <p className="text-[10px] text-zinc-500">ground points</p>
            </Card>
            <Card className="border-white/10 bg-white/5 p-2">
              <p className="tabular-nums text-lg font-semibold text-zinc-100">{data.triangles.toLocaleString()}</p>
              <p className="text-[10px] text-zinc-500">triangles</p>
            </Card>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {(["top", "oblique"] as const).map((fmt) => {
              const url = `/api/splat/jobs/${jobId}/scene/ground/file?fmt=${fmt}`;
              return (
                <button key={fmt} type="button" onClick={() => openLightbox(url)} className="block overflow-hidden rounded-lg border border-white/10">
                  <img src={url} alt={`${fmt} receipt`} className="w-full object-contain" onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")} />
                  <p className="bg-black/40 py-1 text-center text-[10px] uppercase tracking-wide text-zinc-400">{fmt}</p>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Assemble (P6f) ───────────────────────────────────────────────────────────
function AssemblePanel({
  jobId,
  scene,
  busy,
  assembleMutation,
  approveMutation,
  fidelityMode,
  setFidelityMode,
  overrides,
  setOverrides,
}: {
  jobId: string;
  scene: SplatSceneSummary | undefined;
  busy: boolean;
  assembleMutation: ReturnType<typeof useMutation<SceneAssembleReport, Error, Record<string, unknown>>>;
  approveMutation: ReturnType<typeof useMutation<{ job_id: string; state: string }, Error, void>>;
  fidelityMode: "faithful" | "styled";
  setFidelityMode: (m: "faithful" | "styled") => void;
  overrides: Record<string, "captured" | "proxy">;
  setOverrides: (fn: (prev: Record<string, "captured" | "proxy">) => Record<string, "captured" | "proxy">) => void;
  openLightbox: (src: string) => void;
}) {
  const isolateReport = useQuery({
    queryKey: ["scene-isolate", jobId],
    queryFn: () => fetchSceneIsolate(jobId),
    enabled: Boolean(scene?.isolate),
    retry: false,
  });
  const proxyReport = useQuery({
    queryKey: ["scene-proxy", jobId],
    queryFn: () => fetchSceneProxy(jobId),
    enabled: Boolean(scene?.proxy),
    retry: false,
  });
  const assembleReport = useQuery({
    queryKey: ["scene-assemble", jobId],
    queryFn: () => fetchSceneAssemble(jobId),
    enabled: Boolean(scene?.assemble),
    retry: false,
  });
  // Prefer the mutation's own just-returned report -- same reasoning as the
  // other four panels (avoids a window where job.scene hasn't polled fresh
  // yet, e.g. wrongly disabling Approve right after a passing build).
  const assembleData = assembleMutation.data ?? assembleReport.data;

  const prereqMissing = !scene?.isolate;
  const stale = staleBanner(scene, "assemble", "isolate") ?? staleBanner(scene, "assemble", "proxy") ?? staleBanner(scene, "assemble", "ground");

  const builtInstances = (isolateReport.data?.instances ?? []).filter((i) => i.status === "built");
  const proxyBySlug = useMemo(() => {
    const m = new Map<string, SceneProxyReport["instances"][number]>();
    for (const p of proxyReport.data?.instances ?? []) m.set(p.slug, p);
    return m;
  }, [proxyReport.data]);
  const selectionBySlug = useMemo(() => {
    const m = new Map<string, string>();
    for (const el of assembleData?.manifest.elements ?? []) m.set(el.slug, el.selection.reason);
    return m;
  }, [assembleData]);

  const state = assembleData?.manifest.state ?? scene?.assemble?.state ?? null;

  function run() {
    assembleMutation.mutate({ mode: fidelityMode, overrides });
  }

  return (
    <div className="space-y-4">
      <p className="text-xs leading-snug text-zinc-500">
        Assembles isolate/proxy/ground into one scene.blend / scene.glb, per the fidelity dial below. A bad element
        is flagged, never aborts the batch. Never auto-approved — approval is a separate, deliberate step.
      </p>
      <StaleNote text={stale} />
      {prereqMissing && <StaleNote text="Run Isolate first — assembly needs it." />}

      <Card className="space-y-3 border-white/10 bg-white/5 p-3">
        <div>
          <SectionLabel className="mb-1.5">Fidelity mode</SectionLabel>
          <div className="flex gap-1 rounded-xl border border-white/10 bg-black/20 p-1">
            {(["faithful", "styled"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setFidelityMode(m)}
                className={`flex-1 rounded-lg py-1.5 text-xs font-semibold capitalize transition ${
                  fidelityMode === m ? "bg-cyan-400 text-[#04121a]" : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
          <p className="mt-1 text-[10px] leading-snug text-zinc-500">
            {fidelityMode === "faithful"
              ? "Every instance uses its captured file — a pure reality-capture twin. Proxies excluded even when built."
              : "Uses each instance's proxy when one was built, falls back to captured where a proxy wasn't."}
          </p>
        </div>

        {builtInstances.length > 0 && (
          <div>
            <SectionLabel className="mb-1.5">Per-object overrides</SectionLabel>
            <div className="space-y-1.5">
              {builtInstances.map((inst) => {
                const proxy = proxyBySlug.get(inst.slug);
                const proxyBuilt = Boolean(proxy && proxy.status === "built");
                const chosen = overrides[inst.slug] ?? (proxyBuilt && fidelityMode === "styled" ? "proxy" : "captured");
                const reason = selectionBySlug.get(inst.slug);
                return (
                  <div key={inst.slug} className="flex items-center justify-between gap-2 rounded-lg border border-white/10 bg-black/20 px-2.5 py-1.5">
                    <div className="min-w-0">
                      <p className="truncate text-xs text-zinc-200">{inst.label}</p>
                      {reason && <p className="truncate text-[10px] text-zinc-500">{reason}</p>}
                    </div>
                    <div className="flex shrink-0 gap-1 text-[10px]">
                      {(["captured", "proxy"] as const).map((opt) => {
                        const disabled = opt === "proxy" && !proxyBuilt;
                        return (
                          <button
                            key={opt}
                            type="button"
                            disabled={disabled}
                            title={disabled ? `No proxy built for this instance${proxy ? ` (${proxy.status})` : ""}` : undefined}
                            onClick={() => setOverrides((prev) => ({ ...prev, [inst.slug]: opt }))}
                            className={`rounded-full px-2 py-0.5 font-semibold capitalize transition disabled:cursor-not-allowed disabled:opacity-30 ${
                              chosen === opt ? "bg-cyan-400/20 text-cyan-200" : "text-zinc-500 hover:text-zinc-300"
                            }`}
                          >
                            {opt}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <Button type="button" size="sm" onClick={run} disabled={busy || prereqMissing} className="w-full">
          {assembleMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          {state ? "Re-assemble" : "Assemble scene"}
        </Button>
        {state === "approved" && (
          <p className="text-center text-[10px] text-amber-200/80">Re-assembling drops this scene back to unapproved.</p>
        )}
        <ErrorBanner message={assembleMutation.isError ? assembleMutation.error.message : null} />
      </Card>

      {assembleData && (
        <Card className="space-y-2 border-white/10 bg-white/5 p-3">
          <SectionLabel>Build result</SectionLabel>
          <p className="text-[11px] tabular-nums text-zinc-400">
            {assembleData.assemble.n_built}/{assembleData.assemble.n_elements_total} elements built
            {assembleData.assemble.n_flagged > 0 && (
              <span className="ml-1.5 text-amber-300">· {assembleData.assemble.n_flagged} flagged</span>
            )}
            {" · "}
            {assembleData.assemble.contamination_gate.ok ? (
              <span className="text-emerald-300">contamination gate passed</span>
            ) : (
              <span className="text-red-300">contamination gate FAILED</span>
            )}
          </p>
          <div className="flex gap-2 pt-1">
            <a
              href={`/api/splat/jobs/${jobId}/scene/assemble/file?fmt=glb`}
              download={`${jobId}-scene.glb`}
              className="flex flex-1 items-center justify-center gap-1 rounded-lg border border-white/15 bg-white/5 px-2 py-1.5 text-[11px] font-semibold text-zinc-200 hover:bg-white/10"
            >
              <Download className="h-3 w-3" /> Preview .glb
            </a>
            <a
              href={`/api/splat/jobs/${jobId}/scene/assemble/file?fmt=blend`}
              download={`${jobId}-scene.blend`}
              className="flex flex-1 items-center justify-center gap-1 rounded-lg border border-white/15 bg-white/5 px-2 py-1.5 text-[11px] font-semibold text-zinc-200 hover:bg-white/10"
            >
              <Download className="h-3 w-3" /> Preview .blend
            </a>
          </div>
        </Card>
      )}

      {/* approve gate — visually set apart, the one mandatory HITL */}
      {state && (
        <Card className={`space-y-2 border p-3 ${state === "approved" ? "border-emerald-400/30 bg-emerald-400/5" : "border-amber-400/30 bg-amber-400/5"}`}>
          {state === "approved" ? (
            <p className="flex items-center gap-1.5 text-xs font-semibold text-emerald-200">
              <CheckCircle2 className="h-4 w-4" /> Approved — this build is final and available for download in the gallery.
            </p>
          ) : (
            <>
              <p className="flex items-center gap-1.5 text-xs font-semibold text-amber-200">
                <ShieldCheck className="h-4 w-4" /> Not approved yet — nothing outside this panel treats it as final.
              </p>
              <Button
                type="button"
                size="sm"
                onClick={() => approveMutation.mutate()}
                disabled={busy || !assembleData?.assemble.contamination_gate.ok}
                className="w-full"
              >
                {approveMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ShieldCheck className="h-3.5 w-3.5" />}
                Approve this build
              </Button>
              <ErrorBanner message={approveMutation.isError ? approveMutation.error.message : null} />
            </>
          )}
        </Card>
      )}
    </div>
  );
}
