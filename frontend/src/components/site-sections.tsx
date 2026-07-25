// Site sections: an on-demand, "official" trigger for POST /geo/sections —
// two auto-picked principal-axis cross-sections + an isometric shaded ground
// TIN. A small single-purpose modal, not folded into GeoLocateModal (scoped
// to map/anchor concerns) or SceneRegenModal (the P6 object-decomposition
// lane, a different concern) — its own first-class action, same level as
// "Locate" and "Scene".
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { SplatJob } from "@/lib/contracts";
import { Button, SectionLabel } from "@/components/ui";
import ReceiptLightbox from "@/components/receipt-lightbox";
import { AlertTriangle, Loader2, Mountain, X } from "lucide-react";

// Bypasses api.ts's apiRequest() (raw response body on error) so the
// backend's real prerequisite/toolchain detail strings render cleanly —
// same pattern as scene-regen.tsx / spark-scene-viewer.tsx.
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

interface SiteSectionsResponse {
  job_id: string;
  sections_url: string;
  surface_iso_url: string;
  ground_points_url: string;
}

export default function SiteSectionsModal({ job, onClose }: { job: SplatJob; onClose: () => void }) {
  const jobId = job.job_id;
  const queryClient = useQueryClient();
  const [lightbox, setLightbox] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => postJSON<SiteSectionsResponse>(`/api/splat/jobs/${jobId}/geo/sections`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["status"] }),
  });

  // Prefer the mutation's own just-returned URLs; fall back to the polled
  // job payload so a previously-generated set (e.g. from an earlier /geo/
  // contours run) still shows up before anyone re-triggers this.
  const sectionsUrl = mutation.data?.sections_url ?? job.sections_url ?? null;
  const surfaceIsoUrl = mutation.data?.surface_iso_url ?? job.surface_iso_url ?? null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="relative w-full max-w-lg rounded-2xl border border-white/10 bg-surface">
        <div className="flex items-center justify-between gap-2 border-b border-white/10 px-4 py-3">
          <div className="flex items-center gap-2">
            <Mountain className="h-4 w-4 text-cyan-300" />
            <SectionLabel>Site sections</SectionLabel>
          </div>
          <button type="button" onClick={onClose} className="rounded-full p-1 text-zinc-400 transition hover:bg-white/10 hover:text-zinc-100" title="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 p-4">
          <p className="text-xs leading-snug text-zinc-500">
            Two auto-picked cross-sections through the scene's principal axes, plus an isometric shaded ground
            TIN — a quick, shareable "what does this site actually look like" picture. Needs the scene's scale
            calibrated and a Locate anchor set, plus either a built mesh or a language field for the ground
            sample.
          </p>

          <Button type="button" size="sm" onClick={() => mutation.mutate()} disabled={mutation.isPending} className="w-full">
            {mutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Mountain className="h-3.5 w-3.5" />}
            Generate site sections
          </Button>
          {mutation.isError && (
            <p className="flex items-start gap-1.5 rounded-lg border border-red-400/25 bg-red-400/10 px-2.5 py-2 text-xs leading-snug text-red-200">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {mutation.error.message}
            </p>
          )}

          {(sectionsUrl || surfaceIsoUrl) && (
            <div className="grid grid-cols-2 gap-2">
              {sectionsUrl && (
                <button type="button" onClick={() => setLightbox(sectionsUrl)} className="block overflow-hidden rounded-lg border border-white/10">
                  <img
                    src={sectionsUrl}
                    alt="sections receipt"
                    className="w-full object-contain"
                    onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")}
                  />
                  <p className="bg-black/40 py-1 text-center text-[10px] uppercase tracking-wide text-zinc-400">Sections</p>
                </button>
              )}
              {surfaceIsoUrl && (
                <button type="button" onClick={() => setLightbox(surfaceIsoUrl)} className="block overflow-hidden rounded-lg border border-white/10">
                  <img
                    src={surfaceIsoUrl}
                    alt="surface iso receipt"
                    className="w-full object-contain"
                    onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")}
                  />
                  <p className="bg-black/40 py-1 text-center text-[10px] uppercase tracking-wide text-zinc-400">3D surface</p>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
      {lightbox && <ReceiptLightbox src={lightbox} onClose={() => setLightbox(null)} />}
    </div>
  );
}
