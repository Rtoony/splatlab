import { Suspense, lazy, useState } from "react";
import type { SplatJob } from "@/lib/contracts";
import { Badge, Button, Card, SectionLabel } from "@/components/ui";
import { DownloadMenu } from "@/components/gallery/download-menu";
import { SceneCard } from "@/components/gallery/scene-card";
import { CaptureHealthCard } from "@/components/jobs/capture-health-card";
import { Orbit } from "lucide-react";

// Featured live preview — lazy so the 3D engine chunk never loads unless a
// previewable scene exists on the landing page.
const SparkSceneViewer = lazy(() =>
  import("@/components/spark-scene-viewer").then((m) => ({ default: m.SparkSceneViewer })),
);

// ── results gallery ───────────────────────────────────────────────────────────
export function ResultsGallery({
  jobs,
  onRerun,
  onPromote,
  busy,
  computeBlocked,
  onPin,
  onDelete,
}: {
  jobs: SplatJob[];
  onRerun: (job: SplatJob, mult?: number) => void;
  onPromote: (job: SplatJob) => void;
  busy: boolean;
  computeBlocked: boolean;
  onPin: (job: SplatJob) => void;
  onDelete: (id: string) => void;
}) {
  const previewable = jobs.filter((j) => j.preview_available);
  const [featured, setFeatured] = useState<string | null>(null);
  const featuredJob = previewable.find((j) => j.job_id === featured) || previewable[0] || null;

  if (jobs.length === 0) return null;

  return (
    <section className="mt-8">
      <div className="mb-3 flex items-center justify-between">
        <SectionLabel>Your scenes</SectionLabel>
        <Badge>{jobs.length} completed</Badge>
      </div>

      {featuredJob?.preview_web_url && (
        <Card className="mb-4 overflow-hidden">
          <div className="relative h-[420px] overflow-hidden">
            <Suspense fallback={<div className="flex h-full items-center justify-center text-xs uppercase tracking-[0.3em] text-zinc-600">Loading viewer…</div>}>
              <SparkSceneViewer key={featuredJob.job_id} job={featuredJob} toolsVisible={false} safeMode={computeBlocked} />
            </Suspense>
          </div>
          <div className="flex items-center justify-between gap-2 p-3">
            <p className="truncate text-sm text-zinc-300">{featuredJob.input_path.split("/").pop()}</p>
            <div className="flex items-center gap-2">
              <a href={`/view/${featuredJob.job_id}`} target="_blank" rel="noreferrer">
                <Button size="sm">
                  <Orbit className="h-3.5 w-3.5" /> Fullscreen
                </Button>
              </a>
              <DownloadMenu job={featuredJob} />
            </div>
          </div>
        </Card>
      )}
      {featuredJob && <CaptureHealthCard job={featuredJob} />}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
        {jobs.map((j) => (
          <SceneCard
            key={j.job_id}
            job={j}
            active={j.job_id === featuredJob?.job_id}
            onFeature={() => setFeatured(j.job_id)}
            onRerun={onRerun}
            onPromote={onPromote}
            busy={busy}
            computeBlocked={computeBlocked}
            onPin={onPin}
            onDelete={onDelete}
          />
        ))}
      </div>
    </section>
  );
}
