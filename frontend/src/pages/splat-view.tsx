import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useRoute } from "wouter";
import { apiRequest } from "@/lib/api";
import type { SplatJob, SplatStatusResponse } from "@/lib/contracts";
import { SplatViewer } from "@/components/splat-viewer";
import { ArrowLeft, Download, Loader2, Orbit } from "lucide-react";

export default function SplatViewPage() {
  const [, params] = useRoute("/view/:jobId");
  const jobId = params?.jobId ?? "";

  const { data: status, isLoading } = useQuery({
    queryKey: ["status"],
    queryFn: () => apiRequest<SplatStatusResponse>("/api/splat/status"),
    refetchInterval: 4000,
  });

  const job: SplatJob | undefined = useMemo(
    () => status?.jobs.find((j) => j.job_id === jobId),
    [status, jobId],
  );
  const viewUrl = job?.preview_web_url ?? job?.preview_view_url ?? null;
  const title = job ? job.input_path?.split("/").pop() || job.job_id : jobId;

  return (
    <div className="flex h-screen flex-col bg-[#05070d] text-zinc-100">
      <header className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <Link href="/" className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200">
            <ArrowLeft className="h-4 w-4" /> Splat Lab
          </Link>
          <span className="text-white/20">/</span>
          <div className="flex min-w-0 items-center gap-2">
            <Orbit className="h-4 w-4 shrink-0 text-cyan-300" />
            <span className="truncate text-sm font-semibold">{title}</span>
          </div>
        </div>
        {job?.preview_file_url && (
          <a
            href={job.preview_file_url}
            download={`${jobId}.ply`}
            className="flex shrink-0 items-center gap-1.5 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-semibold text-zinc-200 hover:bg-white/10"
          >
            <Download className="h-3.5 w-3.5" /> Full-quality .ply
          </a>
        )}
      </header>
      <main className="relative flex-1 overflow-hidden">
        {isLoading && !job ? (
          <Centered>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading…
          </Centered>
        ) : !job ? (
          <Centered>
            <div className="text-center">
              <p className="font-semibold text-zinc-200">Scene not found</p>
              <Link href="/" className="mt-2 inline-block text-cyan-300 hover:underline">
                Back to Splat Lab
              </Link>
            </div>
          </Centered>
        ) : !viewUrl || !job.preview_available ? (
          <Centered>
            <div className="text-center">
              <Loader2 className="mx-auto mb-2 h-5 w-5 animate-spin text-cyan-300" />
              <p className="font-semibold text-zinc-200">
                {job.status === "completed" ? "Preparing preview…" : `Scene is ${job.status}…`}
              </p>
            </div>
          </Centered>
        ) : (
          <SplatViewer url={viewUrl} format="ply" fill />
        )}
      </main>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full items-center justify-center text-sm text-zinc-400">{children}</div>;
}
