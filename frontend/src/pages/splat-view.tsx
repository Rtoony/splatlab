import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useRoute } from "wouter";
import { apiRequest, queryLangfield } from "@/lib/api";
import type { LangfieldQueryResult, SplatJob, SplatStatusResponse } from "@/lib/contracts";
import { SplatViewer } from "@/components/splat-viewer";
import { Button, Card, Input, SectionLabel } from "@/components/ui";
import { ArrowLeft, Download, Loader2, Orbit, Search, Sparkles, X } from "lucide-react";

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
  // Where the viewer should fly on a search hit (set from a query's 3D focus).
  const [focus, setFocus] = useState<FocusTarget | null>(null);

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
          <>
            <SplatViewer url={viewUrl} format="ply" fill focus={focus} />
            {job.langfield_available && <LangfieldSearch jobId={job.job_id} onFocus={setFocus} />}
          </>
        )}
      </main>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full items-center justify-center text-sm text-zinc-400">{children}</div>;
}

// The 3D point (viewer frame) + spread the viewer should fly to on a search hit.
export type FocusTarget = { point: [number, number, number]; radius: number };

// Text-search panel for scenes that carry an opt-in Language Field. Floats over
// the bottom of the viewer; a query renders a server-side relevancy heatmap strip
// AND flies the 3D view to the match (via onFocus).
function LangfieldSearch({ jobId, onFocus }: { jobId: string; onFocus: (f: FocusTarget) => void }) {
  const [text, setText] = useState("");
  const search = useMutation<LangfieldQueryResult, Error, string>({
    mutationFn: (q: string) => queryLangfield(jobId, q),
    onSuccess: (r) => {
      if (r.focus) onFocus({ point: r.focus, radius: r.radius ?? 0.3 });
    },
  });

  function submit() {
    const q = text.trim();
    if (q) search.mutate(q);
  }

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex justify-center p-3 sm:p-4">
      <Card className="pointer-events-auto w-full max-w-lg border-white/12 bg-[#070b14]/85 p-3 backdrop-blur-md">
        <div className="mb-2 flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5 text-cyan-200" />
          <SectionLabel>Search this scene</SectionLabel>
        </div>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type what you’re looking for…"
            autoComplete="off"
          />
          <Button type="submit" disabled={!text.trim() || search.isPending} className="shrink-0">
            {search.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            Search
          </Button>
        </form>

        {search.isPending && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-zinc-400">
            <Loader2 className="h-3 w-3 animate-spin" /> Searching…
          </p>
        )}
        {search.isError && (
          <p className="mt-2 text-xs text-red-300">{search.error.message || "Search failed."}</p>
        )}
        {search.data && !search.isPending && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <p className="truncate text-xs text-zinc-400">
                Relevancy for <span className="text-zinc-200">“{search.data.query}”</span>
                {!search.data.ready && " · still building…"}
              </p>
              <button
                type="button"
                onClick={() => {
                  setText("");
                  search.reset();
                }}
                className="flex shrink-0 items-center gap-1 text-xs text-zinc-400 transition hover:text-zinc-100"
              >
                <X className="h-3.5 w-3.5" /> Clear
              </button>
            </div>
            {search.data.heatmap_url && (
              <img
                src={search.data.heatmap_url}
                alt={`Relevancy heatmap for "${search.data.query}"`}
                className="w-full rounded-xl border border-white/10"
              />
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
