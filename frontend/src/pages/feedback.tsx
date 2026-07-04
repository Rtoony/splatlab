import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "wouter";
import { ArrowLeft, MessageSquare, Paperclip, RefreshCw, Save, Search, Send } from "lucide-react";
import { addFeedbackComment, getFeedback, listFeedback, updateFeedback, uploadFeedbackAttachment } from "@/lib/feedback-api";
import {
  FEEDBACK_PRIORITIES,
  FEEDBACK_QUEUES,
  FEEDBACK_STATUSES,
  FEEDBACK_TYPES,
  type FeedbackAttachment,
  type FeedbackItem,
  type FeedbackPriority,
  type FeedbackQueueKey,
  type FeedbackStatus,
  type FeedbackType,
} from "@/lib/feedback-contracts";
import { setSplatlabFeedbackContext } from "@/lib/feedback-context";
import { Badge, Button, Card, Input, SectionLabel } from "@/components/ui";
import { cn } from "@/lib/cn";

const queueKeys = Object.keys(FEEDBACK_QUEUES) as FeedbackQueueKey[];

export default function FeedbackPage() {
  const qc = useQueryClient();
  const [queue, setQueue] = useState<FeedbackQueueKey>("active");
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("Any");
  const [priorityFilter, setPriorityFilter] = useState("Any");
  const [typeFilter, setTypeFilter] = useState("Any");
  const [selectedId, setSelectedId] = useState<string | number | null>(null);

  useEffect(() => {
    setSplatlabFeedbackContext({
      page: "feedback-management",
      queue,
      search: search || undefined,
      status_filter: statusFilter,
      priority_filter: priorityFilter,
      type_filter: typeFilter,
      selected_feedback_id: selectedId,
    });
    return () => setSplatlabFeedbackContext(null);
  }, [priorityFilter, queue, search, selectedId, statusFilter, typeFilter]);

  const { data: items = [], isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ["feedback"],
    queryFn: listFeedback,
    refetchInterval: 30000,
  });

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const statuses = FEEDBACK_QUEUES[queue].statuses;
    return items
      .filter((item) => !statuses || statuses.includes(item.status as FeedbackStatus))
      .filter((item) => statusFilter === "Any" || item.status === statusFilter)
      .filter((item) => priorityFilter === "Any" || item.priority === priorityFilter)
      .filter((item) => typeFilter === "Any" || feedbackType(item) === typeFilter)
      .filter((item) => {
        if (!q) return true;
        return [item.title, feedbackBody(item), item.page_path, item.component_label, tagsText(item)]
          .filter(Boolean)
          .join(" ")
          .toLowerCase()
          .includes(q);
      })
      .sort((a, b) => timestamp(b.updated_at ?? b.created_at) - timestamp(a.updated_at ?? a.created_at));
  }, [items, priorityFilter, queue, search, statusFilter, typeFilter]);

  useEffect(() => {
    if (!selectedId && filtered[0]) setSelectedId(filtered[0].id);
  }, [filtered, selectedId]);

  const selected = selectedId ? items.find((item) => String(item.id) === String(selectedId)) ?? null : null;

  return (
    <div className="min-h-screen bg-[#05070d] px-4 py-6 text-zinc-100 sm:px-6">
      <div className="mx-auto max-w-7xl">
        <header className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <Link href="/" className="mb-2 inline-flex items-center gap-1.5 text-xs text-zinc-400 hover:text-cyan-200">
              <ArrowLeft className="h-4 w-4" /> Splat Lab
            </Link>
            <h1 className="display text-3xl font-black tracking-tight text-white">Feedback Loop</h1>
            <p className="mt-1 text-sm text-zinc-400">Triage captured feedback, add notes, and move work through verification.</p>
          </div>
          <Button variant="outline" onClick={() => refetch()} disabled={isFetching}>
            <RefreshCw className={cn("h-4 w-4", isFetching && "animate-spin")} />
            Refresh
          </Button>
        </header>

        <div className="mb-4 flex gap-2 overflow-x-auto pb-1">
          {queueKeys.map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setQueue(key)}
              className={cn(
                "shrink-0 rounded-full border px-3 py-1.5 text-xs font-bold transition",
                queue === key
                  ? "border-cyan-300/45 bg-cyan-300/15 text-cyan-100"
                  : "border-white/10 bg-white/[0.03] text-zinc-400 hover:text-zinc-100",
              )}
            >
              {FEEDBACK_QUEUES[key].label}
            </button>
          ))}
        </div>

        <div className="mb-4 grid gap-2 lg:grid-cols-[minmax(260px,1fr)_160px_160px_160px]">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-zinc-500" />
            <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search title, details, path, tags..." className="pl-9" />
          </div>
          <FilterSelect value={statusFilter} onChange={setStatusFilter} options={["Any", ...FEEDBACK_STATUSES]} />
          <FilterSelect value={priorityFilter} onChange={setPriorityFilter} options={["Any", ...FEEDBACK_PRIORITIES]} />
          <FilterSelect value={typeFilter} onChange={setTypeFilter} options={["Any", ...FEEDBACK_TYPES]} />
        </div>

        <div className="grid gap-4 lg:grid-cols-[minmax(320px,0.9fr)_minmax(0,1.4fr)]">
          <Card className="overflow-hidden">
            <div className="border-b border-white/10 px-4 py-3">
              <SectionLabel>{filtered.length} records</SectionLabel>
            </div>
            <div className="max-h-[72vh] overflow-y-auto">
              {isLoading ? (
                <EmptyState>Loading feedback...</EmptyState>
              ) : isError ? (
                <EmptyState>Could not load `/api/feedback`.</EmptyState>
              ) : filtered.length === 0 ? (
                <EmptyState>No feedback in this view.</EmptyState>
              ) : (
                filtered.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSelectedId(item.id)}
                    className={cn(
                      "block w-full border-b border-white/5 px-4 py-3 text-left transition hover:bg-white/[0.04]",
                      String(selectedId) === String(item.id) && "bg-cyan-300/[0.07]",
                    )}
                  >
                    <div className="mb-1.5 flex items-start justify-between gap-3">
                      <p className="line-clamp-2 text-sm font-semibold text-zinc-100">{item.title || "Untitled feedback"}</p>
                      <Badge className={statusTone(item.status)}>{item.status}</Badge>
                    </div>
                    <p className="line-clamp-2 text-xs text-zinc-500">{feedbackBody(item) || "No details."}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-500">
                      <Badge>{feedbackType(item)}</Badge>
                      <Badge className={priorityTone(item.priority)}>{item.priority}</Badge>
                      {item.page_path && <span className="truncate font-mono">{item.page_path}</span>}
                      {!!item.attachments?.length && (
                        <span className="inline-flex items-center gap-1">
                          <Paperclip className="h-3 w-3" /> {item.attachments.length}
                        </span>
                      )}
                    </div>
                  </button>
                ))
              )}
            </div>
          </Card>

          <FeedbackDetail key={String(selectedId ?? "none")} seed={selected} />
        </div>
      </div>
    </div>
  );
}

function FeedbackDetail({ seed }: { seed: FeedbackItem | null }) {
  const qc = useQueryClient();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [draft, setDraft] = useState<FeedbackItem | null>(seed);
  const [comment, setComment] = useState("");
  const [showRaw, setShowRaw] = useState(false);

  const { data } = useQuery({
    queryKey: ["feedback", seed?.id],
    queryFn: () => getFeedback(seed!.id),
    enabled: Boolean(seed?.id),
  });

  useEffect(() => setDraft(data ?? seed), [data, seed]);

  const save = useMutation({
    mutationFn: () =>
      updateFeedback(draft!.id, {
        title: draft!.title,
        body: feedbackBody(draft!),
        status: draft!.status,
        priority: draft!.priority,
        feedback_type: feedbackType(draft!) as FeedbackType,
        tags_json: splitTags(tagsText(draft!)),
        resolution_notes: draft!.resolution_notes,
      }),
    onSuccess: (item) => {
      qc.setQueryData(["feedback", item.id], item);
      qc.invalidateQueries({ queryKey: ["feedback"] });
    },
  });

  const addComment = useMutation({
    mutationFn: () => addFeedbackComment(draft!.id, comment.trim()),
    onSuccess: () => {
      setComment("");
      qc.invalidateQueries({ queryKey: ["feedback"] });
      qc.invalidateQueries({ queryKey: ["feedback", draft!.id] });
    },
  });

  const attach = useMutation({
    mutationFn: (files: FileList) => Promise.all(Array.from(files).map((file) => uploadFeedbackAttachment(draft!.id, file))),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["feedback"] });
      qc.invalidateQueries({ queryKey: ["feedback", draft!.id] });
    },
  });

  if (!draft) {
    return (
      <Card className="flex min-h-[420px] items-center justify-center p-6 text-sm text-zinc-500">
        Select a feedback item.
      </Card>
    );
  }

  const context = parseJson(draft.context_json);

  return (
    <Card className="min-h-[620px] p-4">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <SectionLabel>Feedback #{draft.id}</SectionLabel>
          <Input
            value={draft.title ?? ""}
            onChange={(e) => setDraft({ ...draft, title: e.target.value })}
            className="mt-1 h-auto rounded-none border-x-0 border-t-0 bg-transparent px-0 text-xl font-black text-white"
          />
        </div>
        <Button onClick={() => save.mutate()} disabled={save.isPending}>
          <Save className="h-4 w-4" /> {save.isPending ? "Saving..." : "Save"}
        </Button>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <EditorSelect label="Status" value={draft.status} options={FEEDBACK_STATUSES} onChange={(v) => setDraft({ ...draft, status: v as FeedbackStatus })} />
        <EditorSelect
          label="Priority"
          value={draft.priority}
          options={FEEDBACK_PRIORITIES}
          onChange={(v) => setDraft({ ...draft, priority: v as FeedbackPriority })}
        />
        <EditorSelect
          label="Type"
          value={feedbackType(draft)}
          options={FEEDBACK_TYPES}
          onChange={(v) => setDraft({ ...draft, feedback_type: v as FeedbackType })}
        />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-4">
          <label className="block">
            <SectionLabel>Details</SectionLabel>
            <textarea
              value={feedbackBody(draft)}
              onChange={(e) => setDraft({ ...draft, body: e.target.value })}
              rows={6}
              className="mt-1 w-full rounded-2xl border border-white/12 bg-white/5 px-3 py-2 text-sm text-zinc-100 focus:border-cyan-400/40 focus:outline-none"
            />
          </label>

          <label className="block">
            <SectionLabel>Tags</SectionLabel>
            <Input value={tagsText(draft)} onChange={(e) => setDraft({ ...draft, tags_json: splitTags(e.target.value) })} placeholder="comma, separated, tags" className="mt-1" />
          </label>

          <label className="block">
            <SectionLabel>Resolution Notes</SectionLabel>
            <textarea
              value={draft.resolution_notes ?? ""}
              onChange={(e) => setDraft({ ...draft, resolution_notes: e.target.value })}
              rows={4}
              placeholder="What changed, where to test, or why this is blocked..."
              className="mt-1 w-full rounded-2xl border border-white/12 bg-white/5 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-cyan-400/40 focus:outline-none"
            />
          </label>

          <section>
            <div className="mb-2 flex items-center justify-between gap-2">
              <SectionLabel>Discussion</SectionLabel>
              <span className="text-xs text-zinc-500">{draft.comments?.length ?? 0} comments</span>
            </div>
            <div className="space-y-2">
              {(draft.comments ?? []).map((entry) => (
                <div key={entry.id} className="rounded-2xl border border-white/10 bg-white/[0.03] p-3">
                  <p className="text-sm text-zinc-200">{entry.body}</p>
                  <p className="mt-1 text-[11px] text-zinc-500">
                    {entry.created_by ?? entry.author ?? "unknown"} · {formatDate(entry.created_at)}
                  </p>
                </div>
              ))}
              <form
                className="flex gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (comment.trim()) addComment.mutate();
                }}
              >
                <Input value={comment} onChange={(e) => setComment(e.target.value)} placeholder="Add a note or question..." />
                <Button type="submit" disabled={!comment.trim() || addComment.isPending}>
                  <Send className="h-4 w-4" />
                </Button>
              </form>
            </div>
          </section>
        </div>

        <aside className="space-y-4">
          <ContextSummary item={draft} context={context} />

          <section>
            <div className="mb-2 flex items-center justify-between">
              <SectionLabel>Attachments</SectionLabel>
              <Button type="button" variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
                <Paperclip className="h-3.5 w-3.5" /> Add
              </Button>
              <input
                ref={inputRef}
                type="file"
                multiple
                className="hidden"
                onChange={(e) => e.target.files && attach.mutate(e.target.files)}
              />
            </div>
            <AttachmentGrid attachments={draft.attachments ?? []} />
          </section>

          <section>
            <button
              type="button"
              onClick={() => setShowRaw((v) => !v)}
              className="mb-2 flex w-full items-center justify-between text-left"
            >
              <SectionLabel>Raw Context</SectionLabel>
              <span className="text-xs text-cyan-200">{showRaw ? "Hide" : "Show"}</span>
            </button>
            {showRaw && (
              <pre className="max-h-80 overflow-auto rounded-2xl border border-white/10 bg-black/30 p-3 text-[11px] text-zinc-400">
                {JSON.stringify(context ?? draft.context_json ?? {}, null, 2)}
              </pre>
            )}
          </section>
        </aside>
      </div>
    </Card>
  );
}

function ContextSummary({ item, context }: { item: FeedbackItem; context: Record<string, unknown> | null }) {
  const route = context?.route as Record<string, unknown> | undefined;
  const ui = context?.ui_state as Record<string, unknown> | undefined;
  const browser = context?.browser as Record<string, unknown> | undefined;
  const errors = Array.isArray(context?.recent_js_errors) ? context.recent_js_errors : [];
  const failed = Array.isArray(context?.recent_failed_api_calls) ? context.recent_failed_api_calls : [];
  const splatlab = ui?.splatlab as Record<string, unknown> | undefined;
  return (
    <section className="rounded-2xl border border-white/10 bg-white/[0.03] p-3">
      <SectionLabel>Captured Context</SectionLabel>
      <dl className="mt-2 space-y-1.5 text-xs">
        <Row label="Path" value={String(route?.path ?? item.page_path ?? "unknown")} mono />
        <Row label="Component" value={String(item.component_label ?? ui?.component_label ?? "unknown")} />
        <Row label="Scene/job" value={String(splatlab?.job_id ?? splatlab?.active_job_id ?? "none")} mono />
        <Row label="Scroll" value={`${String(ui?.scroll_x ?? 0)}, ${String(ui?.scroll_y ?? 0)}`} />
        <Row label="Viewport" value={viewportText(browser)} />
        <Row label="JS errors" value={String(errors.length)} />
        <Row label="Failed API" value={String(failed.length)} />
      </dl>
    </section>
  );
}

function AttachmentGrid({ attachments }: { attachments: FeedbackAttachment[] }) {
  if (attachments.length === 0) {
    return <div className="rounded-2xl border border-dashed border-white/10 p-4 text-center text-xs text-zinc-500">No attachments.</div>;
  }
  return (
    <div className="grid grid-cols-2 gap-2">
      {attachments.map((attachment) => {
        const url = attachment.url ?? attachment.download_url ?? "#";
        const name = attachment.original_name ?? attachment.filename ?? attachment.name ?? `Attachment ${attachment.id}`;
        const type = attachment.content_type ?? attachment.mime_type ?? "";
        return (
          <a key={attachment.id} href={url} target="_blank" rel="noreferrer" className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.03] hover:border-cyan-300/30">
            {type.startsWith("image/") && url !== "#" ? (
              <img src={url} alt={name} className="h-28 w-full object-cover" />
            ) : (
              <div className="flex h-28 items-center justify-center text-zinc-500">
                <Paperclip className="h-6 w-6" />
              </div>
            )}
            <p className="truncate px-2 py-1.5 text-[11px] text-zinc-300">{name}</p>
          </a>
        );
      })}
    </div>
  );
}

function FilterSelect({ value, onChange, options }: { value: string; onChange: (value: string) => void; options: readonly string[] }) {
  return <select value={value} onChange={(e) => onChange(e.target.value)} className="h-9 rounded-xl border border-white/12 bg-[#101822] px-3 text-sm text-zinc-100 focus:border-cyan-400/40 focus:outline-none">{options.map((option) => <option key={option}>{option}</option>)}</select>;
}

function EditorSelect({ label, value, options, onChange }: { label: string; value: string; options: readonly string[]; onChange: (value: string) => void }) {
  return (
    <label>
      <SectionLabel>{label}</SectionLabel>
      <FilterSelect value={value} onChange={onChange} options={options} />
    </label>
  );
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="flex min-h-48 items-center justify-center p-6 text-sm text-zinc-500">{children}</div>;
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[86px_minmax(0,1fr)] gap-2">
      <dt className="text-zinc-500">{label}</dt>
      <dd className={cn("truncate text-zinc-300", mono && "font-mono")}>{value}</dd>
    </div>
  );
}

function feedbackBody(item: FeedbackItem): string {
  return item.body ?? item.details ?? "";
}

function feedbackType(item: FeedbackItem): string {
  return item.feedback_type ?? item.type ?? "Comment";
}

function tagsText(item: FeedbackItem): string {
  const tags = item.tags ?? item.tags_json ?? [];
  if (Array.isArray(tags)) return tags.join(", ");
  return tags;
}

function splitTags(value: string): string[] {
  return value.split(",").map((tag) => tag.trim()).filter(Boolean);
}

function parseJson(value: unknown): Record<string, unknown> | null {
  if (!value) return null;
  if (typeof value === "object" && !Array.isArray(value)) return value as Record<string, unknown>;
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function timestamp(value?: string): number {
  return value ? new Date(value).getTime() || 0 : 0;
}

function formatDate(value?: string): string {
  if (!value) return "unknown time";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function viewportText(browser?: Record<string, unknown>): string {
  const viewport = browser?.viewport as Record<string, unknown> | undefined;
  if (!viewport) return "unknown";
  return `${String(viewport.width)} x ${String(viewport.height)}`;
}

function statusTone(status: string): string {
  if (["Ready to Test", "Fixed", "Needs Info"].includes(status)) return "border-amber-400/25 bg-amber-400/10 text-amber-200";
  if (["Accepted", "Closed"].includes(status)) return "border-emerald-400/25 bg-emerald-400/10 text-emerald-200";
  if (["Won't Fix", "Archived"].includes(status)) return "border-zinc-400/20 bg-zinc-400/10 text-zinc-300";
  return "border-cyan-400/25 bg-cyan-400/10 text-cyan-200";
}

function priorityTone(priority: string): string {
  if (priority === "Critical") return "border-red-400/25 bg-red-400/10 text-red-200";
  if (priority === "High") return "border-amber-400/25 bg-amber-400/10 text-amber-200";
  if (priority === "Low") return "border-zinc-400/20 bg-zinc-400/10 text-zinc-300";
  return "border-cyan-400/25 bg-cyan-400/10 text-cyan-200";
}
