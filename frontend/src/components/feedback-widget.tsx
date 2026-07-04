import { type SyntheticEvent, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "wouter";
import { AlertTriangle, CheckCircle2, MessageSquarePlus, Paperclip, Send, X } from "lucide-react";
import { Button, Card, Input, SectionLabel } from "@/components/ui";
import { createFeedback, uploadFeedbackAttachment } from "@/lib/feedback-api";
import { FEEDBACK_PRIORITIES, FEEDBACK_TYPES, type FeedbackPriority, type FeedbackType } from "@/lib/feedback-contracts";
import { collectFeedbackContext, titleFromBody } from "@/lib/feedback-context";
import { cn } from "@/lib/cn";

const MAX_ATTACHMENTS = 6;

function stopViewerControls(e: SyntheticEvent) {
  e.stopPropagation();
}

export function FeedbackWidget() {
  const qc = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [open, setOpen] = useState(false);
  const [body, setBody] = useState("");
  const [title, setTitle] = useState("");
  const [feedbackType, setFeedbackType] = useState<FeedbackType>("Comment");
  const [priority, setPriority] = useState<FeedbackPriority>("Medium");
  const [files, setFiles] = useState<File[]>([]);
  const [successId, setSuccessId] = useState<string | number | null>(null);
  const [uploadNote, setUploadNote] = useState<string | null>(null);

  const submit = useMutation({
    mutationFn: async () => {
      const details = body.trim();
      const context = await collectFeedbackContext();
      const item = await createFeedback({
        title: title.trim() || titleFromBody(details),
        body: details,
        feedback_type: feedbackType,
        priority,
        status: "New",
        page_url: context.route.url,
        page_path: context.route.path,
        page_tab: context.route.tab,
        component_label: context.ui_state.component_label,
        context_json: context,
      });
      const failures: string[] = [];
      await Promise.all(
        files.map((file) =>
          uploadFeedbackAttachment(item.id, file).catch(() => {
            failures.push(file.name);
          }),
        ),
      );
      return { item, failures };
    },
    onSuccess: ({ item, failures }) => {
      setSuccessId(item.id);
      setBody("");
      setTitle("");
      setFiles([]);
      setUploadNote(failures.length ? `Created, but ${failures.length} attachment upload failed.` : null);
      qc.invalidateQueries({ queryKey: ["feedback"] });
    },
  });

  function addFiles(next: FileList | File[]) {
    const incoming = Array.from(next).filter((file) => file.size > 0);
    setFiles((current) => {
      const merged = [...current, ...incoming];
      const deduped = merged.filter(
        (file, index) => merged.findIndex((f) => f.name === file.name && f.size === file.size && f.lastModified === file.lastModified) === index,
      );
      return deduped.slice(0, MAX_ATTACHMENTS);
    });
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => {
          setOpen(true);
          setSuccessId(null);
        }}
        className="fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-full border border-cyan-300/30 bg-[#07131b]/90 px-4 py-3 text-sm font-bold text-cyan-100 shadow-[0_12px_50px_rgba(34,211,238,0.22)] backdrop-blur transition hover:border-cyan-200/60 hover:bg-[#09202c]"
      >
        <MessageSquarePlus className="h-4 w-4" />
        Feedback
      </button>
    );
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-end bg-black/30 p-3 backdrop-blur-sm sm:p-5"
      onKeyDownCapture={stopViewerControls}
      onKeyUpCapture={stopViewerControls}
      onKeyPressCapture={stopViewerControls}
      onPointerDownCapture={stopViewerControls}
      onPointerMoveCapture={stopViewerControls}
      onPointerUpCapture={stopViewerControls}
      onWheelCapture={stopViewerControls}
      onPasteCapture={stopViewerControls}
    >
      <Card className="w-full max-w-lg border-cyan-300/20 bg-[#071018]/95 p-4 shadow-2xl">
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <SectionLabel>Send Splatlab Feedback</SectionLabel>
            <h2 className="mt-1 text-lg font-black text-white">Capture what happened</h2>
            <p className="mt-0.5 text-xs text-zinc-400">
              Required: details. Context is captured safely without cookies, localStorage, headers, or request bodies.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="rounded-full p-1.5 text-zinc-400 transition hover:bg-white/10 hover:text-zinc-100"
            aria-label="Close feedback"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {successId ? (
          <div className="rounded-2xl border border-emerald-500/25 bg-emerald-500/10 p-4">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-300" />
              <div>
                <p className="font-semibold text-emerald-100">Feedback #{successId} created.</p>
                {uploadNote && <p className="mt-1 text-sm text-amber-200">{uploadNote}</p>}
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button size="sm" onClick={() => setSuccessId(null)}>
                    Add another
                  </Button>
                  <Link
                    href="/feedback"
                    className="inline-flex h-8 items-center justify-center rounded-xl border border-white/15 bg-white/5 px-3 text-xs font-semibold text-zinc-200 hover:bg-white/10"
                  >
                    Open feedback page
                  </Link>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <form
            className="space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              if (body.trim()) submit.mutate();
            }}
          >
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-zinc-300">What should Codex know?</span>
              <textarea
                value={body}
                onChange={(e) => setBody(e.target.value)}
                placeholder="Rough notes are fine: what you did, what looked wrong, what you expected..."
                rows={5}
                className="w-full rounded-2xl border border-white/12 bg-white/5 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-cyan-400/40 focus:outline-none"
                autoFocus
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-zinc-300">Title (optional)</span>
              <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Auto-filled from details if blank" />
            </label>

            <div className="grid gap-2 sm:grid-cols-2">
              <Select label="Type" value={feedbackType} onChange={(v) => setFeedbackType(v as FeedbackType)} options={FEEDBACK_TYPES} />
              <Select label="Priority" value={priority} onChange={(v) => setPriority(v as FeedbackPriority)} options={FEEDBACK_PRIORITIES} />
            </div>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                addFiles(e.dataTransfer.files);
              }}
              onPaste={(e) => addFiles(e.clipboardData.files)}
              className="rounded-2xl border border-dashed border-white/15 bg-white/[0.03] p-3"
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                onChange={(e) => e.target.files && addFiles(e.target.files)}
              />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="min-w-0">
                  <p className="flex items-center gap-1.5 text-xs font-semibold text-zinc-200">
                    <Paperclip className="h-3.5 w-3.5" /> Attach screenshots/files
                  </p>
                  <p className="mt-0.5 text-[11px] text-zinc-500">Drop, paste, or pick files after reviewing sensitive content.</p>
                </div>
                <Button type="button" variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
                  Pick files
                </Button>
              </div>
              {files.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {files.map((file) => (
                    <button
                      key={`${file.name}-${file.size}-${file.lastModified}`}
                      type="button"
                      onClick={() => setFiles((current) => current.filter((f) => f !== file))}
                      className="rounded-full border border-white/10 bg-white/5 px-2 py-1 text-[11px] text-zinc-300 hover:border-red-300/40 hover:text-red-200"
                      title="Remove attachment"
                    >
                      {file.name}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {submit.isError && (
              <div className="flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                {submit.error instanceof Error ? submit.error.message : "Could not submit feedback."}
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <Link href="/feedback" className="text-xs font-semibold text-cyan-200 hover:underline">
                Manage feedback
              </Link>
              <Button type="submit" disabled={!body.trim() || submit.isPending}>
                <Send className={cn("h-4 w-4", submit.isPending && "animate-pulse")} />
                {submit.isPending ? "Sending..." : "Send feedback"}
              </Button>
            </div>
          </form>
        )}
      </Card>
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: readonly string[];
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-semibold text-zinc-300">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-9 w-full rounded-xl border border-white/12 bg-[#101822] px-3 text-sm text-zinc-100 focus:border-cyan-400/40 focus:outline-none"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
